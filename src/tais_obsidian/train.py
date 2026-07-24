"""训练循环：WSD 调度 + AdamW(decay 分组) + bf16 autocast + checkpoint/resume + tensorboard。

标准配方：参数 fp32，autocast(bf16) 前向，grad clip 1.0；
AdamW β=(0.9,0.95) eps=1e-8 wd=0.1（embedding/norm/1D 参数不衰减），fused 若可用；
WSD：warmup 线性升 → 稳定段恒定 → 最后 decay_frac 步数线性降到 0。

用法：
  python -m tais_obsidian.train --config configs/pilot_0p1b.json
  python -m tais_obsidian.train --config configs/pilot_0p1b.json --resume checkpoints/pilot_0p1b/latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM

DEFAULTS = dict(
    run_name="pilot_0p1b",
    seed=42,
    out_dir="checkpoints/pilot_0p1b",
    data_dir="data/shards",
    seq_len=1024,
    max_steps=2000,
    micro_batch=16,
    grad_accum=4,
    lr=1e-3,
    warmup=200,
    decay_frac=0.15,
    weight_decay=0.1,
    grad_clip=1.0,
    val_every=250,
    val_batches=20,
    log_every=20,
    ckpt_every=500,
    attn_only=False,
)


def lr_at(step: int, cfg: dict) -> float:
    """WSD：warmup 线性升到 peak → 恒定 → 末段线性降到 0。"""
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    decay_start = int(cfg["max_steps"] * (1 - cfg["decay_frac"]))
    if step < decay_start:
        return cfg["lr"]
    return cfg["lr"] * (cfg["max_steps"] - step) / (cfg["max_steps"] - decay_start)


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.AdamW:
    """decay 分组：≥2D 且非 embedding 的参数衰减；norm/1D/embedding 不衰减。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    print(f"[opt] AdamW fused={fused}, decay {sum(p.numel() for p in decay)/1e6:.1f}M, "
          f"no_decay {sum(p.numel() for p in no_decay)/1e6:.1f}M")
    return torch.optim.AdamW(
        groups, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=fused
    )


def chunked_ce(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """分块 fp32 cross entropy，避免一次性 fp32 logits 副本撑爆显存。"""
    flat = logits.view(-1, logits.size(-1))
    tgt = targets.view(-1)
    total = None
    for i in range(0, flat.shape[0], chunk):
        part = F.cross_entropy(flat[i : i + chunk].float(), tgt[i : i + chunk], reduction="sum")
        total = part if total is None else total + part
    return total / flat.shape[0]


@torch.no_grad()
def eval_val(model, val: Shards, cfg: dict, device: str, rng: np.random.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg["val_batches"]):
        x, y = val.get_batch(cfg["micro_batch"], cfg["seq_len"], device, rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, model, opt, step: int, cfg: dict, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),  # fp32
            "opt": opt.state_dict(),
            "step": step,
            "train_cfg": cfg,
            "model_cfg": model.config.__dict__,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": rng.bit_generator.state,
            },
        },
        path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="latest.pt 路径；断点续训")
    ap.add_argument("--max_steps", type=int, default=None, help="临时覆盖 max_steps（短跑验证用）")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("max_steps", "run_name", "out_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(f"[cfg] {json.dumps(cfg, ensure_ascii=False)}")

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = "cuda"
    # TF32：GDN 核心算子内部 fp32 GEMM 走 tensor core（训练吞吐 ~1.4x；单元对拍不开启，仍全 fp32）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model_cfg = ModelConfig(attn_only=cfg["attn_only"])
    model = TaisObsidianForCausalLM(model_cfg).to(device)
    opt = build_optimizer(model, cfg)
    train_shards = Shards(cfg["data_dir"], "train")
    val_shards = Shards(cfg["data_dir"], "val")
    print(f"[data] train {train_shards.total/1e6:.1f}M tokens, val {val_shards.total/1e6:.1f}M")

    from torch.utils.tensorboard import SummaryWriter

    out_dir = Path(cfg["out_dir"])
    writer = SummaryWriter(log_dir=f"runs/{cfg['run_name']}")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        torch.set_rng_state(ckpt["rng"]["torch"])
        if ckpt["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(ckpt["rng"]["cuda"])
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng"]["numpy"]
        print(f"[resume] 从 {args.resume} 恢复，step={start_step}")

    # micro_batch OOM 自动降级：16 → 8，grad_accum 翻倍保持 global batch
    oom_checked = False
    step = start_step
    tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
    print(f"[train] global batch {tokens_per_step/1024:.0f}k tokens/step, "
          f"micro {cfg['micro_batch']}×accum {cfg['grad_accum']}×seq {cfg['seq_len']}")

    ema_loss = None
    t_log = time.time()
    while step < cfg["max_steps"]:
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        try:
            for _ in range(cfg["grad_accum"]):
                x, y = train_shards.get_batch(cfg["micro_batch"], cfg["seq_len"], device, rng)
                with torch.autocast("cuda", torch.bfloat16):
                    logits, _ = model(x)
                    loss = chunked_ce(logits, y)
                (loss / cfg["grad_accum"]).backward()
                loss_accum += loss.item() / cfg["grad_accum"]
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
            if "out of memory" not in str(e).lower():
                raise
            if not oom_checked and cfg["micro_batch"] > 8:
                cfg["micro_batch"] //= 2
                cfg["grad_accum"] *= 2
                oom_checked = True
                tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
                try:
                    torch.cuda.empty_cache()
                except torch.AcceleratorError:
                    # CUDA 上下文已被 OOM 污染（kernel 内 OOM 不可恢复），只能重启进程
                    raise RuntimeError(
                        "CUDA OOM 发生在 kernel 内，上下文不可恢复；请以更小 micro_batch 重启"
                    ) from e
                opt = build_optimizer(model, cfg)
                print(f"[oom] micro_batch 降至 {cfg['micro_batch']}，grad_accum 升至 {cfg['grad_accum']}，重试本步")
                continue
            raise
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1

        ema_loss = loss_accum if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_accum
        if step % cfg["log_every"] == 0 or step == 1:
            dt = time.time() - t_log
            n_tok = tokens_per_step * (cfg["log_every"] if step > 1 else 1)
            tok_s = n_tok / dt if step > 1 else tokens_per_step / dt
            mem = torch.cuda.max_memory_allocated() / 1024**3
            print(f"step {step:5d} | loss {loss_accum:.4f} (ema {ema_loss:.4f}) | lr {lr:.2e} | "
                  f"gnorm {grad_norm.item():.2f} | {tok_s/1e3:.1f}k tok/s | mem {mem:.2f}GB")
            writer.add_scalar("train/loss", loss_accum, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", grad_norm.item(), step)
            writer.add_scalar("train/tok_per_s", tok_s, step)
            t_log = time.time()
        if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
            vl = eval_val(model, val_shards, cfg, device, rng)
            print(f"step {step:5d} | val loss {vl:.4f}")
            writer.add_scalar("val/loss", vl, step)
        if step % cfg["ckpt_every"] == 0 or step == cfg["max_steps"]:
            save_checkpoint(out_dir / "latest.pt", model, opt, step, cfg, rng)
            print(f"[ckpt] step {step} → {out_dir/'latest.pt'}")

    # 结束：另存 final bf16 save_pretrained
    model.save_pretrained(out_dir / "final")
    print(f"[done] final 模型（bf16）→ {out_dir/'final'}")
    writer.close()


if __name__ == "__main__":
    main()
