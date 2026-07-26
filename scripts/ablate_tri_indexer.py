"""tri_use_indexer 消融（V4 式独立 indexer vs NSA 式复用分数，0.1B 短跑对比）。

设计：同数据/种子/步数，对比三级栈两种 CSA 选择机制——
- NSA 式（tri_use_indexer=False，基线）：复用压缩注意力分数 Softmax(q·K̃) 选 top-k；
- V4 式（tri_use_indexer=True）：独立 LightningIndexer 在压缩条目上打分选 top-k（DeepSeek V4 正式路径）。

对比指标：val loss（越低越好）、训练吞吐 tok/s、峰值显存 GB、参数量。
诚实标注：短跑（默认 300 步）为早期信号，非架构级结论；正式 2000 步消融在主代理执行。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/ablate_tri_indexer.py [--steps 300]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig  # noqa: E402
from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.train import build_optimizer, chunked_ce, lr_at  # noqa: E402

DEVICE = "cuda"


def build(use_indexer: bool, seed: int = 42) -> TaisObsidianForCausalLM:
    torch.manual_seed(seed)
    cfg = ModelConfig(
        tri_use_indexer=use_indexer,
        tri_index_heads=4,
        tri_index_dim=32,
    )
    return TaisObsidianForCausalLM(cfg).to(DEVICE)


def run_variant(use_indexer: bool, steps: int, seq_len: int = 512, micro: int = 8) -> dict:
    """跑一个变体（NSA 或 V4），返回 {val_loss, tok_per_s, peak_gb, n_params}。"""
    tag = "V4-indexer" if use_indexer else "NSA-reuse"
    model = build(use_indexer)
    n_params = sum(p.numel() for p in model.parameters())
    cfg = dict(lr=1e-3, warmup=50, decay_frac=0.2, max_steps=steps, weight_decay=0.1,
               grad_clip=1.0, grad_accum=1, micro_batch=micro, seq_len=seq_len,
               val_batches=4)
    opt = build_optimizer(model, cfg)
    train_shards = Shards(ROOT / "data" / "shards", "train")
    val_shards = Shards(ROOT / "data" / "shards", "val")
    rng = np.random.default_rng(42)

    model.train()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tokens = 0
    model.train()
    for step in range(steps):
        lr = lr_at(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        x, y = train_shards.get_batch(micro, seq_len, DEVICE, rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        tokens += micro * seq_len
    train_s = time.time() - t0
    tok_per_s = tokens / train_s
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # val
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(cfg["val_batches"]):
            x, y = val_shards.get_batch(micro, seq_len, DEVICE, rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x)
                losses.append(chunked_ce(logits, y).item())
    val_loss = float(np.mean(losses))
    print(f"[{tag}] steps={steps} val_loss={val_loss:.4f} tok/s={tok_per_s:.0f} "
          f"peak={peak_gb:.2f}GB params={n_params/1e6:.2f}M")
    return {"val_loss": val_loss, "tok_per_s": tok_per_s, "peak_gb": peak_gb,
            "n_params": n_params, "tag": tag}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"[ablate] tri_use_indexer 消融：NSA-reuse vs V4-indexer，steps={args.steps}")
    nsa = run_variant(False, args.steps)
    v4 = run_variant(True, args.steps)
    d_val = v4["val_loss"] - nsa["val_loss"]
    d_params = (v4["n_params"] - nsa["n_params"]) / 1e6
    print("\n===== 对比 =====")
    print(f"val_loss: NSA {nsa['val_loss']:.4f} → V4 {v4['val_loss']:.4f}（Δ={d_val:+.4f}）")
    print(f"tok/s:    NSA {nsa['tok_per_s']:.0f} → V4 {v4['tok_per_s']:.0f}"
          f"（{(v4['tok_per_s']/nsa['tok_per_s']-1)*100:+.1f}%）")
    print(f"peak GB:  NSA {nsa['peak_gb']:.2f} → V4 {v4['peak_gb']:.2f}")
    print(f"params:   NSA {nsa['n_params']/1e6:.2f}M → V4 {v4['n_params']/1e6:.2f}M（+{d_params:.3f}M）")
    verdict = ("V4-indexer 不劣化（|Δval|<0.02）" if abs(d_val) < 0.02
               else ("V4-indexer 更优" if d_val < 0 else "⚠️ V4-indexer 略劣（短跑早期信号）"))
    print(f"判定：{verdict}（短跑 {args.steps} 步为早期信号，非架构级结论）")
    out = ROOT / "runs" / "ablate_tri_indexer.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"nsa": nsa, "v4": v4, "d_val": d_val, "steps": args.steps},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ablate] 结果写入 {out}")


if __name__ == "__main__":
    main()
