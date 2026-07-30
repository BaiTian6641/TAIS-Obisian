"""Muon vs AdamW 短训练对比：同配置 tiny 模型，比较 loss 收敛曲线与吞吐 tok/s。

用法：
  $env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe scripts/muon_vs_adamw_bench.py [--steps 150]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import build_optimizer, chunked_ce

MICRO, ACCUM, SEQ = 8, 2, 512


def tiny_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=256, n_layer=4, n_q_heads=4, n_kv_heads=2, head_dim=64,
        n_v_heads=4, n_qk_heads=2, mlp_hidden=688, max_seq=SEQ, check_0p1b_params=False,
    )


def run(opt_kind: str, steps: int) -> tuple[list[float], float]:
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(device)
    cfg = {"lr": 1e-3, "weight_decay": 0.1, "optimizer": opt_kind,
           "muon_lr": 0.02, "muon_momentum": 0.95, "muon_ns_steps": 5}
    opt = build_optimizer(model, cfg)
    shards = Shards("data/shards", "train")
    rng = np.random.default_rng(1234)
    model.train()
    losses = []
    # warmup 3 步
    for _ in range(3):
        x, y = shards.get_batch(MICRO, SEQ, device, rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        (loss / ACCUM).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for step in range(steps):
        lr = 1e-3 if opt_kind == "adamw" else None
        for g in opt.param_groups:
            if opt_kind == "adamw":
                g["lr"] = 1e-3
            else:
                if g.get("use_muon"):
                    g["muon_lr"] = 0.02
                else:
                    g["adamw_lr"] = 1e-3
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for _ in range(ACCUM):
            x, y = shards.get_batch(MICRO, SEQ, device, rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x)
                loss = chunked_ce(logits, y)
            (loss / ACCUM).backward()
            acc += loss.item() / ACCUM
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(acc)
    torch.cuda.synchronize()
    tok_s = MICRO * ACCUM * SEQ * steps / (time.time() - t0)
    return losses, tok_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()
    la, ta = run("adamw", args.steps)
    lm, tm = run("muon", args.steps)
    print(f"[adamw] loss {la[0]:.4f}→{la[-1]:.4f}（末10均值 {np.mean(la[-10:]):.4f}），{ta/1e3:.2f}k tok/s")
    print(f"[muon ] loss {lm[0]:.4f}→{lm[-1]:.4f}（末10均值 {np.mean(lm[-10:]):.4f}），{tm/1e3:.2f}k tok/s")
    print(f"[cmp ] Muon 末10 loss {'更低(更好)' if np.mean(lm[-10:])<np.mean(la[-10:]) else '更高'}，"
          f"吞吐比 muon/adamw = {tm/ta*100:.1f}%（Newton-Schulz 开销）")


if __name__ == "__main__":
    main()
