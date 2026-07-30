"""PM-stream 吞吐基准：对比 pm_stream=1（hybrid 单流）vs pm_stream=5（PM-stream）训练 tok/s。

用法：
  $env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe scripts/pm_throughput_bench.py [--steps 60]
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
from tais_obsidian.train import chunked_ce

# 对齐 D-0 pilot 训练配方（0.1B，micro 16×accum 4×seq 1024）
MICRO = 16
ACCUM = 4
SEQ = 1024


def pilot_cfg(pm_stream: int, pm_sk_t_max: int = 20) -> ModelConfig:
    return ModelConfig(pm_stream=pm_stream, pm_sk_t_max=pm_sk_t_max)  # 其余全默认（0.1B G2G2G2A）


def bench(pm_stream: int, steps: int, warmup: int = 12, pm_sk_t_max: int = 20) -> float:
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TaisObsidianForCausalLM(pilot_cfg(pm_stream, pm_sk_t_max)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), eps=1e-8, fused=True)
    shards = Shards("data/shards", "train")
    rng = np.random.default_rng(1234)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    # warmup（不计时，让 cudnn benchmark / autotune 稳定）
    for _ in range(warmup):
        x, y = shards.get_batch(MICRO, SEQ, device, rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        (loss / ACCUM).backward()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        for _ in range(ACCUM):
            x, y = shards.get_batch(MICRO, SEQ, device, rng)
            with torch.autocast("cuda", torch.bfloat16):
                logits, _ = model(x)
                loss = chunked_ce(logits, y)
            (loss / ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = MICRO * ACCUM * SEQ * steps / dt
    mem = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[bench] pm_stream={pm_stream}: {tok_s/1e3:.2f}k tok/s | 峰值显存 {mem:.2f}GB")
    return tok_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--sk_t_max", type=int, default=20, help="PM Sinkhorn 迭代数（吞吐优化）")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "需 CUDA"
    print(f"[bench] 配方 micro {MICRO}×accum {ACCUM}×seq {SEQ}，{args.steps} 步计时，sk_t_max={args.sk_t_max}")
    t1 = bench(1, args.steps)
    t5 = bench(5, args.steps, pm_sk_t_max=args.sk_t_max)
    print(f"[bench] PM-stream/hybrid = {t5/t1*100:.1f}%（pm_stream=5 {t5/1e3:.2f}k vs hybrid {t1/1e3:.2f}k）")


if __name__ == "__main__":
    main()
