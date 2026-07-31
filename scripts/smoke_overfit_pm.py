"""冒烟测试：tiny PM-stream 配置（pm_stream=5，mHC 多流）过拟合固定真实 batch。

验证 PM 变体的前向/反向/优化器/AMP 全链路：hybrid(GGGA) + pm_stream=5 训 300 步，
断言 final loss（末 10 步均值）< 0.1（与 scripts/smoke_overfit.py 同一判据与数据）。
用法：python scripts/smoke_overfit_pm.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import chunked_ce

STEPS = 300
BATCH = 8
SEQ = 512
LR = 3e-3
WARMUP = 20


def tiny_pm_cfg() -> ModelConfig:
    return ModelConfig(
        d_model=256,
        n_layer=4,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        n_v_heads=4,
        n_qk_heads=2,
        mlp_hidden=688,
        max_seq=SEQ,
        pm_stream=5,  # 4 内容流 + 1 感知-记忆流（mHC 多流残差）
        check_0p1b_params=False,
    )


def main() -> None:
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # 同 train.py：fp32 GEMM 走 tensor core
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    shards = Shards("data/shards", "train")
    rng = np.random.default_rng(1234)  # 固定取同一个真实 batch（与 smoke_overfit.py 一致）
    x, y = shards.get_batch(BATCH, SEQ, device, rng)
    print(f"[data] 固定真实 batch: {x.shape}，来自 train shards")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TaisObsidianForCausalLM(tiny_pm_cfg()).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
    tail: list[float] = []
    t0 = time.time()
    model.train()
    for step in range(STEPS):
        lr = LR * min(1.0, (step + 1) / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tail.append(loss.item())
        if (step + 1) % 50 == 0:
            print(f"[pm5] step {step+1:3d} | loss {loss.item():.4f}")
    final = float(np.mean(tail[-10:]))
    print(f"[pm5] 300 步完成，final loss（末10步均值）= {final:.4f}，耗时 {time.time()-t0:.0f}s")
    assert final < 0.1, f"[pm5] final loss {final:.4f} >= 0.1"
    print("[smoke] PM-stream 变体过拟合通过。")


if __name__ == "__main__":
    main()
