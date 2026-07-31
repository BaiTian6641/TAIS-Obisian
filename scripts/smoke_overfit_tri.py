"""冒烟测试：tiny 三级注意力栈（GGGA，"A" 层换 tri）过拟合固定真实 batch，验证全链路。

300 步，断言 final loss（末 10 步均值）< 0.1（对齐 scripts/smoke_overfit.py 判据）。
用法：python scripts/smoke_overfit_tri.py
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


def tiny_cfg() -> ModelConfig:
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
        # 注意力层已统一为三级栈（2026-07 移除 attn_impl 字段），此处仅压实三分支参数：
        # SEQ=512 下滑窗 128、CSA stride-4 top-32（128 条压 32）、HCA 64:1（8 条）
        tri_window=128,
        tri_csa_stride=4,
        tri_csa_topk=32,
        tri_hca_stride=64,
        check_0p1b_params=False,
    )


def main() -> None:
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True  # 同 train.py：fp32 GEMM 走 tensor core
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(0)
    shards = Shards("data/shards", "train")
    rng = np.random.default_rng(1234)  # 固定取同一个真实 batch
    x, y = shards.get_batch(BATCH, SEQ, device, rng)
    print(f"[data] 固定真实 batch: {x.shape}，来自 train shards")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(device)
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
            print(f"[tri] step {step+1:3d} | loss {loss.item():.4f}")
    final = float(np.mean(tail[-10:]))
    print(f"[tri] 300 步完成，final loss（末10步均值）= {final:.4f}，耗时 {time.time()-t0:.0f}s")
    assert final < 0.1, f"[tri] final loss {final:.4f} >= 0.1"
    print("[smoke] 三级栈过拟合冒烟通过")


if __name__ == "__main__":
    main()
