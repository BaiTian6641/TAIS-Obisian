"""PM-stream 算子级微基准：read/write/sinkhorn 在 fp64 vs fp32 + t_max=20 vs 早停 的延迟对比。

直接定位瓶颈收益（不进整模型），快速量化三管齐下（fp32/迭代裁剪/einsum）效果。
用法：$env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe scripts/pm_op_microbench.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.pmstream import PMStreamMix, sinkhorn_knopp

DEVICE = "cuda"
B, T, N, D = 16, 1024, 5, 768  # 对齐 pilot micro_batch=16×seq=1024，n=5，d=768


def timeit(fn, iters: int = 200, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3  # ms


def main() -> None:
    torch.manual_seed(0)
    S = torch.randn(B, T, N, D, device=DEVICE, dtype=torch.bfloat16)
    mix = PMStreamMix(D, N).to(DEVICE)
    with torch.no_grad():
        h_pre, h_post, h_res = mix(S.float())
    print(f"[shape] S {tuple(S.shape)}，h_res {tuple(h_res.shape)}")

    # --- read：fp64（旧）vs fp32（新） ---
    def read_fp64():
        return torch.einsum("btn,btnd->btd", h_pre.double(), S.double()).to(S.dtype)

    def read_fp32():
        return torch.einsum("btn,btnd->btd", h_pre.float(), S.float()).to(S.dtype)

    t_r64, t_r32 = timeit(read_fp64), timeit(read_fp32)
    print(f"[read ] fp64 {t_r64:.3f}ms → fp32 {t_r32:.3f}ms（×{t_r64/t_r32:.2f} 加速）")

    # --- write：fp64（旧）vs fp32（新） ---
    m = torch.randn(B, T, D, device=DEVICE, dtype=torch.bfloat16)

    def write_fp64():
        o = torch.einsum("btjk,btkd->btjd", h_res.double(), S.double())
        o = o + h_post.double().unsqueeze(-1) * m.double().unsqueeze(2)
        return o.to(S.dtype)

    def write_fp32():
        o = torch.einsum("btjk,btkd->btjd", h_res.float(), S.float())
        o = o + h_post.float().unsqueeze(-1) * m.float().unsqueeze(2)
        return o.to(S.dtype)

    t_w64, t_w32 = timeit(write_fp64), timeit(write_fp32)
    print(f"[write] fp64 {t_w64:.3f}ms → fp32 {t_w32:.3f}ms（×{t_w64/t_w32:.2f} 加速）")

    # --- sinkhorn：t_max=20（旧）vs 早停 tol=1e-3（新） ---
    def sk_20():
        return sinkhorn_knopp(h_res, t_max=20, tol=0.0)

    def sk_early():
        return sinkhorn_knopp(h_res, t_max=20, tol=1e-3)

    t_sk20, t_ske = timeit(sk_20), timeit(sk_early)
    print(f"[sink ] t_max=20 {t_sk20:.3f}ms → 早停(1e-3) {t_ske:.3f}ms（×{t_sk20/t_ske:.2f} 加速）")

    # 双随机精度对比（早停 vs 固定 20 次）
    m20 = sk_20()
    me = sk_early()
    dev20 = max((m20.sum(-1) - 1).abs().max().item(), (m20.sum(-2) - 1).abs().max().item())
    deve = max((me.sum(-1) - 1).abs().max().item(), (me.sum(-2) - 1).abs().max().item())
    diff = (m20 - me).abs().max().item()
    print(f"[prec ] 双随机偏差 固定20={dev20:.2e} vs 早停={deve:.2e}；两者矩阵差 {diff:.2e}")

    # --- 单子层 read+write+sinkhorn 合计（旧 vs 新） ---
    old_total = t_r64 + t_w64 + t_sk20
    new_total = t_r32 + t_w32 + t_ske
    print(f"[total] 单子层 旧 {old_total:.3f}ms → 新 {new_total:.3f}ms（×{old_total/new_total:.2f} 加速）")
    print(f"        12层×2子层=24 次调用：旧 {old_total*24:.2f}ms → 新 {new_total*24:.2f}ms / 前向")


if __name__ == "__main__":
    main()
