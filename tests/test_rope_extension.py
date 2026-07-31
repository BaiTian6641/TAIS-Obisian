"""RoPE 上下文扩充测试（fb1 P1，2026-07-31）：缓存形状/scaling 数值/断点兼容/>1024 前向与增量。

a 缓存形状与旧构造逐 bit 一致（none 且 ≤8192 走 legacy fp32 路径）；
b YaRN 数值：高频维精确不动（γ=1）、低频维插值 1/s、单调有界、mscale 公式；
c RoPE 相对性：scaling 变更后滑窗分数仍只依赖相对距离（位移不变性）——架构扩窗安全性的
  核心论据（滑窗分支 RoPE 负载只承载 ≤tri_window 的相对位置）；
d >1024 整段前向 + 增量生成不越界（旧硬限解除），整段 vs 增量一致；
e 断点兼容：旧 config.json（无 rope_* 字段）from_json 回填默认、行为不变；
f extend_context API + save/load 往返（新字段持久化、缓存重建一致）。
用法：python tests/test_rope_extension.py
"""
from __future__ import annotations

import json
import math
import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tri_attention import (
    TriRetrievalAttention,
    _build_rope_cache,
    _yarn_scaled_inv_freq,
)


def tiny_cfg(**kw) -> ModelConfig:
    base = dict(
        vocab_size=512, d_model=256, n_layer=4, n_q_heads=4, n_kv_heads=2, head_dim=64,
        n_v_heads=4, n_qk_heads=2, mlp_hidden=688, max_seq=128,
        tri_window=32, tri_csa_stride=4, tri_csa_topk=8, tri_hca_stride=16,
        check_0p1b_params=False,
    )
    base.update(kw)
    return ModelConfig(**base)


def _legacy_cache(head_dim: int, max_seq: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """2026-07-31 前版本的 RoPE 缓存构造（逐 bit 对照基准）。"""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def check_cache_shape(device: str) -> None:
    """a) 缓存形状 + legacy 路径逐 bit 一致 + >8192 行走 fp64 构造（大位置辐角精度）。"""
    cfg = tiny_cfg(max_seq=256)
    cos, sin, mscale = _build_rope_cache(cfg, 64)
    assert cos.shape == (256, 32) and sin.shape == (256, 32), (cos.shape, sin.shape)
    assert mscale == 1.0
    ref_cos, ref_sin = _legacy_cache(64, 256, cfg.rope_theta)
    assert torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin), "none 且 ≤8192 应与旧构造逐 bit 一致"
    # 16384 行（>8192）：fp64→fp32 路径，与 fp64 参考 1e-6 内一致；与 fp32 旧式相比
    # 高位置行差异可测（fp32 辐角误差），证明 fp64 构造生效
    cfg2 = tiny_cfg(max_seq=16384)
    cos2, _, _ = _build_rope_cache(cfg2, 64)
    inv = 1.0 / (cfg2.rope_theta ** (torch.arange(0, 64, 2).float() / 64))
    ref64 = torch.outer(torch.arange(16384, dtype=torch.float64), inv.to(torch.float64)).cos().float()
    assert (cos2 - ref64).abs().max().item() == 0.0
    print("[a] 缓存形状/legacy 逐 bit/fp64 长缓存路径通过")


def check_yarn_numerics(device: str) -> None:
    """b) YaRN 逐维 ramp：高频维精确不动、低频维插值 1/s、单调有界、mscale、短窗相位报告。"""
    theta, D, L, s = 10000.0, 64, 1024, 4.0
    inv = 1.0 / (theta ** (torch.arange(0, D, 2).float() / D))
    inv_s, mscale = _yarn_scaled_inv_freq(inv, s, L)
    lam = 2 * math.pi / inv
    r = L / lam
    gamma = ((r - 1.0) / 31.0).clamp(0.0, 1.0)
    untouched = gamma == 1.0
    full_interp = gamma == 0.0
    # 高频维（γ=1）：精确不动（逐 bit）
    assert untouched.sum().item() >= 4, f"高频不动维过少：{untouched.sum()}"
    assert torch.equal(inv_s[untouched], inv[untouched]), "γ=1 维必须精确不变（短上下文局部区分红线）"
    # 低频维（γ=0）：全插值 1/s
    if full_interp.any():
        d = (inv_s[full_interp] - inv[full_interp] / s).abs().max().item()
        assert d < 1e-9, d
    # 全体单调有界：inv/s ≤ inv' ≤ inv（插值无跳变无越界）
    assert (inv_s >= inv / s - 1e-12).all() and (inv_s <= inv + 1e-12).all()
    assert mscale == 0.1 * math.log(s) + 1.0
    # 短窗相位扰动报告（诚实记录：γ<1 的中频维在 window=512 距离上有可测相位偏移，
    # 这正是"训练内 YaRN"需渐进微调吸收的二阶扰动；高频局部区分维逐 bit 不动）
    d_win = 512
    phase_shift = (d_win * (inv - inv_s)).abs()
    n_exact = int(untouched.sum())
    print(f"[b] YaRN s={s} L={L}：{n_exact}/{D // 2} 维精确不动，mscale={mscale:.4f}，"
          f"window={d_win} 内最大相位偏移 {phase_shift.max().item():.3f} rad"
          f"（受影响维均值 {phase_shift[~untouched].mean().item():.3f}）")


def check_relative_invariance(device: str) -> None:
    """c) RoPE 相对性（none 与 yarn 均保持）：同一窗口内容置于不同绝对位置，滑窗输出一致。

    即 q·R(i)·R(j)ᵀ·k = q·R(i−j)·k：滑窗分支注意力只依赖相对距离 → 扩窗（含 YaRN 只改
    inv_freq 表、不改相对性结构）对滑窗分支是安全的；>1024 的适配负担在 GDN 状态/NoPE 分支，
    由渐进微调承担。
    """
    for scaling, scale in (("none", 1.0), ("yarn", 4.0)):
        torch.manual_seed(0)
        cfg = tiny_cfg(max_seq=512, rope_scaling=scaling, rope_scale=scale,
                       rope_original_max_seq=128)
        tri = TriRetrievalAttention(cfg).to(device).eval()
        T, W = 64, cfg.tri_window  # 窗口 32，取后段 t≥W 的位置其窗口内全是 x 自身内容
        torch.manual_seed(1)
        x = torch.randn(1, T, cfg.d_model, device=device)
        prefix = torch.randn(1, W, cfg.d_model, device=device)
        with torch.no_grad():
            aux1: dict = {}
            tri(x, None, 0, aux1)
            aux2: dict = {}
            tri(torch.cat([prefix, x], dim=1), None, 0, aux2)
        # x 位置 t≥W 时其窗口（[i−W+1, i]）内全是 x 自身内容：aux1 的位置 W..T-1
        # 与 aux2（前缀 W + x）的位置 2W..2W+T-W-1 应一致
        d = (aux1["o_win"][0, :, W:, :] - aux2["o_win"][0, :, 2 * W :, :]).abs().max().item()
        assert d < 1e-5, (scaling, d)
        print(f"[c] scaling={scaling}: 位移不变性 max diff {d:.2e}（滑窗只依赖相对距离）")


def check_beyond_1024(device: str) -> None:
    """d) >1024 整段前向 + 增量生成不越界（旧硬限解除），整段 vs 增量一致。"""
    torch.manual_seed(0)
    cfg = tiny_cfg(max_seq=2048)
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    T = 1536  # > 1024 旧硬限
    ids = torch.randint(0, 512, (1, T), device=device)
    with torch.no_grad():
        logits_full, cache_full = model(ids)
        assert logits_full.shape == (1, T, 512) and cache_full["pos"] == T
        # 增量：prefill 1200（已超 1024）+ 逐 token 跨过 1536
        logits_pre, cache = model(ids[:, :1200])
        steps = [logits_pre]
        for t in range(1200, T):
            lg, cache = model(ids[:, t : t + 1], cache)
            steps.append(lg)
        d = (logits_full - torch.cat(steps, dim=1)).abs().max().item()
        assert d < 1e-4, d
        # yarn 配置下 >1024 同样工作
        cfg2 = tiny_cfg(max_seq=2048, rope_scaling="yarn", rope_scale=2.0,
                        rope_original_max_seq=1024)
        model2 = TaisObsidianForCausalLM(cfg2).to(device).eval()
        logits2, _ = model2(ids)
        assert torch.isfinite(logits2).all()
    print(f"[d] >1024 前向/增量不越界，整段 vs 增量 max diff {d:.2e}；yarn 配置前向有限")


def check_old_checkpoint_compat(device: str) -> None:
    """e) 旧 config.json（无 rope_* 字段）from_json 回填默认值，行为与旧版一致。"""
    old_dict = {
        "vocab_size": 512, "d_model": 256, "n_layer": 4,
        "block_pattern": ["G2", "G2", "G2", "A"],
        "n_q_heads": 4, "n_kv_heads": 2, "head_dim": 64, "rope_theta": 10000.0,
        "n_v_heads": 4, "n_qk_heads": 2, "conv_kernel": 4, "mlp_hidden": 688,
        "rms_eps": 1e-06, "max_seq": 128, "grad_checkpoint": True,
        "check_0p1b_params": False, "pm_stream": 1, "pm_constrain": True,
        "tri_window": 32, "tri_csa_stride": 4, "tri_csa_topk": 8, "tri_hca_stride": 16,
        "tri_use_indexer": True, "tri_index_heads": 4, "tri_index_dim": 32,
        "kernel_enabled": False, "kernel_dg_dim": 256, "kernel_dg_topk": 32,
        "kernel_sense_layers": [],
        # 注意：无 rope_scaling/rope_scale/rope_original_max_seq（2026-07-31 前格式）
    }
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "config.json"
        p.write_text(json.dumps(old_dict), encoding="utf-8")
        cfg = ModelConfig.from_json(p)
    assert cfg.rope_scaling == "none" and cfg.rope_scale == 1.0 and cfg.rope_original_max_seq is None
    cos, sin, mscale = _build_rope_cache(cfg, 64)
    ref_cos, ref_sin = _legacy_cache(64, 128, 10000.0)
    assert mscale == 1.0 and torch.equal(cos, ref_cos) and torch.equal(sin, ref_sin)
    print("[e] 旧 config.json（无 rope_* 字段）回填默认 none/1.0/None，缓存与旧版逐 bit 一致")


def check_extend_context_api(device: str) -> None:
    """f) extend_context：原地扩窗重建缓存 → save/load 往返字段持久化、缓存一致、长前向可用。"""
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(device).eval()
    tri = model.layers[3].mixer
    assert tri.rope_cos.shape[0] == 128
    model.extend_context(max_seq=4096, rope_scaling="yarn", rope_scale=4.0,
                         rope_original_max_seq=128)
    assert tri.rope_cos.shape == (4096, 32) and tri.rope_mscale > 1.0
    assert model.config.max_seq == 4096 and model.config.rope_scaling == "yarn"
    cos_after = tri.rope_cos.clone()
    ids = torch.randint(0, 512, (1, 2048), device=device)
    with torch.no_grad():
        logits, cache = model(ids)
        assert torch.isfinite(logits).all() and cache["pos"] == 2048
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        saved = json.loads((Path(tmp) / "config.json").read_text(encoding="utf-8"))
        assert saved["max_seq"] == 4096 and saved["rope_scaling"] == "yarn"
        assert saved["rope_scale"] == 4.0 and saved["rope_original_max_seq"] == 128
        model2 = TaisObsidianForCausalLM.from_pretrained(tmp, device)
        tri2 = model2.layers[3].mixer
        assert tri2.rope_cos.shape == (4096, 32)
        assert torch.equal(tri2.rope_cos, cos_after), "save/load 后 RoPE 缓存应一致"
    print("[f] extend_context 扩窗 + save/load 往返（字段持久化、缓存一致、2048 前向有限）通过")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_cache_shape(device)
    check_yarn_numerics(device)
    check_relative_invariance(device)
    check_beyond_1024(device)
    check_old_checkpoint_compat(device)
    check_extend_context_api(device)
    print("test_rope_extension 全部通过。")


def test_cache_shape() -> None:
    check_cache_shape("cuda" if torch.cuda.is_available() else "cpu")


def test_yarn_numerics() -> None:
    check_yarn_numerics("cuda" if torch.cuda.is_available() else "cpu")


def test_relative_invariance() -> None:
    check_relative_invariance("cuda" if torch.cuda.is_available() else "cpu")


def test_beyond_1024() -> None:
    check_beyond_1024("cuda" if torch.cuda.is_available() else "cpu")


def test_old_checkpoint_compat() -> None:
    check_old_checkpoint_compat("cuda" if torch.cuda.is_available() else "cpu")


def test_extend_context_api() -> None:
    check_extend_context_api("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
