"""HRL Indexer 注意力派生初始化（§11.1 近似）单元测试。

判据：
- init_indexer_from_model 从第一个 "A" 层 q_proj 派生归一化检索方向（非随机初始化）；
- 初始化后 score 权重是 q_proj 聚合方向（归一化，norm≈1）；
- 无 "A" 层时 fail-closed 返回 -1（不初始化）；
- init_from_attention_qproj 的聚合正确性（按头均值→归一）。
"""
from __future__ import annotations

import torch

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tais_kernel import HRLIndexer, make_kernel

D = 32


def _tri_model(with_a: bool = True) -> TaisObsidianForCausalLM:
    torch.manual_seed(0)
    pattern = ["G", "G", "G", "A"] if with_a else ["G", "G", "G", "G"]
    cfg = ModelConfig(
        vocab_size=64, d_model=D, n_layer=4, block_pattern=pattern,
        n_q_heads=4, n_kv_heads=2, head_dim=8, n_v_heads=4, n_qk_heads=2,
        mlp_hidden=64, max_seq=16, grad_checkpoint=False, check_0p1b_params=False,
        attn_impl="tri", kernel_enabled=True, kernel_dg_dim=32, kernel_dg_topk=4,
    )
    return TaisObsidianForCausalLM(cfg).eval()


def test_init_indexer_from_model_loads_direction() -> None:
    m = _tri_model(with_a=True)
    before = m.kernel.hrl_indexer.score.weight.clone()
    idx = m.kernel.init_indexer_from_model(m)
    assert idx == 3  # 第一个 "A" 层索引
    after = m.kernel.hrl_indexer.score.weight
    # 初始化改变了权重（从随机→q_proj 聚合方向）
    assert not torch.allclose(before, after), "初始化应改变 Indexer 权重"
    # 方向归一（norm≈1）
    assert abs(float(after.norm()) - 1.0) < 1e-3, f"检索方向应归一，实际 norm={float(after.norm())}"


def test_init_indexer_direction_matches_qproj_aggregation() -> None:
    m = _tri_model(with_a=True)
    a_layer = m.layers[3]
    W = a_layer.mixer.q_proj.weight.detach().float()  # [n_q*hd, d]
    m.kernel.init_indexer_from_model(m)
    # 手动复算聚合方向
    per_head = W.view(4, 8, D).mean(dim=1).mean(dim=0, keepdim=True)
    expected = per_head / (per_head.norm() + 1e-6)
    assert torch.allclose(m.kernel.hrl_indexer.score.weight, expected, atol=1e-5)


def test_init_indexer_no_attention_layer_fail_closed() -> None:
    m = _tri_model(with_a=False)  # 全 GDN，无 "A" 层
    before = m.kernel.hrl_indexer.score.weight.clone()
    idx = m.kernel.init_indexer_from_model(m)
    assert idx == -1, "无注意力层应返回 -1（fail-closed）"
    assert torch.allclose(before, m.kernel.hrl_indexer.score.weight), "fail-closed 时不应改权重"


def test_route_deterministic_after_init() -> None:
    m = _tri_model(with_a=True)
    m.kernel.init_indexer_from_model(m)
    x = torch.randn(1, 5, D)
    s1 = m.kernel.hrl_indexer(x)
    s2 = m.kernel.hrl_indexer(x)
    assert torch.allclose(s1, s2), "初始化后 route 输出应确定（权重固定）"
    assert s1.shape == (1, 5, 1)


def test_init_from_attention_qproj_shape_check() -> None:
    idx = HRLIndexer(D)
    W = torch.randn(4 * 8, D)
    idx.init_from_attention_qproj(W, d_model=D, n_q_heads=4, head_dim=8)
    assert abs(float(idx.score.weight.norm()) - 1.0) < 1e-3
    # 形状不符应 fail-closed（assert）
    import pytest
    with pytest.raises(AssertionError):
        idx.init_from_attention_qproj(torch.randn(10, D), d_model=D, n_q_heads=4, head_dim=8)
