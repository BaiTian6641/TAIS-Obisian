"""TriAttention 独立 LightningIndexer 选择（V4 CSA 式，tri_use_indexer=True）单元测试。

判据：
- use_indexer=True 时前向形状正确（与 NSA 式一致）；
- 因果性（红线）：扰动位置 j 之后的 token，位置 ≤j 的输出逐点不变；
- indexer 投影可训练（分数可微，照 DSA/V4）；
- use_indexer=False（NSA 式）与 True（V4 式）都可用（可选开关，默认 NSA 保持兼容）。
"""
from __future__ import annotations

import torch

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.tri_attention import TriRetrievalAttention

D = 8


def _cfg(use_indexer: bool) -> ModelConfig:
    return ModelConfig(
        vocab_size=64, d_model=32, n_layer=1, n_q_heads=4, n_kv_heads=2, head_dim=D,
        n_v_heads=4, n_qk_heads=2, mlp_hidden=64, max_seq=64, grad_checkpoint=False,
        check_0p1b_params=False, tri_window=8, tri_csa_stride=2,
        tri_csa_topk=4, tri_hca_stride=4, tri_use_indexer=use_indexer,
        tri_index_heads=2, tri_index_dim=8,
    )


def _x() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(2, 24, 32)


def test_use_indexer_forward_shape() -> None:
    m = TriRetrievalAttention(_cfg(use_indexer=True)).eval()
    x = _x()
    with torch.no_grad():
        o, st = m(x)
    assert o.shape == x.shape
    assert "k" in st and "v" in st


def test_use_indexer_causality() -> None:
    """因果性（红线）：扰动位置 j 之后的 token，位置 ≤j 的输出逐点不变。"""
    m = TriRetrievalAttention(_cfg(use_indexer=True)).eval()
    x = _x()
    x2 = x.clone()
    x2[:, 16:] = torch.randn(2, 8, 32)  # 扰动 16 之后
    with torch.no_grad():
        o1, _ = m(x)
        o2, _ = m(x2)
    assert torch.allclose(o1[:, :16], o2[:, :16], atol=1e-5), "因果性被违反：扰动后段影响前段输出"


def test_use_indexer_projections_trainable_via_kl_warmup() -> None:
    """indexer 经 KL warmup 可训练（V4/NSA 设计：选择离散无梯度，indexer 单独 warmup）。

    注意（架构事实）：V4/NSA 的设计是"indexer 用于离散 top-k 选择（无梯度），
    主注意力分数回流到 q/k/压缩器"——故 o.sum().backward() 不给 indexer 梯度是
    **符合设计**的；indexer 的训练走独立 KL warmup（对齐稠密教师，V3.2 warmup 范式）。
    """
    m = TriRetrievalAttention(_cfg(use_indexer=True)).train()
    # KL warmup：用主注意力分数（NSA 式的 imp）作稠密教师，对齐 indexer 分布
    x = _x()
    B, T, _ = x.shape
    # 构造稠密教师分数（[B,n_kv,T,S]，S=Tk//stride）
    S = T // m.csa_comp.stride
    teacher = torch.randn(B, m.n_kv, T, S)
    # 逐 kv 头 warmup：indexer(x_q, x_k) 的分布对齐教师
    q = m.q_norm(m.q_proj(x).view(B, T, m.n_q, m.head_dim))
    k = m.k_norm(m.k_proj(x).view(B, T, m.n_kv, m.head_dim))
    q_nope = q.transpose(1, 2)
    k_nope = k.transpose(1, 2)
    kc, _ = m.csa_comp(k_nope, m.v_proj(x).view(B, T, m.n_kv, m.head_dim).transpose(1, 2))
    kc = m.k_norm(kc)
    rep = m.n_q // m.n_kv
    q_g = q_nope.view(B, m.n_kv, rep, T, m.head_dim).sum(dim=2)
    losses = []
    for h in range(m.n_kv):
        losses.append(m.csa_indexer.kl_warmup_loss(q_g[:, h], kc[:, h], teacher[:, h]))
    loss = torch.stack(losses).mean()
    loss.backward()
    assert m.csa_indexer.q_index.weight.grad is not None, "indexer 应经 KL warmup 可训练"


def test_both_modes_available() -> None:
    """可选开关：NSA 式（use_indexer=False）与 V4 式（True）都能前向。"""
    for flag in (False, True):
        m = TriRetrievalAttention(_cfg(use_indexer=flag)).eval()
        x = _x()
        with torch.no_grad():
            o, _ = m(x)
        assert o.shape == x.shape
    m_nsa = TriRetrievalAttention(_cfg(use_indexer=False))
    m_v4 = TriRetrievalAttention(_cfg(use_indexer=True))
    assert not hasattr(m_nsa, "csa_indexer")
    assert hasattr(m_v4, "csa_indexer")
