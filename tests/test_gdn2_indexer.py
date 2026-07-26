"""GDN-2 erase/write 解耦（memlayer）+ CSA Indexer（LightningIndexer）单元测试。

判据：
- memlayer.write：tied（默认）向后兼容；erase/write 解耦（独立门）；
- LightningIndexer：打分形状、ReLU 非负（头加权前）、top-k、KL warmup、分数可微。
"""
from __future__ import annotations

import torch

from tais_obsidian.model.hrl_indexer import make_lightning_indexer
from tais_obsidian.model.memlayer import make_memory_layer

KD = 16
D = 32


# ---------------- GDN-2 erase/write 解耦 ----------------

def test_memlayer_tied_backward_compat() -> None:
    """默认（无门）= tied 原版：写 (k,v) 后读出接近 v。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k, v = torch.randn(KD), torch.randn(D)
    kn = torch.nn.functional.normalize(k, dim=-1)
    ml.write(k, v, beta=1.0)
    assert (kn @ ml.state - v).norm() < 1e-3


def test_memlayer_erase_gate_controls_old_readout() -> None:
    """erase_gate（key 侧 b⊙k）：b=0 → kn_eff=0 → 不写入不擦除（状态不变）；b=1 → 正常写。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k, v = torch.randn(KD), torch.randn(D)
    ml.write(k, torch.ones(D) * 5.0, beta=1.0)  # 先写旧值
    state_before = ml.state.clone()
    # erase_gate=0 → kn_eff=0 → outer(kn_eff, ...)=0 → 状态完全不变
    ml.write(k, v, beta=1.0, erase_gate=0.0, write_gate=1.0)
    assert torch.equal(ml.state, state_before), "erase=0 时应不写入（kn_eff=0，状态不变）"
    # erase_gate=1 → 正常写入（读出接近 v）
    ml2 = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    ml2.write(k, v, beta=1.0, erase_gate=1.0, write_gate=1.0)
    kn = torch.nn.functional.normalize(k, dim=-1)
    assert (kn @ ml2.state - v).norm() < 1e-3


def test_memlayer_write_gate_controls_commit() -> None:
    """write_gate=0：不承诺新值（残差=0，状态不变）。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k, v = torch.randn(KD), torch.randn(D)
    ml.write(k, v, beta=1.0, erase_gate=1.0, write_gate=0.0)
    assert ml.state.abs().sum() == 0, "write_gate=0 时不承诺新值，状态应保持为零"


def test_memlayer_decoupled_erase_write_vector_gates() -> None:
    """向量门：write 只承诺部分 value 坐标（承诺坐标读出≈v，未承诺≈0）。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k, v = torch.randn(KD), torch.randn(D)
    w = torch.zeros(D); w[: D // 2] = 1.0      # 只承诺一半 value 坐标
    ml.write(k, v, beta=1.0, erase_gate=1.0, write_gate=w)
    kn = torch.nn.functional.normalize(k, dim=-1)
    readback = kn @ ml.state
    # 承诺的坐标应接近 v，未承诺的坐标应接近 0
    assert (readback[: D // 2] - v[: D // 2]).norm() < 1e-2
    assert readback[D // 2 :].abs().max() < 1e-2


# ---------------- LightningIndexer ----------------

def test_indexer_score_shape() -> None:
    idx = make_lightning_indexer(D, n_heads=4, d_index=16)
    xq, xk = torch.randn(2, 5, D), torch.randn(2, 9, D)
    s = idx(xq, xk)
    assert s.shape == (2, 5, 9)


def test_indexer_topk() -> None:
    idx = make_lightning_indexer(D, n_heads=4, d_index=16)
    xq, xk = torch.randn(2, 5, D), torch.randn(2, 9, D)
    scores, indices = idx.topk_indices(xq, xk, k=3)
    assert scores.shape == (2, 5, 3) and indices.shape == (2, 5, 3)
    # top-k 分数降序
    assert (scores[..., :-1] >= scores[..., 1:]).all()


def test_indexer_scores_differentiable() -> None:
    """分数可微（top-k 离散无梯度，但分数本身可反传到 q/k 投影——DSA/PEER 原文）。"""
    idx = make_lightning_indexer(D, n_heads=4, d_index=16)
    xq, xk = torch.randn(1, 3, D), torch.randn(1, 4, D)
    s = idx(xq, xk)
    s.sum().backward()
    assert idx.q_index.weight.grad is not None
    assert idx.k_index.weight.grad is not None


def test_indexer_kl_warmup() -> None:
    """KL warmup：indexer 分布对齐稠密教师分布，损失有限且可反传。"""
    idx = make_lightning_indexer(D, n_heads=4, d_index=16)
    xq, xk = torch.randn(2, 4, D), torch.randn(2, 6, D)
    teacher = torch.randn(2, 4, 6)  # 稠密教师分数
    loss = idx.kl_warmup_loss(xq, xk, teacher)
    assert torch.isfinite(loss)
    loss.backward()
    assert idx.q_index.weight.grad is not None
