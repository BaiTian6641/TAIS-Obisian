"""TAIS 内核骨架（M1）单元测试：结构、sense/route/inject、监测-执行分置、载体能力边界。

判据（与接口与实现计划 v1.0 / 子系统架构规格 Part B 对齐）：
- 前向不崩、PM 读写通（M1 退出标准）；
- sense() 零副作用（只读，不改输入）；
- route() 的 DG 稀疏 key 严格 top-k 稀疏（防碰撞的去相关，§15.2）；
- inject() 位置不变向量=单次加法（steer 行为），token 寻址载体 fail-closed 拒绝（不静默）；
- BlockPayload.factual_recall 由载体类型推导、不可伪造（向量不能当事实用）。
"""
from __future__ import annotations

import pytest
import torch

from tais_obsidian.model.tais_kernel import (
    ADDRESSED_KINDS,
    VECTOR_KINDS,
    BlockPayload,
    TAISKernel,
    make_kernel,
)

D = 32          # 小 d_model，秒级 CPU 测试
DG_DIM = 64
DG_TOPK = 8


@pytest.fixture
def kernel() -> TAISKernel:
    torch.manual_seed(0)
    return make_kernel(D, dg_dim=DG_DIM, dg_topk=DG_TOPK)


def test_kernel_structure(kernel: TAISKernel) -> None:
    """内核聚合 KAL(L1/L2) + HRL Indexer + DG + 侧信道头簇；参数随 state_dict 存取。"""
    names = {n for n, _ in kernel.named_parameters()}
    assert any("kal_l1" in n for n in names)
    assert any("kal_l2" in n for n in names)
    assert any("hrl_indexer" in n for n in names)
    assert any("dg_proj" in n for n in names)
    assert any("side_heads" in n for n in names)
    sd = kernel.state_dict()
    assert sd, "state_dict 应非空（checkpoint 内生可存取）"


def test_sense_shapes_and_zero_side_effect(kernel: TAISKernel) -> None:
    """sense() 输出形状正确且零副作用（只读，不改输入张量）。"""
    pm = torch.randn(2, 7, D)
    pm_copy = pm.clone()
    out = kernel.sense(pm)
    assert out.pik_logits.shape == (2, 7, 3)      # L1 三态
    assert out.affect_logits.shape == (2, 7, 2)   # L2 情感（valence/arousal）
    assert out.write_salience.shape == (2, 7, 1)
    assert out.conflict_logit.shape == (2, 7, 1)
    # 零副作用：输入张量不被修改（监测=只读，设计 §8.2 读 hidden state 成本≈0）
    assert torch.equal(pm, pm_copy), "sense() 修改了输入张量（违反只读监测红线）"


def test_route_sparse_key_topk(kernel: TAISKernel) -> None:
    """route() 的 DG 稀疏 key 严格 top-k 稀疏（DG 模式分离防碰撞，潜空间去相关）。"""
    q = torch.randn(3, 5, D)
    out = kernel.route(q)
    assert out.sparse_key.shape == (3, 5, DG_DIM)
    assert out.score.shape == (3, 5, 1)
    # 每行非零元素数 ≤ topk（稀疏性）
    nnz = (out.sparse_key != 0).sum(dim=-1)
    assert int(nnz.max()) <= DG_TOPK, f"非零元素 {int(nnz.max())} 超 topk={DG_TOPK}"


def test_inject_vector_adds_once(kernel: TAISKernel) -> None:
    """inject() 位置不变向量 = PM-stream 单次加法（steer 行为，一次加法零上下文开支）。"""
    pm = torch.zeros(1, 4, D)
    v = torch.ones(D)
    p = BlockPayload(block_id="b1", compiled_kind="icv", vector=v)
    out = kernel.inject(pm, [p], alphas=[2.0])
    # 结果 = pm + 2.0 * v（单次加法，位置不变偏移）
    expected = pm + 2.0 * v.view(1, 1, D)
    assert torch.allclose(out, expected), "向量注入应为单次加法（位置不变偏移）"
    # 向量载体 factual_recall=False（只能 steer 行为，不能事实召回）
    assert p.factual_recall is False


@pytest.mark.parametrize("kind", sorted(ADDRESSED_KINDS))
def test_inject_addressed_kinds_fail_closed(kernel: TAISKernel, kind: str) -> None:
    """token 寻址载体（kv/mem_entry/gist 等）应由 M5 注入路径处理，本骨架 fail-closed 拒绝。"""
    p = BlockPayload(block_id="b", compiled_kind=kind)
    assert p.factual_recall is True, f"{kind} 应为 token 寻址（可事实召回）"
    with pytest.raises(NotImplementedError):
        kernel.inject(torch.zeros(1, 1, D), [p])


def test_blockpayload_factual_recall_not_forgeable() -> None:
    """factual_recall 由载体类型推导，不可由调用方伪造（防向量当事实用红线）。"""
    for kind in VECTOR_KINDS:
        assert BlockPayload(block_id="x", compiled_kind=kind).factual_recall is False
    for kind in ADDRESSED_KINDS:
        assert BlockPayload(block_id="x", compiled_kind=kind).factual_recall is True
    with pytest.raises(ValueError):
        BlockPayload(block_id="x", compiled_kind="not_a_kind")


def test_monitoring_execution_separation(kernel: TAISKernel) -> None:
    """监测/执行分置：sense() 纯读、inject() 纯写——两通道独立，互不改对方语义。"""
    pm_gdn = torch.randn(1, 3, D)   # GDN 输出层 PM-stream（监测读点）
    pm_csa = torch.randn(1, 3, D)   # CSA 残差前 PM-stream（执行写点）
    s = kernel.sense(pm_gdn)
    # sense 不返回任何 PM-stream 修改（只读信号）
    assert not hasattr(s, "pm_out"), "sense() 不应返回 PM-stream 修改（只读）"
    v = torch.zeros(D)
    out = kernel.inject(pm_csa, [BlockPayload(block_id="b", compiled_kind="steering", vector=v)])
    # inject 返回修改后 PM-stream，且不触碰 sense 的输出
    assert out.shape == pm_csa.shape
    assert s.pik_logits.shape == (1, 3, 3)


def test_gradient_flow(kernel: TAISKernel) -> None:
    """内生头参数可训练（前向可微）——T2/T3 训练路径的前置条件。"""
    pm = torch.randn(1, 2, D, requires_grad=False)
    out = kernel.sense(pm)
    loss = out.pik_logits.sum() + out.affect_logits.sum() + out.write_salience.sum()
    loss.backward()
    for n, p in kernel.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"{n} 梯度含非有限值"
    # 至少一部分头有梯度（内生头可训）
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for _, p in kernel.named_parameters())
