"""HRL 内生（M3）单元测试：Indexer 梯度隔离、CSA indexer 权重初始化、DG 稀疏性。

判据（与接口与实现计划 v1.0 §4.1 / 部件实现详细计划 Part C 对齐）：
- 红线（MoE-RL 教训，§27.3）：辅助损失梯度只进 HRL Indexer，禁止污染主干——
  detach_input=True 时主干 query 的梯度应为零（隔离）；detach_input=False 时透传。
- 设计 §11.1：HRL 块索引器用 CSA indexer 权重初始化（同构）。
- DG 模式分离：稀疏 key 严格 top-k（防碰撞，潜空间去相关 §15.2）。
"""
from __future__ import annotations

import pytest
import torch

from tais_obsidian.model.tais_kernel import DGProjection, HRLIndexer, make_kernel

D = 32


def test_indexer_gradient_isolation() -> None:
    """红线：detach_input=True 时辅助损失梯度只进 Indexer，主干 query 梯度为零。"""
    torch.manual_seed(0)
    idx = HRLIndexer(D)
    # 主干 query：requires_grad 模拟"主干残差流"
    query = torch.randn(2, 4, D, requires_grad=True)
    score = idx(query, detach_input=True)
    score.sum().backward()
    # Indexer 权重有梯度（可训练），但主干 query 梯度为零（隔离）
    assert idx.score.weight.grad is not None and idx.score.weight.grad.abs().sum() > 0
    assert query.grad is None or query.grad.abs().sum() == 0, \
        "detach_input=True 时主干 query 不应收到梯度（梯度隔离红线被违反）"


def test_indexer_gradient_pass_through_when_disabled() -> None:
    """detach_input=False 时梯度透传主干（T3 统一 RL 端到端路径）。"""
    torch.manual_seed(0)
    idx = HRLIndexer(D)
    query = torch.randn(2, 4, D, requires_grad=True)
    score = idx(query, detach_input=False)
    score.sum().backward()
    assert query.grad is not None and query.grad.abs().sum() > 0, \
        "detach_input=False 时主干 query 应收到梯度（T3 端到端 RL）"


def test_indexer_load_from_csa() -> None:
    """设计 §11.1：HRL Indexer 用 CSA indexer 权重初始化（同构，fail-closed 形状校验）。"""
    idx = HRLIndexer(D)
    csa_w = torch.randn(1, D)  # CSA indexer 打分向量 [1, d]
    idx.load_from_csa_indexer(csa_w)
    assert torch.allclose(idx.score.weight, csa_w), "CSA indexer 权重未正确载入"
    # 形状不匹配应 fail-closed
    with pytest.raises((ValueError, RuntimeError)):
        idx.load_from_csa_indexer(torch.randn(1, D + 8))


def test_dg_projection_sparsity() -> None:
    """DG 模式分离：稀疏 key 严格 top-k，且保留 top-k 绝对值最大的激活。"""
    torch.manual_seed(0)
    dg = DGProjection(D, dg_dim=64, topk=8)
    x = torch.randn(2, 3, D)
    key = dg(x)
    nnz = (key != 0).sum(dim=-1)
    assert int(nnz.max()) <= 8
    # 保留的应是绝对值 top-8
    full = dg.proj(x)
    top8 = full.abs().topk(8, dim=-1).values[..., -1:]  # 每行第 8 大绝对值
    kept = key[key != 0].abs()
    assert (kept >= top8.expand_as(full.abs())[key != 0]).all() or kept.numel() == 0


def test_route_end_to_end_with_isolation() -> None:
    """route() 端到端：默认隔离主干，输出 DG 稀疏 key + Indexer 分数。"""
    torch.manual_seed(0)
    kernel = make_kernel(D, dg_dim=64, dg_topk=8)
    query = torch.randn(1, 5, D, requires_grad=True)
    out = kernel.route(query, detach_input=True)
    out.score.sum().backward()
    assert out.sparse_key.shape == (1, 5, 64)
    assert out.score.shape == (1, 5, 1)
    # Indexer 可训练、主干隔离
    assert kernel.hrl_indexer.score.weight.grad is not None
    assert query.grad is None or query.grad.abs().sum() == 0
    # DG 经 kernel.dg_proj 而非 Indexer，主干梯度经 dg_proj 透传与否取决于其输入——
    # route() 当前对 dg_proj 输入不 detach（DG 是模式分离投影，允许与主干共训）。
