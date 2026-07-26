"""内核 route_candidates（LightningIndexer 接入）+ 命名一致性 单元测试。

判据：
- kernel.route_candidates：全分数形状 + top-k + detach_input 梯度隔离；
- kernel.indexer_kl_warmup_loss：可反传到 lightning 投影；
- 命名一致性：RetrievalAttention（attn_impl=full）与 TriAttention（attn_impl=tri）都在，
  且 init_indexer_from_model 对两者都能取 q_proj（type "A" 层）。
"""
from __future__ import annotations

import pytest
import torch

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tais_kernel import make_kernel

D = 32


def _kernel() -> object:
    torch.manual_seed(0)
    return make_kernel(D, dg_dim=32, dg_topk=4)


def test_route_candidates_full_scores() -> None:
    k = _kernel()
    q = torch.randn(2, 4, D)
    c = torch.randn(2, 7, D)
    s = k.route_candidates(q, c, k=None)
    assert s.shape == (2, 4, 7)


def test_route_candidates_topk() -> None:
    k = _kernel()
    q = torch.randn(2, 4, D)
    c = torch.randn(2, 7, D)
    scores, idx = k.route_candidates(q, c, k=3)
    assert scores.shape == (2, 4, 3) and idx.shape == (2, 4, 3)
    assert (scores[..., :-1] >= scores[..., 1:]).all()  # 降序


def test_route_candidates_gradient_isolation() -> None:
    """detach_input=True：主干 query/candidates 梯度为零（MoE-RL 红线）。"""
    k = _kernel()
    q = torch.randn(1, 3, D, requires_grad=True)
    c = torch.randn(1, 5, D, requires_grad=True)
    s = k.route_candidates(q, c, k=None, detach_input=True)
    s.sum().backward()
    assert q.grad is None or q.grad.abs().sum() == 0
    assert c.grad is None or c.grad.abs().sum() == 0
    # indexer lightning 投影可训练
    assert k.hrl_indexer.lightning.q_index.weight.grad is not None


def test_indexer_kl_warmup_flows() -> None:
    k = _kernel()
    q = torch.randn(2, 4, D)
    c = torch.randn(2, 6, D)
    teacher = torch.randn(2, 4, 6)
    loss = k.indexer_kl_warmup_loss(q, c, teacher)
    assert torch.isfinite(loss)
    loss.backward()
    assert k.hrl_indexer.lightning.q_index.weight.grad is not None


def _mk() -> TaisObsidianForCausalLM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64, d_model=D, n_layer=4, block_pattern=["G", "G", "G", "A"],
        n_q_heads=4, n_kv_heads=2, head_dim=8, n_v_heads=4, n_qk_heads=2,
        mlp_hidden=64, max_seq=16, grad_checkpoint=False, check_0p1b_params=False,
        kernel_enabled=True, kernel_dg_dim=32, kernel_dg_topk=4,
    )
    return TaisObsidianForCausalLM(cfg).eval()


def test_naming_consistency_unified_tri() -> None:
    """命名一致性：唯一 "A" 层统一为 TriRetrievalAttention（旧 RetrievalAttention 已移除），
    init_indexer_from_model 能取其 q_proj（type "A" 层）。"""
    from tais_obsidian.model.tri_attention import TriRetrievalAttention
    m = _mk()
    assert isinstance(m.layers[3].mixer, TriRetrievalAttention)
    assert m.kernel.init_indexer_from_model(m) == 3
