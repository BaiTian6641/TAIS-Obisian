"""concept_slot ↔ HRL route_graph 互动单元测试（动态词表与 HRL 联想检索集成）。

判据（kernel_orchestrator.register_block_to_graph/associative_recall / Part C4 CA3 PPR）：
- concept_slot 注册后接入 route_graph（非孤立节点）；
- register_block_to_graph 双向连边；
- associative_recall（CA3 PPR）：种子分数沿图扩散到 concept_slot 等可达节点；
- fail-closed：空图/孤立节点不崩。
"""
from __future__ import annotations

import torch

from tais_obsidian.model.dyn_vocab import make_dynamic_vocab
from tais_obsidian.model.tais_kernel import TAISKernel
from tais_obsidian.runtime import (
    BlockSpec,
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    make_orchestrator,
)

D = 32
NS = ("m1", 0, 1, "bf16", 10000.0)


def _orch_with_dv():
    k = TAISKernel(D)
    pt = PageTable()
    bs = BlockStore()
    bus = MemoryBus(pt, bs, Pager(bs, pt))
    dv = make_dynamic_vocab(pt, NS, extract_fn=lambda t: torch.ones(D), blockstore=bs)
    orch = make_orchestrator(k, bus, dynamic_vocab=dv)
    return orch, pt


def test_concept_slot_enters_graph() -> None:
    orch, pt = _orch_with_dv()
    ok = orch.assess_vocab_friction("Zorblax", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9)
    assert ok is True
    assert "concept/Zorblax" in orch.route_graph, "concept_slot 应入 route_graph"


def test_register_block_bidirectional_edges() -> None:
    orch, _ = _orch_with_dv()
    orch.register_block_to_graph("A", ["B", "C"])
    assert set(orch.route_graph["A"]) == {"B", "C"}
    assert "A" in orch.route_graph["B"] and "A" in orch.route_graph["C"], "连边须双向"


def test_associative_recall_ppr_diffuses() -> None:
    orch, _ = _orch_with_dv()
    # 构小图：seed → B → concept/C
    orch.register_block_to_graph("seed", ["B"])
    orch.register_block_to_graph("B", ["concept/C"])
    out = orch.associative_recall({"seed": 1.0}, alpha=0.1, iters=30)
    # PPR 扩散：B 与 concept/C 都应有非零分（多跳可达）
    assert out.get("B", 0) > 0
    assert out.get("concept/C", 0) > 0, "concept_slot 应被 PPR 扩散到达"
    # 种子分最高（alpha 回流）
    assert out["seed"] >= out["concept/C"]


def test_concept_slot_reachable_via_ppr() -> None:
    # 端到端：注册 concept_slot → 连边 → PPR 从邻居种子扩散到 concept_slot
    orch, pt = _orch_with_dv()
    # 先注册一个相关块（route_key 含 "Zorblax"）
    pt.register(BlockSpec(block_id="fact/Zorblax mineral", route_key="Zorblax mineral", namespace=NS,
                          compiled_kind="kv", factual_recall=True))
    orch.register_block_to_graph("fact/Zorblax mineral", [])
    ok = orch.assess_vocab_friction("Zorblax", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9)
    assert ok is True
    # concept_slot 应与 route_key 含 "Zorblax" 的块连边
    nbrs = orch.route_graph.get("concept/Zorblax", [])
    assert nbrs, "concept_slot 应有语义邻居（非孤立）"
    # PPR 从邻居种子扩散
    out = orch.associative_recall({nbrs[0]: 1.0}, iters=30)
    assert out.get("concept/Zorblax", 0) > 0, "concept_slot 应被联想检索到达"


def test_empty_graph_fail_closed() -> None:
    orch, _ = _orch_with_dv()
    out = orch.associative_recall({"x": 1.0})
    assert isinstance(out, dict)  # 空图不崩（PPR 对单节点回流）
