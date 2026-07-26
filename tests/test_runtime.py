"""M4 运行时骨架单元测试（纯 CPU、秒级）。

判据（接口与实现计划 v1.0 §4 / 部件实现详细计划 M4）：
- 页表 register/get/usage/pending；未知载体 fail-closed 拒收；
- BlockStore 分层写入 + usage_weighted 淘汰（**非 LRU**）；
- Pager namespace 通过 vs fail-closed（不匹配 → None 且 page_faults 自增）；
- Bus route_to_blocks top-k；fetch_payloads fail-closed；
- ca1_gate 四态（含 drift 触发 QUARANTINE、GATES 共识 REJECT、候选空 DROP）；
- ca3_ppr 把种子分数扩散到邻居、总和守恒；
- state_ckpt 往返 < 1e-5 且 states_equal 成立（M4 退出标准）。
"""
from __future__ import annotations

import torch

from tais_obsidian.runtime import (
    CA1Gate,
    BlockSpec,
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    ca1_gate,
    ca3_ppr,
    restore_state,
    save_state,
    states_equal,
)
from tais_obsidian.runtime.pager import namespace_ok

NS = ("m1", 0, 1, "torch.bfloat16", 10000.0)
NS_BAD = ("m1", 999, 1, "torch.bfloat16", 10000.0)


def _spec(bid: str, kind: str = "kv", usage: int = 0, merged: bool = False) -> BlockSpec:
    return BlockSpec(block_id=bid, route_key=f"key/{bid}", namespace=NS,
                     compiled_kind=kind, usage_count=usage, merged_flag=merged)


# ---------------- 页表 ----------------

def test_pagetable_register_and_get() -> None:
    pt = PageTable()
    assert pt.register(_spec("b1")) is True
    got = pt.get("b1")
    assert got is not None and got.block_id == "b1" and got.namespace == NS
    assert pt.get("nope") is None


def test_pagetable_register_unknown_kind_fail_closed() -> None:
    pt = PageTable()
    assert pt.register(_spec("bad", kind="not_a_kind")) is False
    assert pt.get("bad") is None


def test_pagetable_update_usage() -> None:
    pt = PageTable()
    pt.register(_spec("b1"))
    pt.update_usage("b1")
    pt.update_usage("b1", delta=3)
    assert pt.get("b1").usage_count == 4


def test_pagetable_query_by_route_key() -> None:
    pt = PageTable()
    pt.register(BlockSpec(block_id="a", route_key="physics/entropy", namespace=NS))
    pt.register(BlockSpec(block_id="b", route_key="math/algebra", namespace=NS))
    assert [s.block_id for s in pt.query_by_route_key("physics")] == ["a"]


def test_pagetable_list_pending_promotion() -> None:
    pt = PageTable()
    pt.register(_spec("hot", usage=20))
    pt.register(_spec("cold", usage=2))
    pt.register(_spec("merged", usage=50, merged=True))
    assert {s.block_id for s in pt.list_pending_promotion(min_usage=10)} == {"hot"}


# ---------------- BlockStore ----------------

def test_blockstore_tiered_put_get_and_stats() -> None:
    bs = BlockStore()
    bs.put("x", "payload-x", tier="L0")
    bs.put("y", "payload-y", tier="L2")
    assert bs.get("x") == "payload-x"
    assert bs.tier_of("x") == "L0" and bs.tier_of("y") == "L2"
    assert bs.stats()["L0"] == 1 and bs.stats()["L2"] == 1
    assert bs.get("missing") is None


def test_blockstore_eviction_is_usage_weighted_not_lru() -> None:
    """usage_weighted（非 LRU）：高频旧块保留，低频新块先出局。"""
    bs = BlockStore(caps={"L0": 2})
    bs.put("hot", 1, tier="L0", usage_count=100)
    bs.put("cold", 2, tier="L0", usage_count=0)
    bs.put("new", 3, tier="L0", usage_count=0)
    assert bs.get("hot") == 1, "高频热块不应被淘汰（usage_weighted，非 LRU）"
    assert bs.get("cold") is None, "零频块应优先出局"
    assert bs.get("new") == 3


# ---------------- Pager ----------------

def test_pager_fetch_namespace_pass() -> None:
    pt, bs = PageTable(), BlockStore()
    pt.register(_spec("b1"))
    bs.put("b1", "pay", tier="L1")
    pg = Pager(bs, pt)
    assert pg.fetch("b1", NS) == "pay"
    assert pg.page_faults == 0
    assert pt.get("b1").usage_count == 1


def test_pager_fetch_namespace_mismatch_fail_closed() -> None:
    pt, bs = PageTable(), BlockStore()
    pt.register(_spec("b1"))
    bs.put("b1", "pay", tier="L1")
    pg = Pager(bs, pt)
    assert pg.fetch("b1", NS_BAD) is None
    assert pg.page_faults == 1


def test_pager_fetch_missing_block_fail_closed() -> None:
    pg = Pager(BlockStore(), PageTable())
    assert pg.fetch("ghost", NS) is None
    assert pg.page_faults == 1


def test_namespace_ok_tuple_and_dict() -> None:
    d = dict(zip(("model_id", "layer_idx", "compressor_version", "dtype", "rope_theta"), NS))
    assert namespace_ok(NS, d) is True
    assert namespace_ok(NS, NS_BAD) is False
    assert namespace_ok(NS, "garbage") is False


# ---------------- MemoryBus ----------------

def test_bus_route_to_blocks_topk() -> None:
    bus = MemoryBus(PageTable(), BlockStore(), Pager(BlockStore()))
    keys, scores = ["a", "b", "c", "d"], [0.1, 0.9, 0.4, 0.7]
    assert bus.route_to_blocks(scores, keys, 2) == ["b", "d"]
    assert bus.route_to_blocks(scores, keys, 0) == []


def test_bus_fetch_payloads_fail_closed() -> None:
    pt, bs = PageTable(), BlockStore()
    pt.register(_spec("ok"))
    bs.put("ok", "P", tier="L1")
    pg = Pager(bs, pt)
    bus = MemoryBus(pt, bs, pg)
    assert bus.fetch_payloads(["ok", "ghost"], NS) == ["P"]  # ghost 缺页丢弃
    assert bus.fetch_payloads(["ok"], NS_BAD) == []          # namespace 不匹配丢弃
    assert pg.page_faults >= 1


# ---------------- CA1 巩固门 ----------------

def test_ca1_gate_promote() -> None:
    assert ca1_gate("cand", True, usage_count=20, teacher_consensus=0.9, belief_drift=0.1) == "PROMOTE"


def test_ca1_gate_quarantine_on_drift() -> None:
    assert ca1_gate("cand", True, usage_count=20, teacher_consensus=0.9, belief_drift=0.9) == "QUARANTINE"


def test_ca1_gate_reject_low_usage_or_regression() -> None:
    assert ca1_gate("cand", True, usage_count=1, teacher_consensus=0.9, belief_drift=0.0) == "REJECT"
    assert ca1_gate("cand", False, usage_count=99, teacher_consensus=0.9, belief_drift=0.0) == "REJECT"


def test_ca1_gate_reject_low_consensus_and_drop_none() -> None:
    assert ca1_gate("cand", True, usage_count=99, teacher_consensus=0.3, belief_drift=0.0) == "REJECT"
    assert ca1_gate(None, True, usage_count=99, teacher_consensus=0.9, belief_drift=0.0) == "DROP"
    g = CA1Gate()
    assert (g.min_usage, g.min_consensus, g.max_drift) == (10, 0.7, 0.5)


# ---------------- CA3 PPR ----------------

def test_ca3_ppr_expands_to_neighbors() -> None:
    graph = {"a": ["b", "c"], "b": ["c"], "c": []}
    out = ca3_ppr({"a": 1.0}, graph, alpha=0.1, iters=30)
    assert out["b"] > 0.0 and out["c"] > 0.0
    assert set(out.keys()) == {"a", "b", "c"}
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_ca3_ppr_empty() -> None:
    assert ca3_ppr({}, {}, alpha=0.1) == {}


# ---------------- state_ckpt ----------------

def test_state_ckpt_roundtrip_within_tol() -> None:
    torch.manual_seed(0)
    state = {"gdn_h": torch.randn(4, 8), "gdn_c": torch.randn(4, 8)}
    back = restore_state(save_state(state))
    assert states_equal(state, back, tol=1e-5)
    for k in state:
        assert torch.allclose(state[k], back[k], atol=1e-5)


def test_states_equal_detects_mismatch() -> None:
    a = {"h": torch.zeros(2)}
    b = {"h": torch.ones(2)}
    c = {"h": torch.zeros(2), "x": torch.zeros(1)}
    assert states_equal(a, b) is False
    assert states_equal(a, c) is False
