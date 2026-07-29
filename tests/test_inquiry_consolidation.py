"""求知知识块睡眠固化单元测试（Inquiry Sleep Consolidation，pilot）。

判据（对齐 docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md §4/§5 + 知识内化 §3 阶段3）：
- InquiryW0Adapter：知识块 payload→W0Item 转换正确（content/saliency/credibility/
  conflict 保留映射）；
- PriorConsistencyGate：一致→fast_track（快固化同化），冲突→慢通道（顺应，挡错误经验）；
- TriRewardRL：correct+1/hallucinate−1/abstain 0~0.3 窗口（不重罚拒答）；
- 端到端：求知块经固化→一致块 PROMOTE/冲突块 QUARANTINE 保留双方；累积不覆盖；
- 防错误固化：回归不通过（regression_ok=False）的块不 PROMOTE。
"""
from __future__ import annotations

import time

import pytest

from tais_obsidian.runtime.blockstore import BlockStore
from tais_obsidian.sleep import SleepConsolidator, make_consolidator
from tais_obsidian.sleep.inquiry_consolidation import (
    InquirySleepConsolidation,
    InquiryW0Adapter,
    PriorConsistencyGate,
    TriRewardRL,
    make_inquiry_sleep_consolidation,
)


# ---------------------------------------------------------------------------
# 工具：构造 KnowledgeBlockWriter 风格的知识块 payload
# ---------------------------------------------------------------------------
def _payload(content, credibility=0.9, conflict=False, verified=True, ts=None):
    return {
        "content": content,
        "source": "user",
        "source_credibility": credibility,
        "consistency": 0.8,
        "timestamp": time.time() if ts is None else ts,
        "verified": verified,
        "version": 1,
        "draft": True,
        "conflict": conflict,
        "dispute_note": "与既有知识冲突未决，保留双方标分歧" if conflict else None,
    }


# ===========================================================================
# InquiryW0Adapter：知识块→W0Item 转换
# ===========================================================================
def test_adapter_converts_content_saliency_credibility() -> None:
    """content/saliency/credibility 保留映射到 W0Item。"""
    adapter = InquiryW0Adapter()
    pl = _payload("地球绕太阳公转", credibility=0.9)
    item = adapter.to_w0item("inquiry/abc:v1", pl, prior_knowledge=None,
                             saliency=2.5, usage_count=7)
    assert item.item_id == "inquiry/abc:v1"
    assert item.content == "地球绕太阳公转"
    assert item.saliency == 2.5
    assert item.usage_count == 7
    # 无先验 → 一致性高基线 0.8；teacher_consensus = 0.8×(0.5+0.5×0.9)=0.76
    assert item.teacher_consensus == pytest.approx(0.8 * (0.5 + 0.5 * 0.9))
    assert item.belief_drift == 0.0  # 无冲突 → 无漂移


def test_adapter_conflict_maps_to_belief_drift() -> None:
    """冲突标记 → belief_drift 高（CA1 门拦截到 QUARANTINE 慢通道）。"""
    adapter = InquiryW0Adapter()
    pl = _payload("有冲突的知识", conflict=True)
    item = adapter.to_w0item("inquiry/xyz:v1", pl)
    assert item.belief_drift > 0.5, "冲突块应映射为高信念漂移（慢通道）"


def test_adapter_credibility_weights_consensus() -> None:
    """信任度加权（§4.1 Jeffrey）：低可信度 → 低 teacher_consensus。"""
    adapter = InquiryW0Adapter()
    hi = adapter.to_w0item("i/h:v1", _payload("知识", credibility=0.9))
    lo = adapter.to_w0item("i/l:v1", _payload("知识", credibility=0.2))
    assert hi.teacher_consensus > lo.teacher_consensus, "高可信度应有更高 consensus"


# ===========================================================================
# PriorConsistencyGate：一致→快固化 / 冲突→慢通道（McClelland CLS）
# ===========================================================================
def test_gate_consistent_fast_track() -> None:
    """与先验一致 → fast_track=True（快固化，同化）。"""
    gate = PriorConsistencyGate()
    adapter = InquiryW0Adapter()
    # 同一内容作先验（embed hash 相同 → 余弦相似度 1.0 → 一致性 1.0）
    prior = ["地球绕太阳公转"]
    item = adapter.to_w0item("i/a:v1", _payload("地球绕太阳公转"))
    consistency, fast_track = gate.assess(item, prior_knowledge=prior)
    assert consistency > 0.6
    assert fast_track is True, "与先验一致应快固化（同化）"


def test_gate_conflict_slow_track() -> None:
    """与先验冲突 → fast_track=False（慢通道，顺应，挡单次错误经验）。"""
    gate = PriorConsistencyGate()
    adapter = InquiryW0Adapter()
    prior = ["量子力学波函数坍缩的哥本哈根诠释"] * 3  # 完全不同的先验
    item = adapter.to_w0item("i/b:v1", _payload("烹饪红烧肉的火候控制技巧"))
    consistency, fast_track = gate.assess(item, prior_knowledge=prior)
    assert fast_track is False, "与先验冲突应走慢通道（顺应，挡错误经验）"


# ===========================================================================
# TriRewardRL：correct+1/hallucinate−1/abstain 0~0.3
# ===========================================================================
def test_tri_reward_ternary() -> None:
    """correct+1 / hallucinate−1 / abstain 在 0~0.3 窗口（不重罚拒答）。"""
    rl = TriRewardRL()
    assert rl.reward("correct") == 1.0
    assert rl.reward("hallucinate") == -1.0
    r_abs = rl.reward("abstain")
    assert 0.0 <= r_abs <= 0.3, "abstain 奖励须在 0~0.3 窗口（TruthRL 不重罚拒答）"
    assert r_abs > rl.reward("hallucinate"), "拒答应优于幻觉"


def test_tri_reward_abstain_configurable_and_bounded() -> None:
    """abstain 奖励可配但限 0~0.3 窗口（超窗报错）。"""
    rl = TriRewardRL(abstain_reward=0.3)
    assert rl.reward("abstain") == 0.3
    with pytest.raises(ValueError):
        TriRewardRL(abstain_reward=0.5)  # 超窗（重罚拒答违背 TruthRL）
    with pytest.raises(ValueError):
        rl.reward("unknown_outcome")


# ===========================================================================
# 端到端：求知块经固化→一致 PROMOTE / 冲突 QUARANTINE；累积不覆盖
# ===========================================================================
def test_end_to_end_consistent_promote_conflict_quarantine() -> None:
    """一致块 PROMOTE（快固化）；冲突块 QUARANTINE 保留双方（不静默覆盖）。"""
    store = BlockStore()
    store.put("inquiry/a:v1", _payload("一致知识", conflict=False), tier="L1", usage_count=5)
    store.put("inquiry/b:v1", _payload("冲突知识", conflict=True), tier="L1", usage_count=5)
    isc = make_inquiry_sleep_consolidation()
    con = make_consolidator()
    rep = isc.consolidate_inquiry_blocks(
        store, con, prior_knowledge=None, usage_count=20, regression_ok=True,
    )
    # 一致块（无冲突）→ PROMOTE；冲突块（belief_drift 高）→ QUARANTINE
    assert "inquiry/a:v1" in rep.promoted_ids, "一致块应 PROMOTE（快固化同化）"
    assert rep.n_promoted == 1
    assert rep.n_quarantined == 1, "冲突块应 QUARANTINE（慢通道保留双方，不静默覆盖）"
    # 累积不覆盖：冲突块仍在 BlockStore（未被删除/覆盖，保留双方标分歧）
    assert store.get("inquiry/b:v1") is not None
    assert store.get("inquiry/b:v1")["conflict"] is True
    assert store.get("inquiry/b:v1")["dispute_note"] is not None, "冲突块应保留分歧标注"


def test_end_to_end_only_draft_consolidated() -> None:
    """只固化 draft 态块（已固化块跳过）。"""
    store = BlockStore()
    draft = _payload("draft 知识")
    store.put("inquiry/d:v1", draft, tier="L1", usage_count=5)
    solid = _payload("已固化知识")
    solid["draft"] = False  # 非 draft
    store.put("inquiry/s:v1", solid, tier="L1", usage_count=5)
    isc = make_inquiry_sleep_consolidation()
    con = make_consolidator()
    rep = isc.consolidate_inquiry_blocks(store, con, usage_count=20)
    # 只有 draft 块被固化（n_practiced==1），已固化块跳过
    assert rep.n_practiced == 1


# ===========================================================================
# 防错误固化：回归不通过（regression_ok=False）的块不 PROMOTE
# ===========================================================================
def test_regression_gate_blocks_promotion() -> None:
    """draft→固化验证门：regression_ok=False → 不 PROMOTE（防错误固化红线）。"""
    store = BlockStore()
    store.put("inquiry/a:v1", _payload("一致知识", conflict=False), tier="L1", usage_count=5)
    isc = make_inquiry_sleep_consolidation()
    con = make_consolidator()
    rep = isc.consolidate_inquiry_blocks(
        store, con, prior_knowledge=None, usage_count=20, regression_ok=False,
    )
    # 回归不通过 → CA1 门 REJECT（验证门），不 PROMOTE
    assert rep.n_promoted == 0, "回归不通过的块不应 PROMOTE（防错误固化验证门）"
    assert rep.n_rejected == 1


def test_unverified_block_low_consensus_not_promoted() -> None:
    """未验证/低可信度块 teacher_consensus 低 → 不 PROMOTE（绝不裸自我修正）。"""
    store = BlockStore()
    # 低可信度 + 与先验冲突 → teacher_consensus 低 → CA1 门 REJECT
    store.put("inquiry/w:v1", _payload("弱证据知识", credibility=0.1, conflict=False),
              tier="L1", usage_count=5)
    isc = make_inquiry_sleep_consolidation()
    con = make_consolidator()
    # 提供冲突先验使一致性低 → consensus 更低
    rep = isc.consolidate_inquiry_blocks(
        store, con,
        prior_knowledge=["完全无关的另一领域先验知识内容"] * 3,
        usage_count=20, regression_ok=True,
    )
    # 低可信度+低一致性 → consensus 低 → 不 PROMOTE（外部验证不足，绝不裸自我修正）
    assert "inquiry/w:v1" not in rep.promoted_ids


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
