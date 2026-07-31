"""CA1 门自适应（v1.1）单元测试：边缘带补验证 / 证据加权共识 / 信源可信度在线学习。

背景（信源可信度边缘效应，交互式验证 demo 与 REPL 两路独立复现）：旧口径
teacher_consensus = 先验一致性×(0.5+0.5·cred)，doc 源（cred 0.7）= 0.8×0.85 = 0.68
恰低于 0.7 阈 → 工具来源知识被**系统性 REJECT**（6 条已教事实只能固化 3 条 user 源）。
v1.1 三条机制：
  ① 边缘带 [0.62, 0.7) RE_VERIFY 补验证重试（有上限 max_reverify、有日志、
     fail-closed——无复核回调按 REJECT 落账）；
  ② 证据感知 consensus（0.85·静态主项 + 0.10·usage/20 + 0.05·验证通过率，权重可配）；
  ③ 信源可信度 EMA 在线学习（α=0.2，截断 [0.3,0.95]，initial 映射向后兼容）。
红线（本文件专门回归）：
  - drift>0.5 仍 QUARANTINE——自适应（边缘带/证据加权/可信度学习）不得放行投毒；
  - 弱证据仍弱更新——证据过低（<0.62）进不了补验证带；补验证未过不放行（非必过）；
  - novelty ⊥ correctness 不可平均——证据加权只在 correctness 维度内部。
"""
from __future__ import annotations

import time

import pytest

from tais_obsidian.runtime.blockstore import BlockStore
from tais_obsidian.runtime.ca1_gate import (
    RE_VERIFY,
    EvidenceWeights,
    SourceCredibilityTracker,
    ca1_gate,
    evidence_aware_consensus,
)
from tais_obsidian.sleep import SleepConsolidator, W0Item
from tais_obsidian.sleep.inquiry_consolidation import (
    InquirySleepConsolidation,
    InquiryW0Adapter,
)


def _payload(content, source="doc", credibility=0.7, conflict=False, verified=True):
    return {
        "content": content, "source": source, "source_credibility": credibility,
        "consistency": 0.8, "timestamp": time.time(), "verified": verified,
        "version": 1, "draft": True, "conflict": conflict,
        "dispute_note": "与既有知识冲突未决，保留双方标分歧" if conflict else None,
    }


# ===========================================================================
# ① 边缘带 RE_VERIFY：判定 / 上限 / fail-closed
# ===========================================================================
def test_band_verdict_reverify_then_reject_after_cap() -> None:
    """consensus ∈ [0.62,0.7) → RE_VERIFY；重试上限用尽 → REJECT（防无限重试）。"""
    assert ca1_gate("c", True, 20, 0.65, 0.0) == RE_VERIFY
    assert ca1_gate("c", True, 20, 0.69, 0.0) == RE_VERIFY  # 带内上沿
    assert ca1_gate("c", True, 20, 0.65, 0.0, reverify_attempts=1) == "REJECT"  # 上限=1
    assert ca1_gate("c", True, 20, 0.65, 0.0, max_reverify=0) == "REJECT"  # 关闭边缘带
    assert ca1_gate("c", True, 20, 0.65, 0.0, max_reverify=2,
                    reverify_attempts=1) == RE_VERIFY  # 上限可配


def test_band_edges_outside() -> None:
    """带外行为与旧版一致：<0.62 直接 REJECT（弱证据不进带）；≥0.7 PROMOTE。"""
    assert ca1_gate("c", True, 20, 0.619, 0.0) == "REJECT"
    assert ca1_gate("c", True, 20, 0.3, 0.0) == "REJECT"
    assert ca1_gate("c", True, 20, 0.7, 0.0) == "PROMOTE"


def test_drift_quarantine_overrides_adaptive() -> None:
    """红线：drift>0.5 仍 QUARANTINE——边缘带/补验证不得放行投毒（最优先拦截）。"""
    assert ca1_gate("c", True, 20, 0.65, 0.9) == "QUARANTINE"  # 带内 consensus 也拦截
    assert ca1_gate("c", True, 20, 0.95, 0.9) == "QUARANTINE"
    # 有界加成也不救：漂移判定在共识判定之前
    assert ca1_gate("c", True, 20, 0.65, 0.5001) == "QUARANTINE"


# ===========================================================================
# ② 证据感知共识：公式 / 权重可配 / 单调性
# ===========================================================================
def test_evidence_aware_consensus_default_formula() -> None:
    """默认权重 0.85/0.10/0.05（usage_norm=20）的标定值（边缘效应现场参数）。"""
    # doc 源：base=0.8×(0.5+0.5×0.7)=0.68，usage 12/20=0.6，verify 1/1=1.0
    assert evidence_aware_consensus(0.68, 12, 1, 1) == pytest.approx(
        0.85 * 0.68 + 0.10 * 0.6 + 0.05 * 1.0)  # = 0.688 → 边缘带
    # user 源：base=0.76 → 0.756 → 直接 PROMOTE（不进带，向后兼容旧裁决）
    assert evidence_aware_consensus(0.76, 12, 1, 1) == pytest.approx(0.756)


def test_evidence_aware_consensus_monotonic_and_configurable() -> None:
    """usage/验证通过率单调正贡献；权重可配；结果截断 [0,1]。"""
    lo = evidence_aware_consensus(0.68, 1, 0, 1)
    hi = evidence_aware_consensus(0.68, 20, 1, 1)
    assert lo < hi, "检索证据(usage)与验证通过率高 → consensus 高"
    w = EvidenceWeights(w_base=0.5, w_usage=0.3, w_verify=0.2, usage_norm=10)
    c = evidence_aware_consensus(0.6, 10, 1, 2, weights=w)
    assert c == pytest.approx(0.5 * 0.6 + 0.3 * 1.0 + 0.2 * 0.5)
    assert evidence_aware_consensus(2.0, 999, 9, 1, boost=1.0) == 1.0  # 上截断
    assert evidence_aware_consensus(0.0, 0, 0, 1) == 0.0


def test_adapter_doc_source_lands_in_band() -> None:
    """边缘效应现场：doc 源（0.7）证据加权后落 [0.62,0.7) 边缘带（不再一刀切 REJECT）。"""
    adapter = InquiryW0Adapter()
    doc = adapter.to_w0item("i/d:v1", _payload("工具来源知识", credibility=0.7),
                            usage_count=12)
    assert 0.62 <= doc.teacher_consensus < 0.7, (
        f"doc 源应落边缘带（RE_VERIFY 补验证），实得 {doc.teacher_consensus:.3f}")
    user = adapter.to_w0item("i/u:v1", _payload("用户来源知识", source="user",
                                                credibility=0.9), usage_count=12)
    assert user.teacher_consensus >= 0.7, "user 源仍直接达标（向后兼容）"
    # v1.1 证据分量已挂到 item（RE_VERIFY 重算用）
    assert doc.consensus_base == pytest.approx(0.68)
    assert (doc.verify_passes, doc.verify_attempts, doc.source) == (1, 1, "doc")


# ===========================================================================
# ① consolidator 编排：补验证通过 PROMOTE / 未过 REJECT / 无回调 fail-closed
# ===========================================================================
def _band_item(item_id: str = "band") -> W0Item:
    """构造边缘带 item（doc 源现场参数：base 0.68、usage 12、verify 1/1）。"""
    return W0Item(item_id=item_id, content="工具来源知识", usage_count=12,
                  regression_ok=True, belief_drift=0.0,
                  teacher_consensus=evidence_aware_consensus(0.68, 12, 1, 1),
                  consensus_base=0.68, verify_passes=1, verify_attempts=1, source="doc")


def test_consolidator_reverify_pass_promotes() -> None:
    """边缘带 + 复核通过 → 有界加成 → PROMOTE；日志记录 before/after 与次数。"""
    calls = []
    sc = SleepConsolidator(reverify_fn=lambda it: calls.append(it.item_id) or True)
    item = _band_item()
    rep = sc.consolidate([item], recall_fn=lambda it: True)
    assert rep.verdicts["band"] == "PROMOTE"
    assert "band" in rep.promoted_ids
    assert rep.n_reverified == 1 and calls == ["band"], "补验证恰好一次（上限）"
    log = rep.reverify_log[0]
    assert log["passed"] is True and log["consensus_before"] < 0.7
    assert log["consensus_after"] >= 0.7, "复核通过+有界加成后应过阈"
    assert log["consensus_after"] - log["consensus_before"] <= 0.06, (
        "加成有界（≤0.05 加成+usage 微涨）——补验证不是必过通道")


def test_consolidator_reverify_fail_rejects() -> None:
    """边缘带 + 复核未过 → REJECT（验证通过率摊薄，共识不升反降——不放水）。"""
    sc = SleepConsolidator(reverify_fn=lambda it: False)
    item = _band_item()
    rep = sc.consolidate([item], recall_fn=lambda it: True)
    assert rep.verdicts["band"] == "REJECT"
    assert rep.n_promoted == 0
    log = rep.reverify_log[0]
    assert log["passed"] is False
    assert log["consensus_after"] < log["consensus_before"], "复核失败应摊薄共识"
    assert item.verify_attempts == 2 and item.verify_passes == 1
    assert item.consensus_boost == 0.0, "复核未过不得给加成"


def test_consolidator_no_reverify_fn_fail_closed() -> None:
    """无复核回调：边缘带 fail-closed 按 REJECT 落账（与旧行为一致，向后兼容）。"""
    sc = SleepConsolidator()
    rep = sc.consolidate([_band_item()], recall_fn=lambda it: True)
    assert rep.verdicts["band"] == "REJECT"
    assert rep.n_rejected == 1 and rep.n_reverified == 1
    assert rep.reverify_log[0]["passed"] is None  # 明确标注：无回调而非复核失败


def test_reverify_not_offered_to_garbage() -> None:
    """抗放水①：劣质证据（共识<0.62）直接 REJECT，补验证回调根本不被调用。"""
    calls = []
    sc = SleepConsolidator(reverify_fn=lambda it: calls.append(it.item_id) or True)
    garbage = W0Item(item_id="garbage", content="劣质证据", usage_count=20,
                     regression_ok=True, belief_drift=0.0,
                     teacher_consensus=0.5,  # <0.62 带下沿
                     consensus_base=0.4, verify_passes=0, verify_attempts=1, source="web")
    rep = sc.consolidate([garbage], recall_fn=lambda it: True)
    assert rep.verdicts["garbage"] == "REJECT"
    assert calls == [], "弱证据不进补验证带（弱证据仍弱更新）"
    assert rep.n_reverified == 0


def test_reverify_fail_never_accumulates_pass() -> None:
    """抗放水②：复核恒失败的块重复固化——每次都 REJECT，加成/通过率不累积洗白。"""
    for _ in range(3):  # 每次固化 item 重建（与真实流程一致），无跨次累积
        sc = SleepConsolidator(reverify_fn=lambda it: False)
        rep = sc.consolidate([_band_item()], recall_fn=lambda it: True)
        assert rep.verdicts["band"] == "REJECT"
        assert rep.reverify_log[0]["consensus_after"] < 0.7


# ===========================================================================
# ③ 信源可信度在线学习：EMA / 截断 / 初始兼容 / 持久化
# ===========================================================================
def test_tracker_initial_backward_compatible() -> None:
    """initial 映射不动：user 0.9 / doc 0.7 / web 0.5（与 inquiry_executor 一致）。"""
    t = SourceCredibilityTracker()
    assert t.get("user") == 0.9 and t.get("doc") == 0.7 and t.get("web") == 0.5
    assert t.get("novel_source") == 0.5, "未知信源登记中立先验 0.5（截断内）"


def test_tracker_ema_converges_and_clips() -> None:
    """验证成功上调 / 失败下调（EMA α=0.2），上下界截断 [0.3, 0.95]。"""
    t = SourceCredibilityTracker()
    assert t.update("doc", 1.0) == pytest.approx(0.7 + 0.2 * 0.3)   # 0.76
    assert t.update("doc", 0.0) == pytest.approx(0.76 - 0.2 * 0.76)  # 0.608
    for _ in range(100):
        v = t.update("web", 1.0)
    assert v == 0.95, "反复成功收敛到上界 0.95（不自满到 1.0）"
    for _ in range(100):
        v = t.update("user", 0.0)
    assert v == 0.3, "反复失败收敛到下界 0.3（先验不被打穿到 0）"


def test_tracker_persistence_roundtrip() -> None:
    """to_dict/from_dict 快照往返（正式入页表 SQLite 的序列化口径）。"""
    t = SourceCredibilityTracker(alpha=0.3)
    t.update("doc", 1.0)
    t2 = SourceCredibilityTracker.from_dict(t.to_dict())
    assert t2.cred == t.cred and t2.alpha == 0.3


# ===========================================================================
# 端到端（ISC + consolidator + tracker）：边缘效应修复 + 红线不回退
# ===========================================================================
def test_isc_doc_promotes_via_reverify_conflict_stays_quarantined() -> None:
    """边缘效应修复端到端：doc 源经 RE_VERIFY→PROMOTE；web 冲突块仍 QUARANTINE。"""
    store = BlockStore()
    store.put("inquiry/doc:v1", _payload("工具来源一致知识", credibility=0.7),
              tier="L1", usage_count=5)
    store.put("inquiry/cf:v1", _payload("冲突知识", source="web", credibility=0.5,
                                        conflict=True), tier="L1", usage_count=5)
    tracker = SourceCredibilityTracker()
    isc = InquirySleepConsolidation(credibility_tracker=tracker)
    con = SleepConsolidator(reverify_fn=lambda it: True)  # CrossVerifier 复核通过（mock）
    rep = isc.consolidate_inquiry_blocks(store, con, prior_knowledge=None, usage_count=12)
    assert rep.verdicts["inquiry/doc:v1"] == "PROMOTE", "doc 源应经补验证固化"
    assert rep.verdicts["inquiry/cf:v1"] == "QUARANTINE", "冲突块仍 QUARANTINE（红线）"
    assert rep.n_reverified == 1, "只有 doc 源进边缘带（冲突块漂移拦截在带外）"
    # 可信度在线学习：doc 验证成功上调；web 冲突未决不更新（保留双方非信源惩罚）
    assert tracker.get("doc") == pytest.approx(0.76)
    assert tracker.get("web") == 0.5
    # 冲突块仍存 BlockStore（累积不覆盖，保留双方）
    assert store.get("inquiry/cf:v1")["conflict"] is True


def test_isc_failed_reverify_degrades_source_below_band() -> None:
    """抗放水③：复核恒失败 → 信源可信度逐轮下调 → 最终跌出边缘带，连重试资格都失去。"""
    store = BlockStore()
    store.put("inquiry/doc:v1", _payload("工具来源知识", credibility=0.7),
              tier="L1", usage_count=5)
    tracker = SourceCredibilityTracker()
    isc = InquirySleepConsolidation(credibility_tracker=tracker)
    calls = []
    # 第 1 轮：0.688 带内 → RE_VERIFY → 失败 REJECT；doc 0.7→0.56
    con1 = SleepConsolidator(reverify_fn=lambda it: calls.append(1) or False)
    rep1 = isc.consolidate_inquiry_blocks(store, con1, prior_knowledge=None, usage_count=12)
    assert rep1.verdicts["inquiry/doc:v1"] == "REJECT" and len(calls) == 1
    assert tracker.get("doc") == pytest.approx(0.56)
    # 第 2 轮：consensus≈0.640 仍带内 → RE_VERIFY → 失败；doc 0.56→0.448
    con2 = SleepConsolidator(reverify_fn=lambda it: calls.append(1) or False)
    rep2 = isc.consolidate_inquiry_blocks(store, con2, prior_knowledge=None, usage_count=12)
    assert rep2.verdicts["inquiry/doc:v1"] == "REJECT" and len(calls) == 2
    assert tracker.get("doc") == pytest.approx(0.448)
    # 第 3 轮：consensus≈0.602 <0.62 → 直接 REJECT，补验证不再被调用（弱证据仍弱更新）
    con3 = SleepConsolidator(reverify_fn=lambda it: calls.append(1) or False)
    rep3 = isc.consolidate_inquiry_blocks(store, con3, prior_knowledge=None, usage_count=12)
    assert rep3.verdicts["inquiry/doc:v1"] == "REJECT"
    assert len(calls) == 2, "可信度跌出边缘带后连补验证资格都没有（收敛，不放水）"
    assert rep3.n_reverified == 0


def test_isc_tracker_skips_usage_and_regression_rejects() -> None:
    """usage/回归门 REJECT 不是信源质量问题——可信度不更新（只奖惩证据质量）。"""
    store = BlockStore()
    store.put("inquiry/doc:v1", _payload("工具来源知识", credibility=0.7),
              tier="L1", usage_count=5)
    tracker = SourceCredibilityTracker()
    isc = InquirySleepConsolidation(credibility_tracker=tracker)
    # 回归门未过（regression_ok=False）→ REJECT，但 tracker 不动
    rep = isc.consolidate_inquiry_blocks(store, SleepConsolidator(), prior_knowledge=None,
                                         usage_count=12, regression_ok=False)
    assert rep.n_promoted == 0 and rep.n_rejected == 1
    assert tracker.get("doc") == 0.7, "回归门 REJECT 不惩罚信源（非证据质量问题）"


def test_isc_tracker_learned_credibility_feeds_consensus() -> None:
    """学到的可信度回馈下一轮共识计算：doc 连续成功后可毕业出边缘带（直接 PROMOTE）。"""
    store = BlockStore()
    store.put("inquiry/doc:v1", _payload("工具来源知识", credibility=0.7),
              tier="L1", usage_count=5)
    tracker = SourceCredibilityTracker()
    isc = InquirySleepConsolidation(credibility_tracker=tracker)
    # 两轮补验证成功：doc 0.7→0.76→0.808；第三轮 base=0.8×(0.5+0.404)=0.723
    # consensus=0.85×0.723+0.06+0.05≈0.725 ≥0.7 → 不再进边缘带，直接 PROMOTE
    for _ in range(2):
        isc.consolidate_inquiry_blocks(
            store, SleepConsolidator(reverify_fn=lambda it: True),
            prior_knowledge=None, usage_count=12)
    assert tracker.get("doc") == pytest.approx(0.808)
    calls = []
    rep3 = isc.consolidate_inquiry_blocks(
        store, SleepConsolidator(reverify_fn=lambda it: calls.append(1) or True),
        prior_knowledge=None, usage_count=12)
    assert rep3.verdicts["inquiry/doc:v1"] == "PROMOTE"
    assert calls == [], "可信度毕业后直接 PROMOTE，无需再补验证（自适应收敛）"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
