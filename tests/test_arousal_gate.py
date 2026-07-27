"""arousal 写门接 CA1 巩固单元测试（McGaugh 原理工程落点）。

判据（ca1_gate.salience_usage_boost / consolidator arousal 写门 / article_ref/07 §3）：
- 高 saliency（高 arousal）→ 有效 usage 加成 → 低 usage 也能 PROMOTE（高唤醒优先巩固）；
- saliency 只加成优先级：正确性维度独立（regression_ok=False 仍 REJECT，drift 超阈仍 QUARANTINE）；
- 红线：drift 拦截最优先，高 saliency 不能掩盖投毒；
- consolidator 端到端：高 saliency 项优先升格。
"""
from __future__ import annotations

from tais_obsidian.runtime.ca1_gate import ca1_gate
from tais_obsidian.sleep.consolidator import SleepConsolidator, W0Item


def test_salience_boost_enables_promotion() -> None:
    # usage=6 < min_usage=10，但 salience boost=+5 → 有效 11 ≥ 10 → PROMOTE（高唤醒优先）
    verdict = ca1_gate(
        object(), regression_ok=True, usage_count=6, teacher_consensus=0.9,
        belief_drift=0.0, salience_usage_boost=5, min_usage=10,
    )
    assert verdict == "PROMOTE"
    # 无 boost：usage=6 < 10 → REJECT
    verdict_no = ca1_gate(
        object(), regression_ok=True, usage_count=6, teacher_consensus=0.9,
        belief_drift=0.0, salience_usage_boost=0, min_usage=10,
    )
    assert verdict_no == "REJECT"


def test_salience_does_not_override_correctness() -> None:
    # 高 saliency 但 regression_ok=False → 仍 REJECT（saliency 不触碰正确性）
    v1 = ca1_gate(
        object(), regression_ok=False, usage_count=6, teacher_consensus=0.9,
        belief_drift=0.0, salience_usage_boost=10, min_usage=10,
    )
    assert v1 == "REJECT"
    # 高 saliency 但 drift 超阈 → 仍 QUARANTINE（drift 拦截最优先，投毒不可掩盖）
    v2 = ca1_gate(
        object(), regression_ok=True, usage_count=6, teacher_consensus=0.9,
        belief_drift=0.9, salience_usage_boost=10, min_usage=10, max_drift=0.5,
    )
    assert v2 == "QUARANTINE"


def test_consolidator_arousal_prioritizes() -> None:
    # 端到端：两个低 usage 项，高 saliency 的 PROMOTE、低 saliency 的 REJECT
    sc = SleepConsolidator(ca1_thresholds={"min_usage": 10, "min_consensus": 0.7, "max_drift": 0.5},
                           salience_scale=4.0)
    low = W0Item(item_id="low", content="x", saliency=1.0, usage_count=6,
                 teacher_consensus=0.9, regression_ok=True)
    high = W0Item(item_id="high", content="y", saliency=2.5, usage_count=6,
                  teacher_consensus=0.9, regression_ok=True)
    items = [low, high]
    # recall_fn 恒 True（提取练习通过）
    rep = sc.consolidate(items, recall_fn=lambda it: True)
    # 高 saliency（2.5 → boost=int((2.5-1)*4)=6，usage 6+1(练习)+6 ≥ 10）PROMOTE
    assert "high" in rep.promoted_ids, "高 arousal 项应优先升格"
    # 低 saliency（1.0 → boost=0，usage 6+1=7 < 10）REJECT
    assert "low" not in rep.promoted_ids, "低 arousal 项不应升格"


def test_salience_scale_zero_disables() -> None:
    # salience_scale=0 → 写门关闭，高 saliency 不加权（向后兼容/消融对照）
    sc = SleepConsolidator(ca1_thresholds={"min_usage": 10}, salience_scale=0.0)
    high = W0Item(item_id="high", content="y", saliency=3.0, usage_count=6,
                  teacher_consensus=0.9, regression_ok=True)
    rep = sc.consolidate([high], recall_fn=lambda it: True)
    assert "high" not in rep.promoted_ids, "scale=0 时高 saliency 不加权"
