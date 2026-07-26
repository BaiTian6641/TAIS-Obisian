"""M6 睡眠固化单元测试：分簇回放、间隔提取练习、SHY 归一化、CA1 编排、离线锁定。

判据（部件实现详细计划 Part F2 / 设计 §23.5 / M6 退出标准）：
- cluster_by_temporal 按时间邻近分簇（cSPW-R 涟漪簇分批）；
- next_spacing_delay 扩展间隔递增；
- retrieval_practice 答对→强度+/阶段+，答错→强度-/阶段回退（合意困难）；
- shy_normalize 归一化但 top 强度保护（非 LRU，down-selection）；
- consolidate 编排 分簇→提取练习→CA1 门→归一化，QUARANTINE/PROMOTE/REJECT 计数正确；
- 离线锁定（固化期不可重入）。
"""
from __future__ import annotations

import time

import pytest

from tais_obsidian.sleep import (
    SleepConsolidator,
    W0Item,
    cluster_by_temporal,
    make_consolidator,
    next_spacing_delay,
    retrieval_practice,
    shy_normalize,
)


def _item(i, ts, usage=20, drift=0.0, ok=True, strength=1.0, saliency=1.0):
    return W0Item(item_id=f"it{i}", content=f"c{i}", timestamp=ts, usage_count=usage,
                  belief_drift=drift, regression_ok=ok, strength=strength, saliency=saliency)


def test_cluster_by_temporal() -> None:
    items = [_item(1, 0), _item(2, 100), _item(3, 700), _item(4, 5000)]
    clusters = cluster_by_temporal(items, cluster_gap=600)
    # (0,100) 一簇（差 100<600），(700) 一簇（差 600 不<600），(5000) 一簇
    assert [len(c) for c in clusters] == [2, 1, 1]


def test_next_spacing_delay_increases() -> None:
    d0 = next_spacing_delay(0)
    d1 = next_spacing_delay(1)
    d5 = next_spacing_delay(5)
    assert d1 > d0 and d5 > d1


def test_retrieval_practice_correct_and_wrong() -> None:
    it = _item(1, 0, strength=1.0)
    it.review_stage = 2
    retrieval_practice(it, True)
    assert it.strength > 1.0 and it.review_stage == 3 and it.usage_count > 0
    it2 = _item(2, 0, strength=1.0)
    it2.review_stage = 2
    retrieval_practice(it2, False)
    assert it2.strength < 1.0 and it2.review_stage == 1


def test_shy_normalize_protects_top() -> None:
    items = [_item(1, 0, strength=10.0), _item(2, 0, strength=0.1), _item(3, 0, strength=0.1)]
    shy_normalize(items, protect_top_frac=1 / 3)
    # top 强度块（10.0）受保护不衰减；低强度块被归一化下调
    strengths = {it.item_id: it.strength for it in items}
    assert strengths["it1"] >= 10.0, "top 强度块应受 SHY 保护"
    assert strengths["it2"] <= 0.1 and strengths["it3"] <= 0.1, "低强度块应被归一化下调"


def test_consolidate_end_to_end() -> None:
    base = time.time()
    items = [
        _item(1, base, usage=20, ok=True),            # 应 PROMOTE
        _item(2, base + 50, usage=1, ok=True),        # usage 不足 → REJECT
        _item(3, base + 100, usage=20, ok=False),     # 回归失败 → REJECT
        _item(4, base + 200, usage=20, drift=0.9),    # 漂移 → QUARANTINE
    ]
    con = make_consolidator()
    rep = con.consolidate(items)
    assert rep.n_promoted == 1 and rep.promoted_ids == ["it1"]
    assert rep.n_quarantined == 1
    assert rep.n_rejected == 2
    assert rep.n_practiced == 4
    assert rep.locked is False  # 固化完成解锁


def test_consolidate_offline_lock_no_reentry() -> None:
    con = SleepConsolidator()
    con.lock_offline()
    with pytest.raises(RuntimeError):
        con.consolidate([_item(1, 0)])
    con.unlock()
    rep = con.consolidate([_item(1, 0, usage=20, ok=True)])
    assert rep.n_promoted >= 0  # 解锁后可用


def test_consolidate_recall_fn_overrides() -> None:
    """recall_fn 回调（试着回忆再核对）覆盖 regression_ok 字段。"""
    items = [_item(1, 0, usage=20, ok=False)]  # 字段 False，但回调返回 True
    con = make_consolidator()
    rep = con.consolidate(items, recall_fn=lambda it: True)
    # 回调 True → regression_ok=True → 可 PROMOTE
    assert rep.n_promoted == 1
