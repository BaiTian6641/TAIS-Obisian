"""块载荷存储（L0/L1/L2 分页，🟡 运行时数据）。

设计依据：
- 接口与实现计划 v1.0 §4：BlockStore 块载荷存储；L0 VRAM（常驻个位数块）/
  L1 DRAM / L2 NVMe 分页。
- 设计文档存储层级 L0–L3：L0 工作记忆（hot）↔ L1 短期 ↔ L2 长期。

纪律（红线）：
- 淘汰策略为 **usage_weighted**（SHY 启发，接口计划 §4 睡眠巩固器
  "SHY 归一化（非 LRU）"），**禁止朴素 LRU**。淘汰 hint = usage_count × recency，
  淘汰时选 hint 最低者（低频且久未用优先出局）。
"""
from __future__ import annotations

import time
from collections import OrderedDict

# 各层默认容量（L0 VRAM 常驻个位数热块；L1/L2 骨架默认，正式由硬件标定）
_TIER_CAP = {"L0": 8, "L1": 64, "L2": 1024}


class BlockStore:
    """分层块载荷存储。每层一个 OrderedDict（保插入序作 recency 依据）。

    usage_weighted 淘汰：每层维护 ``{block_id: usage_count}``，淘汰分数
    = usage_count × recency（recency 用最近一次访问的单调时间戳），分数最低者出局。
    """

    def __init__(self, caps: dict | None = None):
        self._caps = dict(_TIER_CAP if caps is None else caps)
        self._store: dict[str, OrderedDict] = {t: OrderedDict() for t in self._caps}
        self._usage: dict[str, dict[str, int]] = {t: {} for t in self._caps}
        self._tick: dict[str, dict[str, float]] = {t: {} for t in self._caps}

    def put(self, block_id: str, payload, tier: str = "L1", usage_count: int = 0) -> None:
        """写入载荷到指定层；层满先按 usage_weighted 淘汰。"""
        self._check_tier(tier)
        self.evict_if_full(tier)
        od = self._store[tier]
        if block_id in od:
            del od[block_id]
        od[block_id] = payload
        self._usage[tier][block_id] = max(usage_count, self._usage[tier].get(block_id, 0))
        self._tick[tier][block_id] = time.monotonic()

    def get(self, block_id: str):
        """取载荷（跨层查找）。命中即刷新 recency 与 usage（hint 用）；未命中返回 None。"""
        for tier in self._store:
            od = self._store[tier]
            if block_id in od:
                od.move_to_end(block_id)
                self._tick[tier][block_id] = time.monotonic()
                self._usage[tier][block_id] = self._usage[tier].get(block_id, 0) + 1
                return od[block_id]
        return None

    def tier_of(self, block_id: str) -> str | None:
        """返回块所在层；不存在返回 None。"""
        for tier, od in self._store.items():
            if block_id in od:
                return tier
        return None

    def evict_if_full(self, tier: str) -> None:
        """层满则淘汰 usage_weighted hint 最低者（usage_count × recency）。

        usage_weighted（非 LRU）：分数 = usage_count × recency，**最低**分出局——
        低频且久未用的块优先淘汰；高频热块即使较旧也保留（SHY 归一化语义）。
        """
        self._check_tier(tier)
        od = self._store[tier]
        while len(od) >= self._caps[tier]:
            victim = min(
                od.keys(),
                key=lambda bid: self._usage[tier].get(bid, 0)
                * max(self._tick[tier].get(bid, 0.0), 1e-9),
            )
            del od[victim]
            self._usage[tier].pop(victim, None)
            self._tick[tier].pop(victim, None)

    def stats(self) -> dict[str, int]:
        """各层当前块数。"""
        return {t: len(od) for t, od in self._store.items()}

    def _check_tier(self, tier: str) -> None:
        if tier not in self._store:
            raise KeyError(f"未知存储层: {tier!r}（合法值 {sorted(self._store)}）")
