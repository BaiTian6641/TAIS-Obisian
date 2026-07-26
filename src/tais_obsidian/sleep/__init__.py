"""睡眠固化包（M6）：离线锁定固化与蒸馏。

- consolidator 睡眠巩固器（间隔提取练习 + CA1 门 + SHY 归一化 + 分簇回放）。
"""
from .consolidator import (
    ConsolidateReport,
    SleepConsolidator,
    W0Item,
    cluster_by_temporal,
    make_consolidator,
    next_spacing_delay,
    retrieval_practice,
    shy_normalize,
)

__all__ = [
    "W0Item",
    "ConsolidateReport",
    "SleepConsolidator",
    "cluster_by_temporal",
    "next_spacing_delay",
    "retrieval_practice",
    "shy_normalize",
    "make_consolidator",
]
