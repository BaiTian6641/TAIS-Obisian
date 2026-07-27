"""CA1 巩固门（🟡 运行时逻辑）：固化准入 + 验证门 + 信念漂移监测（规则骨架）。

设计依据：
- 接口与实现计划 v1.0 §4 / 部件实现详细计划 Part C5：① 升格/并入准入
  （高 usage_count + 回归验证 + ⭐ GATES 共识度）；② 验证门（⭐ Kairos NORA 2025：
  验证通过才强化路径）；③ 信念漂移监测（⭐ MemoryGraft arXiv:2512.16962）。
- 🧠 CA1 巩固。

红线（Kairos 设计原则）：**novelty ⊥ correctness 不可平均**——新颖性与正确性是两个
独立维度，禁止合成为单一标量打分（否则高新颖可掩盖错误/投毒）。本骨架用独立阈值
分别判定，不做加权融合。
"""
from __future__ import annotations

from dataclasses import dataclass

# 判定结果
PROMOTE = "PROMOTE"        # 准入（升格/并入）
REJECT = "REJECT"          # 拒绝（用量/回归/共识不足）
QUARANTINE = "QUARANTINE"  # 隔离（信念漂移超阈，MemoryGraft 腐蚀拦截）
DROP = "DROP"              # 丢弃（候选为空/无效）


@dataclass
class CA1Gate:
    """CA1 巩固门配置（阈值）。"""

    min_usage: int = 10           # 升格最低用量
    min_consensus: float = 0.7    # ⭐ GATES 教师共识度下限
    max_drift: float = 0.5        # 信念漂移上限（MemoryGraft 拦截阈）


def ca1_gate(
    candidate,
    regression_ok: bool,
    usage_count: int,
    teacher_consensus: float,
    belief_drift: float,
    *,
    salience_usage_boost: int = 0,
    min_usage: int = 10,
    min_consensus: float = 0.7,
    max_drift: float = 0.5,
) -> str:
    """CA1 巩固门判定（纯函数，fail-closed）。

    规则（按序，前者优先）：
    1. 候选为空 → DROP；
    2. 信念漂移 > max_drift → QUARANTINE（MemoryGraft 信念腐蚀拦截，最优先拦截）；
    3. 有效 usage（usage_count + salience_usage_boost）< min_usage 或 regression_ok
       为 False → REJECT（验证门）；
    4. teacher_consensus < min_consensus → REJECT（⭐ GATES 共识度）；
    5. 否则 → PROMOTE。

    ⭐ arousal 写门（McGaugh 原理 / Payne&Kensinger 2018 编码时分子标签→睡眠选择性巩固）：
    ``salience_usage_boost`` 是 KAL L2 arousal 显著性对**有效 usage** 的加成——高唤醒
    经验编码时被打上显著性标签，睡眠期据此**优先巩固**（arousal 是巩固增益主驱动）。
    **红线保持**：saliency 只加成巩固**优先级**（usage 维度），绝不触碰正确性维度
    （regression_ok / consensus / drift 独立判定）——novelty ⊥ correctness 不可平均，
    高显著不能掩盖错误/投毒（drift 拦截仍在最优先位）。valence 只调极性不进此门。

    注：novelty 与 correctness 独立判定、绝不平均（Kairos NORA 设计原则）。
    """
    if candidate is None:
        return DROP
    if belief_drift > max_drift:
        return QUARANTINE
    effective_usage = usage_count + max(0, salience_usage_boost)  # arousal 写门：显著性加成
    if effective_usage < min_usage or not regression_ok:
        return REJECT
    if teacher_consensus < min_consensus:
        return REJECT
    return PROMOTE
