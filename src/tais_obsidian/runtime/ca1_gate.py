"""CA1 巩固门（🟡 运行时逻辑）：固化准入 + 验证门 + 信念漂移监测（规则骨架）。

设计依据：
- 接口与实现计划 v1.0 §4 / 部件实现详细计划 Part C5：① 升格/并入准入
  （高 usage_count + 回归验证 + ⭐ GATES 共识度）；② 验证门（⭐ Kairos NORA 2025：
  验证通过才强化路径）；③ 信念漂移监测（⭐ MemoryGraft arXiv:2512.16962）。
- 🧠 CA1 巩固。

红线（Kairos 设计原则）：**novelty ⊥ correctness 不可平均**——新颖性与正确性是两个
独立维度，禁止合成为单一标量打分（否则高新颖可掩盖错误/投毒）。本骨架用独立阈值
分别判定，不做加权融合。

v1.1 自适应扩展（2026-07-31，解决信源可信度边缘效应——doc 源 0.68 恰低于 0.7 阈
被系统性 REJECT）：
- **边缘带补验证重试（RE_VERIFY）**：consensus ∈ [reverify_band_lo, min_consensus)
  时不直接 REJECT，转 RE_VERIFY 由编排方（SleepConsolidator）用检索/复核证据
  二次评估，仍不达标才 REJECT；重试次数有上限（max_reverify），fail-closed
  （无复核回调时 RE_VERIFY 按 REJECT 落账，与旧行为一致）。
- **证据感知 consensus**：`evidence_aware_consensus` 把"信源先验×一致性"静态项与
  动态证据项（usage 归一化、验证通过率）加权组合，权重可配（EvidenceWeights）。
- **信源可信度在线学习**：`SourceCredibilityTracker` 按历史验证结果 EMA 更新
  （成功上调/失败下调，截断 [0.3, 0.95]），initial 映射与 inquiry_executor
  SOURCE_CREDIBILITY 一致（user 0.9 / doc 0.7 / web 0.5，向后兼容）。

红线保持（自适应不得绕过）：
- 信念漂移 > max_drift → QUARANTINE 仍在**最优先位**，RE_VERIFY/证据加权/可信度
  学习都不触碰漂移判定（投毒不可靠补验证洗白）；
- 弱证据仍弱更新：证据过低的块（consensus < reverify_band_lo）**进不了补验证带**，
  直接 REJECT——补验证不是"必过通道"；加成有界（reverify_boost 单次 ≤0.05）；
- novelty ⊥ correctness 仍不可平均：证据加权只作用于 correctness 维度的共识度
  估计，不与 novelty/saliency 合成标量。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 判定结果
PROMOTE = "PROMOTE"        # 准入（升格/并入）
REJECT = "REJECT"          # 拒绝（用量/回归/共识不足）
QUARANTINE = "QUARANTINE"  # 隔离（信念漂移超阈，MemoryGraft 腐蚀拦截）
DROP = "DROP"              # 丢弃（候选为空/无效）
RE_VERIFY = "RE_VERIFY"    # 边缘带补验证重试（v1.1：consensus 落边缘带，二次评估后再裁决）


@dataclass
class CA1Gate:
    """CA1 巩固门配置（阈值）。"""

    min_usage: int = 10           # 升格最低用量
    min_consensus: float = 0.7    # ⭐ GATES 教师共识度下限
    max_drift: float = 0.5        # 信念漂移上限（MemoryGraft 拦截阈）
    reverify_band_lo: float = 0.62  # 边缘带下沿（consensus ∈ [此值, min_consensus) → RE_VERIFY）
    max_reverify: int = 1         # 补验证重试上限（防无限重试；0 = 关闭边缘带）


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
    reverify_band_lo: float = 0.62,
    reverify_attempts: int = 0,
    max_reverify: int = 1,
) -> str:
    """CA1 巩固门判定（纯函数，fail-closed）。

    规则（按序，前者优先）：
    1. 候选为空 → DROP；
    2. 信念漂移 > max_drift → QUARANTINE（MemoryGraft 信念腐蚀拦截，最优先拦截，
       自适应扩展不触碰——投毒不可靠补验证洗白）；
    3. 有效 usage（usage_count + salience_usage_boost）< min_usage 或 regression_ok
       为 False → REJECT（验证门）；
    4. teacher_consensus ≥ min_consensus → PROMOTE；
    5. teacher_consensus ∈ [reverify_band_lo, min_consensus) 且
       reverify_attempts < max_reverify → RE_VERIFY（v1.1 边缘带补验证重试——
       由编排方补充检索/复核证据后重算共识再入本门；低于带下沿的弱证据不进带，
       直接规则 6 REJECT，补验证不是必过通道）；
    6. 否则 → REJECT（⭐ GATES 共识度不足）。

    ⭐ arousal 写门（McGaugh 原理 / Payne&Kensinger 2018 编码时分子标签→睡眠选择性巩固）：
    ``salience_usage_boost`` 是 KAL L2 arousal 显著性对**有效 usage** 的加成——高唤醒
    经验编码时被打上显著性标签，睡眠期据此**优先巩固**（arousal 是巩固增益主驱动）。
    **红线保持**：saliency 只加成巩固**优先级**（usage 维度），绝不触碰正确性维度
    （regression_ok / consensus / drift 独立判定）——novelty ⊥ correctness 不可平均，
    高显著不能掩盖错误/投毒（drift 拦截仍在最优先位）。valence 只调极性不进此门。

    向后兼容：新参数（reverify_band_lo / reverify_attempts / max_reverify）全部
    关键字可选；旧调用方不传时行为与 v1.0 的差异仅在"边缘带内 consensus 返回
    RE_VERIFY 而非 REJECT"——编排方（SleepConsolidator）无复核回调时按 REJECT
    落账（fail-closed），最终裁决不变。max_reverify=0 或
    reverify_band_lo ≥ min_consensus 时边缘带完全关闭（与 v1.0 逐字节一致）。

    注：novelty 与 correctness 独立判定、绝不平均（Kairos NORA 设计原则）。
    """
    if candidate is None:
        return DROP
    if belief_drift > max_drift:
        return QUARANTINE
    effective_usage = usage_count + max(0, salience_usage_boost)  # arousal 写门：显著性加成
    if effective_usage < min_usage or not regression_ok:
        return REJECT
    if teacher_consensus >= min_consensus:
        return PROMOTE
    # v1.1 边缘带：共识度接近阈值 → 补验证重试（弱证据 < 带下沿进不了带，弱证据仍弱更新）
    if (max_reverify > 0 and reverify_attempts < max_reverify
            and reverify_band_lo <= teacher_consensus < min_consensus):
        return RE_VERIFY
    return REJECT


# ---------------------------------------------------------------------------
# v1.1 证据感知共识度（Evidence-aware consensus）
# ---------------------------------------------------------------------------
@dataclass
class EvidenceWeights:
    """证据感知共识度的加权配置（v1.1；均可配，默认值经 0.1B 交互式验证标定）。

    consensus = w_base·base + w_usage·usage_score + w_verify·verify_score + boost
      base         = 先验一致性 × (0.5 + 0.5·信源可信度)（v1.0 静态口径，保留为主项）
      usage_score  = min(1, usage_count / usage_norm)（检索证据强度：HRL 命中/使用计数）
      verify_score = verify_passes / max(1, verify_attempts)（验证通过率）
      boost        = 补验证通过后的有界加成（编排方注入，单次 ≤ reverify_boost=0.05）

    默认标定（usage_norm=20、demo usage_count=12、一次写入验证 passes=1/attempts=1）：
      user 源（0.9）：0.85×0.76 + 0.10×0.6 + 0.05×1.0 = 0.756 ≥ 0.7 → 直接 PROMOTE；
      doc  源（0.7）：0.85×0.68 + 0.10×0.6 + 0.05×1.0 = 0.688 ∈ [0.62,0.7) → 边缘带
        RE_VERIFY，复核通过 +0.05 → ≈0.74 → PROMOTE（修复 0.68 一刀切 REJECT）；
      劣质块（cred 0.2、未验证、usage 1）：≈0.41 < 0.62 → 直接 REJECT，进不了补验证带。
    """

    w_base: float = 0.85      # 静态主项权重（先验一致性×信源可信度）
    w_usage: float = 0.10     # 检索证据权重（usage 归一化）
    w_verify: float = 0.05    # 验证通过率权重
    usage_norm: float = 20.0  # usage 归一化常数（达到此计数视为检索证据充分）


def evidence_aware_consensus(
    base: float,
    usage_count: int,
    verify_passes: int,
    verify_attempts: int,
    *,
    boost: float = 0.0,
    weights: EvidenceWeights | None = None,
) -> float:
    """证据感知共识度 ∈ [0,1]（v1.1；correctness 维度内部加权，不触碰 novelty/drift）。

    参数：
      base: 静态主项（先验一致性 × 信任度加权，v1.0 口径）。
      usage_count: 块被检索/使用次数（HRL 命中计数；检索证据强度）。
      verify_passes / verify_attempts: 验证通过/总次数（含补验证历史）。
      boost: 补验证有界加成（编排方注入，单次 ≤0.05；防"补验证=必过"）。
      weights: EvidenceWeights（None 用默认）。
    """
    w = weights or EvidenceWeights()
    usage_score = min(1.0, max(0.0, usage_count) / max(1.0, w.usage_norm))
    verify_score = max(0.0, verify_passes) / max(1, verify_attempts)
    c = w.w_base * base + w.w_usage * usage_score + w.w_verify * verify_score + boost
    return min(1.0, max(0.0, c))


# ---------------------------------------------------------------------------
# v1.1 信源可信度在线学习（Source credibility online learning）
# ---------------------------------------------------------------------------
# 初始映射与 model/inquiry_executor.SOURCE_CREDIBILITY 保持一致（向后兼容：
# doc 0.7 / user 0.9 / web 0.5 不动；runtime 不反向 import model，此处自含副本）。
SOURCE_CREDIBILITY_INITIAL: dict = {"user": 0.9, "doc": 0.7, "web": 0.5}


@dataclass
class SourceCredibilityTracker:
    """信源可信度在线学习器（v1.1；指数滑动 EMA + 上下界截断）。

    更新规则（每次睡眠固化后按块裁决 outcome ∈ {1.0 验证成功, 0.0 验证失败}）：
        cred ← clip(cred + alpha·(outcome − cred), lo, hi)
    验证成功上调、失败下调；截断 [lo, hi] = [0.3, 0.95]（先验不可被单次结果
    打穿/拉满——弱证据仍弱更新，强先验也不可自满）。QUARANTINE（冲突未决）
    不更新（保留双方标分歧，不作信源惩罚）；usage/回归门 REJECT 不更新
    （年轻块不是信源质量问题）。

    持久化：to_dict()/from_dict()（正式入页表 SQLite；pilot 内存态）。
    """

    initial: dict | None = None   # 信源→初始可信度（None 用 SOURCE_CREDIBILITY_INITIAL）
    alpha: float = 0.2            # EMA 步长
    lo: float = 0.3               # 下界截断
    hi: float = 0.95              # 上界截断
    cred: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cred:
            self.cred = dict(SOURCE_CREDIBILITY_INITIAL if self.initial is None else self.initial)
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha 须 ∈ (0,1]，实得 {self.alpha}")
        if not 0.0 <= self.lo < self.hi <= 1.0:
            raise ValueError(f"截断区间须 0≤lo<hi≤1，实得 [{self.lo}, {self.hi}]")

    # ------------------------------------------------------------------
    def get(self, source: str, default: float | None = None) -> float:
        """信源当前可信度（未知信源：default 缺省 0.5 中立先验，并登记待学习）。"""
        if source in self.cred:
            return float(self.cred[source])
        fallback = 0.5 if default is None else float(default)
        self.cred[source] = min(self.hi, max(self.lo, fallback))  # 登记未知信源（截断内）
        return float(self.cred[source])

    # ------------------------------------------------------------------
    def update(self, source: str, outcome: float) -> float:
        """按验证结果 EMA 更新（outcome=1.0 成功上调 / 0.0 失败下调），返回新值。"""
        outcome = min(1.0, max(0.0, float(outcome)))
        old = self.get(source)
        new = old + self.alpha * (outcome - old)
        self.cred[source] = min(self.hi, max(self.lo, new))
        return float(self.cred[source])

    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        """持久化快照（cred 表 + 超参）。"""
        return {"cred": dict(self.cred), "alpha": self.alpha, "lo": self.lo, "hi": self.hi}

    @classmethod
    def from_dict(cls, d: dict) -> "SourceCredibilityTracker":
        """从 to_dict 快照恢复。"""
        return cls(initial=dict(d.get("cred", {})), alpha=float(d.get("alpha", 0.2)),
                   lo=float(d.get("lo", 0.3)), hi=float(d.get("hi", 0.95)),
                   cred=dict(d.get("cred", {})))
