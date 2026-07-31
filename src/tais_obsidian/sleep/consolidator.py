"""睡眠巩固器（M6，离线锁定）：间隔提取练习 + CA1 门 + 蒸馏 + SHY 归一化。

设计依据（必须逐条对齐，禁止凭记忆扩展）：
- 部件实现详细计划 Part F2 / 设计 §23.5"推理即总结即训练"：睡眠行为=分簇回放 → 验证门
  → 蒸馏固化 → SHY 归一化（离线锁定）。
- 间隔提取练习（提取练习效应，Roediger & Karpicke；间隔重复 d≈0.46）：回放**不是重新
  编码一遍**（=重读），而是**按衰减预测在扩展间隔上以检索形式重激活**（让模型试着回忆
  再核对），回放难度即固化强度信号。
- cSPW-R（Vöröslakos/Buzsáki 2026，全文已核）：海马→皮层输出以**涟漪簇**为单位——
  睡眠固化**按轨迹/时间邻近分批成簇**（路径块优先成簇）；**固化期锁定、不对外服务**
  （DOWN 态=合并锁）。
- SHY 突触稳态假说（Tononi & Cirelli）：淘汰策略 = **强度归一化 + 选择性保护**
  （衰减 ≠ 清理，而是可学习性维护），**非 LRU 删除**。
- 固化用**与预训练同优化器 Muon**（2605.06654：优化器一致显著降遗忘）+ 谱修剪 intruder。
- CA1 门（runtime/ca1_gate）：固化准入 + 验证门 + 信念漂移监测。

纪律：
- 离线锁定（固化期不对外服务）；novelty ⊥ correctness 不可平均（Kairos）。
- 本原型只做离线算法骨架 + 对拍单测（分簇、间隔调度、归一化、门控编排）；
  On-Policy 蒸馏的实际模型前向/梯度、Muon 优化器接入留 D-0 之后的正式实现。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..runtime.ca1_gate import RE_VERIFY, ca1_gate, evidence_aware_consensus

# 间隔提取练习的默认扩展间隔阶梯（秒；正式由衰减模型预测"快被遗忘"时刻）
DEFAULT_SPACING_LADDER: tuple = (300, 900, 3600, 14400, 86400)


@dataclass
class W0Item:
    """W0 日志条目（轨迹日志=资格迹，🧠 三因子 STDP 的 eligibility trace）。"""

    item_id: str
    content: object                 # 待固化内容（轨迹/经验/块草稿）
    timestamp: float = field(default_factory=time.time)
    saliency: float = 1.0           # 写显著性头打分（高 arousal=惊讶=值得记）
    usage_count: int = 0
    belief_drift: float = 0.0       # 信念漂移（MemoryGraft 拦截）
    teacher_consensus: float = 1.0  # GATES 共识度（v1.1 起=证据感知加权值）
    regression_ok: bool = True      # 回归验证（提取练习后由验证门更新）
    # 间隔提取练习状态
    last_review: float = field(default_factory=time.time)
    review_stage: int = 0           # 已到阶梯第几级
    strength: float = 1.0           # SHY 强度（归一化对象）
    # v1.1 证据感知共识度分量（RE_VERIFY 二次评估重算用；缺省 0/空=旧构造兼容）
    consensus_base: float = 0.0     # 静态主项（先验一致性×信源可信度；0=未填，回退 teacher_consensus）
    verify_passes: int = 0          # 验证通过次数（含补验证历史）
    verify_attempts: int = 0        # 验证总次数
    consensus_boost: float = 0.0    # 补验证有界加成（≤ consolidator.reverify_boost）
    reverify_attempts: int = 0      # 已用补验证重试次数（上限 max_reverify）
    source: str = ""                # 信源（可信度在线学习归因用；""=未知）


def cluster_by_temporal(items: list[W0Item], cluster_gap: float = 600.0) -> list[list[W0Item]]:
    """按时间邻近分批成簇（🧠 cSPW-R：涟漪簇分批回放，路径块优先成簇）。

    相邻条目时间差 < cluster_gap 归为一簇；返回按首条时间排序的簇列表。
    """
    if not items:
        return []
    sorted_items = sorted(items, key=lambda x: x.timestamp)
    clusters: list[list[W0Item]] = [[sorted_items[0]]]
    for it in sorted_items[1:]:
        if it.timestamp - clusters[-1][-1].timestamp < cluster_gap:
            clusters[-1].append(it)
        else:
            clusters.append([it])
    return clusters


def next_spacing_delay(review_stage: int, ladder: tuple = DEFAULT_SPACING_LADDER) -> float:
    """间隔提取练习的扩展间隔（阶梯外推：超出末级按几何增长）。"""
    if review_stage < len(ladder):
        return ladder[review_stage]
    return ladder[-1] * (2 ** (review_stage - len(ladder) + 1))


def retrieval_practice(item: W0Item, recall_correct: bool) -> None:
    """一次提取练习：检索形式重激活 + 核对，按对错更新强度与阶段。

    合意困难（desirable difficulty）：答对→强度+、阶段+（间隔加长）；
    答错→强度-、阶段回退（间隔缩短，难度上调）。
    """
    if recall_correct:
        item.strength *= 1.2
        item.review_stage += 1
    else:
        item.strength *= 0.7
        item.review_stage = max(0, item.review_stage - 1)
    item.last_review = time.time()
    item.usage_count += 1


def shy_normalize(items: list[W0Item], protect_top_frac: float = 0.2) -> None:
    """SHY 归一化（非 LRU）：强度整体下调防饱和，但 top 强度块相对受保护。

    down-selection：均值归一到 1.0，保护强度最高的 protect_top_frac 部分不衰减
    （重要突触相对受保护，可学习性维护而非清理）。
    """
    if not items:
        return
    strengths = sorted((it.strength for it in items), reverse=True)
    cutoff_idx = max(0, int(len(strengths) * protect_top_frac) - 1)
    cutoff = strengths[cutoff_idx]
    mean = sum(it.strength for it in items) / len(items)
    if mean <= 0:
        return
    for it in items:
        if it.strength < cutoff:
            it.strength /= mean  # 低强度归一化下调
        # top 强度保护：不衰减


@dataclass
class ConsolidateReport:
    """睡眠固化报告。"""

    n_clusters: int = 0
    n_practiced: int = 0
    n_promoted: int = 0
    n_quarantined: int = 0
    n_rejected: int = 0
    promoted_ids: list = field(default_factory=list)
    locked: bool = False
    # v1.1 自适应 CA1 门：逐块最终裁决 + 补验证日志（原来只有计数，无法归因）
    verdicts: dict = field(default_factory=dict)      # item_id → 最终裁决（PROMOTE/QUARANTINE/REJECT）
    reverify_log: list = field(default_factory=list)  # 边缘带补验证逐条日志（透明可审计）
    n_reverified: int = 0                             # 进入边缘带补验证的块数


class SleepConsolidator:
    """睡眠巩固器（离线锁定）。

    流程：分簇回放（cSPW-R）→ 间隔提取练习（检索形式）→ CA1 验证门 →
    蒸馏固化（占位，On-Policy 蒸馏留正式实现）→ SHY 归一化（非 LRU）。
    固化期锁定（lock_offline：DOWN 态=合并锁）。
    """

    def __init__(self, ca1_thresholds: dict | None = None, cluster_gap: float = 600.0,
                 salience_scale: float = 4.0, reverify_fn=None, max_reverify: int = 1,
                 reverify_boost: float = 0.05, evidence_weights=None):
        self.ca1_thresholds = ca1_thresholds or {}
        self.cluster_gap = cluster_gap
        # arousal 写门增益：saliency 超出基线 1.0 的部分 × salience_scale = 有效 usage 加成
        # （McGaugh 高唤醒优先巩固；默认 4.0，saliency=2.0 → +4 usage）。
        self.salience_scale = salience_scale
        # v1.1 边缘带补验证：reverify_fn(item)->bool（CrossVerifier 二次复核/检索证据
        # 复核等外部信号）；None 时边缘带 fail-closed 按 REJECT 落账（与旧行为一致）。
        self.reverify_fn = reverify_fn
        self.max_reverify = max_reverify          # 补验证重试上限（防无限重试）
        self.reverify_boost = reverify_boost      # 复核通过的共识有界加成（≤0.05，防"补验证=必过"）
        self.evidence_weights = evidence_weights  # EvidenceWeights（None 用默认）
        self._lock = False

    def lock_offline(self) -> None:
        """固化期锁定（🧠 DOWN 态=合并锁，固化期间不对外服务）。"""
        self._lock = True

    def unlock(self) -> None:
        self._lock = False

    @property
    def locked(self) -> bool:
        return self._lock

    def consolidate(
        self,
        items: list[W0Item],
        recall_fn=None,
    ) -> ConsolidateReport:
        """执行一次睡眠固化。

        recall_fn(item) -> bool：提取练习的"试着回忆再核对"回调（返回是否答对）；
        未提供时按 regression_ok 字段（骨架默认）。
        """
        if self._lock:
            raise RuntimeError("固化期锁定中，不可重入（DOWN 态合并锁）")
        self.lock_offline()
        rep = ConsolidateReport(locked=True)
        try:
            clusters = cluster_by_temporal(items, self.cluster_gap)
            rep.n_clusters = len(clusters)
            for cluster in clusters:
                for item in cluster:
                    # 间隔提取练习（检索形式重激活 + 核对）
                    ok = recall_fn(item) if recall_fn is not None else item.regression_ok
                    retrieval_practice(item, ok)
                    rep.n_practiced += 1
                    item.regression_ok = ok
                    # CA1 验证门（novelty ⊥ correctness 不可平均）
                    # ⭐ arousal 写门（McGaugh）：item.saliency（KAL L2 arousal 显著性，
                    # 编码时分子标签）对有效 usage 加成——高唤醒经验睡眠期优先巩固；
                    # saliency 只加成优先级，不触碰正确性判定（drift 拦截仍最优先）。
                    salience_boost = int(round(max(0.0, item.saliency - 1.0) * self.salience_scale))
                    verdict = self._gate(item, salience_boost)
                    # v1.1 边缘带补验证重试（有上限、有日志、有界加成；fail-closed）
                    if verdict == RE_VERIFY:
                        rep.n_reverified += 1
                        verdict = self._reverify(item, salience_boost, rep)
                    rep.verdicts[item.item_id] = verdict
                    if verdict == "PROMOTE":
                        rep.n_promoted += 1
                        rep.promoted_ids.append(item.item_id)
                    elif verdict == "QUARANTINE":
                        rep.n_quarantined += 1
                    else:
                        rep.n_rejected += 1
            # SHY 归一化（非 LRU，top 强度保护）
            shy_normalize(items)
        finally:
            self.unlock()
        rep.locked = False
        return rep

    # ------------------------------------------------------------------
    def _gate(self, item: W0Item, salience_boost: int) -> str:
        """CA1 门调用（统一 kwargs；max_reverify 由编排侧注入，reverify_attempts 随 item）。"""
        gate_kw = dict(self.ca1_thresholds)
        gate_kw.pop("reverify_attempts", None)  # 尝试次数只能来自 item（防外部伪造）
        gate_kw.setdefault("max_reverify", self.max_reverify)
        return ca1_gate(
            item,
            regression_ok=item.regression_ok,
            usage_count=item.usage_count,
            teacher_consensus=item.teacher_consensus,
            belief_drift=item.belief_drift,
            salience_usage_boost=salience_boost,
            reverify_attempts=item.reverify_attempts,
            **gate_kw,
        )

    # ------------------------------------------------------------------
    def _reverify(self, item: W0Item, salience_boost: int, rep: ConsolidateReport) -> str:
        """边缘带补验证重试（v1.1）：外部复核信号 + 证据重算 → 二次入 CA1 门。

        - 无 reverify_fn → fail-closed REJECT（与旧行为一致，不留悬空 RE_VERIFY）；
        - reverify_fn(item)->bool：CrossVerifier 二次复核 / 检索证据复核等外部信号
          （绝不裸自我修正——复核信号来自编排方注入的外部验证，非模型自我判断）；
        - 复核通过：verify_passes+1 且 consensus_boost 加 reverify_boost（有界，
          单次 ≤0.05——补验证不是必过通道）；复核失败：verify_attempts+1 摊薄
          验证通过率（共识度不升反降）；
        - 重算证据感知共识后二次入门（reverify_attempts 已达上限 → 不会再进带），
          仍不达标 → REJECT。全程写 rep.reverify_log（透明可审计）。
        """
        entry = {"block_id": item.item_id, "consensus_before": float(item.teacher_consensus)}
        if self.reverify_fn is None:
            entry.update(passed=None, verdict="REJECT",
                         note="边缘带但无 reverify_fn——fail-closed 按 REJECT 落账（同旧行为）")
            rep.reverify_log.append(entry)
            return "REJECT"
        passed = bool(self.reverify_fn(item))
        item.reverify_attempts += 1
        item.verify_attempts += 1
        if passed:
            item.verify_passes += 1
            # 有界加成：单次 ≤ reverify_boost（默认 0.05），不累积超界
            item.consensus_boost = min(self.reverify_boost,
                                       item.consensus_boost + self.reverify_boost)
        # 证据重算：usage（检索证据，提取练习已 +1）+ 验证通过率 + 有界加成
        base = item.consensus_base if item.consensus_base > 0 else item.teacher_consensus
        item.teacher_consensus = evidence_aware_consensus(
            base, item.usage_count, item.verify_passes, item.verify_attempts,
            boost=item.consensus_boost, weights=self.evidence_weights)
        verdict = self._gate(item, salience_boost)
        if verdict == RE_VERIFY:  # 上限兜底（理论上 attempts 已满不会再进带）
            verdict = "REJECT"
        entry.update(passed=passed, attempts=item.reverify_attempts,
                     consensus_after=float(item.teacher_consensus), verdict=verdict,
                     note=("复核通过+有界加成" if passed else "复核未过，验证通过率摊薄"))
        rep.reverify_log.append(entry)
        return verdict


def make_consolidator(**kw) -> SleepConsolidator:
    """工厂函数。"""
    return SleepConsolidator(**kw)
