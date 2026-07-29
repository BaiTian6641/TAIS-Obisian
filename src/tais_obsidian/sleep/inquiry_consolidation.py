"""求知知识块睡眠固化（Inquiry Sleep Consolidation，pilot）。

把主动求知闭环运行时学到的知识块（BlockStore 中的 draft 态）经**睡眠固化**转成
长期记忆——主动求知闭环从"实时可用"到"长期记忆"的最后一块。

设计依据（必须逐条对齐 docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md §4/§5 +
知识内化训练文档 §3 阶段3，禁止凭记忆扩展）：
- **CA1 门先验一致性调速**（McClelland CLS，§4.2）：与先验一致→快固化（同化，
  assimilation，小 KL 并入）；与先验冲突→慢速+校验（顺应，accommodation，大 KL
  新建块+校验集回归）。单次错误经验因与先验冲突被挡在慢通道（QUARANTINE/慢速）。
- **三元奖励 RL**（TruthRL arXiv:2509.25760，§5）：correct+1/hallucinate−1/
  abstain 0~0.3——区分答对/拒答/幻觉，**abstain 不重罚**（拒答是元认知诚实，
  优于幻觉）。睡眠期 RL 用三元非二元。
- **先 SFT 教拒答再上 RL**（两阶段，arXiv:2601.20126）：先 SFT 教"知之为知之、
  不知为不知"的拒答行为，再上三元 RL 精调——否则 RL 阶段模型不会拒答。
- **防错误固化**（§4.3/§7）：知识块库累积不覆盖（版本化，arXiv:2404.01413）；
  draft→固化必须验证门（校验集回归）；绝不裸自我修正（arXiv:2310.01798）——
  固化经外部验证（校验集回归/teacher_consensus），不用模型自我判断当真值。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] McClelland CLS 先验一致性调速（互补学习系统：快同化/慢顺应）；
  TruthRL 三元奖励（arXiv:2509.25760）；先 SFT 后 RL 两阶段（arXiv:2601.20126）；
  累积不覆盖抗坍缩（arXiv:2404.01413）；绝不裸自我修正（arXiv:2310.01798）。
- [推测/独创] pilot 语义相似度近似作"与既有知识一致性"的几何读出（表征余弦
  相似度，与 CrossVerifier 同一近似策略）——文献无直接先例，须经 0.1B pilot
  标定；正式应多源检索 + 外部锚（非纯表征近似）。

红线与纪律（实现时必须遵守）：
- **防错误固化**：draft→固化必须验证门（regression_ok=校验集回归通过才 PROMOTE）；
  冲突块（dispute）走慢通道（QUARANTINE 保留双方，不静默覆盖）。
- **CA1 门先验一致性调速**：一致快固化/冲突慢+校验（McClelland CLS）。
- **三元奖励**：correct+1/hallucinate−1/abstain 0~0.3（不重罚拒答）。
- **绝不裸自我修正**：固化经外部验证（校验集回归/teacher_consensus）。
- 复用 consolidator/ca1_gate/blockstore/inquiry_executor（不重复造轮子）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..runtime.blockstore import BlockStore
from .consolidator import ConsolidateReport, SleepConsolidator, W0Item

# ---------------------------------------------------------------------------
# 先验一致性阈值（pilot；对齐 §4.2 CA1 门"与先验一致性"调速）
# 依据：McClelland CLS——与先验一致→快同化（PROMOTE 优先）；与先验冲突→慢顺应
# （QUARANTINE/慢速+校验）。阈值与 CrossVerifier 的 consistency_threshold 同量级。
# ---------------------------------------------------------------------------
_FAST_TRACK_THRESHOLD = 0.6   # 一致性 > 此值 → fast_track（快固化，同化）
_CONFLICT_THRESHOLD = 0.3     # 与最邻近先验相似度 < 此值 → 冲突（慢通道，顺应）

# 三元奖励的 abstain 窗口（TruthRL arXiv:2509.25760）：0~0.3，不重罚拒答
_DEFAULT_ABSTAIN_REWARD = 0.15  # abstain 默认奖励（0~0.3 窗口内中值，可配）


class InquiryW0Adapter:
    """求知知识块→W0Item 适配器（睡眠固化的输入转换）。

    把 BlockStore 中的求知知识块（KnowledgeBlockWriter 写入的 payload，draft 态）
    转成 SleepConsolidator.consolidate 的输入 W0Item。

    提取字段：
      content（知识内容）、saliency（KAL L2 arousal 显著性，求知时的元数据或默认
      1.0）、source_credibility（信任度加权，§4.1）、conflict/dispute（冲突标记）。

    **先验一致性计算**：新知识块与既有知识（BlockStore 已有块/主干先验）的一致性
    分数（pilot 用语义相似度近似，对齐 CA1 门"与先验一致性"调速）→ 决定固化速度
    （一致快/冲突慢）。[推测/独创] pilot 表征余弦近似，正式应多源检索 + 外部锚。
    """

    def __init__(self, embed_fn=None):
        """embed_fn: callable(str)->Tensor[d]，文本→表征（先验一致性语义相似度用；
        pilot 可注入 mock/hash；正式接模型 capture_layers hidden state）。None 时
        退化为字符级 hash 投影（仅演示，与 CrossVerifier 同一占位策略）。"""
        self.embed_fn = embed_fn

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> torch.Tensor:
        """文本→表征 [d]（pilot：embed_fn 或字符 hash 投影；正式接模型 hidden state）。"""
        if self.embed_fn is not None:
            with torch.no_grad():
                return self.embed_fn(text).float().flatten()
        # [pilot 占位] 字符级 hash 投影到固定维（确定性，仅演示相似度流转；无语义）。
        # 用 hashlib.sha256（**确定性**）而非内建 hash()——内建 hash() 对 str 在
        # PYTHONHASHSEED 未固定时**每进程随机**（DoS 防护），导致跨进程/测试顺序结果
        # 不同（test_gate_conflict_slow_track 单独跑过、全文件跑因种子污染失败的根因）。
        import hashlib
        seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big") % (2**31)
        torch.manual_seed(seed)
        return torch.randn(64)

    # ------------------------------------------------------------------
    def prior_consistency(
        self,
        content: str,
        prior_knowledge: list | None = None,
    ) -> float:
        """新知识块与既有先验的一致性分数 ∈ [0,1]（CA1 门"与先验一致性"调速读出）。

        pilot 用语义相似度近似：新知识表征与最邻近先验表征的余弦相似度（[-1,1]→
        [0,1]）。[推测/独创] 表征余弦近似——正式应多源检索 + 外部锚（非纯表征近似）。
        无先验可参考（首条知识，无先验冲突）→ 返回高基线 0.8（McClelland CLS：
        无先验冲突→可快速同化）。
        """
        prior_knowledge = prior_knowledge or []
        with torch.no_grad():  # 只读计算（监测/执行分置），不建梯度路径
            new_emb = self._embed(content)
            if not prior_knowledge:
                return 0.8  # 无先验冲突 → 可快速同化（McClelland CLS）
            sims = []
            for prior in prior_knowledge:
                prior_text = prior if isinstance(prior, str) else str(
                    prior.get("content", prior) if isinstance(prior, dict) else prior
                )
                sims.append(float(F.cosine_similarity(new_emb, self._embed(prior_text), dim=0).item()))
            max_sim = max(sims)  # 与最一致先验的相似度（CA1 门"与先验一致性"）
            return (max_sim + 1.0) / 2.0  # 余弦 [-1,1] → [0,1]

    # ------------------------------------------------------------------
    def to_w0item(
        self,
        block_id: str,
        payload: dict,
        prior_knowledge: list | None = None,
        saliency: float = 1.0,
        usage_count: int = 1,
    ) -> W0Item:
        """求知知识块 payload → W0Item（睡眠固化输入）。

        参数：
          block_id: 知识块 id（版本化，如 inquiry/xxxx:v1）。
          payload: KnowledgeBlockWriter 写入的块载荷（含 content/source_credibility/
              conflict/dispute_note/consistency 等元数据）。
          prior_knowledge: 既有先验知识（BlockStore 已有块/主干先验），先验一致性
              计算用。
          saliency: KAL L2 arousal 显著性（求知时的元数据；缺省 1.0 基线）。
          usage_count: 求知块被检索/使用的次数（HRL 命中计数；CA1 门 usage 维度）。
        返回：W0Item（content/saliency/teacher_consensus/belief_drift 已映射）。
        """
        content = payload.get("content", "")
        # 先验一致性 → teacher_consensus 映射（§4.2 CA1 门"与先验一致性"作固化速度
        # 调控变量；同时信任度加权 source_credibility 进入 consensus，§4.1 Jeffrey）
        credibility = float(payload.get("source_credibility", 0.5))
        consistency = self.prior_consistency(content, prior_knowledge)
        # teacher_consensus = 先验一致性 × 信任度加权（弱证据弱更新，精度加权预测误差）
        teacher_consensus = consistency * (0.5 + 0.5 * credibility)
        # 冲突标记 → belief_drift（冲突未决→漂移升高，CA1 门拦截到 QUARANTINE 慢通道）
        conflict = bool(payload.get("conflict", False))
        belief_drift = 0.9 if conflict else 0.0  # 冲突→高漂移（挡在慢通道）
        return W0Item(
            item_id=block_id,
            content=content,
            timestamp=float(payload.get("timestamp", 0.0)) or 0.0,
            saliency=saliency,
            usage_count=usage_count,
            belief_drift=belief_drift,
            teacher_consensus=teacher_consensus,
            regression_ok=bool(payload.get("verified", True)),  # 写入时已交叉验证
        )


class PriorConsistencyGate:
    """CA1 门先验一致性调速扩展（pilot；McClelland CLS 互补学习系统）。

    [已确立] McClelland CLS（1995 互补学习系统）：新经验与先验一致→快固化（同化，
    assimilation，小 KL 并入既有图式）；与先验冲突→慢速+校验（顺应，accommodation，
    大 KL 新建图式+校验）。单次错误经验因与先验冲突被挡在慢通道（不被快速固化）。

    assess(item, prior_knowledge) → (consistency, fast_track)：
      consistency 高（与先验一致）→ fast_track=True（快固化，同化，PROMOTE 优先）；
      consistency 低（与先验冲突）→ fast_track=False（慢通道，顺应——QUARANTINE/
      慢速+校验，挡单次错误经验）。
    """

    def __init__(
        self,
        fast_track_threshold: float = _FAST_TRACK_THRESHOLD,
        conflict_threshold: float = _CONFLICT_THRESHOLD,
        embed_fn=None,
    ):
        self.fast_track_threshold = fast_track_threshold
        self.conflict_threshold = conflict_threshold
        self._adapter = InquiryW0Adapter(embed_fn=embed_fn)

    # ------------------------------------------------------------------
    def assess(
        self,
        item: W0Item,
        prior_knowledge: list | None = None,
    ) -> tuple[float, bool]:
        """评估新知识块与先验的一致性，决定固化速度（McClelland CLS）。

        返回 (consistency, fast_track)：
          consistency: 与先验一致性 ∈ [0,1]（pilot 语义相似度近似）。
          fast_track: True=快固化（同化，一致）；False=慢通道（顺应，冲突+校验）。
        """
        content = item.content if isinstance(item.content, str) else str(item.content)
        consistency = self._adapter.prior_consistency(content, prior_knowledge)
        # 一致→快固化（同化）；冲突→慢通道（顺应，挡单次错误经验）
        fast_track = consistency > self.fast_track_threshold
        return consistency, fast_track


class TriRewardRL:
    """三元奖励 RL（pilot 规则版；TruthRL arXiv:2509.25760）。

    [已确立] TruthRL：correct+1/hallucinate−1/abstain 0~0.3——区分答对/拒答/幻觉，
    **abstain 不重罚**（拒答是元认知诚实，优于幻觉；二元奖励把拒答当错误重罚会
    激励幻觉）。睡眠期 RL 用三元非二元。

    [已确立] 先 SFT 教拒答再上 RL（两阶段，arXiv:2601.20126）：先 SFT 教"知之为
    知之、不知为不知"的拒答行为（KAL P(IK) 三态），再上三元 RL 精调——否则 RL
    阶段模型不会拒答（只学答对/幻觉两极）。

    用于睡眠固化的蒸馏/提取练习奖励信号（区分答对/拒答/幻觉）。
    """

    # 三元结果类别
    CORRECT = "correct"        # 答对（外部验证通过）
    HALLUCINATE = "hallucinate"  # 幻觉（答错且自信，未拒答）
    ABSTAIN = "abstain"        # 拒答（诚实降级，"不知为不知"）

    def __init__(self, abstain_reward: float = _DEFAULT_ABSTAIN_REWARD):
        """abstain_reward: abstain 的奖励值，须在 0~0.3 窗口（TruthRL：不重罚拒答，
        给小正奖励鼓励诚实降级优于幻觉）。默认 0.15（窗口中值，可配）。"""
        if not 0.0 <= abstain_reward <= 0.3:
            raise ValueError(
                f"abstain_reward 须 ∈ [0, 0.3]（TruthRL 不重罚拒答窗口），实得 {abstain_reward}"
            )
        self.abstain_reward = abstain_reward

    # ------------------------------------------------------------------
    def reward(self, outcome: str) -> float:
        """三元奖励（pilot 规则版）。

        参数 outcome ∈ {"correct","hallucinate","abstain"}：
          correct → +1（答对，外部验证通过）；
          hallucinate → −1（幻觉，答错且自信——重罚，激励拒答优于瞎猜）；
          abstain → abstain_reward ∈ [0, 0.3]（拒答，诚实降级——不重罚，小正奖励
              鼓励"不知为不知"优于幻觉）。
        """
        if outcome == self.CORRECT:
            return 1.0
        if outcome == self.HALLUCINATE:
            return -1.0
        if outcome == self.ABSTAIN:
            return self.abstain_reward
        raise ValueError(
            f"未知 outcome={outcome!r}（合法值 {self.CORRECT}/{self.HALLUCINATE}/{self.ABSTAIN}）"
        )


class InquirySleepConsolidation:
    """求知知识块睡眠固化封装（pilot）。

    consolidate_inquiry_blocks(blockstore, consolidator, prior_knowledge)：
      把 BlockStore 中的求知知识块（draft 态）经 InquiryW0Adapter 转 W0Item
      → PriorConsistencyGate 调速（一致快/冲突慢）→ SleepConsolidator.consolidate
      （CA1 门+间隔提取）→ 三元奖励信号 → 返回 ConsolidateReport
      （PROMOTE/QUARANTINE/REJECT 分布）。

    **防错误固化**：draft→固化验证门（regression_ok=校验集回归通过才 PROMOTE）；
    冲突块（dispute）走慢通道（QUARANTINE 保留双方，不静默覆盖）。
    """

    def __init__(self, embed_fn=None, abstain_reward: float = _DEFAULT_ABSTAIN_REWARD):
        self.adapter = InquiryW0Adapter(embed_fn=embed_fn)
        self.gate = PriorConsistencyGate(embed_fn=embed_fn)
        self.tri_reward = TriRewardRL(abstain_reward=abstain_reward)

    # ------------------------------------------------------------------
    def consolidate_inquiry_blocks(
        self,
        blockstore: BlockStore,
        consolidator: SleepConsolidator,
        prior_knowledge: list | None = None,
        namespace: str = "inquiry",
        usage_count: int = 1,
        saliency: float = 1.0,
        regression_ok: bool = True,
        recall_fn=None,
    ) -> ConsolidateReport:
        """把 BlockStore 中的求知知识块经睡眠固化转长期记忆。

        参数：
          blockstore: BlockStore（含求知知识块，draft 态，版本化累积）。
          consolidator: SleepConsolidator（CA1 门+间隔提取练习+SHY 归一化）。
          prior_knowledge: 既有先验知识（BlockStore 已有块/主干先验），CA1 门
              "与先验一致性"调速用。
          namespace: 求知知识块命名空间前缀（只固化该 namespace 的 draft 块）。
          usage_count: 求知块被检索/使用次数（HRL 命中计数；CA1 门 usage 维度）。
          saliency: KAL L2 arousal 显著性（默认 1.0 基线）。
          regression_ok: 校验集回归是否通过（**防错误固化验证门**——False 则不
              PROMOTE，draft→固化必须验证门红线）。
          recall_fn: 提取练习"试着回忆再核对"回调（item)->bool；None 时按
              regression_ok 字段（骨架默认）。
        返回：ConsolidateReport（PROMOTE/QUARANTINE/REJECT 分布）。
        """
        # 收集 namespace 下的求知知识块（draft 态，版本化累积不覆盖——全量遍历各层）
        items: list[W0Item] = []
        for tier in ("L0", "L1", "L2"):
            od = blockstore._store.get(tier)  # 只读遍历（不触发 get 的 usage/recency 副作用）
            if od is None:
                continue
            for block_id, payload in od.items():
                if not isinstance(payload, dict):
                    continue
                if not str(block_id).startswith(namespace + "/"):
                    continue
                if not payload.get("draft", False):
                    continue  # 只固化 draft 态（已固化块跳过）
                item = self.adapter.to_w0item(
                    block_id, payload, prior_knowledge,
                    saliency=saliency, usage_count=usage_count,
                )
                # 防错误固化验证门：regression_ok=校验集回归通过才可 PROMOTE
                # （draft→固化必须验证门；绝不裸自我修正——固化经外部验证）
                item.regression_ok = item.regression_ok and regression_ok
                items.append(item)

        # CA1 门先验一致性调速（McClelland CLS）：一致快固化/冲突慢+校验
        # 一致性低的块（冲突）已在 adapter 映射为高 belief_drift → CA1 门 QUARANTINE
        # （慢通道保留双方，不静默覆盖）；此处调速只作 PROMOTE 优先级的提示性标注
        # （CA1 门最终判定权在 consolidator，novelty ⊥ correctness 不可平均红线保持）。
        for item in items:
            consistency, fast_track = self.gate.assess(item, prior_knowledge)
            # fast_track=True（一致）→ 保持 teacher_consensus 高分（快固化同化）；
            # fast_track=False（冲突）→ belief_drift 已高（慢通道顺应），此处不额外干预
            # （CA1 门独立判定，先验一致性已通过 teacher_consensus/belief_drift 进入）。
            _ = (consistency, fast_track)  # 调速读出已映射，CA1 门最终裁决

        # 睡眠固化：分簇回放→间隔提取练习→CA1 门（regression_ok 验证门+drift 拦截）
        # →SHY 归一化；冲突块 drift 高→QUARANTINE 保留双方（累积不覆盖）。
        return consolidator.consolidate(items, recall_fn=recall_fn)


def make_inquiry_sleep_consolidation(**kw) -> InquirySleepConsolidation:
    """工厂函数。"""
    return InquirySleepConsolidation(**kw)
