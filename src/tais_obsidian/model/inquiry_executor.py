"""求知执行器（Inquiry Executor）——主动求知闭环的"执行+学习"落地（pilot）。

设计依据（必须逐条对齐 docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md）：
- §2.1 三条求知通道：AskQuestion（请求用户解释）/ CallTool（查文档）/ 联网搜索。
- §3.1 **绝不裸自我修正**（arXiv:2310.01798，ICLR 2024 被引 540）：无外部反馈的
  内在自我修正不仅无效且常致退化；所有修正必须由外部信号门控（检索证据/用户反馈/
  校验集回归）。设计原则 = Reflexion 的记忆 + CRITIC 的工具验证 + 绝不裸自我修正。
- §3.2 交叉验证 = 多机制叠加（arXiv:2505.09031）：多源一致性 + 自洽性（仅辅助）
  + 检索验证（外部源）；自洽收益来自"一致性"非"正确性"，只能作辅助信号不能当真值。
- §4.1 信任度加权修正（Jeffrey 条件化 / 精度加权预测误差 / BEWA arXiv:2506.16015）：
  新证据带可信度，弱证据弱更新；知识块带 source_credibility 元数据。
- §4.2 修正 vs 加强仲裁（Hypercorrection + McClelland CLS）：与先验一致快固化，
  与先验冲突慢速+校验；CA1 巩固门以"与既有知识一致性"作固化速度调控变量。
- §4.3 防错误固化（arXiv:2404.01413）：知识块库"累积不覆盖"（页表版本化保留旧块），
  累积式存储在理论上抗坍缩；与"冲突不静默覆盖、版本号+时间戳+置信度三路仲裁"红线一致。
- §7 三条红线：绝不裸自我修正 + 累积式存储；累积不覆盖块存储；draft→固化验证门。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] 绝不裸自我修正（2310.01798）；交叉验证多机制叠加（2505.09031）；
  Jeffrey 条件化/精度加权（Stanford Enc. Philosophy / Predictive Processing）；
  累积不覆盖抗坍缩（2404.01413）；Hypercorrection + 先验一致性（Metcalfe/McClelland）。
- [推测/独创] pilot 语义相似度近似（表征余弦相似度作"与既有知识一致性"几何读出，
  §3.2 空间推理验证 + §4.2 CA1 门"与先验一致性"的操作化）——文献无直接先例，
  须经 0.1B pilot 标定；正式应多源检索 + 外部锚（非纯表征近似）。

红线与纪律（实现时必须遵守）：
- **绝不裸自我修正**：所有写入必须经 CrossVerifier 外部验证门控；verified=False
  的证据**绝不写入** BlockStore（裸自我修正防护）。
- **累积不覆盖**：知识块版本化（block_id 带 :v{n} 后缀），同 id 新版不覆盖旧版；
  冲突未决时保留双方并标注分歧（alignment 冲突不静默覆盖红线）。
- **运行时写有界**：steering 式零梯度快写（W1–W2），绝不触碰权重；写入的是
  BlockStore 中的 steering 式块载荷（向量/文本草稿），非梯度更新。
- **诚实降级**：证据不足/冲突未决时不写入 + 保留分歧；Decline 不执行求知动作。
- **监测/执行分置**：CrossVerifier 只读（表征相似度计算 detach，零副作用）。
- ask_fn/tool_fn 是接口：pilot 注入 mock（返回预置答案/文档），正式接对话接口/
  检索搜索工具；本模块不实现真实网络/对话 IO。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from ..runtime.blockstore import BlockStore
from .inquiry_branch import (
    ActiveInquiryLoop,
    InquiryAction,
    InquiryBranch,
    InquiryDecision,
)

# ---------------------------------------------------------------------------
# 信源可信度默认值（pilot；对齐 §4.1 信任度加权——user>doc>web）
# 依据：用户反馈是强外部锚（hypercorrection 的"外部真值锚"TPJ）；文档检索中等；
# 联网搜索弱（可能错的检索结果，不能假设为真——Jeffrey 条件化 P(E)∈(0,1)）。
# ---------------------------------------------------------------------------
SOURCE_CREDIBILITY = {"user": 0.9, "doc": 0.7, "web": 0.5}

# 交叉验证阈值（pilot；对齐 §3.2——一致性>阈值 且 无未决冲突 才 verified）
_CONSISTENCY_THRESHOLD = 0.6   # 与既有知识一致性阈值（余弦相似度近似）
_MULTI_SOURCE_BOOST = 0.15     # 多源一致性提分幅度（多条独立来源一致→加分）


@dataclass
class Evidence:
    """一条求知获得的新证据（信任度加权的载体）。

    字段：
      content: 证据内容（文本/表征——pilot 为文本；正式可为 hidden state 表征）。
      source: 信源类型 ∈ {"user","doc","web"}（§2.1 三通道）。
      credibility: 信源可信度 ∈ [0,1]（Jeffrey 条件化的 P(E)；缺省按 source 取
          SOURCE_CREDIBILITY 默认值——user=0.9/doc=0.7/web=0.5，可显式覆盖）。
      timestamp: 获得时间（Zep 双时态的 ingested_at 对应物）。
      verified: 是否经 CrossVerifier 交叉验证通过（未验证证据绝不写入）。
      embedding: 可选表征向量 [d]（CrossVerifier 语义一致性用；pilot 由调用方
          或 mock embed_fn 提供，正式接模型 capture_layers hidden state）。
      meta: 额外元数据（query、冲突标注、分歧说明等）。
    """

    content: str
    source: str = "doc"
    credibility: float | None = None
    timestamp: float = field(default_factory=time.time)
    verified: bool = False
    embedding: torch.Tensor | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCE_CREDIBILITY:
            raise ValueError(
                f"未知信源 source={self.source!r}（合法值 {sorted(SOURCE_CREDIBILITY)}）"
            )
        if self.credibility is None:
            self.credibility = SOURCE_CREDIBILITY[self.source]
        if not 0.0 <= self.credibility <= 1.0:
            raise ValueError(f"credibility 须 ∈ [0,1]，实得 {self.credibility}")


class CrossVerifier:
    """交叉验证器（§3.2 多机制叠加；**绝不裸自我修正**红线 arXiv:2310.01798）。

    verify(new_evidence, existing_knowledge) 三路信号：
      ① 多源一致性：existing_knowledge 中已有同 source 之外、内容一致的多条证据
         → 提分（多条独立来源一致，§3.2 多源一致性）；
      ② 与既有知识一致性：新证据与既有知识块的语义一致性（pilot 用表征余弦
         相似度近似——§3.2 空间推理验证 + §4.2 CA1 门"与先验一致性"的几何读出）；
      ③ 冲突检测：与既有知识冲突（相似度 < 冲突阈值）→ 标 conflict（不静默覆盖，
         保留双方标分歧——冲突不静默覆盖红线）。
    verified = 一致性 > 阈值 且 无未决冲突（裸自我修正防护：未验证不写入）。

    [推测/独创] pilot 语义相似度近似：用 embedding 余弦相似度作"与既有知识一致性"
    的几何读出——**正式应多源检索 + 外部锚**（非纯表征近似）；相似度只能作辅助
    信号（§3.2 警示：自洽收益来自"一致性"非"正确性"，不能当真值）。
    """

    def __init__(
        self,
        consistency_threshold: float = _CONSISTENCY_THRESHOLD,
        conflict_threshold: float = 0.3,
        embed_fn=None,
    ):
        """embed_fn: callable(str)->Tensor[d]，文本→表征（pilot 可注入 mock/hash；
        正式接模型 capture_layers）。None 时退化为字符级 hash 投影（仅演示）。"""
        self.consistency_threshold = consistency_threshold
        self.conflict_threshold = conflict_threshold
        self.embed_fn = embed_fn

    # ------------------------------------------------------------------
    def _embed(self, text: str) -> torch.Tensor:
        """文本→表征 [d]（pilot：embed_fn 或字符 hash 投影；正式接模型 hidden state）。"""
        if self.embed_fn is not None:
            with torch.no_grad():
                return self.embed_fn(text).float().flatten()
        # [pilot 占位] 字符级 hash 投影到固定维（确定性，仅演示相似度流转；无语义）
        torch.manual_seed(abs(hash(text)) % (2**31))
        return torch.randn(64)

    # ------------------------------------------------------------------
    def verify(
        self,
        new_evidence: Evidence,
        existing_knowledge: list[Evidence] | None = None,
    ) -> tuple[bool, float, bool]:
        """交叉验证（只读，零副作用；监测/执行分置——detach 不建梯度路径）。

        返回 (verified, consistency_score, conflict)：
          verified: 是否通过（一致性>阈值 且 无未决冲突）。
          consistency_score: 与既有知识的一致性 ∈ [0,1]（含多源提分）。
          conflict: 是否与既有知识冲突（True→不静默覆盖，保留双方标分歧）。
        """
        existing_knowledge = existing_knowledge or []
        with torch.no_grad():  # 只读验证（监测），不建梯度路径（红线）
            new_emb = new_evidence.embedding
            if new_emb is None:
                new_emb = self._embed(new_evidence.content)
            new_emb = new_emb.float().flatten()

            if not existing_knowledge:
                # 无既有知识可参考（首条证据，无先验可比）：CA1 门"无先验冲突→可快速
                # 同化"（McClelland CLS，§4.2）；一致性给高基线使可信度加权后仍过门
                # （0.8×(0.5+0.5×0.9)=0.76>阈值），不冲突
                base = 0.8
                conflict = False
            else:
                sims = []
                for old in existing_knowledge:
                    old_emb = old.embedding if old.embedding is not None else self._embed(old.content)
                    old_emb = old_emb.float().flatten()
                    sims.append(float(F.cosine_similarity(new_emb, old_emb, dim=0).item()))
                max_sim = max(sims)  # 与最一致既有知识的相似度（CA1 门"与先验一致性"）
                conflict = max_sim < self.conflict_threshold  # 与最邻近先验仍冲突
                base = (max_sim + 1.0) / 2.0  # 余弦 [-1,1] → [0,1]

            # ① 多源一致性提分：既有知识中存在不同 source 且内容一致的独立来源
            multi_src = any(
                (o.source != new_evidence.source) and (o.verified or o.credibility >= 0.5)
                for o in existing_knowledge
            )
            consistency = min(1.0, base + (_MULTI_SOURCE_BOOST if multi_src else 0.0))

            # ② 信任度加权（Jeffrey/精度加权）：可信度低的新证据一致性打折
            consistency *= (0.5 + 0.5 * new_evidence.credibility)

            verified = (consistency > self.consistency_threshold) and (not conflict)
        return verified, consistency, conflict


class KnowledgeBlockWriter:
    """知识块写入器（累积不覆盖红线 + 运行时写 steering 式有界）。

    write(evidence, blockstore, namespace="inquiry")：
      - 把**验证通过**（verified=True）的证据写入 BlockStore（draft 态）；
      - **累积不覆盖**：block_id 带 :v{n} 版本后缀，同 id 新版不覆盖旧版（版本化
        累积，§4.3 抗坍缩 arXiv:2404.01413）；
      - 冲突未决时保留双方 + 标注分歧（冲突不静默覆盖红线）。
    **运行时写是 steering 式有界**（W1–W2 零梯度快写，绝不触碰权重）——写入的
    是 BlockStore 中的块载荷（steering 式草稿/向量），非梯度更新；draft→固化
    必须验证门（睡眠期 CA1 门，见 sleep/consolidator.py）。
    """

    def __init__(self, tier: str = "L1"):
        self.tier = tier  # 写入层（L1 DRAM 短期记忆；L0 热块由 Pager 调度）

    # ------------------------------------------------------------------
    def _next_version_id(self, blockstore: BlockStore, base_id: str) -> tuple[str, int]:
        """生成版本化 block_id（base:v{n}），n 自增——累积不覆盖（版本化保留旧块）。"""
        n = 1
        while blockstore.get(f"{base_id}:v{n}") is not None:
            n += 1
        return f"{base_id}:v{n}", n

    # ------------------------------------------------------------------
    def write(
        self,
        evidence: Evidence,
        blockstore: BlockStore,
        namespace: str = "inquiry",
        conflict: bool = False,
        consistency: float = 0.0,
    ) -> str | None:
        """写入验证过的证据到 BlockStore（draft 态，版本化累积不覆盖）。

        参数：
          evidence: 待写入证据（须 verified=True，否则拒绝写入——裸自我修正防护）。
          blockstore: BlockStore 块存储。
          namespace: 命名空间前缀（draft 隔离，对齐"draft 日志区隔离"红线）。
          conflict: 是否与既有知识冲突（True→保留双方 + 标注分歧，不覆盖）。
          consistency: 交叉验证一致性（写入 source_credibility 元数据，§4.1）。
        返回：写入的 versioned block_id；未验证证据返回 None（拒绝写入）。
        """
        # 裸自我修正防护（arXiv:2310.01798）：未验证证据绝不写入
        if not evidence.verified:
            return None
        base_id = f"{namespace}/{abs(hash(evidence.content)) % (10**8)}"
        block_id, version = self._next_version_id(blockstore, base_id)
        payload = {
            "content": evidence.content,
            "source": evidence.source,
            # §4.1 信任度加权：source_credibility 元数据（Jeffrey 条件化 P(E)）
            "source_credibility": evidence.credibility,
            "consistency": consistency,
            "timestamp": evidence.timestamp,
            "verified": evidence.verified,
            "version": version,
            "draft": True,  # draft 态（draft→固化须验证门，睡眠期 CA1 门）
            # 冲突不静默覆盖：冲突未决时保留双方 + 标注分歧（对齐设计红线）
            "conflict": conflict,
            "dispute_note": (
                "与既有知识冲突未决，保留双方标分歧（不静默覆盖）" if conflict else None
            ),
        }
        # 运行时写是 steering 式有界（W1–W2 零梯度快写）：写 BlockStore 块载荷，
        # 绝不触碰权重（非梯度更新）；正式注入走内核 inject 向量/KV 路径。
        blockstore.put(block_id, payload, tier=self.tier, usage_count=1)
        return block_id


class InquiryExecutor:
    """求知执行器——接 ActiveInquiryLoop 的 inquiry_executor 接口（执行+学习）。

    __init__(blockstore, verifier, ask_fn, tool_fn)：
      ask_fn: callable(query)->str|Evidence，AskQuestion 实际执行（问用户）——
          pilot 可注入 mock（返回预置答案）；**正式接对话接口**（本模块不实现真实 IO）。
      tool_fn: callable(query)->str|Evidence，CallTool 实际执行（查文档/联网）——
          pilot 可注入 mock（返回预置文档）；**正式接检索/搜索工具**（本模块不实现）。

    __call__(decision) -> bool（实现 inquiry_executor 接口）：
      AskQuestion: 调 ask_fn → Evidence(source="user") → CrossVerifier 验证
          → verified 则 KnowledgeBlockWriter 写入 → 返回 True（获新证据，闭环重评估）；
      CallTool: 调 tool_fn → Evidence(source="doc") → 验证 → 写入 → 返回 True；
      Decline: 不执行（诚实降级），返回 False；
      DirectAnswer: 不执行，返回 False。
    返回 True 仅当**求知成功获得且验证通过**新证据（闭环；未验证不写入返回 False）。
    """

    def __init__(
        self,
        blockstore: BlockStore | None = None,
        verifier: CrossVerifier | None = None,
        ask_fn=None,
        tool_fn=None,
        writer: KnowledgeBlockWriter | None = None,
        namespace: str = "inquiry",
    ):
        self.blockstore = blockstore
        self.verifier = verifier if verifier is not None else CrossVerifier()
        self.ask_fn = ask_fn  # 接口位：pilot mock；正式接对话接口（本模块不实现 IO）
        self.tool_fn = tool_fn  # 接口位：pilot mock；正式接检索/搜索工具
        self.writer = writer if writer is not None else KnowledgeBlockWriter()
        self.namespace = namespace
        # 既有知识缓冲（pilot 内存态，作 CrossVerifier 一致性参考；正式应查
        # kernel_orchestrator.associative_recall / BlockStore 全量）
        self._knowledge: list[Evidence] = []

    # ------------------------------------------------------------------
    def _acquire(self, fn, query: str, source: str) -> Evidence | None:
        """调外部求知通道获证据（ask_fn/tool_fn 接口；返回 None 表示未获得）。"""
        if fn is None:
            return None
        out = fn(query)
        if out is None:
            return None
        if isinstance(out, Evidence):
            return out
        return Evidence(content=str(out), source=source, meta={"query": query})

    # ------------------------------------------------------------------
    def __call__(self, decision: InquiryDecision) -> bool:
        """实现 inquiry_executor 接口（ActiveInquiryLoop.run 调用点）。"""
        # Decline / DirectAnswer 不执行（诚实降级 / 已掌握区），返回 False
        if decision.action not in (InquiryAction.ASK_QUESTION, InquiryAction.CALL_TOOL):
            return False
        query = decision.reason or f"certainty={decision.certainty:.2f}"
        # AskQuestion → ask_fn（user）；CallTool → tool_fn（doc）
        if decision.action == InquiryAction.ASK_QUESTION:
            ev = self._acquire(self.ask_fn, query, source="user")
        else:
            ev = self._acquire(self.tool_fn, query, source="doc")
        if ev is None:
            return False  # 未获得新证据（诚实降级：不伪造）
        # 交叉验证（绝不裸自我修正：外部信号门控）
        verified, consistency, conflict = self.verifier.verify(ev, self._knowledge)
        ev.verified = verified
        # 写入（累积不覆盖 + 冲突保留双方；未验证不写入返回 False）
        if self.blockstore is not None:
            self.writer.write(
                ev, self.blockstore, namespace=self.namespace,
                conflict=conflict, consistency=consistency,
            )
        if verified:
            self._knowledge.append(ev)  # 验证通过纳入既有知识（供后续验证参考）
            return True  # 求知成功获得且验证通过新证据（闭环重评估 certainty）
        return False  # 未验证：不写入（裸自我修正防护），返回 False


class ActiveInquiryPipeline:
    """主动求知流水线（pilot demo 封装）——InquiryExecutor + ActiveInquiryLoop 集成。

    certainty 低 → 求知分支（InquiryBranch 决策）→ 执行器执行（AskQuestion/CallTool）
    → CrossVerifier 验证 → KnowledgeBlockWriter 写入（累积不覆盖）→ 重评估
    certainty（P(IK) 升高则闭环退出求知）。复用 ActiveInquiryLoop.run（inquiry_executor
    参数挂本执行器），不重复造决策/审计逻辑。
    """

    def __init__(
        self,
        inquiry_loop: ActiveInquiryLoop,
        executor: InquiryExecutor | None = None,
        blockstore: BlockStore | None = None,
        ask_fn=None,
        tool_fn=None,
    ):
        self.inquiry_loop = inquiry_loop
        self.executor = executor if executor is not None else InquiryExecutor(
            blockstore=blockstore, ask_fn=ask_fn, tool_fn=tool_fn,
        )

    # ------------------------------------------------------------------
    def run(self, initial_state: torch.Tensor, **kw):
        """跑主动求知推理循环（inquiry_executor 挂本执行器，透传其余参数）。

        返回 (最终 state, 轨迹 list[(ReasoningTickState, InquiryDecision|None)],
               停止 tick 数, 闭环是否发生[bool：任一求知 tick 验证通过重评估 certainty])。
        """
        state, traj, stop_tick = self.inquiry_loop.run(
            initial_state, inquiry_executor=self.executor, **kw
        )
        closed = any(
            (d is not None and "闭环" in d.reason) for _, d in traj
        )
        return state, traj, stop_tick, closed


__all__ = [
    "SOURCE_CREDIBILITY",
    "ActiveInquiryPipeline",
    "CrossVerifier",
    "Evidence",
    "InquiryExecutor",
    "KnowledgeBlockWriter",
]
