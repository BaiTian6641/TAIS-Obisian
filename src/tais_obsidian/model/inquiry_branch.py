"""推理循环求知分支（Inquiry Branch）——主动求知闭环的真实落地（pilot 规则版）。

设计依据：docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md §1（触发/求知动作）+ §6
（与推理循环整合）。把 reasoning_loop 在"certainty 低 + HRL 未命中"时原本只标
recall_triggered 的死角，扩展为**求知分支**（非硬答）——这是从"检测空白"走向
"主动学习"的关键一步。

核心流程（对齐 §6 思考 tick 整合图）：
  思考 tick（reasoning_loop）：GDN 状态 → glimpse → HRL 提议 → KAL certainty
    ├─ certainty 高（P(IK) 高）→ DirectAnswer（流形位移 → 输出）
    └─ certainty 低 + HRL 未命中 → <|recall|> 显式审计（空白显形化）
        → 求知分支（HRL 路由四选一：AskQuestion/CallTool/Decline）
            → 检索/交互得新证据 → 交叉验证 → 写知识块 → 重评估 P(IK)（闭环）

求知动作空间（HRL 路由四选一，对齐 arXiv:2511.08798 Structured-Uncertainty-GRPO）：
    { AskQuestion（请求用户解释更多）
    , CallTool（查文档/联网搜索自我学习）
    , Decline（诚实拒答——"该部分记忆暂不可用"，对齐诚实降级红线）
    , DirectAnswer（直接回答——P(IK) 高）
    }

触发点=可学习区（RPL/LP）非完全空白（反直觉，§1.2 [已确立]）：
    求知欲最强的不是"完全不知道"（完全空白），而是"中等不确定/差一点就知道"
    （Region of Proximal Learning / TOT 舌尖态，Metcalfe RPL + Oudeyer/Schmidhuber
    Learning Progress，rlPFC/纹状体/ACC 互证）。完全空白区学习成本过高，已掌握区
    无可学。priority = EPIG × LP × (1−P(IK))（§1.2 三信号耦合公式，[推测/独创]）。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] RPL/LP 触发区（§1.2，Metcalfe/Oudeyer/Schmidhuber）；求知动作四选一
  （§1.3，arXiv:2511.08798 / 2512.13159）；诚实降级（TruthRL 2509.25760 abstain）；
  <|recall|>/<|ask|> 显式（Self-RAG reflection tokens + 设计红线"必须显式出现在 CoT"）。
- [推测/独创] pilot 规则版路由阈值（high/mid/call_tool_priority）、priority 三信号
  耦合公式操作化——文献无先例（TAIS 自拟，须经 0.1B pilot 标定）。

红线与纪律：
- **诚实降级红线**：Decline 必须明确声明"该部分记忆暂不可用"，绝不用空白知识硬答。
- **审计红线**：<|ask|>（求知动作）显式 token + <|recall|>（复用 reasoning_loop 的
  RECALL_TOKEN 风格）显式出现在 CoT——可解释性前端渲染求知轨迹。
- **监测/执行分置**：KAL certainty 只读（kernel.sense detach，零副作用）；求知动作
  的实际执行（问用户/查文档）由外部执行器负责，本模块只做**决策与审计**（pilot）。
- **pilot 规则版路由**：InquiryRouter 是**非学习型**规则路由器（阈值常量）；学习型
  HRL 路由头（SFT 教拒答 + RLVR 加固，arXiv:2601.20126 两阶段）留后续 milestone。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch
import torch.nn as nn

from .reasoning_loop import RECALL_TOKEN, ReasoningLoop, ReasoningTickState
from .tais_kernel import TAISKernel

# <|ask|> 审计 token 字面量（§6 审计红线：求知动作显式显形化，复用 RECALL_TOKEN 风格）
ASK_TOKEN = "<|ask|>"

# 诚实降级声明模板（诚实降级红线：明确声明"暂不可用"，含 certainty 与未命中信息）
DECLINE_MESSAGE_TEMPLATE = "该部分记忆暂不可用（certainty={certainty:.2f}，HRL 未命中）"


class InquiryAction(str, Enum):
    """求知动作空间（HRL 路由四选一，对齐 §1.3 / arXiv:2511.08798）。

    继承 str 便于序列化/比较（str 常量语义）；四选一互斥。
    """

    ASK_QUESTION = "AskQuestion"   # 请求用户解释更多（把"解释"当新证据输入）
    CALL_TOOL = "CallTool"         # 查文档/联网搜索自我学习（外部证据源）
    DECLINE = "Decline"            # 诚实拒答——"该部分记忆暂不可用"（诚实降级红线）
    DIRECT_ANSWER = "DirectAnswer"  # 直接回答（P(IK) 高 / 有知识可答）


@dataclass
class InquiryDecision:
    """求知决策记录（审计/可解释性前端用，对齐 §1.3 四选一动作空间）。

    字段：
      action: InquiryAction 四选一求知动作。
      certainty: KAL P(IK) 读出标量 ∈ [0,1]（触发本次决策的元认知信号）。
      priority: 求知优先级（§1.2 三信号 EPIG×LP×(1−P(IK)) 操作化标量；缺省 None
          表示未提供——规则路由退化用 certainty 分档）。
      reason: 决策理由（人类可读，标注触发的规则分支与 RPL/LP 对齐）。
      ask_token: 审计 token——AskQuestion→<|ask|>；Decline→诚实降级声明文本；
          DirectAnswer→None（无求知动作不显形）。CallTool→<|ask|>（同属求知动作显形）。
    """

    action: InquiryAction
    certainty: float
    priority: float | None = None
    reason: str = ""
    ask_token: str | None = None


class InquiryRouter:
    """求知路由器（pilot 规则版，**非学习型**——学习型 HRL 路由头留后续）。

    阈值常量（可配），注释对齐 §1.2 RPL/LP 触发区：
      high_threshold: DirectAnswer 门（certainty ≥ 此值 P(IK) 高 → 直接答）。
      mid_threshold: 可学习区下界——mid < certainty < high 是 RPL/LP"差一点就知道"
          的可学习区（触发求知 AskQuestion/CallTool）；certainty ≤ mid 是完全空白区
          （学习成本过高 → Decline 诚实降级）。
      call_tool_priority: priority ≥ 此值的可学习区求知选 CallTool（自我学习优先），
          否则 AskQuestion（请求解释）——§2.2"只有 priority 足够高的求知才执行"。
    """

    def __init__(
        self,
        high_threshold: float = 0.7,
        mid_threshold: float = 0.4,
        call_tool_priority: float = 0.5,
    ):
        if not 0.0 <= mid_threshold < high_threshold <= 1.0:
            raise ValueError(
                f"须 0≤mid({mid_threshold})<high({high_threshold})≤1（RPL/LP 分档）"
            )
        # 对齐主动求知设计 §1.2 RPL/LP 触发区：[已确立，反直觉]触发点=可学习区非完全空白
        self.high_threshold = high_threshold  # DirectAnswer 门（P(IK) 高）
        self.mid_threshold = mid_threshold    # 可学习区下界（RPL/LP，≤此值=完全空白）
        self.call_tool_priority = call_tool_priority  # CallTool 自我学习优先门

    # ------------------------------------------------------------------
    def decide(
        self,
        certainty: float,
        hrl_hit: bool,
        priority: float | None = None,
    ) -> InquiryDecision:
        """路由决策（pilot 规则版四选一）。

        参数：
          certainty: KAL P(IK) ∈ [0,1]（known 类概率，真值锚校准后语义可靠）。
          hrl_hit: HRL 检索是否命中相关知识块（kernel_orchestrator route/associative_recall
              的命中判定——有知识可答则 DirectAnswer）。
          priority: 求知优先级（§1.2 三信号耦合标量；None 时退化按 certainty 分档）。
        返回：InquiryDecision（动作 + 审计 token + 理由）。
        """
        certainty = float(certainty)
        # ① certainty 高（P(IK) 高）→ DirectAnswer（已掌握区，§1.2 已掌握区无可学）
        if certainty >= self.high_threshold:
            return InquiryDecision(
                action=InquiryAction.DIRECT_ANSWER,
                certainty=certainty,
                priority=priority,
                reason=f"certainty={certainty:.2f}≥high({self.high_threshold})：P(IK) 高直接答",
                ask_token=None,
            )
        # ② certainty 低但 HRL 命中（有知识可答）→ DirectAnswer
        if hrl_hit:
            return InquiryDecision(
                action=InquiryAction.DIRECT_ANSWER,
                certainty=certainty,
                priority=priority,
                reason=f"certainty={certainty:.2f} 低但 HRL 命中相关知识块：有知识可答",
                ask_token=None,
            )
        # ③ certainty 低且未命中：按 RPL/LP 可学习区分档（§1.2 反直觉触发区）
        if certainty > self.mid_threshold:
            # 可学习区（mid < certainty < high，RPL/LP"差一点就知道"）→ 求知
            # §2.2：priority 足够高选 CallTool（自我学习优先），否则 AskQuestion
            if priority is not None and priority >= self.call_tool_priority:
                return InquiryDecision(
                    action=InquiryAction.CALL_TOOL,
                    certainty=certainty,
                    priority=priority,
                    reason=(
                        f"可学习区(mid{certainty:.2f}∈({self.mid_threshold},{self.high_threshold}))"
                        f"且 priority={priority:.2f}≥{self.call_tool_priority}：CallTool 自我学习"
                    ),
                    ask_token=ASK_TOKEN,
                )
            return InquiryDecision(
                action=InquiryAction.ASK_QUESTION,
                certainty=certainty,
                priority=priority,
                reason=(
                    f"可学习区(certainty={certainty:.2f}∈({self.mid_threshold},{self.high_threshold})，"
                    f"RPL/LP 差一点就知道)：AskQuestion 请求解释"
                ),
                ask_token=ASK_TOKEN,
            )
        # ④ 完全空白区（certainty ≤ mid，§1.2 学习成本过高）→ Decline 诚实降级
        return InquiryDecision(
            action=InquiryAction.DECLINE,
            certainty=certainty,
            priority=priority,
            reason=(
                f"完全空白区(certainty={certainty:.2f}≤mid{self.mid_threshold})，"
                f"RPL 学习成本过高：Decline 诚实降级"
            ),
            ask_token=DECLINE_MESSAGE_TEMPLATE.format(certainty=certainty),
        )


class InquiryBranch(nn.Module):
    """求知分支——把求知闭环接入推理循环（sense→决策→审计，pilot）。

    持有：
      router: InquiryRouter（pilot 规则路由器）。
      kernel: TAISKernel | None（真实 KAL certainty 源；None 时由 maybe_inquire 的
          tick_state.certainty 提供——reasoning_loop 已含 mock/真实 certainty 逻辑）。
      blank_message: 诚实降级声明模板（缺省用 DECLINE_MESSAGE_TEMPLATE）。
    """

    def __init__(
        self,
        router: InquiryRouter | None = None,
        kernel: TAISKernel | None = None,
        blank_message: str = DECLINE_MESSAGE_TEMPLATE,
    ):
        super().__init__()
        self.router = router if router is not None else InquiryRouter()
        self.kernel = kernel
        self.blank_message = blank_message

    # ------------------------------------------------------------------
    @torch.no_grad()
    def read_certainty(self, state: torch.Tensor) -> float:
        """从真实 KAL（kernel.sense）读 certainty（只读，监测/执行分置红线）。

        state [B,T,d] → kernel.sense → softmax → known 类（类 0）概率末 token 均值。
        kernel=None 时抛错（调用方应走 tick_state.certainty 路径）。
        """
        if self.kernel is None:
            raise RuntimeError("InquiryBranch.kernel=None，无真实 KAL certainty 源")
        sense = self.kernel.sense(state)
        probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)  # [B,3]
        return float(probs[:, 0].mean().item())  # known 类（类 0）概率均值

    # ------------------------------------------------------------------
    def maybe_inquire(
        self,
        tick_state: ReasoningTickState,
        hrl_hit: bool,
        priority: float | None = None,
    ) -> InquiryDecision | None:
        """在 reasoning_tick 后判定是否进入求知分支。

        若 certainty 低（< high_threshold）且 HRL 未命中 → 返回 InquiryDecision
        （求知动作：AskQuestion/CallTool/Decline）；否则返回 None（继续正常 tick，
        DirectAnswer 路径）。certainty 取自 tick_state（reasoning_loop 已含真实/mock
        KAL 读出）；若持有真实 kernel 且需重评估，可另调 read_certainty（pilot 默认
        用 tick_state 避免重复 sense）。
        """
        if tick_state.certainty >= self.router.high_threshold:
            return None  # P(IK) 高 → DirectAnswer，不进求知分支
        if hrl_hit:
            return None  # HRL 命中有知识可答 → DirectAnswer，不进求知分支
        # certainty 低且未命中 → 求知分支（路由四选一：可学习区求知 / 空白区 Decline）
        decision = self.router.decide(tick_state.certainty, hrl_hit, priority=priority)
        # Decline 用本模块的诚实降级声明模板（router 缺省模板可被覆盖）
        if decision.action == InquiryAction.DECLINE:
            decision.ask_token = self.blank_message.format(certainty=decision.certainty)
        return decision

    # ------------------------------------------------------------------
    def inquiry_token(self, decision: InquiryDecision | None) -> str | None:
        """返回审计 token（显式显形化红线）。

        AskQuestion/CallTool→<|ask|>（求知动作）；Decline→诚实降级声明文本；
        DirectAnswer/None→None（无求知动作不显形，由 reasoning_loop 正常输出）。
        """
        if decision is None:
            return None
        if decision.action == InquiryAction.DIRECT_ANSWER:
            return None
        return decision.ask_token


class ActiveInquiryLoop(nn.Module):
    """主动求知推理循环（pilot 封装）——扩展 ReasoningLoop 接入求知分支。

    持有 ReasoningLoop（复用 §1.3 tick 动力学）+ InquiryBranch（求知决策与审计）。
    在 run 循环中：certainty 低且未命中时触发求知分支（记录 InquiryDecision 到轨迹），
    求知后**重评估 certainty**（检索/交互后 P(IK) 应升高，闭环）。pilot 阶段
    AskQuestion/CallTool 的实际执行（问用户/查文档）留接口——由外部执行器负责，
    本模块负责**决策与审计**（轨迹记录 + 显式 token）。

    返回的轨迹元素为 (ReasoningTickState, InquiryDecision | None) 元组——
    DirectAnswer tick 的 decision 为 None，求知 tick 的 decision 非 None。
    """

    def __init__(
        self,
        reasoning_loop: ReasoningLoop,
        inquiry_branch: InquiryBranch,
    ):
        super().__init__()
        self.reasoning_loop = reasoning_loop
        self.inquiry_branch = inquiry_branch

    # ------------------------------------------------------------------
    def run(
        self,
        initial_state: torch.Tensor,
        context: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        target_coord: torch.Tensor | None = None,
        hrl_hit_fn=None,
        priority_fn=None,
        inquiry_executor=None,
        max_ticks: int | None = None,
        stop_threshold: float = 0.9,
        recall_threshold: float = 0.3,
        hrl_k: int = 4,
        bridge_alpha: float = 0.1,
    ) -> tuple[torch.Tensor, list[tuple[ReasoningTickState, InquiryDecision | None]], int]:
        """多 tick 主动求知推理循环（§6 整合：certainty 低+未命中→求知分支→重评估闭环）。

        参数（reasoning_loop.run 同款 + 求知闭环接口）：
          hrl_hit_fn: callable(tick_index, state) -> bool，HRL 检索命中判定（pilot 缺省
              None = 恒未命中——走求知分支演示路径）。正式应接 kernel_orchestrator 的
              route/associative_recall 命中判定。
          priority_fn: callable(tick_index, tick_state) -> float | None，求知优先级
              （§1.2 三信号 EPIG×LP×(1−P(IK))；None 时退化按 certainty 分档）。
          inquiry_executor: callable(InquiryDecision) -> bool，求知动作外部执行器——
              执行 AskQuestion（问用户）/CallTool（查文档）并返回**求知是否成功获得新证据**。
              pilot 缺省 None = 不实际执行（只决策与审计）；返回 True 表示求知后应
              重评估 certainty（闭环）。Decline 不调用执行器（诚实降级无求知动作）。
          其余参数透传 reasoning_loop.reasoning_tick（见 reasoning_loop.py）。

        返回：(最终 state, 轨迹 list[(ReasoningTickState, InquiryDecision|None)],
               停止 tick 数 stop_tick)。
        """
        rl = self.reasoning_loop
        if initial_state.dim() != 3 or initial_state.shape[-1] != rl.thought_core.core_dim:
            raise ValueError(
                f"initial_state 须为 [B,T,{rl.thought_core.core_dim}]，"
                f"实得 {tuple(initial_state.shape)}"
            )
        K = max_ticks if max_ticks is not None else rl.thought_core.max_ticks
        if not 1 <= K <= rl.thought_core.max_ticks:
            raise ValueError(f"max_ticks={K} 须在 [1,{rl.thought_core.max_ticks}]")
        # 新一轮推理前清空思考核历史缓冲（ChannelGroupHistory 是 FIFO 有状态的）
        rl.thought_core.history.reset()
        state = initial_state.float()  # fp32 关键路径（数值稳定红线）
        trajectory: list[tuple[ReasoningTickState, InquiryDecision | None]] = []
        stop_tick = K
        for k in range(K):
            state, tick_state = rl.reasoning_tick(
                state, k,
                context=context, candidates=candidates, target_coord=target_coord,
                recall_threshold=recall_threshold, hrl_k=hrl_k, bridge_alpha=bridge_alpha,
            )
            # CTM 式自适应算力：certainty 达阈值提前停（确定→提前停）
            if tick_state.certainty > stop_threshold:
                tick_state.early_stop = True
                stop_tick = k + 1
                trajectory.append((tick_state, None))
                break
            # HRL 检索命中判定（pilot 缺省恒未命中；正式接 kernel_orchestrator route）
            hrl_hit = bool(hrl_hit_fn(k, state)) if hrl_hit_fn is not None else False
            priority = priority_fn(k, tick_state) if priority_fn is not None else None
            # 求知分支：certainty 低且未命中 → 决策（AskQuestion/CallTool/Decline）
            decision = self.inquiry_branch.maybe_inquire(tick_state, hrl_hit, priority=priority)
            if decision is not None and decision.action in (
                InquiryAction.ASK_QUESTION,
                InquiryAction.CALL_TOOL,
            ):
                # 求知动作由外部执行器执行（问用户/查文档）；成功获新证据 → 重评估闭环
                acquired = (
                    bool(inquiry_executor(decision)) if inquiry_executor is not None else False
                )
                if acquired:
                    # 闭环：检索/交互得新证据后重评估 certainty（P(IK) 应升高）。
                    # pilot 用当前 state 重读 KAL（reasoning_loop 真实/mock 源）；
                    # 正式应把新证据写知识块 → 流形位移到新知识点 → 重评估（§6）。
                    new_certainty = rl.kal_certainty(state)
                    tick_state.certainty = new_certainty
                    decision.reason += f"｜求知后重评估 certainty→{new_certainty:.2f}（闭环）"
            trajectory.append((tick_state, decision))
        return state, trajectory, stop_tick


__all__ = [
    "ASK_TOKEN",
    "DECLINE_MESSAGE_TEMPLATE",
    "ActiveInquiryLoop",
    "InquiryAction",
    "InquiryBranch",
    "InquiryDecision",
    "InquiryRouter",
    "RECALL_TOKEN",
]
