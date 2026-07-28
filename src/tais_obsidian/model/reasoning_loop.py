"""推理循环形式化（Reasoning Loop）——第二阶段（思维能力强化）迭代④ pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §1.3（推理循环 tick 级动力学）：

    每个思考 tick：
      GDN 状态（位置/工作记忆寄存器）
        → CSA/TriRetrieval 注意力（观察/glimpse——CTM 证明它自发学会"往哪看"）
        → HRL 提议移动方向（Indexer 精确检索 + CA3 PPR 联想 = 地图与指南针）
        → KAL 评估 certainty（CTM 式自适应算力：确定→提前停，空白→<|recall|>）
        → 流形上位移一步 → 写 PM-stream（感知-记忆专用道）

本模块把迭代③的思考核（thought_core.py）串进这条完整 tick 动力学：把"段落级
recall 事件"细化为"tick 级动力学"，由外部提供各部件（kernel/thought_core/bridge），
本模块负责按 §1.3 顺序驱动一个 tick 的完整流转。**pilot 阶段做成独立编排模块，
不接 model.py 主干前向**（正式接入属后续 milestone）。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] CTM（arXiv:2505.05522）：逐步时序动力学自发学会"往哪看"（注意力逐
  tick 沿路径行走）；KAL P(IK) 探针（SAPLMA/量化态探针 0.904–1.000 AUROC，kal.py）；
  ITI steering 加法干预有界写纪律（arXiv:2306.03341 / 2505.22637）。
- [推测/独创] 把 §1.3 五步（读出→glimpse→HRL 提议→KAL 门控→流形位移写 PM）
  串成可编排的 tick 闭环、每 tick 记录可审计 ReasoningTickState——文献无先例
  （TAIS 独创外推，须经 0.1B pilot 验证"tick 级 recall 闭环 + 自适应早停"判据）。

红线与纪律：
- **监测/执行分置**（子系统架构规格 Part B）：sense 只读 GDN 层 PM-stream（KAL/HRL
  信号，零副作用），bridge.tick 只写 PM-stream（W1–W2 零梯度 steering 快写）——
  读写不同通道，防探针读到自己的干预自激。
- **梯度边界**：bridge.tick 的 steering 路径 detach（非训练梯度路径）；本循环的
  可反传路径是 thought_core.group_mlp（残差循环正常反传）。sense/route 只读信号
  默认 detach（MoE-RL 红线：辅助损失梯度禁止污染主干）。
- **空白→<|recall|> 审计红线**：KAL 判空白（certainty < recall_threshold）时
  ReasoningTickState 显式标记 recall_triggered；trajectory_to_recall_tokens 把轨迹
  转成 <|recall|> 显式 token 序列——对齐"`<|recall|>` 必须显式出现在 CoT 中"红线。
- **pilot 占位标注**：glimpse（注意力观察）与 hrl_propose（HRL 提议）在 kernel=None
  时用 mock/None 占位，注释明确"接口位，正式应接 CSA 注意力 / HRL Indexer"。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .manifold_bridge import ThoughtManifoldBridge
from .tais_kernel import TAISKernel
from .thought_core import ThoughtCore

# <|recall|> 审计 token 字面量（对齐设计 §4.2/§17.2 忠实性纪律：空白显式显形化）
RECALL_TOKEN = "<|recall|>"


@dataclass
class ReasoningTickState:
    """单个思考 tick 的状态记录（审计/可解释性前端用，对齐 §1.3 tick 动力学）。

    字段即 §1.3 五步的可审计读出：
      tick_index: 思考 tick 序号（0 起）。
      current_coord: 本 tick 流形坐标读出 [B,T,manifold_dim]（bridge.tick 返回的
          current_coord，位移前坐标）。
      disp: 本 tick 流形位移 [B,T,manifold_dim]（target − current，朝 target 方向）。
      certainty: KAL P(IK) 读出标量 ∈ [0,1]（known 类概率；mock 时见 kal_certainty）。
      hrl_topk_idx: HRL 提议的 top-k 方向索引 [B,Tq,k]（kernel.route_candidates 返回；
          kernel=None 或 candidates=None 时为 None——接口位）。
      early_stop: 本 tick 是否触发早停（certainty > stop_threshold，CTM 式自适应算力）。
      recall_triggered: 本 tick 是否触发 <|recall|>（certainty < recall_threshold，
          空白检测——审计接口，对齐"空白→<|recall|>"红线）。
    """

    tick_index: int
    current_coord: torch.Tensor
    disp: torch.Tensor
    certainty: float
    hrl_topk_idx: torch.Tensor | None = None
    early_stop: bool = False
    recall_triggered: bool = False


def trajectory_to_recall_tokens(trajectory: list[ReasoningTickState]) -> list[str]:
    """把 tick 轨迹转成 <|recall|> 显式 token 序列（审计接口）。

    红线对齐（设计 §4.2/§17.2）："`<|recall|>` 必须显式出现在 CoT 中"——每个触发
    空白的 tick（recall_triggered=True）在输出序列中显式标出 <|recall|>，其余 tick
    以 tick 序号占位（供可解释性前端对照渲染）。返回长度 = 轨迹长度。
    """
    return [
        RECALL_TOKEN if ts.recall_triggered else f"<|tick_{ts.tick_index}|>"
        for ts in trajectory
    ]


class ReasoningLoop(nn.Module):
    """推理循环编排器（§1.3 tick 级动力学，pilot 独立编排模块）。

    持有（全部由外部提供，复用迭代①③已有部件，不重复造）：
      thought_core: ThoughtCore（迭代③思考核，forward_step 演化 + 残差循环）；
      bridge: ThoughtManifoldBridge（思考流形 ↔ PM-stream 桥，tick 驱动位移写回）；
      kernel: TAISKernel | None（KAL/HRL 真实信号源；pilot 可为 None 用 mock——
              glimpse/certainty 占位、propose 返回 None，接口位见各方法注释）。

    pilot 阶段不接 model.py 主干前向（正式接入属后续 milestone）；本模块只负责
    按 §1.3 顺序驱动一个 tick 的完整流转（读出→glimpse→提议→门控→演化→位移写 PM）。
    """

    def __init__(
        self,
        thought_core: ThoughtCore,
        bridge: ThoughtManifoldBridge,
        kernel: TAISKernel | None = None,
    ):
        super().__init__()
        self.thought_core = thought_core
        self.bridge = bridge
        self.kernel = kernel
        # glimpse 观察的轻量读出投影（pilot 占位：mean-pool + Linear，接口位见 glimpse）
        # 维度 = thought_core.core_dim（观察向量与状态同维，供 HRL propose 打分）。
        self.obs_proj = nn.Linear(thought_core.core_dim, thought_core.core_dim)

    # ------------------------------------------------------------------
    def glimpse(
        self,
        state: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """观察步（§1.3"CSA/TriRetrieval 注意力 glimpse——CTM 证明它自发学会往哪看"）。

        [pilot 占位] 简化为对 state 做一次轻量池化读出观察向量：
        mean-pool 时间维 → obs_proj Linear → [B, core_dim]。
        **接口位：迭代④正式应调 CSA/TriRetrieval 注意力 glimpse**（对 context 做
        选择性注意观察，CTM 式"往哪看"）；本占位实现暂不使用 context（保留参数位）。
        """
        # state [B,T,core_dim] → mean-pool T → [B,core_dim] → Linear 观察向量
        obs = state.float().mean(dim=1)  # [B, core_dim]
        return self.obs_proj(obs)

    # ------------------------------------------------------------------
    def hrl_propose(
        self,
        obs: torch.Tensor,
        candidates: torch.Tensor | None,
        k: int = 4,
    ) -> torch.Tensor | None:
        """HRL 提议移动方向（§1.3"Indexer 精确检索 + CA3 PPR 联想 = 地图与指南针"）。

        有 kernel 且 candidates 给定时：调 kernel.route_candidates(obs, candidates, k)
        取 top-k（真正的 CSA Indexer 路径，DSA lightning indexer 式；detach 隔离主干）。
        obs [B,core_dim] → unsqueeze 成 [B,1,core_dim] 作 query；返回 top-k 索引 [B,1,k]。
        无 kernel 或 candidates=None 时返回 None——**接口位：正式应接 HRL Indexer +
        CA3 PPR 联想（kernel_orchestrator.associative_recall）**。
        """
        if self.kernel is None or candidates is None:
            return None
        query = obs.unsqueeze(1) if obs.dim() == 2 else obs  # [B,1,d] 或 [B,Tq,d]
        _, top_idx = self.kernel.route_candidates(query, candidates, k)
        return top_idx

    # ------------------------------------------------------------------
    def kal_certainty(self, state: torch.Tensor) -> float:
        """KAL 评估 certainty（§1.3"CTM 式自适应算力：确定→提前停，空白→<|recall|>"）。

        有 kernel 时：调 kernel.sense(state) 取 pik_logits → softmax → known 类（类 0，
        kal.py L1 三态 知道/不确定/空白）概率，末 token 均值成标量 ∈ [0,1]。
        **监测/执行分置红线：sense 只读（detach，零副作用，不建梯度路径）**。

        无 kernel 时（pilot mock）：用 state 末 token 平均范数的 sigmoid 作 mock
        certainty ∈ [0,1]——**接口位：正式应接 KAL L1 P(IK) 探针（kal.py，isotonic
        校准后概率）**；mock 仅保证输出落在 [0,1] 供早停/空白逻辑流转。
        """
        if self.kernel is not None:
            with torch.no_grad():  # sense 只读（监测），不建梯度路径（红线）
                sense = self.kernel.sense(state)
                probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)  # [B,3]
                return float(probs[:, 0].mean().item())  # known 类（类 0）概率均值
        # [pilot mock] state 末 token 平均范数的 sigmoid ∈ [0,1]（无 kernel 占位）
        with torch.no_grad():
            norm = state.float()[:, -1, :].norm(dim=-1).mean()
            return float(torch.sigmoid(norm).item())

    # ------------------------------------------------------------------
    def should_recall(self, certainty: float, threshold: float = 0.3) -> bool:
        """空白检测（§1.3"空白→<|recall|>"）：certainty < threshold 则触发 <|recall|>。

        审计接口（对齐"空白→<|recall|>"红线）：低 certainty = KAL 判知识空白 →
        显式标记，供 trajectory_to_recall_tokens 转成显式 <|recall|> token。
        """
        return certainty < threshold

    # ------------------------------------------------------------------
    def reasoning_tick(
        self,
        state: torch.Tensor,
        tick_index: int,
        context: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        target_coord: torch.Tensor | None = None,
        recall_threshold: float = 0.3,
        hrl_k: int = 4,
        bridge_alpha: float = 0.1,
    ) -> tuple[torch.Tensor, ReasoningTickState]:
        """单 tick 完整流转（§1.3 顺序：读出→glimpse→提议→门控→演化→位移写 PM）。

        ① GDN 状态读出（state 即持续状态/工作记忆寄存器，本方法入参）→
        ② glimpse 观察（注意力 glimpse，pilot 占位）→
        ③ hrl_propose 方向（HRL Indexer top-k，kernel=None 时 None）→
        ④ kal_certainty 评估（KAL P(IK)，CTM 式自适应算力门控）→
        ⑤ thought_core.forward_step 演化（通道组历史 + 思考时间相位化残差循环）→
        ⑥ bridge.tick 流形位移写 PM-stream（读坐标→算位移→反投影→有界写回）。

        参数：
            state: [B,T,core_dim] 当前思考状态（GDN 持续状态读出）。
            tick_index: 思考 tick 序号（0 ≤ k < thought_core.max_ticks，相位步进用）。
            context: 可选上下文（glimpse 正式注意力的观察对象；pilot 占位未用）。
            candidates: 可选 HRL 候选块表示 [B,Tk,d]（route_candidates 打分对象）。
            target_coord: [B,T,manifold_dim] 或 [B,manifold_dim] 目标流形坐标
                （bridge.tick 位移目标；须给——§1.3"流形上位移一步"的方向）。
            recall_threshold: 空白阈值（certainty < 此值标记 recall_triggered）。
            hrl_k: HRL 提议 top-k 数。
            bridge_alpha: bridge 写入强度（clamp 到 writer 上限，W1–W2 零梯度快写）。
        返回：(新 state [B,T,core_dim], ReasoningTickState 审计记录)。
        """
        if target_coord is None:
            raise ValueError("reasoning_tick 须提供 target_coord（流形位移目标，§1.3）")
        # ① GDN 状态读出 = 入参 state（持续状态/工作记忆寄存器，无需额外操作）
        # ② glimpse 观察（pilot 占位：mean-pool+Linear；正式应接 CSA/TriRetrieval 注意力）
        obs = self.glimpse(state, context)
        # ③ HRL 提议移动方向（kernel=None/candidates=None 时 None——接口位）
        topk_idx = self.hrl_propose(obs, candidates, k=hrl_k)
        # ④ KAL 评估 certainty（sense 只读；mock 时见 kal_certainty）
        certainty = self.kal_certainty(state)
        recall_triggered = self.should_recall(certainty, threshold=recall_threshold)
        # ⑤ thought_core 演化（通道组历史 + 思考时间相位化，残差循环——可反传路径）
        new_state = self.thought_core.forward_step(state, tick_index)
        # ⑥ bridge.tick 流形位移写 PM-stream（steering 有界快写；位移路径 detach）
        new_state, current_coord, disp = self.bridge.tick(
            new_state, target_coord, alpha=bridge_alpha
        )
        tick_state = ReasoningTickState(
            tick_index=tick_index,
            current_coord=current_coord.detach(),
            disp=disp.detach(),
            certainty=certainty,
            hrl_topk_idx=topk_idx.detach() if topk_idx is not None else None,
            early_stop=False,  # 由 run() 在循环层判定后回填
            recall_triggered=recall_triggered,
        )
        return new_state, tick_state

    # ------------------------------------------------------------------
    def run(
        self,
        initial_state: torch.Tensor,
        context: torch.Tensor | None = None,
        candidates: torch.Tensor | None = None,
        target_coord: torch.Tensor | None = None,
        max_ticks: int | None = None,
        stop_threshold: float = 0.9,
        recall_threshold: float = 0.3,
        hrl_k: int = 4,
        bridge_alpha: float = 0.1,
    ) -> tuple[torch.Tensor, list[ReasoningTickState], int]:
        """多 tick 推理循环（§1.3 tick 动力学 + CTM 式自适应算力早停）。

        每 tick 调 reasoning_tick 完整流转；certainty > stop_threshold 时早停
        （确定→提前停，CTM 式自适应算力），并把该 tick 的 ReasoningTickState
        标记 early_stop=True。certainty 始终低则跑满 max_ticks。

        参数：
            initial_state: [B,T,core_dim] 初始思考状态（GDN 持续状态读出）。
            context/candidates/target_coord: 透传 reasoning_tick（见上）。
            max_ticks: 本轮回合最大 tick 数（缺省用 thought_core.max_ticks）。
            stop_threshold: 早停阈值（certainty > 此值提前停，CTM 式自适应算力）。
            recall_threshold: 空白阈值（certainty < 此值标记 <|recall|>）。
        返回：(最终 state [B,T,core_dim], 轨迹 list[ReasoningTickState],
               停止 tick 数 stop_tick)。轨迹长度 = stop_tick（早停则 < max_ticks）。
        """
        if initial_state.dim() != 3 or initial_state.shape[-1] != self.thought_core.core_dim:
            raise ValueError(
                f"initial_state 须为 [B,T,{self.thought_core.core_dim}]，"
                f"实得 {tuple(initial_state.shape)}"
            )
        K = max_ticks if max_ticks is not None else self.thought_core.max_ticks
        if not 1 <= K <= self.thought_core.max_ticks:
            raise ValueError(f"max_ticks={K} 须在 [1,{self.thought_core.max_ticks}]")
        # 新一轮推理前清空思考核历史缓冲（ChannelGroupHistory 是 FIFO 有状态的）
        self.thought_core.history.reset()
        state = initial_state.float()  # fp32 关键路径（数值稳定红线）
        trajectory: list[ReasoningTickState] = []
        stop_tick = K  # 缺省跑满
        for k in range(K):
            state, tick_state = self.reasoning_tick(
                state, k,
                context=context, candidates=candidates, target_coord=target_coord,
                recall_threshold=recall_threshold, hrl_k=hrl_k, bridge_alpha=bridge_alpha,
            )
            # CTM 式自适应算力：certainty 达阈值提前停（确定→提前停）
            if tick_state.certainty > stop_threshold:
                tick_state.early_stop = True
                stop_tick = k + 1
                trajectory.append(tick_state)
                break
            trajectory.append(tick_state)
        final_state = state
        return final_state, trajectory, stop_tick


__all__ = [
    "RECALL_TOKEN",
    "ReasoningLoop",
    "ReasoningTickState",
    "trajectory_to_recall_tokens",
]
