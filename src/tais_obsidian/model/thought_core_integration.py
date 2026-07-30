"""思考核接入主干前向（Thought-Core Backbone Integration）——第二阶段"能力证明"关键一跳。

背景（fb1 硬约束，/memories/repo/fb1-feedback-verification.md P1）：第二阶段思考核/
推理循环此前是 **pilot 独立模块**（thought_core.py / reasoning_loop.py，不接 model.py
主干前向）。真实部件适配（thinking_real_adapter_demo.py）只验证了"思考核多 tick 演化
改变思考动力学（轨迹/位移）"，但 **dist_core≈dist_no_core——对最终任务结果无可测量
增益（"思考核未挣到自己的 FLOPs"）**。fb1 判读："从模块到原生的门槛只有一个：接入
主干前向后，在推理基准上产生可测量的增益（哪怕 1-2 个点）"。

本模块把思考核接入 model.py 主干前向（**可选路径，默认关，向后兼容**）：
  - **接入点**：最终 norm_f 前——对主干最末层输出 hidden state [B,T,768] 做
    ThoughtCore.think 多 tick 演化，把"思考动力学"作用到 logits 前的最后表征上。
  - **维度桥接**：真实模型 d_model=768，ThoughtCore core_dim=384——down_proj 768→384
    降维进核，up_proj 384→768 升维回主干（复用 RealThoughtAdapter 的桥接思路）。
  - **有界演化**：tick 数有界（max_ticks，CTM 式 certainty 早停）；演化增量经
    **zero-init 输出门** 缩放后残差加回（初始恒等——zero-init 保证"随机/未训核在前向
    不改变 logits"，这是 dist_core≈dist_no_core 的结构根因修复点：门可学，思考才挣得到
    FLOPs）；certainty 早停基于核状态范数（sigmoid，有界）。
  - **监测/执行分置**：演化在主干前向内联，但核参数独立于主干（可选 detach
    主干→核输入，防训练期思考核梯度污染主干——默认 detach=True 对齐 HRL 梯度隔离红线）。

证据分级（写作纪律：区分已确立与独创外推）：
  - [已确立] CTM（arXiv:2505.05522）：逐步时序动力学；RoPE 相位（arXiv:2104.09864）；
    zero-init 门控残差（GPT-2 残差投影缩小初始化 / ReZero arXiv:2003.04887 同族——
    初始恒等、训练期门逐渐打开）。
  - [推测/独创] 把思考核作为"最终表征精炼器"内联进主干前向（logits 前多 tick 演化）——
    文献无先例（TAIS 独创外推，须经 0.1B 推理基准有核/无核对照验证增益）。
  - [降预期] CTM 语言域零证据（arXiv:2505.05522 §12 + 民间复现负面）；未训/随机核
    经 zero-init 门后增益≈0——**基准增益须来自已训核（离线训练打开门），随机核仅通路
    验证**（诚实标注：本模块不臆造未训核的增益）。

红线与纪律：
  - **可选路径默认关**：use_thought_core=False（默认）时前向与现状逐行一致（357 测试零
    改动）；True 且已 attach_thought_core 才走核演化路径。
  - **有界写回**：演化增量 = up_proj(think(down_proj(h))) − h 经 zero-init 门 alpha 缩放，
    h_out = h + tanh(alpha)·Δ（tanh 有界，防未训大幅值核污染 logits；alpha 可学）。
  - **梯度边界**：detach_backbone=True（默认）时核输入 detach（思考核梯度不回主干——
    对齐"HRL 梯度隔离：辅助路径梯度禁止污染主干"红线）；核自身参数（group_mlp/proj/门）
    正常反传（供离线训练打开门）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .thought_core import ThoughtCore


class ThoughtCoreIntegration(nn.Module):
    """思考核→主干前向的桥接集成（768↔384 维度桥 + zero-init 输出门 + 有界演化）。

    持有：
      down_proj: Linear(d_model→core_dim)（主干 hidden 降到思考核维度）；
      core: ThoughtCore（CTM 式思考核，多 tick 演化，复用 thought_core.py）；
      up_proj: Linear(core_dim→d_model)（核状态升回主干维度）；
      gate_alpha: nn.Parameter（标量，zero-init——初始恒等，tanh 有界门控；
                   训练期逐渐打开让思考增量流入 logits，"挣到 FLOPs"的结构开关）；
      detach_backbone: 是否 detach 主干→核输入（默认 True，梯度隔离红线）。

    forward(h)：h [B,T,d_model] → 精炼后 h' [B,T,d_model]（同形状，残差有界加回）。
    """

    def __init__(
        self,
        d_model: int,
        core_dim: int = 384,
        n_groups: int = 8,
        history: int = 4,
        max_ticks: int = 8,
        manifold_dim: int = 64,
        use_sync: bool = True,
        detach_backbone: bool = True,
        stop_threshold: float = 0.9,
    ):
        super().__init__()
        if not 256 <= core_dim <= 512:
            raise ValueError(f"core_dim={core_dim} 须在 [256,512]（思考核规格 §1.2）")
        self.d_model = d_model
        self.core_dim = core_dim
        self.max_ticks = max_ticks
        self.stop_threshold = stop_threshold
        self.detach_backbone = detach_backbone

        self.down_proj = nn.Linear(d_model, core_dim)
        self.core = ThoughtCore(
            core_dim=core_dim, n_groups=n_groups, history=history,
            max_ticks=max_ticks, manifold_dim=manifold_dim, use_sync=use_sync,
        )
        self.up_proj = nn.Linear(core_dim, d_model)
        # zero-init 输出门（ReZero 同族）：初始 alpha=0 → tanh(0)=0 → 输出=h（恒等）。
        # 这是结构根因修复：随机/未训核在前向不改变 logits（dist_core≈dist_no_core 的
        # 根源是思考增量无处可去）；门可学 → 离线训练打开 → 思考增量流入 logits。
        self.gate_alpha = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------
    def _certainty(self, state: torch.Tensor) -> float:
        """CTM 式早停 certainty：核状态末 token 平均范数的 sigmoid ∈ [0,1]（有界）。

        [pilot 占位] 正式应接 KAL P(IK)（isotonic 校准概率）；此处用状态范数 sigmoid
        作有界占位（保证 ∈[0,1] 供早停逻辑流转，不依赖外部 kernel 信号）。
        """
        with torch.no_grad():
            norm = state.float()[:, -1, :].norm(dim=-1).mean()
            return float(torch.sigmoid(norm).item())

    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        max_ticks: int | None = None,
        return_diagnostics: bool = False,
    ):
        """对主干 hidden h [B,T,d_model] 做有界思考核演化，残差加回。

        参数：
            h: [B, T, d_model] 主干最末层输出（norm_f 前）。
            max_ticks: 本轮回合最大 tick 数（缺省用 self.max_ticks；有界演化）。
            return_diagnostics: True 时返回 (h_out, diagnostics dict)（演化诊断：
                stop_tick / 增量范数 / 门值，供有核/无核对照审计）。
        返回：h_out [B,T,d_model]（默认）或 (h_out, diagnostics)。
        """
        if h.dim() != 3 or h.shape[-1] != self.d_model:
            raise ValueError(
                f"h 须为 [B,T,{self.d_model}]，实得 {tuple(h.shape)}"
            )
        # 主干→核（可选 detach：梯度隔离红线，默认开）
        h_in = h.detach() if self.detach_backbone else h
        state0 = self.down_proj(h_in.float())  # [B,T,core_dim]，fp32 关键路径
        # 多 tick 演化（CTM 式 certainty 早停；有界 max_ticks）
        final_state, trajectory, stop_tick = self.core.think(
            state0, certainty_fn=self._certainty,
            max_ticks=max_ticks, stop_threshold=self.stop_threshold,
        )
        # 核→主干：演化增量经 zero-init 门（tanh 有界）残差加回
        delta = self.up_proj(final_state.float()) - h_in.float()  # [B,T,d_model]
        gate = torch.tanh(self.gate_alpha)  # 有界 (−1,1)，初始 0（恒等）
        h_out = h.float() + gate * delta
        h_out = h_out.to(h.dtype)
        if return_diagnostics:
            diag = {
                "stop_tick": stop_tick,
                "n_ticks": len(trajectory),
                "gate": float(gate.item()),
                "delta_norm": float(delta.norm().item()),
                "delta_gated_norm": float((gate * delta).norm().item()),
            }
            return h_out, diag
        return h_out


__all__ = ["ThoughtCoreIntegration"]
