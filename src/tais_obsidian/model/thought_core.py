"""CTM 式思考核（Thought Core）——第二阶段（思维能力强化）迭代③ pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §1.2（CTM 式思考核）+ §1.3
（推理循环 tick 级动力学）。

**关键取舍（设计 §1.2，CTM 代价分析）**：CTM（Continuous Thought Machine，
arXiv:2505.05522，Sakana AI）的逐神经元时序处理 + O(D²×T) 同步矩阵在百万参数优雅，
在 1.5B 直接套用是灾难。**不全网套用，只抽取两原理做一个小核**：
1. **通道组级活动历史处理**（非逐神经元 NLM）——把 core_dim 分成 G 个通道组，每组
   维护最近 H 个 tick 的激活历史，用小 MLP 处理历史得到下一状态，降维到可负担；
2. **同步代理表征用相位化**——复用 RoPE 构造（RoPE 本质是相位，tri_attention.py
   half-split NeoX 风格）：给思考核一个额外的"思考时间" rotary 维度，第 k 个 tick =
   第 k 个相位步进，时序动力学以近零成本复用现有机制；
3. **GDN 递归状态天然扮演 CTM 的"持续状态"角色**（工作记忆寄存器，设计 §1.2 定位
   不变——pilot 阶段本核独立运行，迭代④才形式化接入推理循环）。

**思考核规格**（设计 §1.2）：256–512 维（默认 384）、K 个内部 tick（默认 max_ticks=8）、
挂 CSA 层旁。**pilot 阶段做成独立模块，不接 model.py 主干**（迭代④才形式化进推理循环）。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] CTM 论文实证（arXiv:2505.05522）：逐步时序动力学可自发涌现"一步一步推理"
  （39×39 迷宫注意力逐 tick 沿路径行走、超出训练步长泛化、ImageNet 注意力无监督沿物体
  轮廓扫描）；RoPE 相位编码位置（Su et al., arXiv:2104.09864）——本核的"思考时间
  相位化"复用同一数学构造。
- [推测/独创] 把 RoPE 相位从"token 位置"维度改用作"思考 tick"维度（第 k tick 施加
  第 k 相位步进）、以通道组级历史替代逐神经元 NLM——文献无先例（TAIS 独创外推，须经
  0.1B pilot 消融验证"相位化 vs 普通残差循环"的贡献）。
- [降预期] **CTM 语言域零证据**：CTM 论文 §12 自认语言域是 future work，民间复现
  （GitHub 社区反馈）在语言任务上负面——故本核**只做小核不做全网套用**，且
  `use_sync` 自消融开关为**默认必选项**（同步表征 vs 普通残差循环须自消融，验证
  相位化确实改变动力学才有保留价值；若消融无差异，应回检设计文档修订本模块）。

红线与纪律：
- **pilot 独立模块**：不接 model.py 主干前向逻辑；仅复用 manifold.py /
  manifold_bridge.py（不重复造投影/桥接）。
- **写纪律**：bridge.tick 写 PM-stream 是 steering 式有界加法（W1–W2 零梯度快写，
  幅度 clamp），绝不触碰权重——对齐 manifold_bridge.py / iti_head.py 写纪律。
- **梯度边界**：bridge 的 to_hidden 梯度由离线睡眠期显式目标提供，不经 tick 的
  steering 路径回流；think 循环内若启用 bridge 集成，位移路径 detach。
- **数值稳定**：相位化（ThoughtTimeRotary）与 MLP 前向的关键路径在 fp32 下计算。
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold import ThoughtManifoldProjector
from .manifold_bridge import ThoughtManifoldBridge


class ChannelGroupHistory:
    """通道组级活动历史（CTM 原理①的降维实现：非逐神经元，而是逐通道组）。

    把 core_dim 切成 G 个通道组（每组 core_dim/G 维），每组维护最近 H 个 tick 的
    激活历史（FIFO 环形语义：append 满 H 后挤掉最旧）。MLP 处理"历史"而非单点
    激活——对齐 CTM 的"活动历史"思想，但把粒度从神经元级降到通道组级（可负担）。

    历史张量约定：group_dim 维上的历史堆叠成 [B, T, G, H, group_dim]（dim=-2 为
    历史 tick 维，最新在末尾）。H 为容量上限（默认 4），不足 H 时左侧以零填充——
    保留固定形状（便于 MLP 定长输入），掩码由调用方按需处理（pilot 简化：零填充）。
    """

    def __init__(self, core_dim: int, n_groups: int, history: int):
        if core_dim % n_groups != 0:
            raise ValueError(f"core_dim={core_dim} 须被 n_groups={n_groups} 整除")
        if history < 1:
            raise ValueError(f"history 须 ≥1，实得 {history}")
        self.core_dim = core_dim
        self.n_groups = n_groups
        self.group_dim = core_dim // n_groups
        self.history = history
        self._buf: torch.Tensor | None = None  # [B, T, G, H, group_dim]

    def reset(self) -> None:
        """清空历史缓冲（新一轮 think 前调用）。"""
        self._buf = None

    def update(self, activations: torch.Tensor) -> torch.Tensor:
        """追加当前 tick 激活并返回更新后的历史（FIFO，长度 ≤ H）。

        参数：activations [B, T, core_dim]（本 tick 的通道组激活）。
        返回：历史 [B, T, G, H, group_dim]（左零填充至 H，最新 tick 在 dim=-2 末尾）。
        """
        if activations.dim() != 3 or activations.shape[-1] != self.core_dim:
            raise ValueError(
                f"activations 须为 [B,T,{self.core_dim}]，实得 {tuple(activations.shape)}"
            )
        B, T, _ = activations.shape
        grouped = activations.view(B, T, self.n_groups, self.group_dim)  # [B,T,G,gd]
        new_step = grouped.unsqueeze(-2)  # [B,T,G,1,gd]
        if self._buf is None:
            # 首轮：左侧零填充到 H（固定形状），最新在末尾
            pad = torch.zeros(
                B, T, self.n_groups, self.history - 1, self.group_dim,
                dtype=grouped.dtype, device=grouped.device,
            )
            self._buf = torch.cat([pad, new_step], dim=-2)  # [B,T,G,H,gd]
        else:
            if self._buf.shape[0] != B or self._buf.shape[1] != T:
                raise ValueError(
                    f"activations [B={B},T={T}] 与历史缓冲 [B={self._buf.shape[0]},"
                    f"T={self._buf.shape[1]}] 不一致（新一轮 think 前须 reset）"
                )
            # FIFO：挤掉最旧（dim=-2 首），追加最新
            self._buf = torch.cat([self._buf[..., 1:, :], new_step], dim=-2)
        return self._buf

    def get(self) -> torch.Tensor | None:
        """返回当前历史 [B,T,G,H,group_dim]；未 update 过则返回 None。"""
        return self._buf


class ThoughtTimeRotary(nn.Module):
    """思考时间相位化（CTM 原理②：同步代理表征 = 复用 RoPE 相位，近零成本）。

    构造与 tri_attention.py 的 RoPE 同一（half-split NeoX 风格 inv_freq/cos/sin
    buffer），但**维度语义是"思考时间"而非 token 位置**：第 k 个思考 tick 施加第 k
    个相位步进——把 CTM 的"时序动力学"以近零成本编码进表征（无额外参数，仅一次
    逐元素旋转）。

    [推测/独创] 把 RoPE 从 token 位置改用作思考 tick 相位——TAIS 独创外推（须
    use_sync 自消融验证贡献）。
    """

    def __init__(self, core_dim: int, max_ticks: int = 8, base: float = 10000.0):
        super().__init__()
        if core_dim % 2 != 0:
            raise ValueError(f"core_dim={core_dim} 须为偶数（RoPE 半分旋转）")
        self.core_dim = core_dim
        self.max_ticks = max_ticks
        self.base = base
        # 与 TriRetrievalAttention 同一构造：inv_freq [core_dim/2]，half-split NeoX 风格
        inv_freq = 1.0 / (base ** (torch.arange(0, core_dim, 2).float() / core_dim))
        t = torch.arange(max_ticks).float()
        freqs = torch.outer(t, inv_freq)  # [max_ticks, core_dim/2]
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def apply(self, x: torch.Tensor, tick_index: int) -> torch.Tensor:
        """对 x [B,T,core_dim] 施加第 tick_index 个思考相位的旋转（half-split NeoX）。

        参数：
            x: [B, T, core_dim] 待相位化的激活。
            tick_index: 思考 tick 序号（0 ≤ tick_index < max_ticks），取相位步进。
        返回：旋转后的 x（同形状、同 dtype；内部 fp32 关键路径）。
        """
        if x.dim() != 3 or x.shape[-1] != self.core_dim:
            raise ValueError(
                f"x 须为 [B,T,{self.core_dim}]，实得 {tuple(x.shape)}"
            )
        if not 0 <= tick_index < self.max_ticks:
            raise ValueError(
                f"tick_index={tick_index} 越界（须 0 ≤ k < max_ticks={self.max_ticks}）"
            )
        # fp32 关键路径（数值稳定红线）
        xf = x.float()
        cos = self.rope_cos[tick_index]  # [core_dim/2]
        sin = self.rope_sin[tick_index]
        cos = torch.cat([cos, cos], dim=-1)[None, None, :]  # [1,1,core_dim]
        sin = torch.cat([sin, sin], dim=-1)[None, None, :]
        x1, x2 = xf[..., : self.core_dim // 2], xf[..., self.core_dim // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        return (xf * cos + rot * sin).to(x.dtype)


class ThoughtCore(nn.Module):
    """CTM 式思考核主体：通道组历史 + 思考时间相位化 + 流形位移 + 自适应早停。

    持有：
      history: ChannelGroupHistory（通道组级活动历史，CTM 原理①降维）；
      rotary: ThoughtTimeRotary（思考时间相位化，CTM 原理②复用 RoPE）；
      group_mlp: nn.Linear（处理该组历史 → 该组下一状态的候选；参数量
                 H*group_dim → group_dim，逐组独立，组间不混）；
      bridge: ThoughtManifoldBridge（思考流形 ↔ PM-stream，驱动位移，复用迭代①）；
      use_sync: 自消融开关（True=思考时间相位化 [CTM 同步代理]；False=普通残差循环
                [消融对照，验证相位化贡献——诚实边界要求]）。

    pilot 阶段独立模块：不接 model.py 主干，think 循环由外部驱动（迭代④才形式化进
    §1.3 推理循环）。
    """

    def __init__(
        self,
        core_dim: int = 384,
        n_groups: int = 8,
        history: int = 4,
        max_ticks: int = 8,
        manifold_dim: int = 64,
        projector: ThoughtManifoldProjector | None = None,
        use_sync: bool = True,
        rotary_base: float = 10000.0,
    ):
        super().__init__()
        if not 256 <= core_dim <= 512:
            raise ValueError(f"core_dim={core_dim} 须在 [256,512] 区间（设计 §1.2 规格）")
        if core_dim % n_groups != 0:
            raise ValueError(f"core_dim={core_dim} 须被 n_groups={n_groups} 整除")
        if core_dim % 2 != 0:
            raise ValueError(f"core_dim={core_dim} 须为偶数（RoPE 半分旋转）")
        self.core_dim = core_dim
        self.n_groups = n_groups
        self.group_dim = core_dim // n_groups
        self.history_len = history
        self.max_ticks = max_ticks
        self.use_sync = use_sync

        self.history = ChannelGroupHistory(core_dim, n_groups, history)
        self.rotary = ThoughtTimeRotary(core_dim, max_ticks=max_ticks, base=rotary_base)
        # 每组一个小 MLP：处理该组历史 [H*group_dim] → 该组下一状态 [group_dim]
        # （历史经 fp32 平坦化后输入；GELU 非线性，近零成本小核）
        self.group_mlp = nn.Sequential(
            nn.Linear(history * self.group_dim, self.group_dim),
            nn.GELU(),
            nn.Linear(self.group_dim, self.group_dim),
        )
        # 流形桥：思考流形 ↔ PM-stream，驱动位移（复用迭代①的 projector/bridge，
        # 不重复造）。pilot 阶段默认持有（迭代④接入推理循环时由外部共享传入）。
        self.bridge = ThoughtManifoldBridge(
            d_model=core_dim, manifold_dim=manifold_dim, projector=projector
        )
        self.manifold_dim = manifold_dim

    def forward_step(
        self,
        state: torch.Tensor,
        tick_index: int,
    ) -> torch.Tensor:
        """单 tick：通道组历史经 group_mlp 得候选下一状态 → 施加思考时间相位。

        参数：
            state: [B, T, core_dim] 当前思考状态（GDN 持续状态角色，工作记忆寄存器）。
            tick_index: 当前思考 tick 序号（0 ≤ k < max_ticks），取相位步进。
        返回：新状态 [B, T, core_dim]（残差循环：state + 候选增量；use_sync 时增量
              经思考时间相位化）。
        """
        if state.dim() != 3 or state.shape[-1] != self.core_dim:
            raise ValueError(
                f"state 须为 [B,T,{self.core_dim}]，实得 {tuple(state.shape)}"
            )
        B, T, _ = state.shape
        hist = self.history.update(state)  # [B,T,G,H,gd]（FIFO，最新在末尾）
        # 逐组处理：历史平坦化 [B,T,G,H*gd] → group_mlp 逐组 [B,T,G,gd]
        hist_flat = hist.reshape(B, T, self.n_groups, self.history_len * self.group_dim)
        # fp32 关键路径（数值稳定红线）
        cand = self.group_mlp(hist_flat.float())  # [B,T,G,gd]
        cand = cand.reshape(B, T, self.core_dim)
        if self.use_sync:
            # CTM 同步代理：候选增量经思考时间相位化（第 tick_index 个相位步进）
            cand = self.rotary.apply(cand, tick_index)
        # 普通残差循环（use_sync=False 时即纯 MLP 演化，消融对照）
        return state.float() + cand

    def think(
        self,
        initial_state: torch.Tensor,
        certainty_fn: Callable[[torch.Tensor], float] | None = None,
        max_ticks: int | None = None,
        stop_threshold: float = 0.9,
        integrate_bridge: bool = False,
        bridge_target: torch.Tensor | None = None,
        bridge_alpha: float = 0.1,
    ) -> tuple[torch.Tensor, list[torch.Tensor], int]:
        """多 tick 思考循环（CTM 式自适应算力：确定→提前停，空白→跑满）。

        参数：
            initial_state: [B, T, core_dim] 初始思考状态（GDN 持续状态读出）。
            certainty_fn: KAL 式 certainty 评估函数 state → 标量 certainty ∈ [0,1]
                （>stop_threshold 则早停）。缺省则跑满 max_ticks。pilot 阶段为简化
                接口（不强行接真实 KAL，留接口位；迭代④由 KAL P(IK) 实现）。
            max_ticks: 本轮回合的最大 tick 数（缺省用 self.max_ticks）。
            stop_threshold: certainty 早停阈值（>此值提前停，CTM 式自适应算力）。
            integrate_bridge: True 时每 tick 经 bridge.tick 驱动流形位移并写 PM-stream
                （对齐 §1.3"流形上位移一步 → 写 PM-stream"）。pilot 默认 False
                （独立模块，bridge 集成为可选验证项）。
            bridge_target: [B,T,manifold_dim] 或 [B,manifold_dim] 目标流形坐标
                （integrate_bridge=True 时须给）。
            bridge_alpha: bridge 写入强度（clamp 到 writer 上限，W1–W2 零梯度快写）。
        返回：(最终状态 [B,T,core_dim], tick 轨迹 list[Tensor]（每 tick 后的状态），
               停止 tick 数 stop_tick)。轨迹长度 = 停止 tick 数（早停则 < max_ticks）。
        梯度边界：integrate_bridge 时 bridge 位移路径 detach（steering 推理期干预，
        非梯度路径）；group_mlp 参数经 think 的残差循环正常反传。
        """
        if initial_state.dim() != 3 or initial_state.shape[-1] != self.core_dim:
            raise ValueError(
                f"initial_state 须为 [B,T,{self.core_dim}]，实得 {tuple(initial_state.shape)}"
            )
        K = max_ticks if max_ticks is not None else self.max_ticks
        if not 1 <= K <= self.max_ticks:
            raise ValueError(f"max_ticks={K} 须在 [1,{self.max_ticks}]")
        self.history.reset()  # 新一轮思考前清空历史缓冲
        state = initial_state.float()  # fp32 关键路径
        trajectory: list[torch.Tensor] = []
        stop_tick = K  # 缺省跑满
        for k in range(K):
            state = self.forward_step(state, k)  # [B,T,core_dim]
            if integrate_bridge:
                if bridge_target is None:
                    raise ValueError("integrate_bridge=True 须提供 bridge_target")
                # 流形位移 → 写 PM-stream（§1.3 tick 闭环；detach 由 bridge 内部保证）
                state, _, _ = self.bridge.tick(state, bridge_target, alpha=bridge_alpha)
            trajectory.append(state)
            # CTM 式自适应算力：certainty 达阈值提前停
            if certainty_fn is not None:
                cert = float(certainty_fn(state))
                if cert > stop_threshold:
                    stop_tick = k + 1
                    break
        final_state = trajectory[-1] if trajectory else state
        return final_state, trajectory, stop_tick


__all__ = [
    "ChannelGroupHistory",
    "ThoughtCore",
    "ThoughtTimeRotary",
]
