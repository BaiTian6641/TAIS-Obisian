"""ITI 干预头（KAL 执行通道，监测→干预闭环；规范 §5）。

设计依据（article_ref/07 §5，逐条已核实 + 本项目实证）：
- **ITI**（Li et al. 2306.03341，NeurIPS 2023）：沿真实度方向 mass-mean shift 激活
  `h ← h + α(μ⁺−μ⁻)`，TruthfulQA 32.5→65.1%。方向 = diff-in-means（真/假类均值差）。
- **本项目实证（2026-07-26）**：kal_l1 真值方向（W[know]−W[blank]）与 know 轴 cos=0.988、
  blank 轴 −0.989（几何完美的真值分界方向）；沿其 steer 残差，fake 的 P(空白) 在
  α≈0.2×残差 norm 翻转——**0.1B ITI steer 有效**。
- **红线级警示（Braun 2505.22637 + 本项目洞察）**：① steer 是**双刃剑**——同一方向能把
  fake 推向 know（造假，坏）也能把不确定推向真（好）；② 小模型对 steering 退化更敏感，
  α 过大致连贯性/忠实性崩。→ **ITI 必须门控 + α 有界 + 仅触发**：
  * **绝不把"空白" steer 成"知道"**（那是让模型对无知内容自信=造假，违反诚实降级红线）；
  * ITI 应用于**确定正确方向**的场景：L3 冲突时沿参数知识方向、或生成中保持真实度；
  * "steering 后人效不降"纳入退出标准（对齐 M5 Δ+0.0001 纪律）。

监测/执行分置（红线）：sense 读 GDN 输出层（监测只读），ITI steer 写 CSA 残差前层
（执行写入）——不同层，防探针读到自己的干预自激。本模块是**执行侧**（写）。

本模块定位：KAL 的"手"（相对 sense 的"眼"）。方向来自内核已训头（diff-in-means），
触发门来自编排层 KAL 决策（L1 空白/L3 冲突），α 按残差 norm 比例有界。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ITIHead:
    """ITI 干预头（执行通道）：门控的真实度方向 steer。

    非可学习（方向来自内核已训头 diff-in-means，不新增参数——ITI 方向与探针同源，
    避免独立训练导致方向与检测错位）。红线：α 有界、仅触发时、单方向、绝不造假。
    """

    def __init__(self, direction: torch.Tensor, max_alpha_frac: float = 0.2):
        """direction [d]：steer 方向（须已归一化，如 kal_l1 真值方向 W[know]−W[blank]）。

        max_alpha_frac：α 上限 = 该分数 × 残差流 norm（Braun：±0.1–0.2×norm 级，
        小模型保守取 0.2 以下）。>0.5×norm 已实证会翻转探针读数（过强）。
        """
        if not 0.0 < max_alpha_frac <= 1.0:
            raise ValueError("max_alpha_frac 须在 (0,1]")
        self.direction = F.normalize(direction.detach().float(), dim=0)
        self.max_alpha_frac = max_alpha_frac

    @classmethod
    def from_kal_l1(cls, kal_l1_head, max_alpha_frac: float = 0.2) -> "ITIHead":
        """从 KAL L1 真值头派生 ITI 方向（W[know=0]−W[blank=2]，diff-in-means）。

        与探针同源保证方向与检测一致（不独立训练）。kal_l1_head.proj.weight [3,d]。
        """
        W = kal_l1_head.proj.weight.detach().float()
        direction = W[0] - W[2]  # 知道 − 空白（真值分界方向）
        return cls(direction, max_alpha_frac=max_alpha_frac)

    def steer(
        self,
        hidden: torch.Tensor,
        alpha_frac: float,
        reverse: bool = False,
    ) -> torch.Tensor:
        """沿方向 steer 残差流（执行写入）。

        hidden [..., d]（CSA 残差前层，监测/执行分置的写入点）；alpha_frac ∈ [0, max]：
        相对残差 norm 的强度（0=不 steer）；reverse=True 取反方向（默认 False=真值方向）。

        返回 steer 后的 hidden（`h + sign·α·direction`，α=alpha_frac×残差 norm，
        α 钳制到 max_alpha_frac 上限——Braun 红线：α 有界防崩溃）。
        """
        alpha_frac = max(0.0, min(alpha_frac, self.max_alpha_frac))  # α 有界钳制
        if alpha_frac == 0.0:
            return hidden
        res_norm = hidden.float().norm(dim=-1, keepdim=True).mean()
        alpha = alpha_frac * res_norm
        sign = -1.0 if reverse else 1.0
        return hidden + sign * alpha * self.direction.to(hidden.device, hidden.dtype)

    def steer_toward_truth(self, hidden: torch.Tensor, alpha_frac: float = 0.1) -> torch.Tensor:
        """沿真值方向 steer（用于 L3 冲突沿参数知识 / 生成中保持真实度）。

        保守 α（默认 0.1×norm，Braun 安全区）；仅应在编排层判定"方向正确"时调用
        （绝不用于把空白 steer 成知道——那是造假，由编排层门控保证）。
        """
        return self.steer(hidden, alpha_frac, reverse=False)


def make_iti_from_kernel(kernel, max_alpha_frac: float = 0.2) -> ITIHead:
    """从内核 KAL L1 头派生 ITI 干预头（执行通道）。"""
    if kernel is None or not hasattr(kernel, "kal_l1"):
        raise RuntimeError("内核无 kal_l1 头（fail-closed，无法派生 ITI 方向）")
    return ITIHead.from_kal_l1(kernel.kal_l1, max_alpha_frac=max_alpha_frac)
