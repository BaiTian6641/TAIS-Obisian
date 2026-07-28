"""思考流形层（Thought Manifold Layer）——第二阶段（思维能力强化）迭代① pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §1.1（几何底座）+ §6 迭代①。

概念：一个共享的低维坐标空间，知识块 route_key 表征、注意力上下文思考段、
W0 日志轨迹段都经**同一投影器**映射进同一空间；推理 = 该空间上的位移，且
**共形等距**（相邻思考段的流形位移 ∝ 语义关系步长）。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] 网格细胞在路径积分训练下涌现六边形空间码（Banino et al., Nature 2018；
  Sorscher et al. 2023 充分条件）；共形等距假说——网格细胞高维群体活动构成
  2D 物理空间的保距嵌入，神经流形位移与物理位移成比例；Gurnee & Tegmark 2023
  （LLaMA 中间层激活中地点/时间坐标可作线性维度读出）。
- [推测/独创] 把"知识块 / CoT 思考段 / W0 轨迹"三类对象经共享投影器映射进同一
  坐标空间、并以"共形等距"作显式训练目标——文献仅有认知地图隐喻，无 LLM
  实现先例（TAIS 独创外推，须经 pilot 验证）。
- [降预期] 网格码在 transformer 中不会自发涌现（证据全在 RNN/PCN；Sorscher 2022
  仅 ~10% 网络涌现且依赖非负/共形约束 arXiv:2310.19192）——故本模块**显式训练
  诱导**可导航几何，不指望涌现。

维度修正（关键，接住 CTM 搜索的 ⚠️）：神经科学的答案**不是"3 维坐标"**，而是
高维神经活动约束在低维流形上。对应本实现：manifold_dim 默认 64（几十到一百
多维有效维，避免信息瓶颈）；3D 投影（project_3d）**仅**作人类可解释性视图
（归因监测/审计前端），固定不参与训练目标。

坍缩红线（设计 §15.2 同族手段）：共形等距目标若权重失衡，坐标会坍缩到无信息
点（对比学习经典失败模式）。本模块以 decorrelation_loss（坐标维协方差非对角
惩罚，VICReg 谱系 arXiv:2105.04906）作去相关兜底；TAIS 内核 DGProjection
（tais_kernel.py）的 sparse_topk 是同族去相关手段（潜空间几何各向异性去相关）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ThoughtManifoldProjector(nn.Module):
    """高维表征 → 思考流形坐标的共享投影器。

    **同一实例**服务三类输入（共享坐标映射的核心，设计 §1.1）：
    ① 知识块 route_key 表征；② PM-stream 思考段读出（pm_index(n) 末位流）；
    ③ W0 日志轨迹段表征。同输入 → 同坐标 ⇒ 三类对象在同一空间可导航。

    结构：Linear(d_model → manifold_dim) + 可选 LayerNorm（无偏置/无缩放，
    只做标准化——避免归一化参数把坐标几何学成任意仿射，保住位移比例的语义）。
    manifold_dim 默认 64：**不是** 3 维（维度修正，设计 §1.1）。
    """

    def __init__(self, d_model: int, manifold_dim: int = 64, use_layernorm: bool = True):
        super().__init__()
        self.d_model = d_model
        self.manifold_dim = manifold_dim
        self.proj = nn.Linear(d_model, manifold_dim)
        self.norm = nn.LayerNorm(manifold_dim, elementwise_affine=False) if use_layernorm else nn.Identity()
        # 3D 可解释性视图：固定随机投影（不参与训练目标、无梯度）。
        # 固定种子保证同一 run 内视图稳定（可复现的人类审计前端）。
        self.view3d = nn.Linear(manifold_dim, 3, bias=False)
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(0)
            w = torch.randn(3, manifold_dim, generator=g) / (manifold_dim ** 0.5)
            self.view3d.weight.copy_(w)
        for p in self.view3d.parameters():
            p.requires_grad_(False)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """x [..., d_model] → 流形坐标 [..., manifold_dim]。"""
        return self.norm(self.proj(x))

    def project_3d(self, coords: torch.Tensor) -> torch.Tensor:
        """流形坐标 [..., manifold_dim] → [..., 3]（仅人类可视化视图，固定投影）。"""
        return self.view3d(coords)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(x)


def conformal_isometry_loss(
    coords: torch.Tensor,
    semantic_steps: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict]:
    """共形等距损失（**比例**而非相等）：相邻思考段的流形位移 ∝ 语义关系步长。

    参数：
        coords: [B, T, manifold_dim] 轨迹流形坐标序列（T 个思考段），建议 fp32。
        semantic_steps: [B, T-1] 相邻段语义距离的标量监督（来自 W0 日志/CoT 段标注）。
        mask: 可选 [B, T-1]，1=该相邻对计入损失。
        eps: 数值稳定常数。

    尺度不变实现（防全局尺度坍缩的关键）：对每条轨迹分别把位移序列与步长序列
    **归一化为单位 ℓ2 范数向量**，再求二者 MSE——
        min || disp/||disp||₂ − steps/||steps||₂ ||²
    只约束"位移在各相邻对之间的分配比例"，完全不约束绝对尺度：轨迹整体缩放
    不改变损失（共形 = 保角/保比例，不保长度）。这比 raw MSE（会把坐标拉到
    任意小尺度上凑步长，诱发坍缩）稳健，也比 Pearson 目标可微路径更直接。

    返回：(标量 loss, 诊断 dict)。诊断含 `pearson`（位移-步长 Pearson 相关系数，
    跨全部有效相邻对池化计算）——§6 迭代①验证判据"流形位移与语义步长相关性"。
    """
    if coords.dim() != 3:
        raise ValueError(f"coords 须为 [B, T, manifold_dim]，实得 shape={tuple(coords.shape)}")
    B, T, _ = coords.shape
    if semantic_steps.shape != (B, T - 1):
        raise ValueError(
            f"semantic_steps 须为 [B, T-1]={(B, T - 1)}，实得 {tuple(semantic_steps.shape)}"
        )
    coords = coords.float()
    steps = semantic_steps.float().clamp_min(0.0)

    # 相邻段流形位移 disp[b,t] = ||coords[b,t+1] − coords[b,t]||₂，[B, T-1]
    disp = (coords[:, 1:, :] - coords[:, :-1, :]).norm(dim=-1)

    if mask is not None:
        if mask.shape != (B, T - 1):
            raise ValueError(f"mask 须为 [B, T-1]={(B, T - 1)}，实得 {tuple(mask.shape)}")
        m = mask.float()
    else:
        m = torch.ones_like(disp)

    # 每轨迹内归一化（尺度不变）：仅对有效位求范数；范数为 0 的轨迹跳过（置 0 梯度贡献）
    disp_n = (disp * m).norm(dim=-1, keepdim=True)  # [B,1]
    step_n = (steps * m).norm(dim=-1, keepdim=True)
    valid_traj = ((disp_n > eps) & (step_n > eps)).float()  # [B,1]
    disp_hat = (disp * m) / (disp_n + eps)
    step_hat = (steps * m) / (step_n + eps)
    per_traj = ((disp_hat - step_hat) ** 2).sum(dim=-1)  # [B] 轨迹级尺度不变 MSE
    denom = valid_traj.sum().clamp_min(1.0)
    loss = (per_traj * valid_traj.squeeze(-1)).sum() / denom

    # 诊断：位移-步长 Pearson 相关（全部有效相邻对池化；no_grad，纯观测）
    with torch.no_grad():
        sel = m > 0
        d_v, s_v = disp[sel], steps[sel]
        if d_v.numel() >= 2:
            d_c = d_v - d_v.mean()
            s_c = s_v - s_v.mean()
            denom_p = d_c.norm() * s_c.norm()
            pearson = (d_c @ s_c / (denom_p + eps)).item() if denom_p > eps else 0.0
        else:
            pearson = 0.0
    diag = {"pearson": pearson, "mean_disp": disp.mean().item(), "mean_steps": steps.mean().item()}
    return loss, diag


def decorrelation_loss(
    coords: torch.Tensor,
    eps: float = 1e-8,
    var_target: float = 1.0,
) -> torch.Tensor:
    """去相关兜底（防坍缩红线）：相关矩阵非对角惩罚 + 逐维方差铰链。

    对 coords [B, T, manifold_dim]，把 batch×time 展平为样本维：
      1. 协方差项：求各坐标维的**归一化协方差矩阵**（相关矩阵），惩罚非对角
         项平方均值——让坐标维去相关；
      2. 方差铰链：mean(relu(var_target − std_d)²)——每维标准差须达到
         var_target。**此项不可省**：纯非对角惩罚对"全坍缩到一点"的坐标恒为 0
         （协方差矩阵全零，无相关结构可罚），无法单独满足防坍缩红线；
         var_target=1.0 对齐 projector 默认 LayerNorm 后坐标的自然尺度
         （每样本跨维方差≈1 ⇒ 每维跨样本方差≈1）。

    思路同族：VICReg 方差-协方差正则（arXiv:2105.04906）的方差项+协方差项；
    TAIS 内核 DGProjection sparse_topk（tais_kernel.py，§15.2 潜空间去相关）。
    """
    if coords.dim() != 3:
        raise ValueError(f"coords 须为 [B, T, manifold_dim]，实得 shape={tuple(coords.shape)}")
    x = coords.float().reshape(-1, coords.shape[-1])  # [N, D]，N=B*T
    x = x - x.mean(dim=0, keepdim=True)
    d = x.shape[-1]
    # 方差铰链（VICReg 方差项）：逐维 std 不得低于 var_target
    std = x.std(dim=0)
    l_var = F.relu(var_target - std).pow(2).mean()
    # 协方差项（VICReg 协方差项）：相关矩阵非对角平方均值
    cov = (x.T @ x) / max(x.shape[0] - 1, 1)  # [D, D] 协方差
    var = cov.diag().clamp_min(eps)
    corr = cov / (var.sqrt().unsqueeze(0) * var.sqrt().unsqueeze(1) + eps)  # 相关矩阵
    off_diag = corr - torch.eye(d, device=corr.device, dtype=corr.dtype)
    l_off = (off_diag ** 2).sum() / (d * d - d)
    return l_off + l_var


class ThoughtManifold(nn.Module):
    """思考流形封装：持有共享 projector + 组合损失。

    用法（pilot 迭代①，独立训练该投影器，不触碰主干权重）：
        manifold = ThoughtManifold(d_model, manifold_dim=64)
        coords = manifold.project(segment_reprs)          # 三类输入同一 project
        loss, diag = manifold.loss(coords, semantic_steps)
        loss.backward()                                    # 梯度只进 projector
    """

    def __init__(self, d_model: int, manifold_dim: int = 64, use_layernorm: bool = True):
        super().__init__()
        self.projector = ThoughtManifoldProjector(d_model, manifold_dim, use_layernorm)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector.project(x)

    def project_3d(self, coords: torch.Tensor) -> torch.Tensor:
        return self.projector.project_3d(coords)

    def loss(
        self,
        coords: torch.Tensor,
        semantic_steps: torch.Tensor,
        mask: torch.Tensor | None = None,
        w_conformal: float = 1.0,
        w_decor: float = 0.1,
    ) -> tuple[torch.Tensor, dict]:
        """组合损失 = w_conformal·共形等距 + w_decor·去相关兜底。

        权重纪律（坍缩红线）：w_decor 不可为 0——共形目标单独存在时存在
        退化解风险；w_conformal 为主目标。默认 1.0 / 0.1 待 pilot 标定。
        """
        l_conf, diag = conformal_isometry_loss(coords, semantic_steps, mask=mask)
        l_decor = decorrelation_loss(coords)
        total = w_conformal * l_conf + w_decor * l_decor
        diag = {**diag, "conformal": l_conf.item(), "decorrelation": l_decor.item()}
        return total, diag
