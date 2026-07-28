"""路径积分辅助任务（Path Integration Auxiliary Task）——第二阶段（思维能力强化）迭代② pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §6 迭代② + §1.1。

概念：给 HRL indexer 加辅助损失——给定轨迹片段的**位移序列**，预测当前位置
（path integration）。按 **Sorscher et al. 2023 充分条件**（路径积分 + 非负性
是网格码涌现的充分条件），这会诱导 indexer 内部形成类网格的多尺度空间码。
**验证判据（T1 新探针）**：indexer 表征里是否出现周期性空间响应——出现 =
空间导航 substrate 成立；不出现 = 辅助损失权重不足。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] 网格细胞涌现定理（Banino et al., Nature 2018；Sorscher et al. 2023）：
  RNN 只做路径积分训练，六边形网格编码自发涌现；路径积分 + 非负性是充分条件。
  含义：空间码是训练目标的免费副产品。
- [推测/独创] 把路径积分辅助损失挂在 HRL indexer 内部表征上诱导可导航几何——
  文献仅有 RNN/PCN 实现先例，无 LLM indexer 先例（TAIS 独创外推，须经 pilot 验证）。
- [降预期] 网格码在 transformer 中**不会自发涌现**（文献核实
  /memories/repo/verified-literature-thinking-manifold.md：证据全在 RNN/PCN/tPCN；
  Sorscher 2022 仅 ~10% RNN 涌现且依赖非负/共形约束 arXiv:2310.19192）。
  故本模块**显式训练诱导**可导航几何（路径积分辅助损失 + 非负约束），不指望涌现；
  GridCodeProbe 仅作观测验证（趋势观测，非神经科学全精度）。

MoE-RL 红线（设计 §11 / 接口计划）：**辅助损失梯度只进 indexer/encoder，禁污染主干**。
本模块为独立 pilot，不接 model.py 主干；`PathIntegrationTask.loss()` 的梯度路径
只流经自身 encoder/head（任何主干输入须在调用侧 detach，见类 docstring）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathIntegrationData:
    """合成轨迹数据生成器（自监督，免费数据）。

    在 2D 任务空间随机游走（对齐设计"任务空间低维"，网格细胞的 2D 物理空间）：
    每步随机方向单位位移 + 可选小噪声。path integration 一致性由构造保证：
    positions[:, t, :] = positions[:, 0, :] + Σ_{s≤t} displacements[:, s, :]，
    其中 displacements[:, 0, :] = 0（起点列，累积不改变位置）。
    """

    @staticmethod
    def sample_trajectory(
        batch: int,
        n_steps: int,
        dim: int = 2,
        noise_std: float = 0.0,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """采样随机游走轨迹。

        参数：
            batch: 轨迹条数 B。
            n_steps: 每条轨迹步数 T。
            dim: 空间维度（默认 2，网格细胞的物理空间）。
            noise_std: 每步位移叠加的高斯噪声标准差（0 = 严格单位步长）。
            device/seed: 设备与随机种子（seed 在 CPU 生成器上生效，结果可复现）。

        返回：(positions [B,T,dim], displacements [B,T,dim])。
            displacements[:, 0, :] = 0；cumsum(displacements) ≡ positions − positions[:, :1]。
        """
        if dim < 1:
            raise ValueError(f"dim 须 ≥1，实得 {dim}")
        g = torch.Generator(device="cpu")
        if seed is not None:
            g.manual_seed(seed)
        # 每步随机方向单位位移：高斯向量归一化到单位 ℓ2 范数
        steps = torch.randn(batch, n_steps, dim, generator=g)
        steps = steps / steps.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        if noise_std > 0:
            steps = steps + noise_std * torch.randn(batch, n_steps, dim, generator=g)
        steps[:, 0, :] = 0.0  # 起点列：无先验位移（累积 = 绝对位置）
        positions = steps.cumsum(dim=1)
        return positions.to(device), steps.to(device)


class PathIntegrationEncoder(nn.Module):
    """路径积分 encoder：位移序列 → indexer 内部表征序列（非负）。

    结构：input_proj(dim→hidden) → GRU(hidden) → out_proj(hidden→repr_dim) → **ReLU**。
    循环结构对齐 Banino 2018 / Sorscher 2023 的 RNN 路径积分设定（逐步累积位移）；
    其隐藏状态即"indexer 内部表征"的 pilot 替身（真实接入时由 LightningIndexer
    的低维 q/k 表征替换，辅助损失挂法不变）。

    **非负约束（Sorscher 充分条件之一，不可省）**：末层激活 ReLU 保证表征 ≥0——
    路径积分 + 非负性是网格码涌现的充分条件，缺少非负约束时周期性空间码
    无法稳定形成（arXiv:2310.19192 的共形/非负消融）。
    """

    def __init__(self, dim: int = 2, hidden: int = 128, repr_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(dim, hidden)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.out_proj = nn.Linear(hidden, repr_dim)

    def forward(self, displacements: torch.Tensor) -> torch.Tensor:
        """displacements [B,T,dim] → 表征 [B,T,repr_dim]，末维 ≥0（ReLU 非负约束）。"""
        h, _ = self.gru(self.input_proj(displacements))
        return F.relu(self.out_proj(h))  # Sorscher 非负充分条件：表征非负


class PathIntegrationHead(nn.Module):
    """路径积分预测头：indexer 内部表征序列 → 预测的当前 2D 位置。

    Linear(repr_dim→dim) 回归——刻意保持线性读出：探针式读出的逻辑是
    "表征里是否线性可读位置信息"，头部容量过大会把位置信息"学进头里"，
    削弱对 indexer 表征本身的诱导（对齐探针只读纪律的同族考量）。
    """

    def __init__(self, repr_dim: int = 128, dim: int = 2):
        super().__init__()
        self.regressor = nn.Linear(repr_dim, dim)

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        """representations [B,T,repr_dim] → pred_positions [B,T,dim]。"""
        return self.regressor(representations)


def path_integration_loss(
    pred_positions: torch.Tensor,
    true_positions: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """路径积分辅助损失 = 预测位置与真实位置的 MSE。

    返回：(标量 loss, 诊断 dict)。诊断含：
        - rel_error：相对误差 ||pred − true|| / ||true − true_mean||
          （以各轨迹质心为基线的相对偏差，尺度不变，便于跨配置比较）；
        - abs_rmse：绝对 RMSE（任务空间单位）。
    """
    if pred_positions.shape != true_positions.shape:
        raise ValueError(
            f"pred/true 形状不一致：{tuple(pred_positions.shape)} vs {tuple(true_positions.shape)}"
        )
    pred = pred_positions.float()
    true = true_positions.float()
    loss = F.mse_loss(pred, true)
    with torch.no_grad():
        err = (pred - true).norm(dim=-1)  # [B,T]
        centered = true - true.mean(dim=1, keepdim=True)
        denom = centered.norm(dim=-1).clamp_min(1e-8)  # [B,T]
        rel = (err / denom).mean().item()
        rmse = err.pow(2).mean().sqrt().item()
    return loss, {"rel_error": rel, "abs_rmse": rmse}


class GridCodeProbe:
    """T1 新探针：检测 indexer 表征是否出现周期性空间响应（网格码）。

    标准 gridness score（神经科学度量，Sargolini et al. 2006 谱系）：
    对单神经元 2D 发放率图做空间自相关，取自相关图中心环带，比较
    **60°/120° 旋转自相关（六边形对称，网格码应为高）** 与
    **30°/90°/150° 旋转自相关（非六边形，网格码应为低）**：
        grid_score = min(corr60, corr120) − max(corr30, corr90, corr150)
    grid_score > 阈值（如 0.3）= 出现周期性空间响应（网格码成立）。

    [近似声明] 本实现是 **pilot 级 gridness 近似**：2D 直方图率图（非核密度
    平滑）、裁剪方形自相关中心（非精确环带掩膜）、最近邻旋转重采样（非
    双三次插值）——仅作趋势观测与有无判定，非神经科学全精度流程。
    """

    def __init__(self, n_bins: int = 20, threshold: float = 0.3):
        self.n_bins = n_bins
        self.threshold = threshold

    def _rate_map(self, act: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """单神经元发放率图：act [N]（该维发放）、pos [N,2] → [n_bins,n_bins]。

        平均发放率直方图：每格 = 落入该格样本的 act 均值；无样本格 = 0。
        位置域按数据范围归一化到 [0, n_bins)。
        """
        n = self.n_bins
        pmin = pos.min(dim=0).values
        pmax = pos.max(dim=0).values
        span = (pmax - pmin).clamp_min(1e-8)
        uv = (pos - pmin) / span  # [N,2] ∈ [0,1]
        ij = (uv * (n - 1)).long().clamp(0, n - 1)  # [N,2] 格索引
        rate = torch.zeros(n, n, device=act.device, dtype=torch.float32)
        cnt = torch.zeros(n, n, device=act.device, dtype=torch.float32)
        idx = ij[:, 0] * n + ij[:, 1]
        rate.view(-1).index_add_(0, idx, act.float())
        cnt.view(-1).index_add_(0, idx, torch.ones_like(act.float()))
        return rate / cnt.clamp_min(1.0)

    @staticmethod
    def _rotate(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """2D 张量绕中心旋转（最近邻重采样，近似）。"""
        n = x.shape[0]
        theta = torch.tensor(angle_deg * 3.141592653589793 / 180.0, device=x.device)
        c = (n - 1) / 2.0
        ys, xs = torch.meshgrid(
            torch.arange(n, device=x.device, dtype=torch.float32),
            torch.arange(n, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        dy, dx = ys - c, xs - c
        # 逆旋转采样坐标
        sy = c + dx * torch.sin(theta) + dy * torch.cos(theta)
        sx = c + dx * torch.cos(theta) - dy * torch.sin(theta)
        sy = sy.round().long().clamp(0, n - 1)
        sx = sx.round().long().clamp(0, n - 1)
        return x[sy, sx]

    @staticmethod
    def _crop_center(x: torch.Tensor, frac: float = 0.6) -> torch.Tensor:
        """裁剪中心方形区域（近似标准流程的中心环带：排除中心峰与外圈伪影）。"""
        n = x.shape[0]
        half = max(1, int(n * frac / 2))
        c = n // 2
        return x[c - half : c + half + 1, c - half : c + half + 1]

    @staticmethod
    def _flat_corr(a: torch.Tensor, b: torch.Tensor) -> float:
        """两同形张量的 Pearson 相关（展平）；零方差返回 0。"""
        a = a.flatten().float()
        b = b.flatten().float()
        a = a - a.mean()
        b = b - b.mean()
        denom = a.norm() * b.norm()
        if denom < 1e-12:
            return 0.0
        return (a.dot(b) / denom).item()

    def grid_score(self, activations: torch.Tensor, positions: torch.Tensor) -> float:
        """单维表征的 gridness score：activations [N]，positions [N,2] → 标量。

        六边形网格模式 → 高分；随机/条纹（非六边形）→ 低分或负分。
        """
        if activations.dim() != 1 or positions.shape != (activations.shape[0], 2):
            raise ValueError(
                f"activations 须 [N] 且 positions 须 [N,2]，实得 {tuple(activations.shape)} / {tuple(positions.shape)}"
            )
        rate = self._rate_map(activations, positions)
        # 空间自相关（fft 实现：循环相关数学等价于线性自相关，无 conv padding 边界错位）
        n = self.n_bins
        f = torch.fft.rfft2(rate, s=(2 * n - 1, 2 * n - 1))
        ac = torch.fft.fftshift(torch.fft.irfft2(f * f.conj(), s=(2 * n - 1, 2 * n - 1)))
        ac = self._crop_center(ac)
        # 六边形对称（60°/120°）减非六边形（30°/90°/150°）
        c60 = self._flat_corr(ac, self._rotate(ac, 60.0))
        c120 = self._flat_corr(ac, self._rotate(ac, 120.0))
        c30 = self._flat_corr(ac, self._rotate(ac, 30.0))
        c90 = self._flat_corr(ac, self._rotate(ac, 90.0))
        c150 = self._flat_corr(ac, self._rotate(ac, 150.0))
        return min(c60, c120) - max(c30, c90, c150)

    def probe(
        self, representations: torch.Tensor, positions: torch.Tensor, top_k: int = 8
    ) -> tuple[float, torch.Tensor, bool]:
        """批量表征维度的网格码检测。

        参数：
            representations: [N, D]（或 [B,T,D]，自动展平前两维）表征样本；
            positions: [N, 2]（或 [B,T,2]）对应 2D 位置；
            top_k: 返回 grid score 最高的维度索引数。

        返回：(平均 grid score, top-k 维度索引 [top_k], 是否超过阈值)。
            超过阈值 = indexer 表征出现周期性空间响应（T1 判据成立）。
        """
        if representations.dim() == 3:
            representations = representations.reshape(-1, representations.shape[-1])
        if positions.dim() == 3:
            positions = positions.reshape(-1, positions.shape[-1])
        if representations.shape[0] != positions.shape[0]:
            raise ValueError(
                f"样本数不一致：repr {representations.shape[0]} vs pos {positions.shape[0]}"
            )
        reps = representations.detach().float()
        pos = positions.detach().float()
        D = reps.shape[1]
        scores = torch.stack([self._score_dim(reps[:, d], pos) for d in range(D)])
        mean_score = scores.mean().item()
        k = min(top_k, D)
        top_idx = scores.topk(k).indices
        return mean_score, top_idx, bool(mean_score > self.threshold)

    def _score_dim(self, act: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self.grid_score(act, pos), device=act.device)


class PathIntegrationTask(nn.Module):
    """路径积分辅助任务封装：encoder + head + loss() + probe()。

    pilot 独立模块，**不接 model.py 主干**——这是 indexer 的辅助任务。
    MoE-RL 红线：辅助损失梯度只进 indexer/encoder，禁污染主干。
    - 本模块自身参数：encoder/head 全量可训练（梯度正常流入）；
    - 若未来接入真实 LightningIndexer 表征：调用侧须对主干产出
      `representations = indexer_repr.detach()` 后再喂本任务（detach 输入
      纪律同 tais_kernel.HRLIndexer），保证梯度只停在本辅助头内。
    """

    def __init__(self, dim: int = 2, hidden: int = 128, repr_dim: int = 128):
        super().__init__()
        self.dim = dim
        self.encoder = PathIntegrationEncoder(dim=dim, hidden=hidden, repr_dim=repr_dim)
        self.head = PathIntegrationHead(repr_dim=repr_dim, dim=dim)
        self.probe_impl = GridCodeProbe()

    def forward(self, displacements: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """displacements [B,T,dim] → (pred_positions [B,T,dim], repr [B,T,repr_dim])。"""
        repr_seq = self.encoder(displacements)
        return self.head(repr_seq), repr_seq

    def loss(
        self, displacements: torch.Tensor, true_positions: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """辅助损失：位移序列 → 位置预测 MSE。梯度只进本模块 encoder/head。"""
        pred, _ = self.forward(displacements)
        return path_integration_loss(pred, true_positions)

    def probe(
        self, displacements: torch.Tensor, positions: torch.Tensor, top_k: int = 8
    ) -> tuple[float, torch.Tensor, bool]:
        """T1 观测探针：encoder 表征的网格码检测（no_grad，只读不训练）。"""
        with torch.no_grad():
            repr_seq = self.encoder(displacements)
        return self.probe_impl.probe(repr_seq, positions, top_k=top_k)


def make_path_integration_task(
    dim: int = 2, hidden: int = 128, repr_dim: int = 128
) -> PathIntegrationTask:
    """工厂函数。"""
    return PathIntegrationTask(dim=dim, hidden=hidden, repr_dim=repr_dim)
