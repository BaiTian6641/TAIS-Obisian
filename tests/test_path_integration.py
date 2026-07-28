"""路径积分辅助任务（Path Integration）测试：第二阶段迭代② pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 数据生成：轨迹形状正确、位移累积=位置（path integration 一致性）；
  b) 预测头：随机初始化下 loss 有限；简单训练几步后 loss 下降（可学性）；
  c) 非负约束：encoder 末层激活 ≥0（Sorscher 充分条件）；
  d) GridCodeProbe 判别力：人工六边形网格模式 → 高 grid score；
     随机噪声/条纹（非六边形）→ 低分；
  e) 端到端：训练 PathIntegrationTask 若干步后 probe，记录 grid score 变化
     （pilot 趋势观测，不强制涌现，因 [降预期]）；
  f) 梯度隔离（MoE-RL 红线）：loss.backward() 后 encoder/head 有梯度，
     模拟主干参数无梯度。
用法：python -m pytest tests/test_path_integration.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.path_integration import (
    GridCodeProbe,
    PathIntegrationData,
    PathIntegrationTask,
    path_integration_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _sample_pos_probe_grid(n_side: int = 24, device: str = DEVICE):
    """均匀覆盖 2D 区域的探针采样点（n_side² 个点，密度足够避免粗采样伪影）。"""
    g = torch.linspace(0.0, 4.0, n_side, device=device)
    ys, xs = torch.meshgrid(g, g, indexing="ij")
    pos = torch.stack([ys, xs], dim=-1).reshape(-1, 2)  # [N,2]
    return pos


def _hex_pattern(pos: torch.Tensor) -> torch.Tensor:
    """人工六边形网格发放模式：三组互成 60° 的平面波叠加（标准网格场构造）。"""
    import math

    x, y = pos[:, 0], pos[:, 1]
    k = 2 * math.pi / 1.5  # 空间周期 1.5
    out = torch.zeros_like(x)
    for ang in (0.0, math.pi / 3, 2 * math.pi / 3):  # 0°/60°/120° 三组波矢
        out = out + torch.cos(k * (x * math.cos(ang) + y * math.sin(ang)))
    return out


def test_a_data_generation() -> None:
    """a) 轨迹形状正确、位移累积=位置（path integration 一致性）。"""
    B, T, DIM = 5, 17, 2
    pos, disp = PathIntegrationData.sample_trajectory(B, T, dim=DIM, device=DEVICE, seed=42)
    assert pos.shape == (B, T, DIM), f"positions 形状错误: {tuple(pos.shape)}"
    assert disp.shape == (B, T, DIM), f"displacements 形状错误: {tuple(disp.shape)}"
    # 一致性：cumsum(displacements) ≡ positions（起点位移为 0，累积即绝对位置）
    err = (disp.cumsum(dim=1) - pos).abs().max().item()
    assert err < 1e-5, f"path integration 一致性破坏: {err}"
    # 无噪声时步长严格为单位位移
    _, disp_clean = PathIntegrationData.sample_trajectory(4, 9, dim=2, device=DEVICE, seed=1)
    norms = disp_clean[:, 1:, :].norm(dim=-1)
    assert (norms - 1.0).abs().max().item() < 1e-5, "非起点步长应为单位位移"
    # 起点位置恒为 0（累积基准）
    assert pos[:, 0, :].abs().max().item() == 0.0
    print(f"[a] 数据生成 OK：形状 {tuple(pos.shape)}，一致性误差 {err:.2e}，单位步长 ✓")


def test_b_head_learnable() -> None:
    """b) 随机初始化 loss 有限；训练几步后 loss 下降（可学性）。"""
    torch.manual_seed(0)
    task = PathIntegrationTask(dim=2, hidden=64, repr_dim=64).to(DEVICE)
    pos, disp = PathIntegrationData.sample_trajectory(16, 12, device=DEVICE, seed=7)
    loss0, diag0 = task.loss(disp, pos)
    assert torch.isfinite(loss0), f"初始 loss 非有限: {loss0.item()}"
    opt = torch.optim.Adam(task.parameters(), lr=3e-3)
    for _ in range(80):
        opt.zero_grad()
        loss, _ = task.loss(disp, pos)
        loss.backward()
        opt.step()
    lossN, diagN = task.loss(disp, pos)
    assert lossN.item() < loss0.item() * 0.5, (
        f"loss 未明显下降: {loss0.item():.4f} → {lossN.item():.4f}"
    )
    print(
        f"[b] 可学性 OK：loss {loss0.item():.4f} → {lossN.item():.4f}，"
        f"rel_error {diag0['rel_error']:.3f} → {diagN['rel_error']:.3f}"
    )


def test_c_nonneg_constraint() -> None:
    """c) 非负约束：encoder 末层激活 ≥0（Sorscher 充分条件关键）。"""
    torch.manual_seed(0)
    task = PathIntegrationTask(dim=2, hidden=64, repr_dim=96).to(DEVICE)
    _, disp = PathIntegrationData.sample_trajectory(8, 10, device=DEVICE, seed=3)
    repr_seq = task.encoder(disp)
    assert repr_seq.shape == (8, 10, 96), f"表征形状错误: {tuple(repr_seq.shape)}"
    assert repr_seq.min().item() >= 0.0, f"非负约束破坏: min={repr_seq.min().item()}"
    print(f"[c] 非负约束 OK：表征最小值 {repr_seq.min().item():.3f} ≥ 0")


def test_d_probe_discrimination() -> None:
    """d) GridCodeProbe 判别力：六边形网格 → 高分；随机噪声/条纹 → 低分。"""
    probe = GridCodeProbe(n_bins=20, threshold=0.3)
    pos = _sample_pos_probe_grid(n_side=24)  # 576 个均匀覆盖点
    hex_act = _hex_pattern(pos)
    g_hex = probe.grid_score(hex_act, pos)
    torch.manual_seed(0)
    g_rand = probe.grid_score(torch.randn(pos.shape[0], device=DEVICE), pos)
    g_stripe = probe.grid_score(torch.cos(4.0 * pos[:, 0]), pos)  # 单方向条纹（90° 对称）
    assert g_hex > probe.threshold, f"六边形模式 grid score 过低: {g_hex:.3f}"
    assert g_rand < g_hex, f"随机噪声应低于六边形: {g_rand:.3f} vs {g_hex:.3f}"
    assert g_stripe < g_hex, f"条纹应低于六边形: {g_stripe:.3f} vs {g_hex:.3f}"
    # probe() 批量接口：hex 维度应进 top-k，均值随噪声维度拉低
    reps = torch.stack([hex_act, torch.randn_like(hex_act), g_stripe_act := torch.cos(4.0 * pos[:, 0])], dim=-1)
    mean_score, top_idx, is_grid = probe.probe(reps, pos, top_k=1)
    assert top_idx[0].item() == 0, f"top-1 应为六边形维度 0，实得 {top_idx[0].item()}"
    print(
        f"[d] 探针判别力 OK：hex={g_hex:.3f} > 阈值 0.3；"
        f"rand={g_rand:.3f}，stripe={g_stripe:.3f}；batch probe top1=dim{top_idx[0].item()}"
    )


def test_e_end_to_end_probe_trend() -> None:
    """e) 端到端：训练若干步后 probe，记录 grid score 变化（趋势观测，不强制涌现）。"""
    torch.manual_seed(0)
    task = PathIntegrationTask(dim=2, hidden=128, repr_dim=128).to(DEVICE)
    pos, disp = PathIntegrationData.sample_trajectory(32, 16, device=DEVICE, seed=11)
    mean0, _, _ = task.probe(disp, pos)
    opt = torch.optim.Adam(task.parameters(), lr=3e-3)
    for _ in range(120):
        opt.zero_grad()
        loss, _ = task.loss(disp, pos)
        loss.backward()
        opt.step()
    lossN, diagN = task.loss(disp, pos)
    mean1, top_idx, is_grid = task.probe(disp, pos)
    # 判据：任务本身须学会（rel_error 显著下降）；grid score 只记录趋势（[降预期] 不强制涌现）
    assert diagN["rel_error"] < 0.5, f"路径积分未学会: rel_error={diagN['rel_error']:.3f}"
    print(
        f"[e] 端到端 OK：rel_error={diagN['rel_error']:.3f}；"
        f"grid score {mean0:.3f} → {mean1:.3f}（趋势观测，阈值判定={is_grid}，"
        f"top dims={top_idx[:4].tolist()}）"
    )


def test_f_gradient_isolation() -> None:
    """f) 梯度隔离（MoE-RL 红线）：辅助 loss 只进 encoder/head，模拟主干参数无梯度。"""
    torch.manual_seed(0)
    task = PathIntegrationTask(dim=2, hidden=64, repr_dim=64).to(DEVICE)
    # 模拟"主干"参数：产出 indexer 表征的上游（detach 后喂入，照 tais_kernel 纪律）
    backbone = torch.nn.Linear(2, 2).to(DEVICE)
    pos, disp = PathIntegrationData.sample_trajectory(8, 10, device=DEVICE, seed=5)
    disp_detached = backbone(disp).detach()  # detach 边界：辅助任务只见 detach 后输入
    loss, _ = task.loss(disp_detached, pos)
    loss.backward()
    # encoder/head 有梯度
    enc_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.encoder.parameters()
    )
    head_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in task.head.parameters()
    )
    assert enc_has_grad, "encoder 应有梯度"
    assert head_has_grad, "head 应有梯度"
    # 主干参数无梯度（detach 边界外，辅助 loss 不回流）
    backbone_clean = all(
        p.grad is None or p.grad.abs().sum().item() == 0 for p in backbone.parameters()
    )
    assert backbone_clean, "MoE-RL 红线破坏：主干参数收到辅助损失梯度"
    print("[f] 梯度隔离 OK：encoder/head 有梯度，模拟主干参数零梯度（MoE-RL 红线 ✓）")
