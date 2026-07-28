"""思考流形层（Thought Manifold）测试：第二阶段迭代① pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 投影形状：[B,T,d]→[B,T,manifold_dim]；project_3d→[B,T,3]；
  b) 共享坐标：三类输入用同一 projector，同输入→同坐标；
  c) 共形等距：位移∝步长的合成轨迹损失 < 打乱者；诊断 Pearson 接近 1；
  d) 去相关兜底：坍缩坐标的去相关损失 > 分散坐标（防坍缩红线）；
  e) 组合 loss 为标量、可反传（projector 参数有梯度）；
  f) 尺度不变性：轨迹整体缩放不改变 conformal 损失（比例语义，非相等）。
用法：python -m pytest tests/test_manifold.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import (
    ThoughtManifold,
    ThoughtManifoldProjector,
    conformal_isometry_loss,
    decorrelation_loss,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 256
MANIFOLD_DIM = 64


def make_proportional_traj(B: int = 4, T: int = 9, seed: int = 0):
    """合成"位移∝步长"轨迹：1D 潜线上等距方向随机、步长按 steps 走，再嵌入
    manifold_dim 维空间（随机正交嵌入保持欧氏位移）。返回 (coords, steps)。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    steps = torch.rand(B, T - 1, generator=g) + 0.2  # 语义步长 >0
    pos = torch.cat([torch.zeros(B, 1), steps.cumsum(dim=1)], dim=1)  # [B,T] 1D 位置
    # 随机正交嵌入到 manifold_dim 维（位移保持）
    E = torch.linalg.qr(torch.randn(MANIFOLD_DIM, MANIFOLD_DIM, generator=g))[0][:, 0]  # 单位向量
    coords = pos.unsqueeze(-1) * E.view(1, 1, -1)  # [B,T,manifold_dim]
    return coords.to(DEVICE), steps.to(DEVICE)


def test_a_projection_shapes() -> None:
    """a) 投影形状：[B,T,d_model]→[B,T,manifold_dim]；project_3d→[B,T,3]。"""
    proj = ThoughtManifoldProjector(D_MODEL, MANIFOLD_DIM).to(DEVICE)
    x = torch.randn(2, 7, D_MODEL, device=DEVICE)
    coords = proj.project(x)
    assert coords.shape == (2, 7, MANIFOLD_DIM), f"project 形状错误: {tuple(coords.shape)}"
    c3d = proj.project_3d(coords)
    assert c3d.shape == (2, 7, 3), f"project_3d 形状错误: {tuple(c3d.shape)}"
    # 3D 视图不参与训练（固定投影无梯度）
    assert all(not p.requires_grad for p in proj.view3d.parameters())
    # forward 与 project 一致
    assert torch.equal(proj(x), coords)
    print(f"[a] 投影形状 OK：{tuple(coords.shape)} / 3D 视图 {tuple(c3d.shape)}（固定无梯度）")


def test_b_shared_projector_same_space() -> None:
    """b) 共享坐标映射：三类输入（route_key / PM 思考段 / W0 轨迹段表征）走同一
    projector 实例 ⇒ 同输入必得同坐标（同一空间的必要条件）。"""
    proj = ThoughtManifoldProjector(D_MODEL, MANIFOLD_DIM).to(DEVICE).eval()
    torch.manual_seed(1)
    x = torch.randn(3, 5, D_MODEL, device=DEVICE)  # 同一批表征
    with torch.no_grad():
        c_routekey = proj.project(x)   # ① 知识块 route_key 表征
        c_pmseg = proj.project(x)      # ② PM-stream 思考段读出
        c_w0traj = proj.project(x)     # ③ W0 日志轨迹段
    assert torch.equal(c_routekey, c_pmseg) and torch.equal(c_routekey, c_w0traj)
    # 且 3D 视图确定：project_3d(project(x)) 两次调用一致
    assert torch.equal(proj.project_3d(c_routekey), proj.project_3d(c_pmseg))
    # 共享性反证：不同实例（不同随机初始化）不应给出同坐标
    torch.manual_seed(2)
    proj2 = ThoughtManifoldProjector(D_MODEL, MANIFOLD_DIM).to(DEVICE).eval()
    with torch.no_grad():
        c_other = proj2.project(x)
    assert not torch.allclose(c_routekey, c_other), "不同实例坐标应不同（反证共享性来自同一实例）"
    print("[b] 三类输入同一 projector ⇒ 同输入同坐标 OK；不同实例坐标不同（反证）OK")


def test_c_conformal_isometry_proportional_vs_shuffled() -> None:
    """c) 共形等距：位移∝步长轨迹损失 < 打乱步长者；诊断 Pearson 接近 1。"""
    coords, steps = make_proportional_traj()
    loss_prop, diag_prop = conformal_isometry_loss(coords, steps)
    # 打乱步长序列（破坏比例关系）
    g = torch.Generator(device="cpu").manual_seed(99)
    perm = torch.randperm(steps.shape[1], generator=g)
    steps_shuf = steps[:, perm].to(DEVICE)
    loss_shuf, diag_shuf = conformal_isometry_loss(coords, steps_shuf)
    print(f"[c] 成比例 loss={loss_prop.item():.3e} pearson={diag_prop['pearson']:.4f} | "
          f"打乱 loss={loss_shuf.item():.3e} pearson={diag_shuf['pearson']:.4f}")
    assert loss_prop.item() < 1e-6, f"完全成比例轨迹损失应≈0，实得 {loss_prop.item():.3e}"
    assert loss_prop.item() < loss_shuf.item(), "成比例者损失必须 < 打乱者"
    assert diag_prop["pearson"] > 0.999, f"成比例者 Pearson 应≈1，实得 {diag_prop['pearson']:.4f}"
    # mask：屏蔽一半相邻对后，成比例轨迹损失仍应≈0
    mask = torch.ones_like(steps)
    mask[:, ::2] = 0.0
    loss_masked, _ = conformal_isometry_loss(coords, steps, mask=mask)
    assert loss_masked.item() < 1e-6, f"mask 后成比例损失仍应≈0，实得 {loss_masked.item():.3e}"


def test_d_decorrelation_collapsed_vs_spread() -> None:
    """d) 去相关兜底（防坍缩红线）：坍缩（全同点）去相关损失 > 分散坐标。"""
    B, T = 4, 9
    collapsed = torch.full((B, T, MANIFOLD_DIM), 0.7, device=DEVICE)
    spread = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE)
    l_col = decorrelation_loss(collapsed)
    l_spr = decorrelation_loss(spread)
    print(f"[d] 坍缩去相关 loss={l_col.item():.3e} vs 分散 {l_spr.item():.3e}")
    assert l_col.item() > l_spr.item(), "坍缩坐标的去相关损失必须 > 分散坐标"
    # 随机分散坐标应近去相关且各维方差≈1（N=B*T=36 时理论残差 ~1/N≈0.03）
    assert l_spr.item() < 0.15, f"随机分散坐标去相关损失应很小，实得 {l_spr.item():.3e}"
    # 线上轨迹（各维完全相关）也应被罚
    line = torch.linspace(0, 1, T, device=DEVICE).view(1, T, 1).expand(B, T, MANIFOLD_DIM).contiguous()
    l_line = decorrelation_loss(line)
    assert l_line.item() > l_spr.item(), "完全共线轨迹（维间全相关）应被去相关项惩罚"


def test_e_combined_loss_scalar_and_backward() -> None:
    """e) 组合 loss：标量、可反传，projector 参数获得梯度（view3d 固定无梯度）。"""
    torch.manual_seed(3)
    manifold = ThoughtManifold(D_MODEL, MANIFOLD_DIM).to(DEVICE)
    x = torch.randn(4, 9, D_MODEL, device=DEVICE)
    # 用随机步长监督（pilot 阶段由 W0 日志/CoT 段标注提供）
    steps = torch.rand(4, 8, device=DEVICE) + 0.1
    coords = manifold.project(x)
    loss, diag = manifold.loss(coords, steps, w_conformal=1.0, w_decor=0.1)
    assert loss.dim() == 0, f"组合 loss 须为标量，实得 dim={loss.dim()}"
    loss.backward()
    g = manifold.projector.proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "projector 权重必须获得非零梯度"
    assert manifold.projector.view3d.weight.grad is None, "3D 视图固定，不得有梯度"
    assert {"pearson", "conformal", "decorrelation"} <= set(diag), f"诊断键缺失: {set(diag)}"
    print(f"[e] 组合 loss={loss.item():.4f}（conformal={diag['conformal']:.4f}, "
          f"decor={diag['decorrelation']:.4f}, pearson={diag['pearson']:.4f}），反传 OK")


def test_f_scale_invariance() -> None:
    """f) 尺度不变性：轨迹整体缩放 α 倍，conformal 损失不变（比例语义，非相等）。"""
    coords, steps = make_proportional_traj(B=3, T=8, seed=5)
    # 加小扰动使损失明确非零（~1e-3），避开 fp32 零点附近相对漂移的数值地板
    g = torch.Generator(device="cpu").manual_seed(6)
    coords = coords + 0.05 * torch.randn(coords.shape, generator=g).to(DEVICE)
    loss0, _ = conformal_isometry_loss(coords, steps)
    assert loss0.item() > 1e-5, f"扰动后损失应明确非零，实得 {loss0.item():.3e}"
    for alpha in (0.01, 0.5, 7.3, 100.0):
        loss_a, _ = conformal_isometry_loss(coords * alpha, steps)
        rel = abs(loss_a.item() - loss0.item()) / loss0.item()
        assert rel < 1e-4, f"缩放 {alpha} 后损失相对漂移 {rel:.2e}（尺度不变性被破坏）"
    # 完全成比例轨迹在任意缩放下损失都≈0（绝对容差：零点附近的相对比较无数值意义）
    coords_p, steps_p = make_proportional_traj(B=3, T=8, seed=7)
    for alpha in (0.01, 50.0):
        loss_a, _ = conformal_isometry_loss(coords_p * alpha, steps_p)
        assert loss_a.item() < 1e-6, f"成比例轨迹缩放 {alpha} 后损失应≈0，实得 {loss_a.item():.3e}"
    print(f"[f] 尺度不变 OK：α∈{{0.01,0.5,7.3,100}} 损失漂移 <1e-4（loss0={loss0.item():.3e}），"
          f"成比例轨迹任意缩放 loss≈0")


if __name__ == "__main__":
    test_a_projection_shapes()
    test_b_shared_projector_same_space()
    test_c_conformal_isometry_proportional_vs_shuffled()
    test_d_decorrelation_collapsed_vs_spread()
    test_e_combined_loss_scalar_and_backward()
    test_f_scale_invariance()
    print("全部思考流形测试通过")
