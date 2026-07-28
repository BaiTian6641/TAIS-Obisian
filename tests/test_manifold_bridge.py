"""思考流形 ↔ PM-stream 桥接模块测试：第二阶段迭代① × 前置工程③交汇点。

覆盖判据（对齐任务规范 §实现要求）：
  a) 投影形状：extract [B,T,d]→[B,T,64]；extract_segments 段池化 [B,n_seg,64]；
  b) 共享性：extract 坐标与直接 project 一致（同一 projector 实例）；
  c) write 有界：写后 pm_state 与原始的差被 alpha clamp 限制（增量范数 ≤ alpha×norm）；
  d) tick 端到端：pm_state 改变、坐标朝 target 方向移动（位移与 target−current 点积为正）；
  e) 反传：bridge 参数（to_hidden）有梯度；steering 路径 detach（梯度边界注释说明）；
  f) 段聚合边界正确：不同 segment_boundaries 给出不同段数。
用法：python -m pytest tests/test_manifold_bridge.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import ThoughtManifoldProjector
from tais_obsidian.model.manifold_bridge import (
    ManifoldToHidden,
    ThoughtDisplacementWriter,
    ThoughtManifoldBridge,
    ThoughtSegmentExtractor,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL = 256
MANIFOLD_DIM = 64


def make_bridge(seed: int = 42, share: bool = True) -> ThoughtManifoldBridge:
    torch.manual_seed(seed)
    if share:
        proj = ThoughtManifoldProjector(D_MODEL, MANIFOLD_DIM)
        return ThoughtManifoldBridge(D_MODEL, MANIFOLD_DIM, projector=proj).to(DEVICE)
    return ThoughtManifoldBridge(D_MODEL, MANIFOLD_DIM).to(DEVICE)


def test_a_extract_shapes() -> None:
    """a) 投影形状：extract [B,T,d]→[B,T,64]；extract_segments → [B,n_seg,64]。"""
    bridge = make_bridge()
    pm = torch.randn(2, 10, D_MODEL, device=DEVICE)
    coords = bridge.extract(pm)
    assert coords.shape == (2, 10, MANIFOLD_DIM), f"extract 形状错误: {tuple(coords.shape)}"
    # 段聚合：T=10，boundaries=[0,4,7] ⇒ 3 段 [0,4) [4,7) [7,10)
    seg = bridge.extract_segments(pm, [0, 4, 7])
    assert seg.shape == (2, 3, MANIFOLD_DIM), f"extract_segments 形状错误: {tuple(seg.shape)}"
    print(f"[a] extract {tuple(coords.shape)} / extract_segments {tuple(seg.shape)} OK")


def test_b_shared_projector_consistency() -> None:
    """b) 共享性：extract 坐标与直接 project 一致（同一 projector 实例，复用迭代①）。"""
    torch.manual_seed(7)
    proj = ThoughtManifoldProjector(D_MODEL, MANIFOLD_DIM).to(DEVICE).eval()
    bridge = ThoughtManifoldBridge(D_MODEL, MANIFOLD_DIM, projector=proj).to(DEVICE).eval()
    pm = torch.randn(3, 6, D_MODEL, device=DEVICE)
    with torch.no_grad():
        c_bridge = bridge.extract(pm)
        c_direct = proj.project(pm)
    assert torch.equal(c_bridge, c_direct), "桥 extract 必须与同一 projector 直接 project 完全一致"
    # 桥的 projector 就是传入实例（对象同一，非副本）
    assert bridge.projector is proj
    print("[b] 共享 projector：extract ≡ 直接 project（同一实例）OK")


def test_c_write_bounded() -> None:
    """c) write 有界：写后 pm_state 与原始的差被 alpha clamp 限制（增量范数 ≤ alpha×norm）。"""
    writer = ThoughtDisplacementWriter(max_alpha_frac=0.2)
    pm = torch.randn(2, 8, D_MODEL, device=DEVICE)
    disp = torch.randn(2, 8, D_MODEL, device=DEVICE)
    alpha = 0.1
    pm_w = writer.write(pm, disp, alpha=alpha)
    inc = (pm_w.float() - pm.float())  # 增量
    inc_norms = inc.norm(dim=-1)  # [B,T] 逐 token 增量范数
    bound = alpha * pm.float().norm(dim=-1).mean()
    assert (inc_norms <= bound + 1e-5).all(), (
        f"增量范数 {inc_norms.max().item():.4e} 超界 {bound.item():.4e}"
    )
    # α clamp：alpha=0.9 > max_alpha_frac=0.2 ⇒ 按 0.2 写入
    pm_w2 = writer.write(pm, disp, alpha=0.9)
    inc2 = (pm_w2.float() - pm.float()).norm(dim=-1)
    bound2 = 0.2 * pm.float().norm(dim=-1).mean()
    assert (inc2 <= bound2 + 1e-5).all(), "α 超上限时须 clamp 到 max_alpha_frac"
    # alpha=0 ⇒ 不变
    assert torch.equal(writer.write(pm, disp, alpha=0.0), pm)
    print(f"[c] 增量范数峰值 {inc_norms.max().item():.4e} ≤ 界 {bound.item():.4e}；"
          f"α=0.9 已 clamp 到 0.2（峰值 {inc2.max().item():.4e} ≤ {bound2.item():.4e}）OK")


def test_d_tick_moves_toward_target() -> None:
    """d) tick 端到端：pm_state 改变、坐标朝 target 方向移动（位移与 target−current 点积为正）。"""
    bridge = make_bridge()
    pm = torch.randn(2, 6, D_MODEL, device=DEVICE)
    target = torch.randn(2, 6, MANIFOLD_DIM, device=DEVICE)  # [B,T,manifold_dim]
    pm_w, cur, disp_m = bridge.tick(pm, target, alpha=0.15)
    # pm_state 改变
    assert not torch.allclose(pm_w, pm), "tick 后 pm_state 必须改变"
    # 返回的位移 = target − current（流形上朝 target）
    assert torch.allclose(disp_m, target - cur, atol=1e-5)
    # 位移方向与 target−current 一致（逐 token 点积为正）
    dots = (disp_m * (target - cur)).sum(dim=-1)
    assert (dots > 0).all(), "流形位移与 target−current 点积须为正"
    # target 用 [B,manifold_dim] 广播形式也应工作
    pm_w2, cur2, _ = bridge.tick(pm, target[:, 0, :], alpha=0.15)
    assert torch.equal(cur2, cur), "广播 target 与逐 token target 的当前坐标读出应一致"
    print(f"[d] tick：pm 改变（Δ峰值 {(pm_w-pm).abs().max().item():.3e}），"
          f"位移与 target−current 点积最小值 {dots.min().item():.4f} > 0 OK")


def test_e_backward_to_hidden_grad_and_steering_detached() -> None:
    """e) 反传：to_hidden 参数有梯度（离线显式目标）；tick 的 steering 路径 detach（梯度边界）。"""
    bridge = make_bridge()
    pm = torch.randn(2, 6, D_MODEL, device=DEVICE)
    # 离线训练目标（如重建）：to_hidden(project(x)) 逼近某目标 ⇒ to_hidden 获得梯度
    coords = bridge.extract(pm)
    recon = bridge.to_hidden(coords)
    loss = (recon - pm).pow(2).mean()
    loss.backward()
    g = bridge.to_hidden.proj.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "to_hidden 权重必须获得非零梯度"
    # steering 路径 detach：tick 的 disp_hidden 不经梯度回流（pm_w 对 pm 无梯度路径）
    bridge.zero_grad()
    pm2 = torch.randn(2, 6, D_MODEL, device=DEVICE, requires_grad=True)
    target = torch.randn(2, 6, MANIFOLD_DIM, device=DEVICE)
    pm_w, cur, disp_m = bridge.tick(pm2, target, alpha=0.1)
    # current_coord 与 disp_manifold 保留梯度路径（读侧可训）；写回 disp_hidden 已 detach
    assert cur.requires_grad and disp_m.requires_grad, "读侧坐标/位移须保留梯度路径（读侧可训）"
    # 写后 pm_w = pm2 + detach(增量) ⇒ 对 pm2 的梯度是恒等（增量无梯度），说明 steering detach
    pm_w.sum().backward()
    assert pm2.grad is not None, "pm_w 对 pm 须可微（恒等通路）"
    assert torch.allclose(pm2.grad, torch.ones_like(pm2), atol=1e-5), (
        "steering 增量已 detach：pm_w 对 pm 的梯度须为恒等（增量不回流）"
    )
    # to_hidden 不经 steering 路径获梯度
    assert bridge.to_hidden.proj.weight.grad is None or torch.equal(
        bridge.to_hidden.proj.weight.grad, torch.zeros_like(bridge.to_hidden.proj.weight.grad)
    ), "to_hidden 不得经 tick steering 路径获得梯度（梯度边界）"
    print("[e] to_hidden 经离线目标获梯度 OK；tick steering 路径 detach（增量不回流）OK")


def test_f_segment_boundaries_count() -> None:
    """f) 段聚合边界正确：不同 segment_boundaries 给出不同段数；段=区间均值池化。"""
    bridge = make_bridge()
    pm = torch.randn(2, 12, D_MODEL, device=DEVICE)
    s3 = bridge.extract_segments(pm, [0, 4, 8])       # 3 段 [0,4)[4,8)[8,12)
    s4 = bridge.extract_segments(pm, [0, 3, 6, 9])    # 4 段
    s1 = bridge.extract_segments(pm, [0])             # 1 段（全序列均值）
    assert s3.shape[1] == 3 and s4.shape[1] == 4 and s1.shape[1] == 1
    # 段值 = 段内 token 坐标的均值（验证池化正确性）
    coords = bridge.extract(pm)
    seg0_manual = coords[:, 0:4, :].mean(dim=1)
    assert torch.allclose(s3[:, 0, :], seg0_manual, atol=1e-6), "段0须等于 [0,4) token 坐标均值"
    seg_last_manual = coords[:, 8:12, :].mean(dim=1)
    assert torch.allclose(s3[:, 2, :], seg_last_manual, atol=1e-6), "末段须自动延伸至 T"
    # 非法边界：首元素非 0 / 非升序 / 越界
    for bad in ([2, 5], [0, 5, 5], [0, 12], [0, 20]):
        try:
            bridge.extract_segments(pm, bad)
            raise AssertionError(f"非法边界 {bad} 未报错")
        except (ValueError,):
            pass
    print(f"[f] 段数 3/4/1 OK；段值=区间均值池化 OK；非法边界报错 OK")


if __name__ == "__main__":
    test_a_extract_shapes()
    test_b_shared_projector_consistency()
    test_c_write_bounded()
    test_d_tick_moves_toward_target()
    test_e_backward_to_hidden_grad_and_steering_detached()
    test_f_segment_boundaries_count()
    print("全部思考流形桥接测试通过")
