"""CTM 式思考核（Thought Core）测试：第二阶段（思维能力强化）迭代③ pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 形状：forward_step [B,T,core_dim]→同形状；think 返回轨迹长度=停止 tick 数；
  b) 思考时间相位化：不同 tick_index 施加不同相位（同输入不同 tick 输出不同）；
     use_sync=False 时不同 tick 仅靠 MLP 演化；
  c) 自适应早停：certainty_fn 返回递增 certainty，达阈值提前停（停止 tick < max_ticks）；
     certainty 始终低则跑满；
  d) 通道组历史：update/get 历史缓冲正确（长度≤H，FIFO）；
  e) 反传：think 后 loss.backward()，group_mlp 参数有梯度；
  f) 自消融：use_sync True/False 同种子同输入产出不同轨迹（验证相位化确实改变动力学）；
  g) 与 bridge 集成：think 内调用 bridge.tick 写 PM-stream，PM 状态被更新（真实 bridge）。
用法：python -m pytest tests/test_thought_core.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.thought_core import (
    ChannelGroupHistory,
    ThoughtCore,
    ThoughtTimeRotary,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORE_DIM = 384
N_GROUPS = 8
HISTORY = 4
MAX_TICKS = 8
MANIFOLD_DIM = 64


def make_core(seed: int = 42, use_sync: bool = True) -> ThoughtCore:
    torch.manual_seed(seed)
    return ThoughtCore(
        core_dim=CORE_DIM,
        n_groups=N_GROUPS,
        history=HISTORY,
        max_ticks=MAX_TICKS,
        manifold_dim=MANIFOLD_DIM,
        use_sync=use_sync,
    ).to(DEVICE)


# ---------- a) 形状 ----------

def test_a_forward_step_shape() -> None:
    """a) forward_step [B,T,core_dim]→同形状。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    out = core.forward_step(state, tick_index=0)
    assert out.shape == (2, 10, CORE_DIM), f"forward_step 形状错误: {tuple(out.shape)}"
    print(f"[a] forward_step {tuple(state.shape)}→{tuple(out.shape)} OK")


def test_a_think_shapes() -> None:
    """a) think 返回轨迹长度=停止 tick 数；最终状态/轨迹元素同 [B,T,core_dim]。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    final, traj, stop_tick = core.think(state, max_ticks=5)
    assert final.shape == (2, 10, CORE_DIM), f"final 形状错误: {tuple(final.shape)}"
    assert len(traj) == stop_tick == 5, f"轨迹长度={len(traj)} ≠ 停止 tick={stop_tick}"
    for i, t in enumerate(traj):
        assert t.shape == (2, 10, CORE_DIM), f"traj[{i}] 形状错误: {tuple(t.shape)}"
    assert torch.equal(final, traj[-1]), "final 应等于轨迹最后一个状态"
    print(f"[a] think 轨迹长度={len(traj)}=stop_tick={stop_tick} OK")


# ---------- b) 思考时间相位化 ----------

def test_b_sync_phase_differs() -> None:
    """b) use_sync=True：同输入不同 tick_index → 不同相位 → 不同输出。"""
    core = make_core(use_sync=True)
    core.history.reset()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    out0 = core.forward_step(state, tick_index=0)
    core.history.reset()
    out1 = core.forward_step(state, tick_index=1)
    assert not torch.allclose(out0, out1, atol=1e-5), (
        "use_sync=True 时不同 tick_index 应产生不同相位化输出"
    )
    print("[b] use_sync=True：不同 tick 相位不同 → 输出不同 OK")


def test_b_nosync_mlp_only() -> None:
    """b) use_sync=False：不同 tick 仅靠 MLP 演化（无相位化，输出由历史驱动）。"""
    core = make_core(use_sync=False)
    core.history.reset()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    out0 = core.forward_step(state, tick_index=0)
    core.history.reset()
    out1 = core.forward_step(state, tick_index=5)
    # use_sync=False 时 forward_step 不依赖 tick_index（相位化关闭），首轮历史仅含
    # 当前 state（零填充），故同 state + 不同 tick_index → 同输出（纯 MLP 演化）。
    assert torch.allclose(out0, out1, atol=1e-5), (
        "use_sync=False 时 forward_step 不应依赖 tick_index（无相位化）"
    )
    print("[b] use_sync=False：forward_step 不依赖 tick_index（纯 MLP 演化）OK")


# ---------- c) 自适应早停 ----------

def test_c_early_stop() -> None:
    """c) certainty_fn 返回递增 certainty，达阈值提前停（停止 tick < max_ticks）。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    calls = {"n": 0}

    def certainty_fn(s: torch.Tensor) -> float:
        calls["n"] += 1
        # certainty 递增：0.3, 0.6, 0.9, 1.2——第 4 次调用（k=3）1.2 > 阈值 0.9 → 提前停
        return 0.3 * calls["n"]

    final, traj, stop_tick = core.think(state, certainty_fn=certainty_fn, max_ticks=8)
    assert stop_tick < 8, f"应提前停，实得 stop_tick={stop_tick}"
    assert stop_tick == 4, f"certainty 第 4 tick 达 1.2>0.9，实得 stop_tick={stop_tick}"
    assert len(traj) == stop_tick
    print(f"[c] 递增 certainty：stop_tick={stop_tick} < max_ticks=8 OK")


def test_c_full_ticks_when_low_certainty() -> None:
    """c) certainty 始终低 → 跑满 max_ticks。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    final, traj, stop_tick = core.think(
        state, certainty_fn=lambda s: 0.1, max_ticks=6
    )
    assert stop_tick == 6 == len(traj), f"低 certainty 应跑满，实得 stop_tick={stop_tick}"
    print(f"[c] 低 certainty：跑满 max_ticks={stop_tick} OK")


def test_c_no_certainty_fn_runs_full() -> None:
    """c) certainty_fn 缺省 → 跑满 max_ticks。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    final, traj, stop_tick = core.think(state, max_ticks=7)
    assert stop_tick == 7 == len(traj)
    print(f"[c] certainty_fn 缺省：跑满 max_ticks={stop_tick} OK")


# ---------- d) 通道组历史 ----------

def test_d_history_fifo() -> None:
    """d) ChannelGroupHistory：update/get 历史缓冲正确（长度≤H，FIFO）。"""
    hist = ChannelGroupHistory(core_dim=CORE_DIM, n_groups=N_GROUPS, history=HISTORY)
    assert hist.get() is None, "未 update 前应返回 None"
    B, T = 2, 10
    acts = [torch.randn(B, T, CORE_DIM, device=DEVICE) for _ in range(6)]
    for i, a in enumerate(acts):
        h = hist.update(a)
        assert h.shape == (B, T, N_GROUPS, HISTORY, CORE_DIM // N_GROUPS), (
            f"第{i}次 update 历史形状错误: {tuple(h.shape)}"
        )
    # FIFO：最近 H=4 次保留，最旧被挤出——最新在 dim=-2 末尾
    h_final = hist.get()
    gd = CORE_DIM // N_GROUPS
    for j in range(HISTORY):
        expected = acts[6 - HISTORY + j].view(B, T, N_GROUPS, gd)
        assert torch.allclose(h_final[..., j, :], expected, atol=1e-6), (
            f"FIFO 错位：历史槽 {j} 应为第 {6-HISTORY+j} 次激活"
        )
    print("[d] 通道组历史 FIFO（长度≤H，最新在末尾）OK")


def test_d_history_zero_pad_first() -> None:
    """d) 首轮 update 左侧零填充到 H（固定形状，最新在末尾）。"""
    hist = ChannelGroupHistory(core_dim=CORE_DIM, n_groups=N_GROUPS, history=HISTORY)
    a = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    h = hist.update(a)
    gd = CORE_DIM // N_GROUPS
    assert h.shape == (2, 10, N_GROUPS, HISTORY, gd)
    # 前 H-1 槽为零，末槽为当前激活
    for j in range(HISTORY - 1):
        assert (h[..., j, :] == 0).all(), f"首轮历史槽 {j} 应为零填充"
    assert torch.allclose(h[..., -1, :], a.view(2, 10, N_GROUPS, gd), atol=1e-6)
    print("[d] 首轮零填充（前 H-1 槽=0，末槽=当前）OK")


# ---------- e) 反传 ----------

def test_e_backward_group_mlp() -> None:
    """e) think 后 loss.backward()，group_mlp 参数有梯度。"""
    core = make_core()
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE, requires_grad=True)
    final, traj, stop_tick = core.think(state, max_ticks=4)
    loss = final.square().mean()
    loss.backward()
    grads = [
        p.grad for name, p in core.named_parameters()
        if "group_mlp" in name and p.requires_grad
    ]
    assert len(grads) > 0, "group_mlp 无可训练参数"
    assert all(g is not None for g in grads), "group_mlp 参数梯度为 None"
    assert all(g.abs().sum() > 0 for g in grads), "group_mlp 参数梯度全零"
    print(f"[e] 反传：group_mlp {len(grads)} 个参数均有非零梯度 OK")


# ---------- f) 自消融 ----------

def test_f_sync_ablation() -> None:
    """f) use_sync True/False 同种子同输入 → 不同轨迹（验证相位化确实改变动力学）。"""
    torch.manual_seed(123)
    core_sync = make_core(seed=123, use_sync=True)
    torch.manual_seed(123)
    core_nosync = make_core(seed=123, use_sync=False)
    torch.manual_seed(999)
    state = torch.randn(2, 10, CORE_DIM, device=DEVICE)
    _, traj_sync, _ = core_sync.think(state.clone(), max_ticks=4)
    _, traj_nosync, _ = core_nosync.think(state.clone(), max_ticks=4)
    assert len(traj_sync) == len(traj_nosync) == 4
    differs = [
        not torch.allclose(traj_sync[k], traj_nosync[k], atol=1e-4)
        for k in range(4)
    ]
    assert any(differs), (
        "use_sync True/False 同种子同输入应产出不同轨迹（相位化贡献自消融）"
    )
    n_diff = sum(differs)
    print(f"[f] 自消融：use_sync 开关改变 {n_diff}/4 个 tick 的轨迹 OK")


# ---------- g) bridge 集成 ----------

def test_g_bridge_integration() -> None:
    """g) think 内 integrate_bridge=True：bridge.tick 写 PM-stream，状态被更新。"""
    core = make_core()
    B, T = 2, 10
    pm_state = torch.randn(B, T, CORE_DIM, device=DEVICE)
    target = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE)
    final, traj, stop_tick = core.think(
        pm_state.clone(),
        max_ticks=3,
        integrate_bridge=True,
        bridge_target=target,
    )
    # bridge.tick 有界写回：final 应与初始 pm_state 不同（PM 状态被更新）
    assert not torch.allclose(final, pm_state, atol=1e-4), (
        "integrate_bridge=True 时 bridge.tick 应更新 PM 状态"
    )
    # 有界纪律：增量范数 ≤ max_alpha_frac × norm（steering clamp）
    inc = (final.float() - pm_state.float()).norm(dim=-1)
    bound = core.bridge.writer.max_alpha_frac * pm_state.float().norm(dim=-1).mean()
    # 3 tick 累积写，每 tick 增量 ≤ bound，累积 ≤ 3×bound（粗略上界）
    assert (inc <= 3 * bound + 1e-4).all(), (
        f"bridge 累积写增量应受 alpha clamp 限制：max inc={inc.max().item():.4f} "
        f"vs 3×bound={3*bound.item():.4f}"
    )
    print(f"[g] bridge 集成：PM 状态被更新（max 增量={inc.max().item():.4f}）OK")


if __name__ == "__main__":
    # 手动跑全部测试（无 pytest 时的冒烟路径）
    for name, fn in sorted(
        {k: v for k, v in globals().items() if k.startswith("test_")}.items()
    ):
        fn()
        print(f"  ✓ {name}")
    print("全部通过（手动模式）")
