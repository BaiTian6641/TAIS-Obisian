"""推理循环形式化（Reasoning Loop）测试：第二阶段（思维能力强化）迭代④ pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 形状：reasoning_tick 返回 state 同输入形状 + ReasoningTickState 字段齐全；
  b) §1.3 顺序：用 mock 部件验证调用顺序（glimpse→propose→certainty→forward_step→bridge.tick）；
  c) HRL 提议：candidates 给定时 hrl_propose 返回 top-k idx（有 kernel）或 None（无 kernel）；
  d) KAL certainty：有 kernel 时 certainty 来自 sense P(IK)；无 kernel 时 mock ∈ [0,1]；
  e) 早停：mock certainty 递增，达阈值提前停（stop_tick < max_ticks）；
  f) 空白 recall：certainty 低时 ReasoningTickState 标记 recall 触发；
     trajectory_to_recall_tokens 正确标出 <|recall|>；
  g) bridge 集成：reasoning_tick 后 PM-stream state 被 bridge.tick 更新（位移朝 target）；
  h) 反传：run 后 loss.backward()，thought_core.group_mlp 有梯度。
用法：python -m pytest tests/test_reasoning_loop.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold_bridge import ThoughtManifoldBridge
from tais_obsidian.model.reasoning_loop import (
    RECALL_TOKEN,
    ReasoningLoop,
    ReasoningTickState,
    trajectory_to_recall_tokens,
)
from tais_obsidian.model.tais_kernel import TAISKernel, SenseOut
from tais_obsidian.model.thought_core import ThoughtCore

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORE_DIM = 384
N_GROUPS = 8
HISTORY = 4
MAX_TICKS = 8
MANIFOLD_DIM = 64


def make_loop(seed: int = 42, with_kernel: bool = False) -> ReasoningLoop:
    torch.manual_seed(seed)
    core = ThoughtCore(
        core_dim=CORE_DIM, n_groups=N_GROUPS, history=HISTORY,
        max_ticks=MAX_TICKS, manifold_dim=MANIFOLD_DIM, use_sync=True,
    )
    bridge = ThoughtManifoldBridge(d_model=CORE_DIM, manifold_dim=MANIFOLD_DIM)
    kernel = TAISKernel(CORE_DIM) if with_kernel else None
    return ReasoningLoop(core, bridge, kernel).to(DEVICE)


def make_inputs(B: int = 2, T: int = 10, seed: int = 0):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    state = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    target = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
    return state, target


# ---------- a) 形状与字段 ----------

def test_a_tick_shape_and_fields() -> None:
    """a) reasoning_tick 返回 state 同输入形状 + ReasoningTickState 字段齐全。"""
    loop = make_loop()
    state, target = make_inputs()
    loop.thought_core.history.reset()
    new_state, ts = loop.reasoning_tick(state, 0, target_coord=target)
    assert new_state.shape == state.shape, (
        f"new_state 形状 {tuple(new_state.shape)} ≠ 输入 {tuple(state.shape)}"
    )
    assert isinstance(ts, ReasoningTickState)
    assert ts.tick_index == 0
    assert ts.current_coord.shape == (2, 10, MANIFOLD_DIM), (
        f"current_coord 形状错误: {tuple(ts.current_coord.shape)}"
    )
    assert ts.disp.shape == (2, 10, MANIFOLD_DIM)
    assert 0.0 <= ts.certainty <= 1.0, f"certainty 须在 [0,1]，实得 {ts.certainty}"
    assert ts.hrl_topk_idx is None  # 无 kernel → None（接口位）
    assert ts.early_stop is False
    assert isinstance(ts.recall_triggered, bool)
    print(f"[a] tick 形状 {tuple(new_state.shape)} + 字段齐全（certainty={ts.certainty:.3f}）OK")


# ---------- b) §1.3 顺序 ----------

def test_b_tick_order() -> None:
    """b) §1.3 顺序：mock 部件验证 glimpse→propose→certainty→forward_step→bridge.tick。"""
    loop = make_loop()
    calls: list[str] = []

    # 逐方法打桩记录调用顺序（mock 编排层，不触碰真实部件逻辑）
    orig_glimpse = loop.glimpse
    orig_propose = loop.hrl_propose
    orig_certainty = loop.kal_certainty
    orig_step = loop.thought_core.forward_step
    orig_tick = loop.bridge.tick

    def spy_glimpse(s, c=None):
        calls.append("glimpse")
        return orig_glimpse(s, c)

    def spy_propose(o, c, k=4):
        calls.append("propose")
        return orig_propose(o, c, k)

    def spy_certainty(s):
        calls.append("certainty")
        return orig_certainty(s)

    def spy_step(s, k):
        calls.append("forward_step")
        return orig_step(s, k)

    def spy_tick(s, t, alpha=0.1):
        calls.append("bridge_tick")
        return orig_tick(s, t, alpha=alpha)

    loop.glimpse = spy_glimpse
    loop.hrl_propose = spy_propose
    loop.kal_certainty = spy_certainty
    loop.thought_core.forward_step = spy_step
    loop.bridge.tick = spy_tick

    state, target = make_inputs()
    loop.thought_core.history.reset()
    loop.reasoning_tick(state, 0, target_coord=target)
    assert calls == ["glimpse", "propose", "certainty", "forward_step", "bridge_tick"], (
        f"§1.3 顺序错误: {calls}"
    )
    print(f"[b] §1.3 调用顺序 {'→'.join(calls)} OK")


# ---------- c) HRL 提议 ----------

def test_c_hrl_propose_no_kernel() -> None:
    """c) 无 kernel：hrl_propose 返回 None（接口位）。"""
    loop = make_loop(with_kernel=False)
    state, _ = make_inputs()
    obs = loop.glimpse(state)
    cands = torch.randn(2, 6, CORE_DIM, device=DEVICE)
    assert loop.hrl_propose(obs, cands, k=3) is None, "无 kernel 应返回 None（接口位）"
    assert loop.hrl_propose(obs, None, k=3) is None, "candidates=None 应返回 None"
    print("[c] 无 kernel → hrl_propose=None（接口位）OK")


def test_c_hrl_propose_with_kernel() -> None:
    """c) 有 kernel：hrl_propose 调 route_candidates 返回 top-k idx，形状 [B,1,k]。"""
    loop = make_loop(with_kernel=True)
    state, _ = make_inputs()
    obs = loop.glimpse(state)  # [B, core_dim]
    cands = torch.randn(2, 6, CORE_DIM, device=DEVICE)
    idx = loop.hrl_propose(obs, cands, k=3)
    assert idx is not None, "有 kernel + candidates 应返回 top-k idx"
    assert idx.shape == (2, 1, 3), f"top-k idx 形状错误: {tuple(idx.shape)}"
    assert idx.dtype in (torch.int64, torch.int32), "top-k idx 应为整型索引"
    assert (idx >= 0).all() and (idx < 6).all(), "top-k idx 应在候选数范围内"
    print(f"[c] 有 kernel → top-k idx {tuple(idx.shape)}（范围 [0,6)）OK")


# ---------- d) KAL certainty ----------

def test_d_certainty_from_kernel_sense() -> None:
    """d) 有 kernel：certainty 来自 sense P(IK)（softmax known 类概率 ∈ [0,1]）。"""
    loop = make_loop(with_kernel=True)
    state, _ = make_inputs()
    cert = loop.kal_certainty(state)
    # 对照：直接调 kernel.sense 手算 known 类概率
    with torch.no_grad():
        sense = loop.kernel.sense(state)
        expect = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)[:, 0].mean().item()
    assert abs(cert - expect) < 1e-6, (
        f"certainty {cert} 应等于 sense P(IK) known 类概率 {expect}"
    )
    assert 0.0 <= cert <= 1.0
    print(f"[d] 有 kernel：certainty={cert:.4f}=sense P(IK) known 概率 OK")


def test_d_certainty_mock_no_kernel() -> None:
    """d) 无 kernel：mock certainty（state norm 的 sigmoid）∈ [0,1]。"""
    loop = make_loop(with_kernel=False)
    state, _ = make_inputs()
    cert = loop.kal_certainty(state)
    assert 0.0 <= cert <= 1.0, f"mock certainty 须在 [0,1]，实得 {cert}"
    # 与 sigmoid(norm) 口径一致
    with torch.no_grad():
        norm = state.float()[:, -1, :].norm(dim=-1).mean()
        expect = float(torch.sigmoid(norm).item())
    assert abs(cert - expect) < 1e-6, f"mock certainty {cert} ≠ sigmoid(norm) {expect}"
    print(f"[d] 无 kernel：mock certainty={cert:.4f}=sigmoid(norm) ∈ [0,1] OK")


# ---------- e) 早停 ----------

def test_e_early_stop_increasing_certainty() -> None:
    """e) mock certainty 递增，达阈值提前停（stop_tick < max_ticks）。"""
    loop = make_loop()
    calls = {"n": 0}

    def increasing_certainty(s: torch.Tensor) -> float:
        calls["n"] += 1
        return 0.3 * calls["n"]  # 0.3, 0.6, 0.9, 1.2 —— 第 4 次 1.2 > 0.9 → 提前停

    loop.kal_certainty = increasing_certainty
    state, target = make_inputs()
    final, traj, stop_tick = loop.run(
        state, target_coord=target, max_ticks=8, stop_threshold=0.9
    )
    assert stop_tick == 4, f"第 4 tick certainty=1.2>0.9 应提前停，实得 stop_tick={stop_tick}"
    assert len(traj) == stop_tick
    assert traj[-1].early_stop is True, "末 tick 应标记 early_stop"
    assert all(not t.early_stop for t in traj[:-1]), "非末 tick 不应标记 early_stop"
    print(f"[e] 递增 certainty：stop_tick={stop_tick} < max_ticks=8，末 tick early_stop OK")


def test_e_full_ticks_low_certainty() -> None:
    """e) certainty 始终低 → 跑满 max_ticks。"""
    loop = make_loop()
    loop.kal_certainty = lambda s: 0.1
    state, target = make_inputs()
    final, traj, stop_tick = loop.run(state, target_coord=target, max_ticks=5)
    assert stop_tick == 5 == len(traj), f"低 certainty 应跑满，实得 stop_tick={stop_tick}"
    assert not any(t.early_stop for t in traj)
    print(f"[e] 低 certainty：跑满 max_ticks={stop_tick} OK")


# ---------- f) 空白 recall 审计 ----------

def test_f_recall_triggered_and_tokens() -> None:
    """f) certainty 低 → recall_triggered；trajectory_to_recall_tokens 正确标出 <|recall|>。"""
    loop = make_loop()
    # certainty 序列：高, 低, 高, 低 → 第 2/4 tick 触发空白
    certs = iter([0.8, 0.1, 0.7, 0.2])
    loop.kal_certainty = lambda s: next(certs)
    state, target = make_inputs()
    final, traj, stop_tick = loop.run(
        state, target_coord=target, max_ticks=4, recall_threshold=0.3
    )
    assert stop_tick == 4 == len(traj)
    flags = [t.recall_triggered for t in traj]
    assert flags == [False, True, False, True], f"recall 触发标记错误: {flags}"
    tokens = trajectory_to_recall_tokens(traj)
    assert len(tokens) == 4
    assert tokens[1] == RECALL_TOKEN and tokens[3] == RECALL_TOKEN, (
        f"空白 tick 应显式标出 {RECALL_TOKEN}: {tokens}"
    )
    assert tokens[0] != RECALL_TOKEN and tokens[2] != RECALL_TOKEN, (
        f"非空白 tick 不应标 {RECALL_TOKEN}: {tokens}"
    )
    print(f"[f] recall 触发 {flags} → tokens {tokens} OK")


# ---------- g) bridge 集成 ----------

def test_g_bridge_updates_pm_toward_target() -> None:
    """g) reasoning_tick 后 PM-stream state 被 bridge.tick 更新（位移朝 target）。"""
    loop = make_loop()
    state, target = make_inputs()
    loop.thought_core.history.reset()
    new_state, ts = loop.reasoning_tick(state, 0, target_coord=target)
    # state 被更新（bridge 写回改变 PM-stream）
    assert not torch.allclose(new_state, state), "reasoning_tick 后 state 必须改变"
    # 流形位移 = target − current（朝 target 方向，点积为正）
    dots = (ts.disp * (target - ts.current_coord)).sum(dim=-1)
    assert (dots > 0).all(), "流形位移与 target−current 点积须为正（朝 target）"
    # 写后坐标比写前更接近 target（bridge.tick 有界写回的直接后果）
    core = loop.thought_core.forward_step  # noqa: F841（明示演化步存在）
    loop.thought_core.history.reset()
    evolved = loop.thought_core.forward_step(state, 0)
    coord_before = loop.bridge.extract(evolved)
    coord_after = loop.bridge.extract(new_state)
    dist_before = (coord_before - target).norm(dim=-1).mean()
    dist_after = (coord_after - target).norm(dim=-1).mean()
    assert dist_after < dist_before, (
        f"写后坐标应更接近 target：{dist_after.item():.4f} 不 < {dist_before.item():.4f}"
    )
    print(f"[g] bridge 写回：位移朝 target（点积 min {dots.min().item():.3f}>0），"
          f"target 距离 {dist_before.item():.4f}→{dist_after.item():.4f} OK")


# ---------- h) 反传 ----------

def test_h_backward_group_mlp() -> None:
    """h) run 后 loss.backward()，thought_core.group_mlp 有梯度。"""
    loop = make_loop()
    state, target = make_inputs()
    state.requires_grad_(True)
    final, traj, stop_tick = loop.run(state, target_coord=target, max_ticks=4)
    loss = final.square().mean()
    loss.backward()
    grads = [
        p.grad for name, p in loop.thought_core.named_parameters()
        if "group_mlp" in name and p.requires_grad
    ]
    assert len(grads) > 0, "group_mlp 无可训练参数"
    assert all(g is not None for g in grads), "group_mlp 参数梯度为 None"
    assert all(g.abs().sum() > 0 for g in grads), "group_mlp 参数梯度全零"
    print(f"[h] 反传：group_mlp {len(grads)} 个参数梯度非零 OK")
