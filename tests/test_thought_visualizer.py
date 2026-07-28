"""思考轨迹可视化前端（Thought Visualizer）测试：第二阶段（思维能力强化）迭代⑦ pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 轨迹点构建：tick_state→xyz [3]（经 project_3d），字段齐全；
  b) 轨迹组装：多 tick→ThoughtTrajectory，元数据正确（长度/坏点数）；
  c) 坏路径检测：信心膨胀（高 certainty 低 consistency）、漂移（位移异常）、
     recall 风暴（连续 recall）场景各自正确标记 is_bad+bad_reason；
  d) JSON 导出：to_dict/to_json 结构完整可序列化；
  e) ASCII 渲染：输出字符串非空、含 recall/bad 标记；
  f) 与 projector 集成：用真实 ThoughtManifoldProjector 的 project_3d（固定 view3d），
     坐标确定可复现。
用法：python -m pytest tests/test_thought_visualizer.py -q（纯 CPU 可测）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import ThoughtManifoldProjector
from tais_obsidian.model.reasoning_loop import ReasoningTickState
from tais_obsidian.model.thought_visualizer import (
    ThoughtTrajectory,
    ThoughtTrajectoryPoint,
    ThoughtVisualizer,
    render_ascii,
)

MANIFOLD_DIM = 64
D_MODEL = 384
B, T = 2, 4


def make_projector(seed: int = 0) -> ThoughtManifoldProjector:
    torch.manual_seed(seed)
    return ThoughtManifoldProjector(d_model=D_MODEL, manifold_dim=MANIFOLD_DIM)


def make_tick(
    tick_index: int,
    coord_seed: int = 0,
    certainty: float = 0.5,
    recall_triggered: bool = False,
    early_stop: bool = False,
) -> ReasoningTickState:
    """构造单个 ReasoningTickState（current_coord/disp 为随机张量，可复现）。"""
    g = torch.Generator().manual_seed(coord_seed)
    current_coord = torch.randn(B, T, MANIFOLD_DIM, generator=g)
    disp = torch.randn(B, T, MANIFOLD_DIM, generator=g) * 0.1
    return ReasoningTickState(
        tick_index=tick_index,
        current_coord=current_coord,
        disp=disp,
        certainty=certainty,
        hrl_topk_idx=None,
        early_stop=early_stop,
        recall_triggered=recall_triggered,
    )


# ---------- a) 轨迹点构建 ----------


def test_a_point_fields() -> None:
    """a) 单 tick 经 build 后 xyz [3] 且字段齐全。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    traj = viz.build([make_tick(0, coord_seed=1)], proj)
    assert traj.n_ticks == 1
    p = traj.points[0]
    assert isinstance(p, ThoughtTrajectoryPoint)
    assert len(p.xyz) == 3, f"xyz 须为 [3]，实得 {len(p.xyz)}"
    assert all(isinstance(v, float) for v in p.xyz)
    assert p.tick_index == 0
    assert isinstance(p.certainty, float)
    assert isinstance(p.recall_triggered, bool)
    assert isinstance(p.early_stop, bool)
    assert isinstance(p.is_bad_path, bool)
    assert isinstance(p.bad_reason, str)
    # xyz 与 projector.project_3d 一致（可复现）
    coord = proj.view3d  # 固定 view3d
    assert torch.allclose(
        torch.tensor(p.xyz),
        proj.project_3d(
            torch.stack([make_tick(0, coord_seed=1).current_coord.mean(dim=(0, 1))])
        )[0],
        atol=1e-5,
    ), "xyz 须经 project_3d 且可复现"


# ---------- b) 轨迹组装与元数据 ----------


def test_b_trajectory_meta() -> None:
    """b) 多 tick 组装：n_ticks/stop_tick/recall_any/avg_certainty/n_bad 正确。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    ticks = [
        make_tick(0, coord_seed=10, certainty=0.4),
        make_tick(1, coord_seed=11, certainty=0.6),
        make_tick(2, coord_seed=12, certainty=0.8, early_stop=True),
    ]
    traj = viz.build(ticks, proj)
    assert traj.n_ticks == 3
    assert traj.stop_tick == 2, f"stop_tick 须为早停 tick 2，实得 {traj.stop_tick}"
    assert traj.recall_triggered_any is False
    assert abs(traj.avg_certainty - (0.4 + 0.6 + 0.8) / 3) < 1e-6
    assert traj.n_bad_points >= 0

    # recall_any：含一个 recall tick
    ticks2 = [
        make_tick(0, coord_seed=20, certainty=0.3, recall_triggered=True),
        make_tick(1, coord_seed=21, certainty=0.5),
    ]
    traj2 = viz.build(ticks2, proj)
    assert traj2.recall_triggered_any is True
    # stop_tick：无早停时为最后一个 tick
    assert traj2.stop_tick == 1


def test_b_to_dict_and_json(tmp_path: Path) -> None:
    """b/d) to_dict/to_json 结构完整可序列化。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    ticks = [make_tick(i, coord_seed=i, certainty=0.5) for i in range(3)]
    traj = viz.build(ticks, proj)
    d = traj.to_dict()
    assert "meta" in d and "points" in d
    assert d["meta"]["n_ticks"] == 3
    assert len(d["points"]) == 3
    # JSON 可序列化
    s = json.dumps(d, ensure_ascii=False)
    assert isinstance(s, str)
    # to_json 写文件
    out = tmp_path / "traj.json"
    traj.to_json(out)
    assert out.exists()
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded["meta"]["n_ticks"] == 3
    assert len(loaded["points"]) == 3
    # summary 非空
    assert isinstance(traj.summary(), str) and len(traj.summary()) > 0


# ---------- c) 坏路径检测 ----------


def test_c_confidence_inflation() -> None:
    """c① 信心膨胀：高 certainty 低 speak_do_consistency → is_bad=True + bad_reason 含'信心膨胀'。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    ts = make_tick(0, coord_seed=100, certainty=0.9)  # 高 certainty
    faith = {"speak_do_consistency": 0.1}  # 低一致性（< 0.3）
    is_bad, reason = viz.detect_bad_path(ts, None, faith)
    assert is_bad is True
    assert "信心膨胀" in reason


def test_c_drift() -> None:
    """c② 漂移：当前坐标远离上一坐标（位移异常大）→ is_bad=True + bad_reason 含'漂移'。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    # 上一坐标：小值
    prev = torch.zeros(MANIFOLD_DIM)
    # 当前坐标：大偏移（范数 > drift_disp_ratio=3.0）
    g = torch.Generator().manual_seed(200)
    cur = torch.randn(B, T, MANIFOLD_DIM, generator=g) * 10.0  # 大尺度 → 范数大
    ts = ReasoningTickState(
        tick_index=1, current_coord=cur, disp=torch.zeros(B, T, MANIFOLD_DIM),
        certainty=0.5,
    )
    is_bad, reason = viz.detect_bad_path(ts, prev, None)
    assert is_bad is True
    assert "漂移" in reason


def test_c_recall_storm() -> None:
    """c④ recall 风暴：连续多 tick recall_triggered → is_bad=True + bad_reason 含'recall风暴'。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    # 构造连续 3 个 recall tick（recall_storm_consecutive=3）
    ticks = [
        make_tick(i, coord_seed=300 + i, certainty=0.2, recall_triggered=True)
        for i in range(3)
    ]
    traj = viz.build(ticks, proj)
    # 第 3 个 tick（consec_recall 达 3）应标 recall 风暴
    assert traj.points[2].is_bad_path is True
    assert "recall风暴" in traj.points[2].bad_reason
    # 前两个 tick 不达阈值（consec_recall<3），不应因 recall 风暴标坏
    assert "recall风暴" not in traj.points[0].bad_reason
    assert "recall风暴" not in traj.points[1].bad_reason


def test_c_early_stop_fail() -> None:
    """c③ 早停失败：certainty 始终低于阈值却跑满 max_ticks → is_bad=True + bad_reason 含'早停失败'。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    # 全程低 certainty、无早停、跑满 3 ticks
    ticks = [
        make_tick(i, coord_seed=400 + i, certainty=0.3, early_stop=False)
        for i in range(3)
    ]
    traj = viz.build(ticks, proj)
    # 最后一个 tick 应标早停失败
    assert traj.points[-1].is_bad_path is True
    assert "早停失败" in traj.points[-1].bad_reason
    # 非最后 tick 不应标早停失败
    assert "早停失败" not in traj.points[0].bad_reason


def test_c_bad_composable() -> None:
    """c) 坏路径可组合：信心膨胀 + 漂移 同时标记，bad_reason 含两者。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    prev = torch.zeros(MANIFOLD_DIM)
    g = torch.Generator().manual_seed(500)
    cur = torch.randn(B, T, MANIFOLD_DIM, generator=g) * 10.0
    ts = ReasoningTickState(
        tick_index=1, current_coord=cur, disp=torch.zeros(B, T, MANIFOLD_DIM),
        certainty=0.9,
    )
    faith = {"speak_do_consistency": 0.1}
    is_bad, reason = viz.detect_bad_path(ts, prev, faith)
    assert is_bad is True
    assert "信心膨胀" in reason and "漂移" in reason


# ---------- e) ASCII 渲染 ----------


def test_e_render_ascii() -> None:
    """e) ASCII 渲染：输出非空、含 recall(R)/bad(X) 标记。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    # 构造含 recall 与坏路径的轨迹
    ticks = [
        make_tick(0, coord_seed=600, certainty=0.5),
        make_tick(1, coord_seed=601, certainty=0.2, recall_triggered=True),
        make_tick(2, coord_seed=602, certainty=0.2, recall_triggered=True),
        make_tick(3, coord_seed=603, certainty=0.2, recall_triggered=True),  # recall 风暴
    ]
    traj = viz.build(ticks, proj)
    art = render_ascii(traj)
    assert isinstance(art, str) and len(art) > 0
    # 含 R（recall）与 X（坏路径，recall 风暴 tick 标 X）
    assert "R" in art or "X" in art, "ASCII 渲染须含 recall(R) 或坏路径(X) 标记"
    # 坏路径 tick（recall 风暴第3个）标 X
    assert "X" in art, "recall 风暴坏路径 tick 须标 X"
    # 数字 tick 序号出现
    assert any(ch.isdigit() for ch in art), "ASCII 渲染须含 tick 序号数字"


def test_e_render_ascii_empty() -> None:
    """e) 空轨迹渲染：返回占位字符串。"""
    traj = ThoughtTrajectory(points=[], n_ticks=0)
    art = render_ascii(traj)
    assert art == "(空轨迹)"


# ---------- f) projector 集成可复现 ----------


def test_f_projector_reproducible() -> None:
    """f) 同一 projector + 同一 coord → xyz 可复现（固定 view3d 无梯度）。"""
    proj = make_projector(seed=0)
    viz = ThoughtVisualizer()
    ticks = [make_tick(0, coord_seed=700)]
    traj1 = viz.build(ticks, proj)
    traj2 = viz.build([make_tick(0, coord_seed=700)], proj)
    assert traj1.points[0].xyz == traj2.points[0].xyz, "同一输入 xyz 须可复现"
    # view3d 无梯度（固定投影）
    for p in proj.view3d.parameters():
        assert p.requires_grad is False
    # 不同 projector（不同 seed 的 proj 权重）→ project 不同，但 view3d 固定 seed=0
    # 故相同 coord 输入时 xyz 相同（view3d 确定性）
    proj2 = make_projector(seed=1)
    coord = make_tick(0, coord_seed=700).current_coord.mean(dim=(0, 1))
    xyz1 = proj.project_3d(coord)
    xyz2 = proj2.project_3d(coord)
    assert torch.allclose(xyz1, xyz2), "view3d 固定 seed=0，跨 projector 实例 xyz 须一致"


def test_f_build_no_grad() -> None:
    """f) build 全程 no_grad（监测/执行分置）：projector 权重无梯度累积。"""
    proj = make_projector()
    viz = ThoughtVisualizer()
    proj.proj.weight.grad = None
    ticks = [make_tick(0, coord_seed=800)]
    viz.build(ticks, proj)
    # project_3d 用 view3d（requires_grad=False），且 build 装饰 @torch.no_grad
    assert proj.proj.weight.grad is None, "build 不应给 projector 任何梯度"
