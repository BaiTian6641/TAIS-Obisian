"""流形推理预览可视化（manifold_preview）测试：小尺度合成数据，不依赖模型/checkpoint。

覆盖判据（对齐任务规范 §验证）：
  a) 投影维度正确：hidden [T,d] → coords64 [T,64] → xyz [T,3]（合成小维度验证）；
  b) 渲染出图：3D 与 2D 三视图均产出非空 PNG（matplotlib Agg，无 GUI）；
  c) 知识块叠加路径不崩：project_blocks + 带块渲染 + 轨迹-块距离（含空块边界）；
  d) 坏路径检测接口返回结构：n_bad/classes/bad_idx/trajectory 键齐全、类别合法，
     且合成 recall 风暴/早停失败样例能触发对应类别；
  e) slugify / npz 保存回读。
用法：python -m pytest tests/test_manifold_preview.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import ThoughtManifoldProjector  # noqa: E402

import manifold_preview as mp  # noqa: E402

D_MODEL = 32       # 小尺度（正式 768，测试不依赖模型）
M_DIM = 16         # 小流形维（正式 64）
T_STEPS = 12
BAD_CLASSES = {"信心膨胀", "漂移", "早停失败", "recall风暴"}


def _proj() -> ThoughtManifoldProjector:
    torch.manual_seed(0)
    return ThoughtManifoldProjector(D_MODEL, M_DIM)


def _synth_hidden(T: int = T_STEPS, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(T, D_MODEL, generator=g)


def test_a_projection_dims() -> None:
    """a) 投影维度：hidden [T,d] → coords64 [T,M] → xyz [T,3]；tick states 字段齐全。"""
    proj = _proj()
    h = _synth_hidden()
    with torch.no_grad():
        coords64 = proj.project(h)
        xyz = proj.project_3d(coords64)
    assert coords64.shape == (T_STEPS, M_DIM), f"coords64 形状错误 {tuple(coords64.shape)}"
    assert xyz.shape == (T_STEPS, 3), f"xyz 形状错误 {tuple(xyz.shape)}"
    ticks = mp.build_tick_states(coords64, [0.6] * T_STEPS, early_stop_last=True)
    assert len(ticks) == T_STEPS
    ts = ticks[3]
    assert ts.current_coord.shape == (1, 1, M_DIM)
    assert ts.disp.shape == (1, 1, M_DIM)
    # 末步 disp 为零向量（无后继）；早停回填正确
    assert float(ticks[-1].disp.norm()) == 0.0
    assert ticks[-1].early_stop is True and ticks[0].early_stop is False
    print(f"[a] 投影维度 OK：[T,{D_MODEL}]→[T,{M_DIM}]→[T,3]，tick states 字段 OK")


def test_b_render_outputs(tmp_path) -> None:
    """b) 渲染出图：3D 与 2D 三视图均产出非空 PNG。"""
    g = np.random.default_rng(2)
    xyz = g.normal(size=(T_STEPS, 3)).cumsum(axis=0)
    p3d = mp.render_trajectory(xyz, tmp_path / "traj3d.png", title="t3d",
                               bad_idx=[2, 5], bad_classes=["漂移"])
    p2d = mp.render_trajectory(xyz, tmp_path / "traj2d.png", title="t2d", views2d=True)
    for p in (p3d, p2d):
        assert p.exists() and p.stat().st_size > 5000, f"{p} 未生成或过小"
    print(f"[b] 渲染 OK：{p3d.name} {p3d.stat().st_size}B / {p2d.name} {p2d.stat().st_size}B")


def test_c_block_overlay(tmp_path) -> None:
    """c) 知识块叠加：project_blocks 维度、带块渲染不崩、轨迹-块距离（含空块）。"""
    proj = _proj()
    reps = torch.randn(4, D_MODEL, generator=torch.Generator().manual_seed(3))
    b64, b3 = mp.project_blocks(proj, reps)
    assert b64.shape == (4, M_DIM) and b3.shape == (4, 3)
    # 渲染叠加路径不崩
    xyz = np.random.default_rng(4).normal(size=(T_STEPS, 3))
    p = mp.render_trajectory(xyz, tmp_path / "overlay.png", blocks3d=b3,
                             block_labels=["b0", "b1", "b2", "b3"], title="overlay")
    assert p.exists() and p.stat().st_size > 5000
    # 轨迹-块距离：形状 [T]、非负；空块 → NaN
    c64 = np.random.default_rng(5).normal(size=(T_STEPS, M_DIM))
    dist = mp.trajectory_block_distances(c64, b64)
    assert dist.shape == (T_STEPS,) and (dist >= 0).all()
    dist_empty = mp.trajectory_block_distances(c64, np.zeros((0, M_DIM)))
    assert np.isnan(dist_empty).all()
    print(f"[c] 知识块叠加 OK：4 块 b64{ b64.shape }/b3{ b3.shape }，距离 min {dist.min():.3f}")


def test_d_bad_path_detection_structure() -> None:
    """d) 坏路径检测接口：返回结构键齐全、类别合法；合成样例触发对应类别。"""
    proj = _proj()
    h = _synth_hidden()
    with torch.no_grad():
        coords64 = proj.project(h)
    # 低 certainty 全程 → recall 风暴（连续≥3）+ 末步非早停 → 早停失败
    ticks = mp.build_tick_states(coords64, [0.1] * T_STEPS, early_stop_last=False)
    det = mp.detect_bad_path(ticks, proj)
    assert {"n_bad", "classes", "bad_idx", "trajectory"} <= set(det), f"键缺失 {set(det)}"
    assert isinstance(det["n_bad"], int) and isinstance(det["bad_idx"], list)
    assert set(det["classes"]) <= BAD_CLASSES, f"非法类别 {det['classes']}"
    assert "recall风暴" in det["classes"], "全程低 certainty 应触发 recall 风暴"
    assert "早停失败" in det["classes"], "低 certainty 跑满应触发早停失败"
    assert det["trajectory"].n_ticks == T_STEPS
    # 高 certainty + eot 早停 → 无 recall/早停失败类（漂移取决于位移，不强行断言）
    ticks_ok = mp.build_tick_states(coords64, [0.9] * T_STEPS, early_stop_last=True)
    det_ok = mp.detect_bad_path(ticks_ok, proj)
    assert "recall风暴" not in det_ok["classes"] and "早停失败" not in det_ok["classes"]
    print(f"[d] 坏路径检测 OK：低 cert 触发 {det['classes']}；高 cert {det_ok['classes'] or '无'}")


def test_e_slugify_and_npz(tmp_path) -> None:
    """e) slugify 文件名安全；save_npz 回读一致。"""
    assert mp.slugify("The derivative of x^2 is 2x!") == "the-derivative-of-x-2-is-2x"
    assert mp.slugify("!!!") == "empty"
    long_slug = mp.slugify("a " * 100 + "tail")
    assert len(long_slug) <= 40
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    p = mp.save_npz(tmp_path / "t.npz", coords=arr, label=np.array(["x"]))
    z = np.load(p)
    assert np.array_equal(z["coords"], arr) and z["label"][0] == "x"
    print(f"[e] slugify/npz OK：{mp.slugify('The derivative of x^2 is 2x!')}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
