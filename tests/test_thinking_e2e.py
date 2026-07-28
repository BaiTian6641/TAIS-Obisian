"""第二阶段端到端集成测试：验证迭代①–⑦ pilot 模块协同工作（非各自孤立）。

覆盖判据（对齐任务规范 §实现要求）：
  a) 共享 projector 一致性：ThoughtCore.bridge.projector、ReasoningLoop.bridge.projector、
     Visualizer 用的 projector 是**同一实例**（坐标系统一）；
  b) 端到端跑通：合成任务 run → 轨迹非空 → decode → audit → build → render，全链路无异常；
  c) 坐标一致性：轨迹中某 tick 的 current_coord 经共享 projector.project_3d 与
     Visualizer 该点 xyz 一致（证明同一坐标系）；
  d) recall 贯通：构造低 certainty 场景，轨迹有 recall_triggered，decode 出 <|recall|>，
     Visualizer 标坏路径（recall 风暴若连续）；
  e) 坏路径监测贯通：信心膨胀场景（高 certainty 低 consistency）Visualizer 标记；
  f) demo 脚本可运行：AST 通过 + 函数级导入测试（不实际跑长循环，测 build_demo 返回结构）。
用法：.venv/Scripts/python.exe -m pytest tests/test_thinking_e2e.py -q
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.cot_projection import CotProjectionLayer
from tais_obsidian.model.manifold import ThoughtManifoldProjector
from tais_obsidian.model.reasoning_loop import RECALL_TOKEN, ReasoningLoop
from tais_obsidian.model.thought_core import ThoughtCore
from tais_obsidian.model.thought_visualizer import ThoughtVisualizer, render_ascii

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CORE_DIM = 384
N_GROUPS = 8
HISTORY = 4
MAX_TICKS = 8
MANIFOLD_DIM = 64
B, T = 2, 10


def build_shared(seed: int = 42):
    """构建共享 projector 的集成部件（与 demo 同一集成路径）。"""
    torch.manual_seed(seed)
    shared_projector = ThoughtManifoldProjector(
        d_model=CORE_DIM, manifold_dim=MANIFOLD_DIM
    ).to(DEVICE)
    thought_core = ThoughtCore(
        core_dim=CORE_DIM,
        n_groups=N_GROUPS,
        history=HISTORY,
        max_ticks=MAX_TICKS,
        manifold_dim=MANIFOLD_DIM,
        projector=shared_projector,
        use_sync=True,
    ).to(DEVICE)
    reasoning_loop = ReasoningLoop(
        thought_core=thought_core,
        bridge=thought_core.bridge,
        kernel=None,
    ).to(DEVICE)
    cot_layer = CotProjectionLayer(
        manifold_dim=MANIFOLD_DIM, d_model=CORE_DIM, use_mlp=True
    ).to(DEVICE)
    visualizer = ThoughtVisualizer()
    return shared_projector, thought_core, reasoning_loop, cot_layer, visualizer


def run_full_loop(seed: int = 42, mock_certainty: float = 0.2):
    """跑完整推理循环，返回集成结果元组（供各测试复用）。"""
    sp, tc, rl, cl, viz = build_shared(seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    initial_state = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    target_coord = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)

    rl.kal_certainty = lambda s: mock_certainty
    final_state, trajectory, stop_tick = rl.run(
        initial_state,
        target_coord=target_coord,
        max_ticks=MAX_TICKS,
        stop_threshold=0.9,
        recall_threshold=0.3,
        bridge_alpha=0.1,
    )
    decoded_segs, tokens = cl.decode_trajectory(trajectory)

    # 拟合审计伪逆（说-做一致性在流形空间度量）
    with torch.no_grad():
        fit_coords = torch.randn(64, MANIFOLD_DIM, device=DEVICE, generator=g)
        fit_segs = cl.decoder(fit_coords)
    cl.auditor.fit_back_projection(fit_segs, fit_coords)

    output_repr = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    context_repr = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    faithfulness = cl.audit(
        trajectory, decoded_segs, output_repr=output_repr, context_repr=context_repr
    )
    traj_vis = viz.build(trajectory, sp, faithfulness_diag=faithfulness)
    ascii_art = render_ascii(traj_vis)

    return {
        "shared_projector": sp,
        "thought_core": tc,
        "reasoning_loop": rl,
        "cot_layer": cl,
        "visualizer": viz,
        "trajectory": trajectory,
        "stop_tick": stop_tick,
        "decoded_segs": decoded_segs,
        "tokens": tokens,
        "faithfulness": faithfulness,
        "traj_vis": traj_vis,
        "ascii_art": ascii_art,
    }


# ---------- a) 共享 projector 一致性 ----------


def test_a_shared_projector_identity() -> None:
    """a) 共享 projector 一致性：三处 projector 是同一实例（坐标系统一）。"""
    sp, tc, rl, cl, viz = build_shared()
    assert tc.bridge.projector is sp, "ThoughtCore.bridge.projector 须为共享实例"
    assert rl.bridge.projector is sp, "ReasoningLoop.bridge.projector 须为共享实例"
    assert rl.thought_core.bridge.projector is sp, (
        "ReasoningLoop.thought_core.bridge.projector 须为共享实例"
    )
    # ReasoningLoop.bridge 与 ThoughtCore.bridge 是同一实例（复用不另造）
    assert rl.bridge is tc.bridge, "ReasoningLoop.bridge 须复用 ThoughtCore.bridge"
    print("[a] 共享 projector 一致性：三处同一实例 OK")


# ---------- b) 端到端跑通 ----------


def test_b_end_to_end_pipeline() -> None:
    """b) 端到端跑通：run → decode → audit → build → render 全链路无异常。"""
    r = run_full_loop(seed=1, mock_certainty=0.5)  # 中等 certainty 不触发 recall
    assert len(r["trajectory"]) > 0, "轨迹非空"
    assert len(r["decoded_segs"]) == len(r["trajectory"]), "段序列与轨迹等长"
    assert len(r["tokens"]) == len(r["trajectory"]), "tokens 与轨迹等长"
    for key in ("speak_do_consistency", "cmi_approx", "divergence_penalty", "faithfulness_rate"):
        assert key in r["faithfulness"], f"忠实性诊断缺键 {key}"
    assert r["traj_vis"].n_ticks == len(r["trajectory"]), "可视化轨迹点数与轨迹等长"
    assert isinstance(r["ascii_art"], str) and len(r["ascii_art"]) > 0, "ASCII 渲染非空"
    print(
        f"[b] 端到端跑通：{r['traj_vis'].n_ticks} ticks，"
        f"faith={r['faithfulness']['faithfulness_rate']:.3f} OK"
    )


# ---------- c) 坐标一致性 ----------


def test_c_coordinate_consistency() -> None:
    """c) 坐标一致性：tick current_coord 经 project_3d 与 Visualizer xyz 一致。"""
    r = run_full_loop(seed=2, mock_certainty=0.5)
    sp = r["shared_projector"]
    traj_vis = r["traj_vis"]
    trajectory = r["trajectory"]
    # 验证前 3 个 tick（或全部若 < 3）
    for i in range(min(3, len(trajectory))):
        ts = trajectory[i]
        coord = ts.current_coord.float().mean(dim=(0, 1))  # [manifold_dim]
        xyz_manual = sp.project_3d(coord).tolist()
        xyz_vis = traj_vis.points[i].xyz
        assert all(abs(a - b) < 1e-5 for a, b in zip(xyz_manual, xyz_vis)), (
            f"tick{i} 坐标不一致：manual={xyz_manual} vs vis={xyz_vis}"
        )
    print(f"[c] 坐标一致性：前 {min(3, len(trajectory))} ticks project_3d == Visualizer xyz OK")


# ---------- d) recall 贯通 ----------


def test_d_recall_pipeline() -> None:
    """d) recall 贯通：低 certainty → recall_triggered → decode <|recall|> → Visualizer 标坏路径。"""
    r = run_full_loop(seed=3, mock_certainty=0.2)  # 低 certainty < recall_threshold=0.3
    trajectory = r["trajectory"]
    tokens = r["tokens"]
    traj_vis = r["traj_vis"]

    # 轨迹有 recall_triggered
    recall_flags = [ts.recall_triggered for ts in trajectory]
    assert any(recall_flags), "低 certainty 应触发 recall"

    # decode 出 <|recall|>
    assert RECALL_TOKEN in tokens, f"decode 应产出 <|recall|>，tokens={tokens}"
    recall_idx = [i for i, t in enumerate(tokens) if t == RECALL_TOKEN]
    assert len(recall_idx) == sum(recall_flags), "recall token 数与 recall_triggered 数一致"

    # Visualizer 标坏路径（连续 recall ≥ 3 → recall 风暴）
    assert traj_vis.recall_triggered_any is True
    bad_reasons = [p.bad_reason for p in traj_vis.points if p.is_bad_path]
    # 连续 recall 应触发 recall 风暴（MAX_TICKS=8 全 recall，第 3 个起标风暴）
    assert any("recall风暴" in reason for reason in bad_reasons), (
        f"连续 recall 应标 recall 风暴，bad_reasons={bad_reasons}"
    )
    print(
        f"[d] recall 贯通：{sum(recall_flags)}/{len(trajectory)} ticks recall，"
        f"tokens 含 {len(recall_idx)} 个 <|recall|>，风暴标记 OK"
    )


# ---------- e) 坏路径监测贯通（信心膨胀） ----------


def test_e_confidence_inflation_flagged() -> None:
    """e) 信心膨胀场景（高 certainty 低 consistency）Visualizer 标记。"""
    sp, tc, rl, cl, viz = build_shared(seed=4)
    g = torch.Generator(device=DEVICE).manual_seed(4)
    initial_state = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    target_coord = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)

    # 高 certainty（> 0.7 信心膨胀阈值）但跑满（< 0.9 不早停）
    rl.kal_certainty = lambda s: 0.8
    _, trajectory, _ = rl.run(
        initial_state, target_coord=target_coord, max_ticks=MAX_TICKS,
        stop_threshold=0.9, recall_threshold=0.3,
    )
    decoded_segs, _ = cl.decode_trajectory(trajectory)
    with torch.no_grad():
        fit_coords = torch.randn(64, MANIFOLD_DIM, device=DEVICE, generator=g)
        fit_segs = cl.decoder(fit_coords)
    cl.auditor.fit_back_projection(fit_segs, fit_coords)
    faithfulness = cl.audit(trajectory, decoded_segs)

    # 注入低 speak_do_consistency 触发信心膨胀（pilot 随机 decoder 一致性本低）
    low_faith = {"speak_do_consistency": 0.1}
    traj_vis = viz.build(trajectory, sp, faithfulness_diag=low_faith)
    bad_reasons = [p.bad_reason for p in traj_vis.points if p.is_bad_path]
    assert any("信心膨胀" in reason for reason in bad_reasons), (
        f"高 certainty + 低 consistency 应标信心膨胀，bad_reasons={bad_reasons}"
    )
    print(
        f"[e] 信心膨胀标记：certainty=0.8 + speak_do=0.1 → "
        f"{sum(1 for p in traj_vis.points if p.is_bad_path)} 个坏点 OK"
    )


# ---------- f) demo 脚本可运行 ----------


def test_f_demo_script_ast_and_import() -> None:
    """f) demo 脚本 AST 通过 + build_demo 函数级导入返回结构完整。"""
    demo_path = Path(__file__).resolve().parents[1] / "scripts" / "thinking_e2e_demo.py"
    assert demo_path.exists(), f"demo 脚本不存在: {demo_path}"
    # AST 语法检查
    source = demo_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert isinstance(tree, ast.Module), "AST 解析须得 Module"
    # 含 build_demo 与 main 函数
    func_names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "build_demo" in func_names and "main" in func_names, (
        f"demo 脚本须含 build_demo/main，实得 {func_names}"
    )

    # 函数级导入测试：导入 build_demo 并验证返回结构（不跑 main 打印）
    scripts_dir = str(demo_path.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import thinking_e2e_demo

    result = thinking_e2e_demo.build_demo(seed=99)
    # 返回结构关键键齐全
    for key in (
        "shared_projector", "thought_core", "reasoning_loop", "cot_layer",
        "visualizer", "trajectory", "decoded_segs", "tokens", "faithfulness",
        "traj_vis", "ascii_art", "json_path",
    ):
        assert key in result, f"build_demo 返回缺键 {key}"
    # 共享 projector 一致性（demo 路径同样满足）
    assert result["thought_core"].bridge.projector is result["shared_projector"]
    assert result["reasoning_loop"].bridge.projector is result["shared_projector"]
    # 轨迹非空 + JSON 已写出
    assert result["traj_vis"].n_ticks > 0
    assert Path(result["json_path"]).exists(), "demo 应导出 trajectory.json"
    # JSON 可加载且结构完整
    loaded = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    assert "meta" in loaded and "points" in loaded
    assert loaded["meta"]["n_ticks"] == result["traj_vis"].n_ticks
    print(
        f"[f] demo 脚本 AST 通过 + build_demo 返回 {result['traj_vis'].n_ticks} ticks，"
        f"JSON 已导出 OK"
    )
