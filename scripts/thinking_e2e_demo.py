"""第二阶段端到端集成 demo：把迭代①–⑦ pilot 模块串成完整推理循环。

验证目标：模块协同工作（非各自孤立）——共享 projector 坐标一致性、
ReasoningLoop 多 tick 轨迹 → CoT 投影 → 忠实性审计 → 可视化。

运行：.venv/Scripts/python.exe scripts/thinking_e2e_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

# 把 src 加入 import 路径（脚本直接运行用）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.cot_projection import CotProjectionLayer
from tais_obsidian.model.manifold import ThoughtManifoldProjector
from tais_obsidian.model.reasoning_loop import ReasoningLoop
from tais_obsidian.model.thought_core import ThoughtCore
from tais_obsidian.model.thought_visualizer import ThoughtVisualizer, render_ascii

# ---------------------------------------------------------------------------
# 维度/设备常量（pilot 用 core_dim 作 state 维度，不接 model.py 主干）
# ---------------------------------------------------------------------------
CORE_DIM = 384          # ThoughtCore 维度（GDN 持续状态的 pilot 替身）
N_GROUPS = 8
HISTORY = 4
MAX_TICKS = 8
MANIFOLD_DIM = 64       # 思考流形维度（几十维有效维，避免信息瓶颈）
B, T = 2, 10            # batch / seq 维度
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_demo(seed: int = 42) -> dict:
    """构建共享部件并跑通完整推理循环，返回结果 dict（供 demo 打印与测试断言）。

    集成对齐点：
      1. 共享 projector：同一 ThoughtManifoldProjector 实例注入 ThoughtCore
         （其内部 bridge）→ ReasoningLoop（复用 core.bridge）→ Visualizer。
      2. 维度一致：state 用 core_dim=384（GDN 持续状态 pilot 替身，不接主干）。
      3. 监测/执行分置：sense 只读、bridge.tick 写 PM-stream detach、可视化只读。
    """
    torch.manual_seed(seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)

    # ------------------------------------------------------------------
    # ① 构建共享 projector（同一实例服务全链路，保证坐标系一致）
    # ------------------------------------------------------------------
    shared_projector = ThoughtManifoldProjector(
        d_model=CORE_DIM, manifold_dim=MANIFOLD_DIM
    ).to(DEVICE)

    # ② ThoughtCore：内部自建 bridge，projector=shared → core.bridge.projector is shared
    thought_core = ThoughtCore(
        core_dim=CORE_DIM,
        n_groups=N_GROUPS,
        history=HISTORY,
        max_ticks=MAX_TICKS,
        manifold_dim=MANIFOLD_DIM,
        projector=shared_projector,
        use_sync=True,
    ).to(DEVICE)

    # ③ ReasoningLoop：复用 thought_core.bridge（同一 bridge 实例，
    #    故 loop.bridge.projector is shared_projector；不另造 bridge 避免坐标分叉）
    reasoning_loop = ReasoningLoop(
        thought_core=thought_core,
        bridge=thought_core.bridge,
        kernel=None,  # pilot 用 mock certainty/glimpse（接口位，正式接 KAL/HRL）
    ).to(DEVICE)

    # ④ CoT 投影层：manifold_dim→core_dim 反投影解码（不用 projector，
    #    但 Visualizer 的 project_3d 与 shared_projector 同一坐标系）
    cot_layer = CotProjectionLayer(
        manifold_dim=MANIFOLD_DIM, d_model=CORE_DIM, use_mlp=True
    ).to(DEVICE)

    # ⑤ 可视化器：纯只读，build 时用 shared_projector.project_3d
    visualizer = ThoughtVisualizer()

    # ------------------------------------------------------------------
    # 共享 projector 一致性断言（集成红线：坐标系必须统一）
    # ------------------------------------------------------------------
    assert thought_core.bridge.projector is shared_projector
    assert reasoning_loop.bridge.projector is shared_projector
    assert reasoning_loop.thought_core.bridge.projector is shared_projector

    # ------------------------------------------------------------------
    # ⑥ 合成思考任务：初始 state + target_coord 目标流形坐标
    # ------------------------------------------------------------------
    initial_state = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    target_coord = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)

    # ------------------------------------------------------------------
    # ⑦ 跑完整推理循环（多 tick）
    # ------------------------------------------------------------------
    # pilot 阶段 mock certainty：让循环跑满 max_ticks（低 certainty 不早停）
    reasoning_loop.kal_certainty = lambda s: 0.2  # 低 certainty → 全 tick recall
    final_state, trajectory, stop_tick = reasoning_loop.run(
        initial_state,
        target_coord=target_coord,
        max_ticks=MAX_TICKS,
        stop_threshold=0.9,
        recall_threshold=0.3,
        bridge_alpha=0.1,
    )

    # ------------------------------------------------------------------
    # ⑧ CoT 投影：轨迹 → 思考段序列（含 <|recall|> 标记）
    # ------------------------------------------------------------------
    decoded_segs, tokens = cot_layer.decode_trajectory(trajectory)

    # ------------------------------------------------------------------
    # ⑨ 忠实性审计：须先 fit Hidden→manifold 伪逆（说-做一致性在流形空间度量）
    # ------------------------------------------------------------------
    # 用真实 decoder 生成的 (段, 坐标) 对拟合伪逆，保证审计几何对齐
    with torch.no_grad():
        fit_coords = torch.randn(64, MANIFOLD_DIM, device=DEVICE, generator=g)
        fit_segs = cot_layer.decoder(fit_coords)
    cot_layer.auditor.fit_back_projection(fit_segs, fit_coords)

    # 合成 output/context 表征供 CMI 近似（pilot 占位，正式应接真实推理输出）
    output_repr = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    context_repr = torch.randn(B, T, CORE_DIM, device=DEVICE, generator=g)
    faithfulness = cot_layer.audit(
        trajectory,
        decoded_segs,
        output_repr=output_repr,
        context_repr=context_repr,
    )

    # ------------------------------------------------------------------
    # ⑩ 可视化：构建轨迹 → render_ascii + to_json
    # ------------------------------------------------------------------
    traj_vis = visualizer.build(
        trajectory, shared_projector, faithfulness_diag=faithfulness
    )
    ascii_art = render_ascii(traj_vis)

    # JSON 导出到 runs/thinking_e2e/（gitignore 目录，不入库）
    out_dir = Path(__file__).resolve().parents[1] / "runs" / "thinking_e2e"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "trajectory.json"
    traj_vis.to_json(json_path)

    return {
        "shared_projector": shared_projector,
        "thought_core": thought_core,
        "reasoning_loop": reasoning_loop,
        "cot_layer": cot_layer,
        "visualizer": visualizer,
        "initial_state": initial_state,
        "target_coord": target_coord,
        "final_state": final_state,
        "trajectory": trajectory,
        "stop_tick": stop_tick,
        "decoded_segs": decoded_segs,
        "tokens": tokens,
        "faithfulness": faithfulness,
        "traj_vis": traj_vis,
        "ascii_art": ascii_art,
        "json_path": str(json_path),
    }


def main() -> None:
    print("=" * 70)
    print("第二阶段端到端集成 demo：思考流形 → 推理循环 → CoT 投影 → 可视化")
    print("=" * 70)
    print(f"设备: {DEVICE} | core_dim={CORE_DIM} manifold_dim={MANIFOLD_DIM} "
          f"max_ticks={MAX_TICKS} B={B} T={T}")

    result = build_demo(seed=42)

    # ------------------------------------------------------------------
    # 共享 projector 一致性验证
    # ------------------------------------------------------------------
    sp = result["shared_projector"]
    tc = result["thought_core"]
    rl = result["reasoning_loop"]
    print("\n[① 共享 projector 一致性]")
    print(f"  thought_core.bridge.projector is shared_projector: "
          f"{tc.bridge.projector is sp}")
    print(f"  reasoning_loop.bridge.projector is shared_projector: "
          f"{rl.bridge.projector is sp}")
    print(f"  reasoning_loop.thought_core.bridge.projector is shared_projector: "
          f"{rl.thought_core.bridge.projector is sp}")

    # ------------------------------------------------------------------
    # 轨迹统计
    # ------------------------------------------------------------------
    traj_vis = result["traj_vis"]
    trajectory = result["trajectory"]
    print("\n[② 轨迹统计]")
    print(f"  tick 数: {traj_vis.n_ticks}")
    print(f"  stop_tick: {result['stop_tick']}")
    print(f"  早停 tick: {traj_vis.stop_tick}")
    print(f"  平均 certainty: {traj_vis.avg_certainty:.4f}")
    print(f"  recall 触发: {traj_vis.recall_triggered_any}")
    print(f"  坏路径点数: {traj_vis.n_bad_points}")

    # ------------------------------------------------------------------
    # CoT 投影 tokens
    # ------------------------------------------------------------------
    tokens = result["tokens"]
    print("\n[③ CoT 投影 tokens]")
    print(f"  {tokens}")

    # ------------------------------------------------------------------
    # 忠实性审计
    # ------------------------------------------------------------------
    faith = result["faithfulness"]
    print("\n[④ 忠实性审计]")
    print(f"  speak_do_consistency: {faith['speak_do_consistency']:.4f}")
    print(f"  cmi_approx:           {faith['cmi_approx']:.4f}")
    print(f"  divergence_penalty:   {faith['divergence_penalty']:.4f}")
    print(f"  faithfulness_rate:    {faith['faithfulness_rate']:.4f}")

    # ------------------------------------------------------------------
    # 坐标一致性验证（Visualizer xyz 与 shared_projector.project_3d 一致）
    # ------------------------------------------------------------------
    print("\n[⑤ 坐标一致性验证]")
    ts0 = trajectory[0]
    coord0 = ts0.current_coord.float().mean(dim=(0, 1))  # [manifold_dim]
    xyz_manual = sp.project_3d(coord0).tolist()
    xyz_vis = traj_vis.points[0].xyz
    match = all(abs(a - b) < 1e-5 for a, b in zip(xyz_manual, xyz_vis))
    print(f"  tick0 project_3d 手动: {[round(v, 6) for v in xyz_manual]}")
    print(f"  tick0 Visualizer xyz:  {[round(v, 6) for v in xyz_vis]}")
    print(f"  一致: {match}")

    # ------------------------------------------------------------------
    # ASCII 轨迹图
    # ------------------------------------------------------------------
    print("\n[⑥ ASCII 轨迹图]")
    print(result["ascii_art"])

    # ------------------------------------------------------------------
    # JSON 导出确认
    # ------------------------------------------------------------------
    print(f"\n[⑦ JSON 导出] {result['json_path']}")
    loaded = json.loads(Path(result["json_path"]).read_text(encoding="utf-8"))
    print(f"  meta: {loaded['meta']}")

    print("\n" + "=" * 70)
    print("端到端集成 demo 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
