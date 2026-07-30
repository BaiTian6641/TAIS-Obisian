# 第二阶段端到端集成（2026-07-28 落地）

## 产出
`scripts/thinking_e2e_demo.py`（255 行）+ `tests/test_thinking_e2e.py`（286 行，6 项全绿）。集成子代理实现，验证迭代①–⑦ pilot 模块协同工作。全量 243 绿（237+6）。**已亲自验收**：读码（共享 projector 单实例+复用 core.bridge 统一）+ 独立重跑 demo（端到端跑通 ASCII/JSON）+ 全量回归 243 绿。

## 意义
第二阶段从"7 个 pilot 模块落地"到"系统贯通"的关键一步——验证模块协同骨架（glimpse/certainty 仍 pilot mock 接口位，非真实推理能力）。待接：真实部件替换 mock（CSA glimpse/KAL P(IK)/HRL PPR/GDN 状态）→ 0.1B pilot 消融。

## 集成路径（无缝隙，构造签名已支持共享 projector）
- 共享 `ThoughtManifoldProjector(d_model=384, manifold_dim=64)` 单实例
- → `ThoughtCore(projector=shared)`（内部 `ThoughtManifoldBridge` 复用同一 projector）
- → `ReasoningLoop(thought_core=core, bridge=core.bridge)`（**复用 core.bridge 不另造**，故 `loop.bridge.projector is shared`）
- → `CotProjectionLayer(manifold_dim=64, d_model=384)`（不用 projector，走 ManifoldToHidden 反投影）
- → `ThoughtVisualizer.build(trajectory, shared)`（project_3d 与 shared 同坐标系）
- 三处 projector 同一实例断言：`core.bridge.projector is loop.bridge.projector is shared`

## 端到端闭环
run(8 ticks) → decode_trajectory（含 <|recall|>）→ fit_back_projection 后 audit（四键）→ build → render_ascii + to_json(runs/thinking_e2e/trajectory.json)。

## 关键实现点
- **audit 前必须 fit_back_projection**：用 decoder 生成的 (段, 坐标) 对拟合 Hidden→manifold 伪逆（说-做一致性在流形空间度量，cot_projection 设计纪律），否则 `_project_decoded_to_manifold` 抛错。
- mock certainty=0.2 低值 → 全 8 tick recall → 连续 ≥3 触发 recall 风暴坏路径标记（Visualizer ④）。
- 坐标一致性验证：tick.current_coord.mean(dim=(0,1)) 经 shared.project_3d 与 Visualizer 该点 xyz 完全相等（<1e-5）。
- demo 输出 runs/thinking_e2e/（gitignore 不入库）。

## 集成缝隙
**无**。所有模块构造签名已支持共享 projector 注入（ThoughtCore(projector=)、ThoughtManifoldBridge(projector=)、ReasoningLoop(bridge=)、Visualizer.build(projector=)），未需"构造后替换 .projector 引用"的最小侵入方案。

## 红线
监测/执行分置（sense 只读/visualizer 只读）；bridge.tick detach（W1–W2 零梯度快写）；3D 仅人类视图不参与计算。

---
*导出自 /memories/repo/thinking-e2e-integration.md（2026-07-30 同步快照）。*
