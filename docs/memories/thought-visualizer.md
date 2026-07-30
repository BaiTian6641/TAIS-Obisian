# 可解释性前端 pilot（第二阶段迭代⑦，2026-07-28 落地）

## 产出
`src/tais_obsidian/model/thought_visualizer.py`（403 行）+ `tests/test_thought_visualizer.py`（298 行，12 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 237 绿）。pilot 纯只读模块。

## 核心组件
- **ThoughtVisualizer.build**：ReasoningTickState 轨迹 + projector.project_3d → ThoughtTrajectory。current_coord[B,T,64] 均值池化成[64]再 project_3d（每 tick 一个 3D 代表点）+ certainty/recall/early_stop。
- **detect_bad_path**（坏路径四类检测，可组合 `;` 分隔，阈值模块常量可配标注"pilot 经验值待 T1 标定"）：
  ①信心膨胀（certainty>0.7 且 speak_do_consistency<0.3，Coda-Forno 信号，复用迭代⑤审计）；
  ②漂移（流形位移范数超阈值 3.0，简化固定阈值——逐 tick 调用时轨迹均值不可得，正式应按轨迹平均位移自适应）；
  ③早停失败（certainty<0.5 跑满 max_ticks）；
  ④recall 风暴（连续≥3 tick recall）。
- **ThoughtTrajectory**：list[ThoughtTrajectoryPoint]+元数据（n_ticks/stop_tick/recall_any/avg_certainty/n_bad）。to_dict/to_json（UTF-8 中文不转义，外部渲染消费）+ summary。
- **render_ascii**：xy 投影到 60×20 字符画，y 翻转，符号优先级 X(坏)>R(recall)>tick 序号(0-9a-z 循环)。

## 红线
3D 仅人类视图不参与计算（§1.1 维度修正）；纯只读 no_grad（监测/执行分置）；不依赖 matplotlib/GUI（数据→JSON+ASCII，渲染外部做）。

## 待接
外部 3D 实时渲染前端（消费 JSON）；漂移阈值按轨迹平均位移自适应；逐 tick 精确一致性（现取轨迹均值）。

## 第二阶段 7 迭代全部落地
①流形层 ②路径积分 ③思考核 ④推理循环 ⑤CoT投影 ⑦可视化 ✅（⑥合成导航课程=T2–T3 训练阶段任务，非 pilot 模块）。前置①②③ ✅。测试 237 全绿。

---
*导出自 /memories/repo/thought-visualizer.md（2026-07-30 同步快照）。*
