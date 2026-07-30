# CoT 投影层 pilot（第二阶段迭代⑤，2026-07-28 落地）

## 产出
`src/tais_obsidian/model/cot_projection.py`（371 行）+ `tests/test_cot_projection.py`（240 行，7 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 225 绿）。pilot 独立模块。

## 核心理念
**CoT 是投影层非计算层**：计算在思考流形（潜在思考），每 tick 强制解码显式思考段作 grounded 监督。文本/段表征是思考的压缩投影与审计接口，**不是思考本身**；pilot 运行时只读投影，不回流改变思考动力学。

## 核心组件
- **ThoughtSegmentDecoder**：流形坐标 [B,T,64]→Linear→MLP→思考段表征 [B,T,d_model]。读侧投影（区别于 ManifoldToHidden 写侧 steering）。文本解码留接口。
- **grounded_supervision_loss**：MSE 锚定真实推理 hidden（接住 P2 防漂移），可反传 decoder。
- **CotFaithfulnessAudit**：
  - **说-做一致性**：解码段经最小二乘岭回归闭式解伪逆映回流形空间（审计器固定非可学习防混淆）与真实 disp 求余弦 ∈[-1,1]。
  - **CMI 近似**：corr(段,输出)−corr(上下文,输出)。正值=健康，≈0/负=信心膨胀信号（Coda-Forno）。**标注非精确互信息仅趋势观测**。
  - **说-做分歧惩罚** + faithfulness_rate ∈[0,1]（迭代⑤验证判据）。
- **CotProjectionLayer**：decode_tick/decode_trajectory/audit。

## 接住潜在推理三批评
①信心膨胀（Coda-Forno）→CMI 审计+说-做分歧惩罚；②P2 防漂移→每 tick grounded 监督；③探索抑制（Zou）→仅记录 early_stop（迭代④负责）。

## 红线
`<|recall|>` 显形化原样保留（recall tick 解码出 `<|recall|>`）；audit 全程 no_grad 只读（监测/执行分置）。

## 子代理踩坑（已修复）
test_f 初版不一致数据把 disp 与思考状态同步取反（说做仍同向，faithfulness 仍 1.0）→ 改为"说"恒 +v 仅"做"变号，真正说做脱钩，一致 1.000>不一致 0.000。

## 待接
语言解码成 CoT 文本 + grounded 监督离线训练 decoder + 与推理循环 tick 集成。

---
*导出自 /memories/repo/cot-projection.md（2026-07-30 同步快照）。*
