# 思考流形层 pilot（第二阶段迭代①，2026-07-27 落地）

## 产出
`src/tais_obsidian/model/manifold.py`（220 行）+ `tests/test_manifold.py`（168 行，6 项全绿）+ config 追加 `manifold_dim=64` + model/__init__ 导出。子代理实现，我亲自验收（读码+独立重跑+全量回归 183 全绿）。

## 核心组件
- **ThoughtManifoldProjector**：d→manifold_dim=64 共享投影（Linear+LayerNorm 无仿射，保位移比例语义）。**同一实例服务三类输入**（知识块 route_key / PM-stream 思考段 / W0 轨迹段）=共享坐标核心。view3d 固定随机投影、requires_grad=False（仅人类可视化视图，不进训练目标）。
- **conformal_isometry_loss**：尺度不变共形等距——每轨迹内归一化位移向量∝归一化步长向量（min ||disp/||disp|| − steps/||steps||||²），只约束比例分配不约束绝对尺度（防全局尺度坍缩）。诊断 Pearson 相关=§6 迭代①验证判据。
- **decorrelation_loss**：VICReg 谱系（arXiv:2105.04906）双项——相关矩阵非对角平方均值 + 逐维方差铰链 relu(1−std)²。**方差铰链不可省**：纯非对角惩罚对全坍缩点恒为 0（协方差全零无相关可罚），子代理初版踩坑后补上。
- **ThoughtManifold.loss**：w_conformal(1.0)·共形 + w_decor(0.1)·去相关，w_decor 不可为 0。

## 设计要点（诚实分级）
- 维度修正：manifold_dim=64（几十到一百多维有效维避免瓶颈），**不是 3 维**；3D 仅人类视图。
- [降预期] 网格码在 transformer 不会自发涌现（证据全在 RNN/PCN）→ 显式训练诱导，不指望涌现。
- [推测/独创] 三类对象共享投影+共形等距显式目标，无 LLM 先例（TAIS 独创外推，待 pilot 验证）。

## 待接（下一步）
- W0 日志轨迹数据管线（提取 {思考段表征, 语义步长} 对）。
- 独立训练该投影器（不触碰主干权重），验证"流形位移-语义步长相关性"判据（Pearson 接近 1）。
- 之后接迭代②（路径积分辅助任务，HRL indexer 网格码探针）。

## 桥接模块（2026-07-27 落地，迭代①×③交汇，PM-stream 端到端贯通）
`src/tais_obsidian/model/manifold_bridge.py`（218 行）+ `tests/test_manifold_bridge.py`（6 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 189 绿）。
- **设计交汇**：PM-stream 末位流=思考段载体/寄存器；思考流形=思考段几何坐标系。
- **ThoughtSegmentExtractor**：PM 末位流 [B,T,d]→共享 projector→流形坐标；extract_segments 按思考段边界（起始索引升序，首=0，末段延至 T）段内均值池化。
- **ThoughtDisplacementWriter**：流形位移经 ManifoldToHidden（独立 Linear 反投影，读写解耦非伪逆）回 d_model，steering 有界加法写回 PM-stream。α clamp≤0.2×pm norm（对齐 ITI max_alpha_frac 安全区）；**detach 梯度边界双保险**（steering 是推理期干预 W1–W2 零梯度快写，绝不触碰权重；to_hidden 走离线显式目标）。
- **ThoughtManifoldBridge.tick**：单 tick 闭环（读 current_coord→disp=target−current→反投影 detach→有界写回）。disp_manifold.detach() 经 to_hidden。
- **踩坑**：初版 write 的 scale=pm.norm().mean() 未 detach 致增量回流梯度非恒等 → scale/disp 双 detach 修复。
- **PM-stream 端到端全貌**（已贯通）：多流残差(_forward_pm)+sense 读 GDN PM 流+inject 写 CSA PM 流+capture{content,pm}+增量 cache(流入 cache 无需,test_e<1e-4)+训练读 config+桥接。
- **待接**：思考核（迭代③）用 tick 驱动多步导航；to_hidden 离线训练目标（重建 to_hidden(project(x))≈x 或对比）。

---
*导出自 /memories/repo/thinking-manifold-layer.md（2026-07-30 同步快照）。*
