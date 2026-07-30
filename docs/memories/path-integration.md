# 路径积分辅助任务 pilot（第二阶段迭代②，2026-07-28 落地）

## 产出
`src/tais_obsidian/model/path_integration.py`（325 行）+ `tests/test_path_integration.py`（176 行，6 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 218 绿）。**pilot 独立模块，不接 model.py 主干**（indexer 辅助任务）。

## 核心组件
- **PathIntegrationData**：2D 随机游走合成轨迹（自监督免费数据）。displacements[.,0]=0，cumsum=positions−起点（path integration 一致性）。
- **PathIntegrationEncoder**：Linear→GRU→Linear→**ReLU 非负约束**（Sorscher 2023 充分条件关键，不可省）。GRU 对齐 Banino 2018/Sorscher RNN 路径积分设定；隐藏态=indexer 内部表征 pilot 替身（真实接入换 LightningIndexer 低维表征）。
- **PathIntegrationHead**：刻意线性 Linear(repr→2)（探针式读出，防位置学进头里）。
- **path_integration_loss**：MSE + 诊断 rel_error（尺度不变相对误差）。
- **GridCodeProbe**（T1 新探针）：gridness score = min(corr60,corr120) − max(corr30,corr90,corr150)。2D 直方图率图+**fft 自相关**+中心裁剪+最近邻旋转。>0.3=网格码成立。pilot 级近似（非神经科学全精度，仅趋势观测）。
- **PathIntegrationTask**：encoder+head+loss+probe。

## 验证判据（T1）
indexer 表征是否出现周期性空间响应（grid score>0.3）。出现=空间导航 substrate 成立；不出现=辅助损失权重不足。

## 关键结果
- **探针判别力验证通过**：人工六边形模式 grid score **0.867>0.3**，随机噪声 −0.008、条纹 −0.057 均低分。
- **[降预期] 诚实记录**：端到端训练 120 步 grid score 未见上升（−0.032→−0.036）——符合文献（Sorscher 2022 仅 ~10% RNN 涌现且需更长训练+特定约束）。pilot 判据是"任务学会+探针可判别"（达成），涌现与否留 T1 真实 indexer 长训观测。

## 红线
**MoE-RL**：辅助损失梯度只进 encoder/head（调用侧 detach 隔离主干，同 tais_kernel detach_input 纪律）。

## 子代理踩坑（已修复）
- 初版自相关用 conv2d 与 fft 基准仅 0.53 相关（边界错位）→ fft 自相关修复（hex 0.867）。
- 16×16 粗采样率图 60° 相关被采样伪影破坏 → 24×24 采样。

## 待接
真实 LightningIndexer 长训（10k GDN-2 checkpoint 的 indexer）+ 路径积分辅助损失 → T1 观测 grid score 是否涌现。

---
*导出自 /memories/repo/path-integration.md（2026-07-30 同步快照）。*
