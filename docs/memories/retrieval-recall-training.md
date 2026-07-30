# HRL indexer 块检索 + HCA 召回头训练（兑现实时可用，2026-07-29）

## 产出
`scripts/train_retrieval_recall.py`（484 行）+ `tests/test_retrieval_recall.py`（118 行，4 项全绿）+ report `runs/retrieval_recall/report.json`。子代理实现，我验收（report 数据确认+独立重跑+全量 310 绿）。基座=teaching checkpoint，两缺口训练。

## 结果（真实跑出）
| 指标 | 训练前 | 训练后 | 参考 |
|---|---|---|---|
| **indexer 块检索 top-1 命中率** | 0.062（随机 1/16） | **1.000** | 对齐 embedding 余弦基线 1.000 |
| **KV 注入答对率** | 0.000 | **0.188** | in-context 上界 0.70（未达） |
| **闭环（检索∧召回）** | — | **0.188** | 实时可用兑现 |
| **主干污染** | — | **权重逐位不变（drift=0.0）** | frozen 红线✅ |

## 两缺口训练方法
**① HRL indexer 块检索**：N=16 虚构事实作候选块（harvest 成块表征，首 CSA 层均值隐藏态），query=依赖某事实的问题，正例=对应块。**余弦蒸馏**（MSE 回归 indexer 打分到 query×block 余弦相似度，z-score 标准化），query **均值池化**（实体语义主载体）。只训 indexer（主干 frozen+detach_input 梯度隔离）。
**② HCA 召回头**：事实块 KV 注入各 CSA 层 HCA 区（inject_hca_entries），prefill 问题（不带 K 文本）→prompt 法逐 token 前向（logits 可微）→答案段 CE。**只训各 A 层 TriRetrievalAttention.gate_w/gate_b（585 参数）**——门控是注入块参与注意力的唯一入口，主干全冻。

## 诚实边界
召回 0.188 显著>0 但 << 上界 0.70——**门控仅 585 参数是容量瓶颈**（pilot 级，判"显著改善"非"完美"）。val loss 漂移 0.43 是门控语义变化（训练目标，非污染）。

## 关键意义
**"实时可用"兑现**：知识块写入后，推理时 HRL 检索命中（1.000）→HCA 注入→同一对话立即可用（0.188），无需等睡眠固化、无需重新 SFT。检索与召回协同打通主动求知+知识内化两大闭环。

## 子代理踩坑（已修复，重要经验）
①**LightningIndexer 的 ReLU 门控在 InfoNCE/KL 下坍缩到常数**——必须 MSE 余弦蒸馏（逐元素回归无 ReLU 退化）；②**query 末 token("?")语义弱**致坍缩——改均值池化（实体语义主载体，余弦对角 0.83 vs 非对角 0.57 可分）；③init_indexer_from_model 的 q 方向聚合对块域对比引入偏置→随机初始化更稳；④主干污染检查须排除内核+门控（方案 B 边界），否则误报；⑤召回训练随机采样同一问题反复致 CE 振荡→改循环遍历+恒定 lr。

## 待接
①扩大门控容量（突破 585 参数瓶颈，召回→上界 0.70）；②更多事实/更长沙盒验证；③内化-检索-注入-睡眠固化全链端到端。

---
*导出自 /memories/repo/retrieval-recall-training.md（2026-07-30 同步快照）。*
