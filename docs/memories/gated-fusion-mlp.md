# HCA 门控扩容（GatedFusionMLP）——突破 585 线性门控瓶颈（2026-07-29）

## 产出
- `src/tais_obsidian/model/tri_attention_gated.py`（232 行）：GatedFusionMLP + attach/detach 注入式挂载（不改原 tri_attention.py）。
- `scripts/train_recall_gated.py`（286 行）：扩容门控重训召回头（复用 train_retrieval_recall 逻辑）。
- `tests/test_gated_fusion.py`（207 行，6 项全绿）；report `runs/recall_gated/report.json`。**已亲自验收**：report 数据确认（召回 0.062→0.625，585 基线 0.188，主干 drift=0.0）+独立重跑+全量 338 绿。

## 验收确认的关键突破
召回 0.188→**0.625**（逼近 in-context 上界 0.70，差距从 0.51 收窄到 0.075）——585 线性门控容量瓶颈确证突破。**恒等初始化设计正确**（fc1 小随机破对称+fc2=0 保 g=1/3）：attach 后整层前向与原线性门控逐位一致（atol=1e-5），不破坏既有 checkpoint 行为；这是"可选升级、默认行为不变"红线的正确实现。

## 结果（真实跑出，PRO 4000，CUDA_VISIBLE_DEVICES=1，500 步 lr=5e-3 hidden=128）
| 指标 | 585 线性门控 | 扩容 GatedFusionMLP | 参考 |
|---|---|---|---|
| KV 注入答对率 | 0.188 | **0.625**（init 0.062）| in-context 上界 0.70 |
| 门控参数 | 585 | **26121**（3 A 层 × 8707）| head_dim64→hidden128→3 |
| 主干污染 | — | **逐位不变 drift=0.0** | frozen 红线✅ |
| 训练稳定性 | 收敛 | CE 0.32→0.12 稳定 | 全量 338 项 pytest 绿 |

0.625 远超 585 基线 0.188，逼近上界 0.70（差距收窄到 0.075）。

## 方案
原门控 `g=sigmoid(q@gate_w.T+gate_b)`（线性 195/层）→ MLP `g=sigmoid(fc2(GELU(fc1(q))))`（Linear64→128+GELU+Linear128→3，8707/层）。注入式：attach_gated_fusion 预绑定 mixer.forward（types.MethodType）替换门控行，三分支融合语义不变；detach 恢复。原 gate_w/b 保留 state_dict（向后兼容，旧 checkpoint strict=False 加载仅缺 gate_mlp.* 键）。

## 关键坑（重要经验）
**恒等初始化必须 fc1 随机破对称、fc2=0+bias=-ln2**：初版 fc1=fc2=0 恒等初始化在召回训练下 lr 5e-3/5e-4 均 CE 爆炸（step1 0.32→2.9 发散、答对率卡 0.062）——fc1 全零致隐藏单元梯度相同退化为线性门控、fc2 单点强梯度不稳。**fc1 小随机 std=0.02 破对称 + fc2=0** 保初始 g 精确 1/3（fc2=0 屏蔽 fc1 随机性），训练转稳、MLP 表达力真正可用。教训：恒等初始化≠全零，需破对称让 MLP 非平凡。

## 红线合规
不改原 tri_attention.py；默认行为不变（恒等初始化 g=1/3，attach 后整层前向与原逐位一致 atol=1e-5）；主干 frozen 逐位不变；旧 checkpoint 兼容。双卡：训练 PRO 4000 / 测试 4070。

## 待接
①0.625→0.70 余量（更多步/更大 hidden/更多事实）；②与已训 indexer 联合闭环（检索∧召回）；③内化-检索-注入-睡眠固化全链端到端。

---
*导出自 /memories/repo/gated-fusion-mlp.md（2026-07-30 同步快照）。*
