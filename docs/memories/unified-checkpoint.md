# 统一 checkpoint（pilot_0p1b_gdn2_10k_unified）已训强度合并

2026-07-29 集成子代理任务：把分散已训部件合并为一个统一 checkpoint，演示主动求知闭环全链已训强度。**已亲自验收**：report 数据确认（KAL 0.769 如实保留不臆造 0.8、检索 0.938=余弦基线、召回 0.625 vs kaltruth 对照 0.000、门控副作用明确）+独立重跑+全量 357 绿。

## 验收确认的意义与关键权衡
主动求知闭环从"分散 pilot 验证"到"统一能力展示"——统一 checkpoint 一站集齐 KAL 校准+HRL 检索+HCA 召回+内化 SFT+睡眠固化。**门控副作用是重要诚实发现**：扩容门控对注入条目开权重（召回 0.625 所需）但也对长文本 gist 开权重（干扰纯文本精确召回 0.688→0.250）——可按需 attach/detach（推理用召回则 attach，纯文本则 detach）；正式需门控上下文自适应（注入开/长文本关），这是后续改进点非缺陷。

## 三个纯新增文件
- `scripts/build_unified_checkpoint.py`（~292 行）：合并 + 验证；产出 checkpoints/pilot_0p1b_gdn2_10k_unified + runs/unified_checkpoint/build_report.json。含 `load_unified()` 标准加载（供复用）。
- `scripts/unified_full_chain_demo.py`（~428 行）：全链五项强度 demo；产出 runs/unified_checkpoint/full_chain_report.json。
- `tests/test_unified_checkpoint.py`（~217 行）：13 项集成测试，产物缺失时整模块 skip。

## 合并方案（四步）
① 基座=teaching（主干 GDN-2 10k+内化 SFT），config.kernel_enabled=True 复制；teaching 无 kernel.* 故主干逐键 copy_（非 load_state_dict strict）。
② 从 kaltruth 注入 21 个 kernel.*（kal_l1 校准+kal_l2+dg_proj+side_heads+indexer 占位）。
③ trained_indexer.pt 覆盖 kernel.hrl_indexer 5 键（fp32→bf16）。
④ trained_gate_mlp.pt 对 A 层 [3,7,11] 先 attach_gated_fusion 再 load_state_dict。
state_dict 266 键 = 主干233 + kernel.*21 + gate_mlp.*12。

## 全链已训强度（n=16，实测）
- KAL ℓ10 AUROC 0.769（kaltruth 报告 final=0.75945 如实保留；其 verdict=未达0.8，不臆造0.8）；certainty 方向 known 0.879/fake 0.000。
- HRL 检索 top-1 = 0.938（15/16，train_retrieval_recall 同款均值池化协议，训练 1.000）。
- HCA 注入召回 = 0.625（扩容门控）vs 基线 0.062 vs **kaltruth 对照 0.000**。
- 睡眠固化 PROMOTE 8 / QUARANTINE 1 / REJECT 8。
- 内化 in-context：带门控 0.250 / 拆门控 0.688 ≈ teaching 0.6875。

## 关键坑（防御记录）
1. **config grad_checkpoint**：kaltruth(True) vs teaching(False) 训练 recipe 差异，非结构差异，合并时忽略（推理 forward 不走 checkpoint 分支）。
2. **teaching 无 kernel.***：主干载入须逐键 copy_（strict load_state_dict 会因 kernel 缺失键报错）。
3. **gate_mlp 复挂坑**：from_pretrained 构建的原 TriRetrievalAttention 无 gate_mlp 属性 → 加载统一 ckpt 须先 attach_gated_fusion 再 load_state_dict(strict=True)。`load_unified()` 已实现，demo/test 复用。
4. **评估协议即强度**：检索须用 train_retrieval_recall 的**均值池化** query 协议（末token max 协议会把 1.000 测成 0.438）；in-context 用 internalization_e2e 原版 answer_incontext。
5. **AUROC/softmax 需 no_grad**：kal_l1 head requires_grad 默认 True，eval 前向 logits 带 grad，.numpy() 前须 detach/no_grad。

## 门控副作用（诚实权衡，重要）
扩容门控让 HCA 分支开权重（KV 召回 0.625 所需），但 **in-context 下 HCA 对长文本 gist 也开权重、干扰 win 分支逐 token 精确召回** → 带门控 in-context 从 teaching 0.6875 降到 0.250。拆门控（detach_gated_fusion）即回 0.688。**主干内化 SFT 未退化，是"注入召回"与"纯文本精确召回"的门控权衡**。train_recall_gated 报告 val_loss 漂移 0.037 亦印证。

## 测试
全量 pytest 357 项全绿（含新增 13；原 344）。命令 CUDA_VISIBLE_DEVICES=0。

---
*导出自 /memories/repo/unified-checkpoint.md（2026-07-30 同步快照）。*
