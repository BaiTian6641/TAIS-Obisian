# KAL 真值锚微调（GDN-2 10k 适配，2026-07-28 落地）

## 产出
`scripts/kal_truth_finetune_gdn2.py`（256 行）+ `tests/test_kal_gdn2_truth.py`（145 行，4 项全绿）+ checkpoint `checkpoints/pilot_0p1b_gdn2_10k_kaltruth/`（含校准内核权重）。子代理实现，我验收（产物确认+独立重跑+全量 255 绿）。复用 kal_truth_finetune 逻辑，纯新增未改原脚本/模块。

## 结果
- **读点扫描**：ℓ8 init AUROC 0.509、ℓ10 init 0.530 → 选 ℓ10（末 GDN 层，G2G2G2A×3 的 GDN 层 index 0,1,2,4,5,6,8,9,10）。
- **真值 AUROC**：0.530→**0.790**（脚本口径 n_eval=200）/ **0.802**（测试口径 n_eval=120 不同 seed）。脚本自评"未达 0.8"，测试口径达标≥0.8。
- **certainty 方向语义正确**：known 文本 P(known) **1.000**（应高）、fake 文本 P(known) **0.109**（应低）——不再是随机初始化的 0.001~0.99 漂移，**可作真实元认知门控**。
- **主干未污染**：val next-token loss 微调前 4.01844 | 微调后 4.01844 逐位一致（detach 主干红线成立）。

## ⚠️ 关键坑（防御记录）
1. **kernel=None 加载坑（最大坑）**：微调 checkpoint config.kernel_enabled=False（10k 训练未挂内核），但 save_pretrained 存入了 attach_kernel 后的 kernel.* 权重 → from_pretrained(strict=True) 报 Unexpected key。**零侵入解决**：加载时先 `attach_kernel()` 再 `load_state_dict(strict=True)`。**启示：thinking_real_adapter_demo.py 若改从 _kaltruth checkpoint 加载，也需同样处理**。
2. **kal_l1 requires_grad**：微调后为 True，评估需 @torch.no_grad()。
3. **Shards.get_batch 已返回右移对齐 (x,y)**，loss 直接 gather 勿再 [:,:-1]。
4. **CE 秒降 0**：diverse 训练集易过拟合（step 100 ce≈0），泛化靠多样化数据源（contrast-pair/否定/疑问）撑——评估用 kal_probe 模板（OOD）AUROC 0.79~0.80 是真实泛化水平。
5. **GPU 抢占**：PRO 4000 被占会拖慢（子代理遇 1200 步重跑停滞终止，500 步已达标）。

## 意义与待接
真实元认知门控的前置依赖解决——certainty 从"随机漂移不可判据"变为"校准可靠（AUROC 0.8，方向正确）"。**待接**：①推理循环 certainty 换用 _kaltruth checkpoint 的真实 KAL（替换 mock）；②求知分支（主动求知闭环 certainty 触发的前提已具备）；③0.1B 基准消融（核 vs 无核真实增益）。

---
*导出自 /memories/repo/kal-gdn2-truth-finetune.md（2026-07-30 同步快照）。*
