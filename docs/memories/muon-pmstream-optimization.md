# Muon 优化器 + PM-stream 吞吐优化（P0，2026-07-30）

## 产出
**Muon**：`src/tais_obsidian/optim/muon.py`（210 行）+ `optim/__init__.py` + train.py 接入（config `"optimizer":"adamw"|"muon"`，缺省 adamw 向后兼容）+ `tests/test_muon.py`（234 行，8 项全绿）+ bench 脚本。
**PM-stream**：pmstream.py 优化（read/write fp64→fp32、Sinkhorn t_max 20→10 可调、einsum 融合）+ config.py `pm_sk_t_max` 字段（缺省 20 向后兼容）+ model.py 接入。
子代理实现，我验收（读码+独立重跑+全量 382 绿）。

## Muon（arXiv:2412.02684 Keller Jordan 谱系 + K3 Per-Head 借鉴）
- **原理**：2D 矩阵参数（线性层权重）的**动量**经 Newton-Schulz 迭代（zeropower_via_newtonschulz5，5 步 quintic，系数 3.4445/−4.7750/2.0315）正交化后更新（隐式谱归一化，收敛快、对 lr 不敏感）；embedding/norm/bias/1D 走内部 AdamW。
- **分组**：ndim≥2 且非 embedding→Muon（lr=0.02）；embedding/norm/bias/1D→AdamW（对齐 train.py decay 语义）。可选 per-head（Q/K/V 按头分块正交化）。
- **收敛对比（4070 tiny 120 步）**：**Muon loss→6.523（更好）/ 20.81k tok/s（95.4%，Newton-Schulz 开销仅 4.6%）** vs AdamW loss→6.868 / 21.80k tok/s。
- **0.1B 真实训练冒烟（40 步）**：loss 10.55→7.14、val 7.15，GDN-2+三级栈+grad ckpt+checkpoint/resume+save_pretrained 全链通过。
- **红线**：Muon 只影响优化器更新（前向/反向计算图不变，GDN 递归/PM-stream/checkpoint 天然兼容）。

## PM-stream 吞吐优化（三管齐下）
read/write **fp64→fp32**（×1.8/×1.9）+ Sinkhorn **t_max 20→10**（config 可调，无 GPU 同步）+ einsum 单次融合。
| 版本 | pm_stream=5 | hybrid | 比值 | 显存 |
|---|---|---|---|---|
| 优化前（fp64+t_max20） | 2.92k | 8.64k | 33.8% | 11.92GB |
| **优化后（fp32+t_max10）** | **4.89k** | 8.68k | **56.4%** | 9.75GB |
- **PM-stream 吞吐 ×1.68（2.92k→4.89k），相对 hybrid 33.8%→56.4%，显存 ↓2.2GB**。
- **恒等判据保**：fp32 后 rel=1.46e-06 ≪ 1e-5（余量 ~7×）；谱范数=1.0（信号守恒红线≤1.6 保持）。
- **未达 90%**：剩余瓶颈是 PM-stream 固有 5× 流状态存储/带宽（S[B,T,5,768] 反复读写），纯 PyTorch einsum 已达融合极限；mHC 原文 6.7% 开销是 27B 大模型+定制 CUDA kernel 后（0.1B 小模型固定开销摊薄少）。进一步需 torch.compile/自定义 kernel（受"纯 PyTorch 无 triton"约束，留后续）。

## 子代理踩坑（验收记录）
①**Sinkhorn 早停弃用**：tol 早停用 .item() 判定每次同步 GPU（4.173ms 比固定 20 次 1.502ms 更慢）——改调小固定迭代数 t_max=10（无同步纯 GPU 流水，谱范数仍=1.0）；②Newton-Schulz 判据修正（5 步是近似正交化非严格，改谱归一化判据）；③hybrid 吞吐测量噪声（warmup 不足测 4.85k，充分 warmup 后稳定 8.64–8.68k）。

## 待接
①PM-stream 进一步吞吐（torch.compile/自定义 kernel 突破 56.4%→90%+，受无 triton 约束）；②Muon 用于 1.5B 预训练（与 W4 固化同优化器，降遗忘）；③Muon vs AdamW 长训练对比（收敛优势在长跑是否保持）。

---
*导出自 /memories/repo/muon-pmstream-optimization.md（2026-07-30 同步快照）。*
