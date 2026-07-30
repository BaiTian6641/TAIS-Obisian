# 双卡分工（2026-07-29 用户指示，同步记忆）

## 硬件
- **RTX PRO 4000 Blackwell SFF**（24GB，sm_120，70W）：主计算卡，**专用于训练**（GDN 预训练/长跑 run）。torch 设备序 cuda:1（须 `CUDA_VISIBLE_DEVICES=1` 使其成为单卡视图 cuda:0）。
- **RTX 4070 Laptop**（8GB）：副卡，**用于轻量任务**——微调（KAL 真值锚等小规模）、推理测试、评估脚本。**避免抢占 PRO 4000 的训练资源**。

## 分工原则
- 训练/长 run → PRO 4000（cuda:1，CUDA_VISIBLE_DEVICES=1）。
- 微调/推理测试/评估/NIAH/AUROC → RTX 4070（cuda:0，CUDA_VISIBLE_DEVICES=0 或不设）。
- **背景**：此前 KAL 微调（1200 步）在 PRO 4000 上跑时与 bounded 训练抢占，吞吐从 8.5k 降到 5.0k（子代理遇 1200 步重跑停滞终止）——双卡分工可避免此问题。
- 注意：两卡 torch 设备序 ≠ nvidia-smi 序；torch cuda:0=4070、cuda:1=PRO 4000。
- RTX 4070 是 sm_89（Ada），非 sm_120——torch cu128 wheel 兼容，但显存仅 8GB，微调/推理需控制 batch/seq（0.1B 模型 bf16 ~0.5GB 权重，推理/小 batch 微调可行；大 batch 训练不行）。

---
*导出自 /memories/repo/hardware-dual-gpu.md（2026-07-30 同步快照）。*
