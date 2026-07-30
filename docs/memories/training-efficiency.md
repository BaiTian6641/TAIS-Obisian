# 训练效率优化（RTX PRO 4000 Blackwell sm_120，2026-07-26 Tavily 检索）

> 目标：在 RTX PRO 4000 Blackwell（sm_120，24GB）上提高 PyTorch 训练吞吐。当前 train.py 已落实多数项。

## 已在 train.py 落实
- **TF32**：`torch.backends.cuda.matmul.allow_tf32=True` + `cudnn.allow_tf32=True`（train.py 已有；GDN fp32 GEMM 走 tensor core ~1.4×）
- **bf16 autocast**：`torch.autocast("cuda", torch.bfloat16)` 前向（混合精度，tensor core）
- **fused AdamW**：`torch.optim.AdamW(..., fused=True)`（CUDA 可用时）
- **grad checkpoint**：逐 Block 重算换显存（config.grad_checkpoint=True）

## 可补充（检索确认有效）
- **`torch.set_float32_matmul_precision("high")`**：等价且更现代的 TF32 开关（一行，与 allow_tf32 同效，建议统一）。
- **`torch.backends.cudnn.benchmark=True`**：固定输入形状下 cuDNN autotune 选最快卷积/注意力 kernel（seq_len 固定时收益）。
- **`torch.compile(mode="reduce-overhead")`**：kernel 融合 + 减 Python 开销；稳定后升 `max-autotune`。注意：GDN 递归/chunked 自定义路径、PM-stream sinkhorn、grad checkpoint 可能 graph break，需先小步验证（先 reduce-overhead，再排 .item()/shape-if/numpy 障碍）。
- **Liger Kernel / FLCE**：fused linear cross-entropy 可省大 logits 副本（我们的 chunked_ce 已手动分块，功效类似；Liger 再省 ~内存）。
- **x.to(device, non_blocking=True)** + pin_memory：数据加载异步（当前 get_batch 直接 from_numpy 到 device，量大时可加）。
- **Blackwell 特有**：`TORCH_CUDA_ARCH_LIST="12.0"`（编译自定义 kernel 时）；vLLM/FA3 在 Blackwell 需 `VLLM_FLASH_ATTN_VERSION=2`（推理侧，训练不涉及）；NGC 容器预装优化驱动。
- **NVIDIA cuDNN NSA kernel**（article_ref/01 已记）：为稀疏注意力优化 sm_100+，sm_120 加速路径，可缓解纯 PyTorch GDN 吞吐痛点（9.5k vs SDPA 19.7k tok/s）。

## 建议优先级（0.1B pilot，当前 9.5k tok/s）
1. `set_float32_matmul_precision("high")` + `cudnn.benchmark=True`（零风险，一行）
2. `torch.compile(mode="reduce-overhead")` 小步验证（先排 graph break）
3. 数据加载 non_blocking（量大时）
4. cuDNN NSA kernel / Liger（中期，需评估兼容性）

---
*导出自 /memories/repo/training-efficiency.md（2026-07-30 同步快照）。*
