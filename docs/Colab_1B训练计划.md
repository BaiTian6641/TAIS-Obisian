# Colab 1B 训练计划（2026-07-31，v1.0）

> 决策依据 = 全架构只读审阅（P0/P1/P2 全修）+ 文献/事实交叉验证（tavily/WebSearch 7 节核查）。
> 执行载体：`notebooks/TAIS_1B_Colab.ipynb`（28 cells，纯 Python cell 无 shell 魔法，逐 cell AST 校验过）。

## 1. 目标与约束

- **目标**：1B 模型（实测 **1017.7M** 参数，tied embedding）预训练 9B + 中训练退火 1B = **10B tokens** 一体跑完，上传 HuggingFace，拿到权重直接推理测试。
- **硬件**：Google Colab G4（RTX PRO 6000 Blackwell，96GB，sm_120）——已核实 Colab 确有此机型（付费档，供应波动；arXiv:2606.08638 同款机型实证）。
- **硬约束**：Colab 单会话 ≤24h（断连是常态）→ **checkpoint/resume 是一等公民**（每 500 步落盘 + 15 分钟 Drive 同步 + resume 全链加固）；运行时每次清空 → 环境/代码/数据恢复全部幂等可重跑。

## 2. 模型与训练配置（configs/pilot_1b_gdn2.json）

| 项 | 值 | 依据 |
|---|---|---|
| 参数 | 1017.7M（实例化实测） | d1536×32 层 = 8×{G2G2G2A}（24 GDN-2 + 8 三级栈），mlp 4096，头 24q:8kv / 24v:12qk，head_dim 64，vocab 32768 tied |
| tokens | 10B（9B 预训练 + 1B 中训练） | 用户指定；**研究性欠训**（Chinchilla 20B、当代 1B 实践 4T–11T）——模型卡如实标注，评测不对标市售 |
| 优化器 | Muon（muon_lr 0.02）+ 非矩阵 AdamW 6e-4 | 设计"预训练与 W4 同优化器"；set_lr 已修 WSD 生效 |
| 调度 | 预训练 Stable（decay_frac=0）→ 中训练 decay_frac=1.0 线性到 0 | SmolLM2 多阶段 WSD / OLMo Dolmino 同构：退火段 = 高质量数据上移 + lr 衰减（arXiv:2512.13961、arXiv:2501.00656 核实） |
| batch | micro 32×accum 8 = 262k tok/step（Colab cell 8 实测标定，显存 <85GB 约束） | 步数：预训练 34300 + 中训练 3800 |
| 中训练初始化 | `--init_from` Phase-1 final（仅权重、step=0、新优化器） | OLMo Dolmino 独立退火 run 惯例；避免 resume 的 lr 步数纠缠 |

## 3. 数据（scripts/prepare_data_1b.py）

- **配比**：fineweb_edu **sample-100BT** 73%（7.3B）/ math 12%（NuminaMath-CoT ~430M + FineMath-4+ 补足）/ cosmopedia 10% / FineWeb2-HQ cmn_Hani 5%。
- **中训练退火混合**（1B）：fineweb_edu 35%/math 40%/code 20%/zh 5%（Dolmino 式质量上移）。
- **容量核实**（子代理 HfApi/datasets-server 实测）：fineweb-edu 有 sample-100BT/350BT（10BT 档不够）；FineMath-4+ 9.6B tokens；cosmopedia 25B；cmn_Hani 54.2M 文档。
- **工程**：HTTP GET 直下 parquet→本地流式（hf-mirror 不暴露 auto-converted parquet）；断流重试 12 次；原子写（*.part→rename）；**断点续跑**（_progress.json 跳过已完成源）；**max_id 全量扫描 <32768**；shard 20GB 本地 NVMe（勿放 Drive FUSE，memmap 极慢）；Drive 缓存可选（CACHE_DATA_ON_DRIVE）。
- 本机冒烟实测：四源全通、max_id 32767、续跑跳过验证、**10B 全量 ETA 3-5h**。

## 4. 审阅修复清单（437 pytest 全绿，主代理复验）

**P0**：① tokenizer.json 随权重产物（train.py 复制 + generate 三级回退 `<ckpt>/tokenizer.json`）；② Colab torch 必须 cu128（notebook cell 2 自检+换装+重启提示）；③ 新增 `scripts/export_final.py`（latest.pt→save_pretrained，断连抢救权重）。

**P1**：④ OOM 折半不再重建优化器（此前静默丢 Muon 动量/AdamW 状态）；⑤ generate 守卫（超长 prompt 清晰报错、越界 id assert——tokenizer 32773 > 模型 32768 的 5 个保留特殊 token、max_seq 触顶受控停止）；⑥ resume 加固（CUDA RNG 切片到设备数 + **map_location ByteTensor 修复——此前 GPU 上 resume 必炸 TypeError**；model_cfg 13 关键字段一致性校验）；⑦ build_model_config 透传补齐（pm_sk_t_max/grad_checkpoint/rope_theta/rms_eps/conv_kernel/kernel_dg_*/manifold_dim——pm_sk_t_max 此前是"死旋钮"）；⑧ attach_kernel 同步 kernel_enabled=True；⑨ reconfigure hasattr 守卫 46 处（ipykernel 兼容）；⑩ 冒烟脚本死字段修复（attn_only/attn_impl）；⑪ extend_context --resume；⑫ train.py --data_dir/--init_from/--micro_batch/--grad_accum CLI；⑬ writer try/finally。

**新增回归测试 13 项**：set_lr WSD×Muon、resume e2e（loss 逐 bit 接续）、非默认尺寸 save/load 往返、generate 守卫、export_final 全链。

**遗留（记录在案，不影响本计划）**：tests/ 旧文件的 reconfigure 未加守卫（仅 notebook import 才触发）；muon per_head_qkv 语义粗糙（默认关）；lm-eval 生态接入需 auto_map+trust_remote_code 或 PreTrainedModel 化——**列为后续工程**（拿到权重先用仓库内 generate/val 评测）。

## 5. 事实核查结论（与计划的冲突点与处置）

1. **10B 欠训**：已核实低于 Chinchilla 一半、低于当代 2-3 个数量级 → 定位研究 pilot，模型卡声明，不对标。
2. **Colab G4 RTX PRO 6000 存在**（付费档、供应波动、≤24h 会话）→ checkpoint/resume 一等公民（本计划核心设计）。
3. **中训练做法**：OLMo 3 Dolmino（100B 高质量+线性衰减）/ SmolLM2（WSD 末段质量上移）——本计划两阶段正是该模式。
4. **YaRN 是 256K 扩展主流**（Qwen2.5-1M 用 YaRN rescaling；Llama 3 分 6 阶段 8K→128K）——RoPE 初版已落地（rope_scaling none/yarn + extend_context.py），256K 是 1B 之后的独立阶段。
5. **HF 上传**：upload_folder 断点续传、2.5GB 单文件远低于 50GB 限；模型卡必须显式 `library_name`（自研格式不自动推断）。
6. **safetensors 分片**：新版 transformers 默认 50GB（旧 5GB）——我们单文件 2GB 无感。

## 6. Colab 执行要点（notebook 使用）

1. 配置 cell 填 `GIT_URL`（私有仓库用 PAT）或传 `TAIS_Obsidian.zip` 到 Drive/TAIS_1B/；Secrets 加 `HF_TOKEN`。
2. 自上而下跑；**cell 2 若换装 cu128 必须重启运行时后从头重跑**。
3. cell 6 数据（3-5h，可断点续跑）→ cell 7 冒烟 → cell 8 标定（打印 tok/s 与 ETA）→ cell 9 预训练（跨会话续训）→ cell 10 中训练 → cell 11 推理自验 → cell 12 模型卡 → cell 13 上传。
4. 断连恢复：重跑 cell 1-5 → cell 6（幂等）→ 当前 Phase cell（自动从 Drive ckpt --resume）。详见 notebook 末尾 Runbook。

## 7. 拿到权重后的测试路线（下一步）
- 即日可做：`generate --ckpt` 文本抽验、val loss 终值 vs 0.1B 基线、KAL 探针读点扫描（kal_probe）+ kal_truth_finetune_v2 迁移（0.1B 是 ℓ10/AUROC 0.845 范式）。
- 短期：NIAH 长度扫描复测（扩窗前 1024 基线）、RoPE 渐进扩窗（4K→…→256K，extend_context.py）。
- 生态（可选工程）：auto_map+trust_remote_code 导出 → lm-eval（ARC/HellaSwag/PIQA/BoolQ 当天可跑套餐）。

---
*依据：审阅报告（agent-4）+ 事实核查（agent-5，7 节，来源 URL 见会话记录）；代码状态 commit 见 git log。*
