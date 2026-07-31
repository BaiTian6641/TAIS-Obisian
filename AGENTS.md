# AGENTS.md — TAIS Obsidian（tais-obsidian）

> 本文件面向 AI 编码代理，假定读者对本项目一无所知。全部项目文档为中文，与文档相关的工作请使用中文。

## 1. 项目概览

**TAIS Obsidian**（泰斯人工智能复合体 · 黑曜石框架）是一个处于**原型验证阶段**的研究项目（0.1B 先导与原生部件消融已落地，见 §2），目标是构建一个"面向持续学习 LLM 的权重虚拟内存"架构，并最终从零训练一台 **1.5B 原生 1M 上下文的自学习验证机**。

核心设想：为大语言模型提供一个动态接入外部权重的接口（类比操作系统的页表/虚拟内存），使模型在推理过程中能以"知识块"（KnowledgeBlock）为单位按需换入/换出知识与经验，而不是依赖外挂 markdown / RAG 文本。关键设计点（对齐设计文档 v2.5）：

- **双形态知识块**：markdown 源代码形态（ground truth，可审计）+ 编译产物形态（可失效重建）；**v2.0 起编译产物 = 零梯度记忆栈**（KV 块 CSA harvest / 增强A记忆层 delta / 向量块 ICV-steering / 路径块·概念槽），**LoRA 从必备载体降级为可选的睡眠期深度固化产物**；编译在离线"睡眠时间"进行。
- **存储层级**：L0 VRAM（工作记忆）↔ L1 DRAM（短期记忆）↔ L2 NVMe（长期记忆）↔ L3 远端（档案）。
- **双向接口（读写不对称）**：运行时只读 + 只写 W0 日志 + W1–W2 零梯度快写；W3+（LoRA 梯度、合并入主干）仅离线睡眠期执行。人格块运行时只读。
- **冷启动"苏醒序列"**：参考麻醉苏醒研究，按"路由器/接口层 → 人格块+元数据块 → 高频陈述性块 → 长尾惰性加载"的顺序恢复。
- **元认知（KAL）**：分层元认知 L1 P(IK) 三态 + L2 情感 + L3 冲突 + ITI 干预，内部探针检测知识空白（SAPLMA/量化态探针 0.904–1.000 AUROC 证据），模型主动"回想"而非盲目猜测。
- **TAIS Obsidian 1.5B 模型**：28 层 = 7 × {3 GDN-MemBlock + 1 三级注意力栈（滑窗 512 L0 + CSA stride-4 选择检索 L1 + HCA 128:1 L2 gist）}，hidden 2048，词表 **129280 = 127232 基础 + 2048 reserved**（动态词表三级生长阶梯）tied embedding，原生 1M 上下文；KAL/HRL/知识块注入点/PM-stream（mHC n=5 感知-记忆专用道）为 checkpoint 内生部件（非外挂服务）；预训练与 W4 固化**同优化器 Muon**。

模型谱系：TAIS Obsidian 1B / 1.5B /（远期）4B-A1B。本机硬件（2026-07-24 到位）：**RTX PRO 4000 Blackwell SFF（24GB，sm_120，70W）** + RTX 4070 8GB 副卡（显示/杂务，不用于训练），驱动 596.36（CUDA 13.2）——PyTorch 须用 **cu128+** wheel（cu126 无 sm_120 内核）。

## 2. 仓库当前状态（2026-07-26 更新）

**代码已落地并通过 0.1B 消融矩阵**：设计文档（`docs/`）+ 自研训练/推理框架（纯 PyTorch，无 triton，Windows 原生可跑）。**《部件实现详细计划》M0（主干可跑）已达成**——hybrid 基线 val 3.768；E+-5 PM-stream（3.744）与 E+-7 三级注意力栈（3.762）双消融达标且组合相容（3.743）；E+-3 KAL 探针 ℓ8 AUROC 0.945（语义空白子集 0.979 超 FLARE 基线）；25 项 pytest 全绿。详见《D0_0p1B先导实验报告.md》§6.4 消融矩阵全表。

**本机工作区状态**：`.venv`（Python 3.12.13 + torch 2.11.0+cu128）/ `data/`（tokenizer 32773 词表 + 120M tokens shards）/ `checkpoints/`（pilot_0p1b_ws、pilot_0p1b_attn、pilot_0p1b_pm、pilot_0p1b_tri、pilot_0p1b_pmtri 五个 run 的 final）/ `runs/` 均已就位。

### 2.1 代码结构

- `src/tais_obsidian/`：框架包（`uv pip install -e .` 后可 import）
  - `config.py`：ModelConfig dataclass + JSON 读写（含 `pm_stream`/`pm_constrain`/`tri_*`/`tri_use_indexer`/`kernel_*` 开关；`rope_scaling`/`rope_scale`/`rope_original_max_seq` 上下文扩充开关，默认 none 向后兼容；**2026-07 起移除 `attn_only`/`attn_impl`，注意力层统一为三级栈**）
  - `model/`：`gdn.py`（GDN-MemBlock：naive_recurrent + chunked 双路径，纯 PyTorch）、`tri_attention.py`（**TriRetrievalAttention 三级检索注意力**：滑窗 L0 + CSA 选择检索 L1 + HCA 128:1 gist L2 + NSA 式门控融合 + `inject_hca_entries` + 可选 `tri_use_indexer` 独立 LightningIndexer）、`model.py`（`TaisObsidianForCausalLM`，tied embedding，自研 `save_pretrained`/`from_pretrained`；`forward(capture_layers=..., run_kernel=..., inject_payloads=...)` hidden-state 捕获 + 内核挂点）、`blockpath.py`（块通路：BlockCompressor stride-4 压缩器 + 块 KV 收割/注入 + namespace 五元组 fail-closed）、`pmstream.py`（E+-5 PM-stream：mHC 多流残差 arXiv:2512.24880，恒等初始化 <1e-6）、`kal.py`（E+-3 KAL 分层元认知头：L1 P(IK) W[d,3] + L2 valence/arousal W[d,2]）、`tais_kernel.py`（TAIS 内核：聚合 KAL + HRL Indexer + DG + 侧信道头簇，sense/route/inject）、`hrl_indexer.py`（LightningIndexer：DSA 式独立检索打分器）、`memlayer.py`（增强 A 记忆层：product-key KV + GDN-2 erase/write 解耦 delta 写）、`injection.py`（注入闭环）、`dyn_vocab.py`（动态词表 concept_slot）
  - `runtime/`：运行时服务（`pagetable.py` 页表 SQLite、`blockstore.py` 块存储 usage_weighted、`pager.py` 缺页 fail-closed、`bus.py` Memory Bus、`ca1_gate.py` 巩固门、`ca3_ppr.py` 联想、`state_ckpt.py` GDN 状态持久化、`safety.py` 安全管线）
  - `sleep/`：`consolidator.py`（睡眠巩固器：分簇回放 + 间隔提取练习 + CA1 门 + SHY 归一化）
  - `data/memmap.py`：uint16 bin shard 读写与 batch 采样；`tokenizer_io.py`：tokenizer 封装
  - `train.py`：训练循环（bf16 autocast + fp32 参数、AdamW 分组、WSD 调度、grad clip 1.0、checkpoint/resume、tensorboard；config JSON 可加 `pm_stream`/`tri_*`/`kal_aux_weight` 开关；**注意：1.5B T1 起按设计文档改 Muon 优化器**）
  - `generate.py`：cache 增量生成（temperature/top-k）
- `configs/`：`pilot_0p1b.json`（hybrid 基线）、`pilot_0p1b_pm.json`（PM-stream 消融）、`pilot_0p1b_tri.json`（三级栈消融）、`pilot_0p1b_pmtri.json`（组合）；**`pilot_0p1b_attn.json`（attn_only 孪生）已于 2026-07 移除（对照组废弃）**；`pilot_0p5b_gdn2.json`（**0.5B GDN-2 预训练**：512.8M 参数 d1280×24 层，Muon，max_steps 22900 ≈ 3B tokens 单卡口径）
- `scripts/`：`check_env.py`（环境自检）、`prepare_data.py`（FineWeb-Edu → 训 32k BPE → 120M tokens shards）、`prepare_data_0p5b.py`（**0.5B 3B tokens 多领域混合**：fineweb_edu 70%/NuminaMath 15%/cosmopedia 10%/FineWeb2-HQ 中文 5%，断流重试加固，输出 data/shards_0p5b）、`train_dp.py`（**双卡手动 DP**：WorkerNode 线程化+时间均衡 accum，3.1k tok/s +24%，Windows 无 NCCL 单进程实现）、`smoke_overfit.py` / `smoke_overfit_pm.py` / `smoke_overfit_tri.py`（三变体过拟合冒烟）、`extend_tokenizer.py`（E+-2 特殊 token 扩容，幂等）、`kal_probe.py`（E+-3 KAL 探针管线，输出 `runs/kal_probe/report.json`）、`kal_truth_finetune_v2.py`（**P1 校准 v2**：锚集扩充 AUROC 0.845/0.829 双口径达标；预测反馈循环诚实负结果）、`extend_context.py`（**渐进扩窗**：RoPE 缓存扩容至 256K + YaRN scaling + 阶段课程微调，复用 train.py 组件）、`bench_long_seq_cost.py`（长 seq 成本实测：CSA/HCA 打分随 T²、滑窗 math-SDPA 显存随 T²）
- `tests/`（**418 项 pytest 全绿**，2026-07-30）：第一阶段 M0–M8 测试（test_gdn/cache/capture/blockpath/pmstream/tri_attention/tri_indexer/kal/tokenizer_ext/tais_kernel/kernel_wiring/kernel_route_candidates/hrl/hrl_init/gdn2_indexer/runtime/injection/sleep/dyn_vocab/safety）+ 第二阶段（test_manifold/manifold_bridge/thought_core/reasoning_loop/cot_projection/thought_visualizer/path_integration/thinking_e2e/thinking_real_adapter/thought_core_integration）+ 主动求知（test_inquiry_branch/inquiry_executor/inquiry_consolidation/active_inquiry_full_chain）+ 知识内化（test_teaching_data/internalization_e2e/retrieval_recall/gated_fusion/kaplan_extract/unified_checkpoint/memlayer_internalization）+ 门控自适应（test_decoupled_gate/fully_decoupled/niah_length_scan）+ 优化（test_muon）+ 校准（test_kal_calibration_v2）
- 不入库：`data/`（tokenizer + shards）、`checkpoints/`、`runs/`、`logs/`
- **待建（对齐《接口与实现计划》§1 包结构）**：`model/` 增 `tais_kernel.py`（M1 内核骨架）、`hrl_heads.py`、`memlayer.py`、`injection.py`；新建 `runtime/`（bus/pager/pagetable/blockstore/ca3_ppr/ca1_gate/awakener/**state_ckpt**）与 `sleep/`（consolidator/distill）包。

### 2.2 常用命令

先 `source .venv/Scripts/activate`（Git Bash；venv 由 uv 创建，Python 3.12）。**工作站（Blackwell sm_120）装 torch 用 `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`**；cu126 wheel 不含 sm_120 内核，不可用。**双卡注意：torch 设备序 ≠ nvidia-smi 序——cuda:0 = RTX 4070（8GB 副卡），cuda:1 = RTX PRO 4000（24GB 计算卡）；所有训练/测试/生成命令必须前缀 `CUDA_VISIBLE_DEVICES=1`**（单卡视图下 PRO 4000 即 cuda:0）。HF 直连不稳定时数据脚本前缀 `HF_ENDPOINT=https://hf-mirror.com`。

- 环境自检：`python scripts/check_env.py`
- 数据准备：`python scripts/prepare_data.py`
- 冒烟测试：`python scripts/smoke_overfit.py`（`_pm`/`_tri` 变体同理）
- 单元测试：`python -m pytest tests/ -q`（418 项）
- 训练：`python -m tais_obsidian.train --config configs/<cfg>.json`（续训加 `--resume checkpoints/<run>/latest.pt`；长任务加 `python -u` 防输出缓冲）
- 推理：`python -m tais_obsidian.generate --ckpt checkpoints/<run>/final --prompt "..."`

工作站实测基线（0.1B，micro 16×accum 4×seq 1024）：hybrid 训练 9.5k tok/s（峰值 7.0GB）、attn_only 19.7k、PM-stream 3.0k、三级栈 8.6k、组合 2.9k；生成 hybrid 37.8 tok/s。**不要臆造**未列出的命令。所有实现工作应严格遵循 `docs/` 中的 M0–M8 里程碑链（《部件实现详细计划》§IV）与《从零构建TAIS-Obsidian_总体实施计划.md》的阶段检查点。

## 3. 文档清单与内容地图

全部文档位于 `docs/`，均为中文 Markdown，是本项目唯一的"事实来源"：

| 文件 | 版本 | 内容 |
|---|---|---|
| `docs/动态知识块记忆系统_设计文档.md` | v0.4 | DKB-MS 核心设计（What & Why）：体系结构类比、Block Spec v0.1（字段规范、生命周期状态机）、存储层级 L0–L3、苏醒序列、路由与学习、空白检测三通道、写通道 W0–W4 与页保护位（v0.4 修订注记：EXP-PERSONA 沙箱例外）、接口 ABI v0.1、人格块、可行性证据汇总（§11）、开放问题（§12） |
| `docs/TAIS_Obsidian_细致框架设计文档.md` | v2.5 | TAIS Obsidian 1.5B 模型设计：模型配置（§2）、原生 1M（§3）、数据集（§4）、BF16 配方（§5）、KAL/HRL/视觉空间区（§6）、T0–T5（§7）、原生集成收益（§8）、市面对比（§10）、CSA 块通路与 HRL 头簇（§11）、动态参数增长与 mHC PM-stream（§12）、Reasoning-native（§13）、部署后学习终验（§14）、理论桥与增强 A/B/C（§15）、情感总线与 KAL 分层（§16）、CoT 与路径块（§17）、本地可行性/MoE 对比/半固定半动态/受控并入/边缘运行时/脑区映射（§18–23）、**零梯度记忆栈（§24，v2.0）、动态调用边界（§25）、原生通路 vs tokenizer（§26）、KAL/HRL 联合训练（§27）、动态词表三级阶梯（§28，v2.4）、五大命题独立交叉验证（§29，v2.5）** |
| `docs/TAIS_Obsidian_子系统架构规格.md` | v1.0 | 部件→子系统→整机工程规格（Part A–H + 交叉干扰红线 Part Z + 神经科学承重审计），双镜头（体系结构+神经科学） |
| `docs/TAIS_Obsidian_接口与实现计划.md` | v1.0 | checkpoint 边界判定（HRL 方案 B）+ TAIS 内核/Injector 接口签名 + KAL/HRL 训练数据协议 + 信号清单 + 红线 |
| `docs/TAIS_Obsidian_部件实现详细计划.md` | v1.0 | 32 部件逐一 7 维（是什么/做什么/怎么实现/怎么训练/注意/信号/数据）+ dataclass + 损失公式 + 8 milestone 路线图 + 10 红线总表 |
| `docs/从零构建TAIS-Obsidian_总体实施计划.md` | v0.4 | 从零构建/训练/推理的总实施计划（How，工程向）：工作站环境现状与检查清单（§1–2）、阶段 A–H、§6.6 D-0 逐步执行方案（S0 环境→S1 数据→S2 pilot+孪生，**已完成**）、§7.5 阶段 E+ 原生部件 0.1B 原型序列（E+-1/2/4/5/7）、§8A EXP-PERSONA 极其实验（人格块可读写 + KAL 道德约束块）、显存/吞吐/超参公式（§6、§12）、风险登记册（§11） |
| `docs/TAIS_Obsidian_架构详图.png` | v0.4.1 | Carbon 设计语言架构详图：主干、KAL、HRL、知识块库、DKB-Runtime、记忆层级、睡眠巩固器、T0–T5 流水线、苏醒序列 |
| `docs/D0_0p1B先导实验报告.md` | 2026-07-24 | D-0 0.1B 先导实验报告（工作站 S2）：hybrid vs attn_only 对照（val 3.768 vs 3.818，−0.050 nats）、训练/生成吞吐与显存基线、micro batch 标定、S2 退出判定通过 |
| `docs/AGENT_PLAN_E+-5_PM-stream.md` | 2026-07-24 | E+-5 PM-stream（mHC n=5）实现规范与验收判据——交 GLM5.2（GitHub Copilot）交叉验证核心算法用，自包含 |

注意：文档中引用的另一份配套文档《自我学习LLM框架构想_HippoK》(v0.2) **不在本仓库中**；旧命名 HippoK 已废止，统一为 TAIS Obsidian（tais-obsidian）。

## 4. 技术栈与实现路线（对齐设计文档 v2.5 与 M0–M8 里程碑链）

> 注：原《DKB-MS 实施规划与路线图》已于 2026-07 废弃；实现路线以《部件实现详细计划》§IV 的 **M0–M8 milestone 链**为准（M0 已达成，当前在 M1 内核骨架）。

- **语言**：Python 3.12（uv venv，不污染 Espressif 的 3.11）；
- **训练**：BF16 mixed（bf16 计算 + FP32 主权重/优化器/梯度累积，禁止纯 bf16）、grad clip 1.0、QK-Norm、WSD 调度；**优化器 Muon（预训练与 W4 固化同优化器，arXiv:2605.06654 降遗忘）**——0.1B pilot 用的 AdamW 在 D-2 起切换；8-bit AdamW 仅作显存紧张时的备选；OLMo 3 数据课程（Dolma 3 Mix / Dolmino / Longmino、Dolci 后训练套件）；
- **记忆载体（v2.0 零梯度栈）**：KV 块（CSA harvest）/ 增强A记忆层（delta 写）/ 向量块（ICV-steering）/ 路径块·概念槽；**LoRA 降级为可选的睡眠期深度固化产物**；
- **包结构（checkpoint 边界 = 方案 B）**：`model/` 只放前向可微、随 state_dict 存取的部件（KAL/HRL 学习型头内生 checkpoint）；`runtime/` 放数据/算法/IO（页表 SQLite、BlockStore、CA3 PPR、CA1 门、苏醒调度、**state_ckpt 自研 GDN 状态持久化——引擎空白的关键缺口**）；`sleep/` 放离线固化（间隔提取练习 + On-Policy Context Distillation + SHY 归一化）；
- **推理栈（远期）**：vLLM（WSL2，serving + 动态 LoRA）+ HuggingFace Transformers（hooks 捕获 hidden state）；llama.cpp/GGUF 端侧备选；
- **存储**：SQLite（页表/元数据）+ 向量库（route_key 检索）+ 文件存储（块载荷）；
- **实验底座模型（底座实验线）**：主力 Qwen3.5-9B，管线调试用 Qwen3.5-4B，对照 Qwen3-8B（全注意力）；冻结 vision tower，纯文本实验；
- **硬件**：开发机 = RTX PRO 4000 Blackwell SFF（24GB，sm_120）已到位；9B 底座 bf16 推理（~19GB）与 ≤2B 全流程本机可行（设计 §18：1.5B×30B tokens ≈ 78–104 天月级任务）；正式 1.5B 预训练可按本机月级或云端短租取舍；1M 长上下文阶段需云端短租多卡。

**GDN 层关键约束**（写代码时必须遵守）：GDN 层无 KV cache，KV prefix 注入只适用于注意力层；增强A记忆层与 LoRA 注入各层通用。知识块的"载体适用性"与"事实召回能力"（token 寻址载体可事实召回，位置不变向量不可）必须按层类型与载体类型标注。

## 5. 开发流程约定

- **M0–M8 里程碑链**（《部件实现详细计划》§IV，每级有退出标准，不达标不进下一级）：
  - ~~M0 主干可跑~~ ✅（GDN-MemBlock+TriRetrievalAttention+PM-stream，25 项 tests 全绿 + 消融矩阵）；
  - ~~M1 内核骨架~~ ✅（TAISKernel sense/route/inject，前向不崩、PM 读写通，120 项全绿）；
  - ~~M2 KAL 内生~~ ✅（L1/L2 头+挂点+kal_probe，探针 AUROC 0.945≥0.8 @0.1B ℓ8）；
  - ~~M3 HRL 内生~~ ✅（Indexer+DG+侧信道头簇+LightningIndexer，梯度隔离验证）；
  - ~~M4 运行时骨架~~ ✅（Bus+Pager+页表+BlockStore+state_ckpt，缺页 fail-closed、state 往返 <1e-5）；
  - ~~M5 注入闭环~~ ✅（KV拼接+记忆层+向量加法，注入后人效不降 Δ+0.0001）；
  - ~~M6 睡眠固化~~ ✅（间隔提取练习+CA1门+SHY，回归通过、归一化稳定）；
  - ~~M7 动态词表~~ ✅（concept_slot+注册，输入侧提取通路）；
  - ~~M8 安全管线~~ ✅（签名+namespace+扫描器接口，投毒检出、漂移报警）。
  - **当前进度（2026-07-30，412 项 pytest 全绿）**：M0–M8 全部落地；其后完成——
    - **前置工程（第二阶段地基）**：① GDN-2 门收敛验证（10k 训练 NIAH 0.207 反超 GDN-1 0.177，三阶段证据链）；② GDN decay 有界化（K3 借鉴 g_min=-5 sigmoid，**4× 加速门收敛**+保 1M 数值范围，断点兼容 from_json 回填）；③ PM-stream 端到端（多流+sense/inject+桥接）；④ **PM-stream 吞吐优化**（fp32 Sinkhorn+t_max 10+einsum 融合，×0.35→×0.56）；⑤ **Muon 优化器**（Newton-Schulz 正交化动量，收敛更好 6.523<AdamW 6.868，吞吐仅 4.6% 开销，对齐设计"预训练与 W4 同优化器"）。
    - **第二阶段（思维能力强化）7 迭代 pilot 全落地**：思考流形（manifold.py，共享投影+共形等距+VICReg 去相关）→ 思考流形↔PM-stream 桥接（manifold_bridge.py）→ CTM 式思考核（thought_core.py，通道组历史+RoPE 相位化思考时间+certainty 早停）→ 推理循环（reasoning_loop.py，§1.3 五步 tick）→ CoT 投影层（cot_projection.py，投影非计算+忠实性审计）→ 路径积分辅助任务（path_integration.py，GridCodeProbe）→ 可解释性前端（thought_visualizer.py，3D 轨迹+坏路径四类检测）。端到端集成（thinking_e2e_demo）+真实部件适配（thinking_real_adapter）。**思考核接入主干前向**（thought_core_integration.py，推理增益 +0.078~+0.125 达 fb1"从模块到原生"门槛）。
    - **主动求知闭环（自我学习）**：certainty（KAL 真值锚校准 AUROC 0.769）→ 求知分支（inquiry_branch 四选一 RPL/LP）→ 求知执行器（inquiry_executor 交叉验证[绝不裸自我修正]+KnowledgeBlockWriter[累积不覆盖]）→ 知识内化（teaching_sft 内化行为可训）→ HRL 检索（train_retrieval_recall 已训 0.938）→ HCA 召回（GatedFusionMLP 扩容门控 0.625 破 585 瓶颈）→ 实时可用→ 睡眠固化（inquiry_consolidation CA1 门调速+三元奖励 RL+防错误固化）。统一 checkpoint（pilot_0p1b_gdn2_10k_unified）全链已训强度验证。
    - **门控副作用根治**（fb1 P0）：记忆层条目迁移（memlayer_internalization_e2e.py，**根治**——in-context 0.688=基线零干扰，结构上无 gist 门控被波及）；解耦双通道门控（tri_attention_decoupled.py，注入召回隔离 0.625，副作用权衡）；彻底解耦（tri_attention_fully_decoupled.py，**诚实负结果**——0.1B 注入召回依赖扩容门控整体开权重状态，无法拆进独立 csa 通道复刻）；记忆层路径是正确方向（读出接口未训，召回待训）。
    - **NIAH 长度扫描**（fb1 P1）：eval_niah_length_scan.py（512→4096×keys 8/32×双判据批量扫描）；**max_seq=1024 是真实架构硬限**（RoPE 缓存仅 1024 行，>1024 需扩容）；0.217 低值=GDN 状态饱和+first-token 判据过严双重。
    - **动态 tokenizer**：concept_slot 真实启用（kaplan_extract.py 真实 Kaplan 内词典提取，ℓ3 实测最强），接入自我学习闭环（与求知知识块同存 BlockStore，载体边界：concept_slot=位置不变向量 vs 知识块=token 寻址）。
    - **fb1 学术报告评审交叉验证**：整体可信（文献全核实），3 处措辞修正（TIAR 概念混用/校准漂移归因/副作用机理错配）；P0–P3 critical path（记忆层根治 P0、PM-stream 吞吐 P0、思考核接入 P1、校准 P1、NIAH 长度扫描 P1）。
    - **0.5B 训练管线（2026-07-30）**：3B tokens 多领域数据集（data/shards_0p5b：fineweb_edu 70%/NuminaMath 15%/cosmopedia 10%/中文 5%，断流重试加固）+ 0.5B 配置（512.8M，d1280×24，Muon）+ 双卡手动 DP（train_dp.py 线程化重叠 3.1k tok/s +24%）+ **Muon×WSD set_lr 修复**（Muon 组 lr 此前恒值不衰减）+ 单卡 PRO 4000 训练中。**P1 校准达标**：锚集扩充 AUROC 0.845/0.829（预测反馈循环诚实负结果，双臂择优保存 A 臂）。**Unsloth 评估诚实负结果**（仅支持标准 HF 架构，自研 GDN-2+三级栈不适用）；替代加速：FP8 _scaled_mm 双卡可用 ~1.7×、torch.compile 评估中。
    - **关键文档**：架构详图 v2.2（IBM Carbon）、0.1B 学术报告 v1.0（IEEE 引用）、数据集选型、知识内化训练、主动求知闭环、架构接入状态评估、思维能力强化设计文档（第二阶段）。
  - **当前阶段**：0.5B 预训练进行中（22900 步 ≈ 3B tokens，单卡 PRO 4000）；P1 校准 ✅ 达标；③ **上下文扩充工程已落地**（2026-07-31：RoPE 缓存扩容至 256K + YaRN scaling + extend_context.py 渐进扩窗 + NIAH 复测，见 docs/上下文扩充256K_实施计划.md；0.5B 256K 课程待 0.5B 预训练完成后执行）；推进 ④ 记忆层读出/寻址接口训练（门控副作用根治+召回兼得统一解）⑤ 1.5B 扩展规划。
- **GPU 纪律（2026-07-30 事故教训）**：①训练期间严禁其他进程碰训练 GPU（4070 近满时一个 FP8 基准崩掉 82 分钟训练）；②bench 测速必须 GPU 空闲时做（并发测速致 DP 配比失真慢 35%）；③训练循环改动必须 3 步冒烟再长跑（set_lr 重构 NameError 事故）。
- **首要观测（T1）**：① KAL 探针强度（1.5B 未知）；② 内词典提取强度；③ PM-stream n=5 稳定性（0.1B 已通过）；④ 运行时记忆位置∈{HCA前/HCA后/并行}消融（Part Z）。
- **设计冻结纪律**：Block Spec 是系统的"ISA"——页表、路由器、编译器、缺页处理各自迭代时不得偏离规范；任何里程碑退出标准未达成时，回检设计文档对应章节并修订。

## 6. 文档编写约定

- 文档用**简体中文**撰写，技术术语保留英文原文（如 KV prefix、LoRA、route_key）；
- 论文引用一律标注 arXiv 编号（如 arXiv:2304.13734）或正式出处；
- 架构图优先使用 Mermaid（预览器可渲染），详图为 PNG 并与 Markdown 同目录存放、以相对路径引用；
- 文档头部带版本号与日期，版本演进在文中以"v0.x 新增/修订"标注；文档为"活文档"，随讨论迭代，更新时保持与配套文档的交叉引用一致；
- 文档明确区分"已有研究证据"与"本设计独创部分"——写作或修改设计内容时不得把未验证的设想表述为已验证事实。

## 7. 安全与设计红线（来自设计文档与《部件实现详细计划》红线总表，实现时必须遵守）

- **读写不对称**：运行时仅允许 W0–W2 写原语（日志追加、steering vector、KV prefix / 记忆层 delta 写），绝不触碰正在运行的权重；W3+（LoRA 梯度更新、合并入主干）仅离线睡眠期执行且需审计；
- **页保护位**：人格块运行时只读，元数据块写入需验证门，知识块可写，draft 日志区隔离；**（沙箱例外：EXP-PERSONA 极其实验——KAL 道德约束块 MCB 作为强制闸门时分级开放人格写通道，规范见总体实施计划 §8A；实验外红线维持原状，MCB 自身永不可写）**
- **防记忆投毒**：知识块带防篡改签名；draft→固化之间必须有验证门（校验集回归测试）；源代码形态是最终审计与回滚依据，编译产物可随时废弃重建；**注入即攻击面（MemoryGraft 已实证，时间解耦）——离线筛查不可省**；
- **冲突不静默覆盖**：版本号 + 时间戳 + 置信度三路仲裁，冲突未决时保留双方并标注分歧；
- **诚实降级**：缺页/超时时代理应明确声明"该部分记忆暂不可用"，而非用空白知识作答；苏醒序列阶段 2 完成后须显式声明"记忆部分加载"状态；
- **监测/执行分置**：探针只读 GDN 输出层，干预头只写 CSA 残差前层——读写不同层，防探针读到自己的干预自激；
- **探针冻结**：不对探针信号加生成损失（防模型把特质重编码到不可读基底）；
- **Part Z 交叉干扰**：运行时学习（W-State/≤W2）只从 CSA/HCA 输出读取、或以独立 KV 分支注入，绝不改动任何冻结学习型压缩器下游所依赖的残差；
- **载体能力边界**：token 寻址载体（KV/记忆层）能事实召回，位置不变向量（ICV/steering）只能 steer 行为；Block Spec 必须标 `factual_recall` 字段；
- **HRL 梯度隔离**：辅助损失梯度只进 Indexer/路由器，禁止污染主干；
- **固化纪律**：W4/词表升格用与预训练同优化器 Muon + 谱修剪（intruder 维度）；`<|recall|>` 必须显式出现在 CoT 中（审计接口）；跨设备 reserved 槽命名空间须中心协调。
