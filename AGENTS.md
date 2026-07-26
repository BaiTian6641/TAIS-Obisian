# AGENTS.md — TAIS Obsidian（tais-obsidian）

> 本文件面向 AI 编码代理，假定读者对本项目一无所知。全部项目文档为中文，与文档相关的工作请使用中文。

## 1. 项目概览

**TAIS Obsidian**（泰斯人工智能复合体 · 黑曜石框架）是一个处于**概念设计阶段**的研究项目，目标是构建一个"面向持续学习 LLM 的权重虚拟内存"架构，并最终从零训练一台 **1.5B 原生 1M 上下文的自学习验证机**。

核心设想：为大语言模型提供一个动态接入外部权重的接口（类比操作系统的页表/虚拟内存），使模型在推理过程中能以"知识块"（KnowledgeBlock）为单位按需换入/换出知识与经验，而不是依赖外挂 markdown / RAG 文本。关键设计点：

- **双形态知识块**：markdown 源代码形态（ground truth，可审计）+ 编译产物形态（KV prefix / LoRA / steering vector，可失效重建）；编译在离线"睡眠时间"进行。
- **存储层级**：L0 VRAM（工作记忆）↔ L1 DRAM（短期记忆）↔ L2 NVMe（长期记忆）↔ L3 远端（档案）。
- **双向接口（读写不对称）**：运行时只读 + 只写 W0 日志；W3 以上写原语（LoRA 梯度更新、合并入主干）仅离线执行。人格块运行时只读。
- **冷启动"苏醒序列"**：参考麻醉苏醒研究，按"路由器/接口层 → 人格块+元数据块 → 高频陈述性块 → 长尾惰性加载"的顺序恢复。
- **元认知（KAL）**：内部探针检测知识空白（SAPLMA 证据），模型主动"回想"而非盲目猜测。
- **TAIS Obsidian 1.5B 模型**：28 层 = 7 × {3 GDN-MemBlock + 1 CSA-AttnBlock}（Gated DeltaNet : 全注意力 = 3:1），hidden 2048，词表 129280 tied embedding，原生 1M 上下文；KAL/HRL/知识块注入点为 checkpoint 内生部件（非外挂服务）。

模型谱系：TAIS Obsidian 1B / 1.5B /（远期）4B-A1B。本机硬件（2026-07-24 到位）：**RTX PRO 4000 Blackwell SFF（24GB，sm_120，70W）** + RTX 4070 8GB 副卡（显示/杂务，不用于训练），驱动 596.36（CUDA 13.2）——PyTorch 须用 **cu128+** wheel（cu126 无 sm_120 内核）。

## 2. 仓库当前状态（2026-07-24 更新）

**代码已落地**：设计文档（`docs/`）+ 自研训练/推理框架首个可运行版本（D-0 级 0.1B 先导实验，纯 PyTorch，无 triton，Windows 原生可跑）。

**本机工作区状态**：全新 clone——`.venv` / `data/` / `checkpoints/` / `runs/` 尚未在本机创建；环境重建、数据准备、0.1B pilot 重启按《从零构建TAIS-Obsidian_总体实施计划.md》§6.6（S0→S1→S2）逐步执行。旧 4060 笔记本的 D-0 基线不迁移。

### 2.1 代码结构

- `src/tais_obsidian/`：框架包（`uv pip install -e .` 后可 import）
  - `config.py`：ModelConfig dataclass + JSON 读写
  - `model/`：`attention.py`（CSA：RoPE+GQA+QK-Norm+SDPA，KV cache）、`gdn.py`（GDN：naive_recurrent + chunked 双路径，纯 PyTorch）、`model.py`（`TaisObsidianForCausalLM`，tied embedding，自研 `save_pretrained`/`from_pretrained`，不依赖 transformers 建模；`forward(capture_layers=...)` hidden-state 捕获挂点）、`blockpath.py`（CSA 块通路原型：stride-4 压缩器 + 块 KV 收割/注入 + namespace fail-closed，设计 §11.1）、`pmstream.py`（E+-5 PM-stream：mHC 多流残差 arXiv:2512.24880，`ModelConfig.pm_stream=5` 启用，默认 1 单流零改动；恒等初始化与单流基线 <1e-6）、`kal.py`（E+-3 KAL 分层元认知头：L1 P(IK) 三态 W[d,3] + L2 valence/arousal W[d,2]，nn.Linear 可随 state_dict 存取；设计 §8.3-1/§16.2，探针管线见 scripts/kal_probe.py）
  - `data/memmap.py`：uint16 bin shard 读写与 batch 采样；`tokenizer_io.py`：tokenizer 封装
  - `train.py`：训练循环（bf16 autocast + fp32 参数、AdamW 分组、WSD 调度、grad clip 1.0、checkpoint/resume、tensorboard；config JSON 加 `"pm_stream": 5` 可开 PM 消融）
  - `generate.py`：cache 增量生成（temperature/top-k）
- `configs/pilot_0p1b.json`：0.1B pilot 配置（12 层 = 3×{3 GDN + 1 CSA}，hidden 768，vocab 32768）；`configs/pilot_0p1b_pm.json`：PM-stream 消融变体（同基线 + `pm_stream: 5`）
- `scripts/`：`check_env.py`（环境自检）、`prepare_data.py`（FineWeb-Edu → 训 32k BPE → 120M tokens shards）、`smoke_overfit.py`（双变体过拟合冒烟）、`smoke_overfit_pm.py`（PM-stream 变体过拟合冒烟）、`kal_probe.py`（E+-3 KAL 探针管线：L1 已知/未知 AUROC + FLARE 基线对比、L2 emotion→valence/arousal，输出 `runs/kal_probe/report.json`）
- `tests/`：`test_gdn.py`（GDN 两路径对拍 <1e-4）、`test_cache.py`（增量 vs 整段一致性、save/load 往返）、`test_capture.py`（hidden-state 捕获挂点）、`test_blockpath.py`（CSA 块通路机制）、`test_pmstream.py`（PM-stream：恒等初始化/稳定性/反向/save-load/增量/捕获）、`test_kal.py`（KAL 头形状/管线 smoke/report schema）——均 pytest 可收集
- 不入库：`data/`（tokenizer + shards）、`checkpoints/`、`runs/`、`logs/`

### 2.2 常用命令

先 `source .venv/Scripts/activate`（Git Bash；venv 由 uv 创建，Python 3.12）。**工作站（Blackwell sm_120）装 torch 用 `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`**；cu126 wheel 不含 sm_120 内核，不可用。**双卡注意：torch 设备序 ≠ nvidia-smi 序——cuda:0 = RTX 4070（8GB 副卡），cuda:1 = RTX PRO 4000（24GB 计算卡）；所有训练/测试/生成命令必须前缀 `CUDA_VISIBLE_DEVICES=1`**（单卡视图下 PRO 4000 即 cuda:0）。HF 直连不稳定时数据脚本前缀 `HF_ENDPOINT=https://hf-mirror.com`。

- 环境自检：`python scripts/check_env.py`
- 数据准备：`python scripts/prepare_data.py`
- 冒烟测试：`python scripts/smoke_overfit.py`
- 单元测试：`python -m pytest tests/`（未装 pytest 时直接 `python tests/test_gdn.py`）
- 训练：`python -m tais_obsidian.train --config configs/pilot_0p1b.json --run_name <name>`（续训加 `--resume checkpoints/<run>/latest.pt`）
- 推理：`python -m tais_obsidian.generate --ckpt checkpoints/<run>/final --prompt "..."`

工作站吞吐基线待 S2 实测（历史参考：4060 Laptop 8GB 训练 ~1.8k tok/s、生成 ~43 tok/s）。**不要臆造**未列出的命令。所有实现工作应严格遵循 `docs/` 中的路线图（见 §4）与《从零构建TAIS-Obsidian_总体实施计划.md》的阶段检查点。

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

## 4. 计划中的技术栈（来自路线图，尚未落地）

Phase 0（基础设施，第一步）规划的技术选型：

- **语言**：Python（ML 工程惯例，文档未显式声明但路线图的栈均为 Python 生态）；
- **推理栈双轨**：vLLM（serving + 运行时动态加载 LoRA，`/v1/load_lora_adapter`）+ HuggingFace Transformers（hooks 捕获 hidden state，用于探针训练）；
- **存储**：SQLite（页表/元数据）+ 向量库（route_key 检索）+ 文件存储（块载荷）；
- **实验底座模型**：主力 Qwen3.5-9B，管线调试用 Qwen3.5-4B，对照 Qwen3-8B（全注意力，用于分离混合架构效应）；冻结 vision tower，纯文本实验；
- **训练**：BF16 mixed（bf16 计算 + FP32 主权重/优化器/梯度累积，禁止纯 bf16）、8-bit AdamW、grad clip 1.0、QK-Norm；OLMo 3 数据课程（Dolma 3 Mix / Dolmino / Longmino、Dolci 后训练套件）；
- **硬件**：开发机 = RTX PRO 4000 Blackwell SFF（24GB，sm_120）已到位（2026-07-24），9B 底座 bf16 推理（~19GB）与 1.5B 机制短训本机解锁；正式 1.5B 预训练走多卡/云端；1M 长上下文阶段需云端短租多卡。

**GDN 层关键约束**（写代码时必须遵守）：GDN 层无 KV cache，KV prefix 注入只适用于占 1/4 的全注意力（CSA）层；LoRA 注入各层通用。知识块的"载体适用性"必须按层类型标注。

## 5. 开发流程约定

- **分阶段推进，每阶段有退出标准，不达标不进下一阶段**（避免在沙地上盖楼）：
  - Phase 0 基础设施 → 退出标准：跑通"提问→捕获 hidden state→落盘"端到端管线；
  - Phase 1 知识空白探针（最高优先级，首个实验）→ 退出标准：中间层线性探针 AUROC ≥ 0.8 且显著优于 token 概率、自报置信度两个基线；
  - Phase 2 读通道（页表+路由+注入）→ 注入后目标领域准确率显著提升、切换开销 < 思考段生成时间的 20%；
  - Phase 3 写通道与睡眠固化 → 完成"记录→固化→次日无文本提示直接答对"闭环；
  - Phase 4 闭环与进化（RL 路由、联想路由、人格块、长期运行实验）。
- **下一步行动**（见路线图 §8）：环境搭建（vLLM + HF 双栈、下载权重、验证 hooks 管线）→ Phase 1 数据集"已知/未知"事实构造协议 → Block Spec v0.2（落实载体适用性字段）→ Phase 1 实验方案细稿。
- **设计冻结纪律**：Block Spec 是系统的"ISA"——页表、路由器、编译器、缺页处理各自迭代时不得偏离规范；任何 Phase 退出标准未达成时，回检设计文档对应章节并修订。

## 6. 文档编写约定

- 文档用**简体中文**撰写，技术术语保留英文原文（如 KV prefix、LoRA、route_key）；
- 论文引用一律标注 arXiv 编号（如 arXiv:2304.13734）或正式出处；
- 架构图优先使用 Mermaid（预览器可渲染），详图为 PNG 并与 Markdown 同目录存放、以相对路径引用；
- 文档头部带版本号与日期，版本演进在文中以"v0.x 新增/修订"标注；文档为"活文档"，随讨论迭代，更新时保持与配套文档的交叉引用一致；
- 文档明确区分"已有研究证据"与"本设计独创部分"——写作或修改设计内容时不得把未验证的设想表述为已验证事实。

## 7. 安全与设计红线（来自设计文档，实现时必须遵守）

- **读写不对称**：运行时仅允许 W0–W2 写原语（日志追加、steering vector、KV prefix 追加），绝不触碰正在运行的权重；W3+（LoRA 梯度更新、合并入主干）仅离线执行且需审计；
- **页保护位**：人格块运行时只读，元数据块写入需验证门，知识块可写，draft 日志区隔离；**（沙箱例外：EXP-PERSONA 极其实验——KAL 道德约束块 MCB 作为强制闸门时分级开放人格写通道，规范见总体实施计划 §8A；实验外红线维持原状，MCB 自身永不可写）**
- **防记忆投毒**：知识块带防篡改签名；draft→固化之间必须有验证门（校验集回归测试）；源代码形态是最终审计与回滚依据，编译产物可随时废弃重建；
- **冲突不静默覆盖**：版本号 + 时间戳 + 置信度三路仲裁，冲突未决时保留双方并标注分歧；
- **诚实降级**：缺页/超时时代理应明确声明"该部分记忆暂不可用"，而非用空白知识作答；苏醒序列阶段 2 完成后须显式声明"记忆部分加载"状态。
