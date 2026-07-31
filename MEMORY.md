# MEMORY.md — TAIS Obsidian 迁移状态总档（2026-07-30）

> **用途**：本文件是跨机器迁移的交接总档。新机上的 agent 请先读本文件，再读 `AGENTS.md`（项目约定）与 `docs/memories/README.md`（46+1 份记忆索引）。
> 本机（旧工作站）：Windows + Git Bash，RTX PRO 4000 Blackwell SFF（24GB，sm_120）+ RTX 4070 Laptop（8GB，sm_89）。
> 训练已于 2026-07-30 傍晚主动停止（0.5B 仅跑到早期步数，**无 0.5B checkpoint 损失——新机从头训即可**）。

## 0. 三十秒速览

- 项目：自研纯 PyTorch LLM 研究框架（GDN-2 线性注意力 + 三级检索注意力 + PM-stream + KAL 元认知 + HRL 检索），目标 1.5B 原生 1M 上下文自学习验证机；当前在 **1B/10B tokens Colab 训练**阶段。
- **最新主线（2026-07-31）**：放弃本地双卡路线，转 **Google Colab G4（RTX PRO 6000 Blackwell 96GB）** 一次跑完 1B 预训练 9B + 中训练退火 1B，训完上传 HuggingFace。**全部前置已就绪**——
  - 代码：P0/P1 审阅修复全落地（437 pytest 全绿），tokenizer 随权重、export_final.py、resume 加固、generate 守卫、`--init_from` 中训练初始化；
  - 配置：`configs/pilot_1b_gdn2.json`（实测 1017.7M，d1536×32，Muon）；
  - 数据：`scripts/prepare_data_1b.py`（10B 四源流式、断点续跑、max_id 扫描，全量 ETA 3-5h）；
  - 执行：`notebooks/TAIS_1B_Colab.ipynb`（28 cells 纯 Python，逐 cell AST 校验）；
  - 计划详情：`docs/Colab_1B训练计划.md`（含事实核查冲突点处置）。
- **此前完成**：3B 数据集、0.5B 配置（512.8M）、双卡 DP（3.1k tok/s）、Muon×WSD set_lr 修复、**P1 校准达标 0.845/0.829**、RoPE 扩容初版（rope_scaling none/yarn + extend_context.py）、Unsloth/FP8/torch.compile 评估。
- **1B 训练后任务**：①推理测试（generate/val/KAL 探针迁移）→ ②RoPE 渐进扩窗到 256K → ③记忆层读出训练 → ④ lm-eval 生态接入（需 auto_map/PreTrainedModel 化）→ ⑤ 1.5B 规划。
- 云端启动：Colab 用 `notebooks/TAIS_1B_Colab.ipynb`；普通新机用 `scripts/cloud/README.md`。

## 1. 本次会话关键产出与数据（新机继续的依赖）

### 1.1 3B tokens 数据集（data/shards_0p5b，5.6GB，必传）
- `scripts/prepare_data_0p5b.py`：fineweb_edu 70%（2100M）/ NuminaMath-CoT 15%（450M）/ cosmopedia 10%（300M，the-stack-v2 gated 替代）/ FineWeb2-HQ cmn_Hani 5%（150M）；val 10M；60 train shards + 1 val shard（50M tok/片，uint16 <u2，与 0.1B 同格式）。
- 复用 32k BPE（data/tokenizer/tokenizer.json，vocab 32773，**不重训**）；EOT(id 0) 拼文档。
- 实测：`data/shards_0p5b/_stats.txt`（47 min 收满 3B）。已加固断流重试（12 次×指数退避）；`data/raw/`（3.5GB parquet 缓存）可复用也可重下（非必传）。
- 加载器：`src/tais_obsidian/data/memmap.py` 的 `Shards`（`get_batch` 已右移对齐 (x,y)，**勿再 [:,:-1]**）。

### 1.2 0.5B 配置（configs/pilot_0p5b_gdn2.json）
- **512.81M 参数**（实例化确认）：d_model 1280 / 24 层 = 6×{G2,G2,G2,A}（18 GDN-2 + 6 三级栈）/ mlp 3072 / 注意力头 20q:5kv、GDN 头 20v:10qk（GVA 2:1）/ head_dim 64 / vocab 32768 tied / seq 1024 / gdn_decay_g_min=-5（有界 decay）。
- Muon：矩阵组 470.8M（muon_lr 0.02）+ 非矩阵 AdamW 组 42.0M（lr 7e-4）。
- **max_steps 22900 ≈ 3B tokens ÷ 131k tok/step（单卡 micro 16×accum 8×1024）**；若用双卡 DP（184k tok/step）需改 ~16000 步。
- train.py 新钩子 `build_model_config(cfg)` 读模型尺寸字段（缺省回退 0.1B，向后兼容）。

### 1.3 训练实测数据（新机标定基准）
| 项 | 数值 |
|---|---|
| 单卡 PRO 4000（micro 16） | 2.4–2.6k tok/s，峰值 14.5GB（micro 32 反而 2.0k/22.5GB） |
| 双卡 DP（train_dp.py，线程化重叠+时间均衡 accum） | **3.1k tok/s（+24%）**，worker(4070) micro 2 上限/峰值 6.8GB |
| 串行 DP（无重叠） | 2.1k（负收益，勿用） |
| FP8 `_scaled_mm` matmul 基准 | 4070: 52.8 vs bf16 29.8 TFLOPS；PRO 4000: 85.9 vs 51.2（~1.7×，未集成训练） |
| torch.compile（llm 环境 triton 3.6） | **30 分钟未完成编译被掐**（自研模型图复杂，未得结果，慎试） |
| Unsloth 2026.3.17（conda env `llm` 已装） | **诚实负结果**：仅 monkey-patch 标准 HF 架构（llama/qwen/gemma 等），自研架构结构性不适用 |
| 0.5B 训练（中断前） | v1 双卡 DP 50 步 loss 10.63→9.14（gnorm 19.6→7.4 正常）；后续因事故/切换重启，无 checkpoint 留存 |

### 1.4 Muon×WSD set_lr 修复（已落地，新机直接受益）
- Bug：Muon 组读 `muon_lr`/`adamw_lr` 键，旧循环只设 `g["lr"]` → Muon 全程恒 lr、WSD warmup/decay 失效（长跑致命）。
- 修复：`src/tais_obsidian/train.py` 的 `set_lr(opt, lr, cfg)`（AdamW 组设 lr；Muon 组按 lr/peak 比例缩放 muon_lr、adamw_lr 跟随 lr）；train.py 与 train_dp.py 均已改用；test_muon 8 项全绿。

### 1.5 P1 校准达标（0.769 → 0.845/0.829）
- `scripts/kal_truth_finetune_v2.py` + `scripts/diverse_truth_data_v2.py` + `tests/test_kal_calibration_v2.py`（6 项新增，全套 10 项全绿）。
- **锚集扩充（A 臂，最终保存）**：双口径 AUROC 脚本 n200 = **0.845±0.024**（3 seed 最低 0.816）/ 测试 n400 = **0.829±0.007**（最低 0.820）；certainty 方向 known P(known)≈1.000 / fake 0.129；val loss 微调前后 diff=0.0（主干 detach 红线）。
- **预测反馈循环（B 臂）诚实负结果**：候选打分式反馈无 OOD 增益（0.843/0.823），双臂择优回滚保存 A 臂；收益或在行为层（TIAR 拒答），留作 T1 议题。
- checkpoint：`checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2/`；报告 `runs/kal_truth_v2/report.json`。
- 坑：from_pretrained 前先 `attach_kernel()`（kernel=None 加载坑）；评估 @torch.no_grad()；锚集多样性是全部杠杆（near-miss 细粒度错误+跨域混搭+领域伪事实）。

### 1.6 GPU 事故纪律（三次真实事故，新机同样适用）
1. **训练期间严禁其他进程碰训练 GPU**——4070 近满时一个 FP8 基准崩掉 82 分钟训练（exit 1 无 traceback，CUDA 上下文级死亡）。
2. **bench 测速必须 GPU 空闲时做**——并发测速致 DP 配比失真（worker accum 50 vs 正确 28），全程慢 35%。
3. **训练循环改动必须 3 步冒烟再长跑**（`--max_steps 3`）——set_lr 重构 NameError 在 step 1 才炸，AST 查不出。

## 2. 仓库状态与传输清单

### 2.1 git 状态提醒（迁移前建议在旧机提交）
未提交改动：`AGENTS.md`（M）、`docs/memories/`（新增 47 份）、`configs/pilot_0p5b_gdn2.json`、`scripts/{prepare_data_0p5b,train_dp,_dp_grad_check,_bench_torch_compile,kal_truth_finetune_v2,diverse_truth_data_v2}.py`、`src/tais_obsidian/train.py`（build_model_config+set_lr）、`tests/test_kal_calibration_v2.py`、若干 `scripts/_probe_*.py`（一次性探测，可不传）。**建议旧机 `git add -A && git commit` 后新机 clone/pull；或按下表 rsync。**

### 2.2 传输优先级
| 优先级 | 路径 | 大小 | 说明 |
|---|---|---|---|
| 必需 | `src/ configs/ scripts/ tests/ docs/ article_ref/ AGENTS.md MEMORY.md pyproject.toml` | <1GB | 代码+文档+记忆（不含 .git 也可，建议含 .git） |
| 必需 | `data/shards_0p5b/` | 5.6GB | 0.5B 训练数据（3B tokens） |
| 必需 | `data/tokenizer/` | 4.5MB | 32k BPE |
| 必需 | `data/shards/` | 229MB | 0.1B 数据（扩窗/微调/KAL 用） |
| 建议 | `checkpoints/pilot_0p1b_gdn2_10k/` | 1.6GB | RoPE 扩窗底座 |
| 建议 | `checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2/` | ~1.6GB | P1 校准达标内核 |
| 建议 | `checkpoints/pilot_0p1b_gdn2_10k_unified/` | ~1.6GB | 全链统一 checkpoint |
| 可选 | `checkpoints/pilot_0p1b_gdn2_10k_kaltruth/`、`pilot_0p1b_gdn2_10k_teaching/` | ~3GB | 校准 v1/教学 SFT |
| 可选 | `data/raw/` | 3.5GB | parquet 缓存（可重下，不传则数据脚本自动重下） |
| 不传 | 其余 checkpoints（消融历史）、runs/（报告）、`articles/`（论文原文） | — | runs 报告体积小可酌情带（kal_truth_v2/report.json 建议带） |

### 2.3 环境要点（新机）
- Python 3.12（uv venv）；`uv pip install -e .`；torch 必须 **cu128+**（若新机是 Blackwell/sm_120 必需；Hopper/Ada 可 cu126+，按 `nvidia-smi` 的 CUDA 版本选 wheel）。
- 依赖：torch、numpy、tokenizers、datasets、huggingface_hub、tensorboard、pytest（见 pyproject.toml）。
- HF 直连不稳时数据脚本前缀 `HF_ENDPOINT=https://hf-mirror.com`；HTTP GET 直下 parquet 的路径已内建于 prepare_data_0p5b.py。
- 多卡注意：**torch 设备序 ≠ nvidia-smi 序**，用 `torch.cuda.get_device_name(i)` 确认后再设 CUDA_VISIBLE_DEVICES。

## 3. 新机任务队列（详细）

### 任务 ① 1B 预训练+中训练（最高优先，Colab 执行）
**已被 2026-07-31 的 1B/Colab 计划取代 0.5B 本地路线**（0.5B 配置/数据/双卡 DP 全部保留可复用）：
执行载体 `notebooks/TAIS_1B_Colab.ipynb` + `docs/Colab_1B训练计划.md`。要点：G4 RTX PRO 6000（96GB sm_120，torch 必须 cu128）；10B tokens（shards_1b，prepare_data_1b.py 流式制备 3-5h）；预训练 34300 步 Stable（decay_frac=0）→ 中训练 3800 步 decay_frac=1.0（--init_from 独立退火 run，退火混合 math 40%/code 20%）；checkpoint 每 500 步 + 15 min Drive 同步（断连一等公民）；训完 export_final → 模型卡 → upload_folder 上传 HF（library_name 显式声明自研格式）。**10B 对 1B 是研究性欠训（Chinchilla 20B/当代 4T+），模型卡已如实标注**。
<details><summary>（存档）0.5B 本地训练命令</summary>

```bash
# 单卡（>24GB 显存可调大 micro_batch，但注意 micro 32 在 24GB 上反而降速，需重新标定）
python -u -m tais_obsidian.train --config configs/pilot_0p5b_gdn2.json > logs_train_0p5b.txt 2>&1
```
- 验收：loss 从 ~10.6 正常下降、gnorm <20 收敛；22900 步 ≈ 3B tokens。
</details>

### 任务 ② RoPE 扩容+NTK → 256K（**已部分落地**，继续推进）
- 根因：`src/tais_obsidian/model/tri_attention.py` 的 `rope_cos/rope_sin` 缓冲按 `cfg.max_seq`（1024）构建，`_rope(k,0)` 全量重算 → >1024 越界。`rope_theta=10000`（config.py:28）。
- 架构事实：RoPE 只用于滑窗 L0 分支（512 窗，绝对位置）；CSA/HCA 分支 NoPE；GDN-2 无 RoPE → **扩容负载只在滑窗分支**。
- **已实现**（中断子代理产出，主代理验收 20 项 pytest 绿）：config.py 增 `rope_scaling`/`rope_scale`/`rope_original_max_seq`（默认 none/1.0 与旧版逐 bit 一致、断点兼容）；`rope_scaling="none"` 仅扩缓存行数——滑窗注意力分数只依赖相对距离 ≤tri_window，RoPE 相对性保证数学上精确，扩窗是纯工程解除硬限；`"yarn"` 为 YaRN 逐维 ramp 插值（供 1.5B CSA partial-RoPE 及消融）。`scripts/extend_context.py`（渐进扩窗微调）+ `scripts/bench_long_seq_cost.py`（长 seq 成本实测）+ `tests/test_rope_extension.py`。
- **待做**：extend_context.py 扩窗实测（4K→16K→64K→256K，底座 pilot_0p1b_gdn2_10k 或 0.5B checkpoint）+ NIAH 复测（2048/4096 first-token 从 0.000 变非零即工程成功）+ `docs/上下文扩充256K_实施计划.md` + tri_attention_decoupled/fully_decoupled 变体副本同步检查。
- 参考：docs/memories/niah-length-scan-gate-adaptive.md、设计文档 §3、docs/update/k3_tech_report.md（K3 渐进课程）。

### 任务 ③ 记忆层读出/寻址接口训练（统一最优解候选）
- 背景：记忆层条目迁移已根治门控副作用（in-context 0.688=基线零干扰，token 寻址可事实召回），但**读出接口未训、召回待训**；目标 = 无副作用（0.688）+ 召回强（对齐 KV 0.625）兼得。
- 参考：docs/memories/memlayer-internalization.md、fully-decoupled-gate.md（诚实负结果：扩容门控不可拆）、gated-fusion-mlp.md。

### 任务 ④ 0.5B 上重测 KAL 校准
- 0.5B checkpoint 出来后：kal_probe 复测 ℓ 读点（0.1B 是 ℓ10 最优）→ kal_truth_finetune_v2 迁移 → 期望探针强度随规模上升（设计 §29 预测）。

### 任务 ⑤ 1.5B 扩展规划
- 基于统一能力基线 + 数据集选型（docs/memories/dataset-research.md，OLMo 课程）：28 层 d2048、词表 129280（127232+2048 reserved）、30B tokens、Muon、T0–T5。

## 4. 历史基线数据（0.1B，勿丢）
- hybrid 基线 val **3.768**；E+-5 PM-stream 3.744；E+-7 三级栈 3.762；组合 3.743；KAL 探针 ℓ8 AUROC 0.945（语义空白子集 0.979）。
- 吞吐（0.1B，micro 16×accum 4×seq 1024）：hybrid 9.5k tok/s（峰值 7.0GB）、PM-stream 3.0k、三级栈 8.6k、组合 2.9k；生成 37.8 tok/s。
- 统一 checkpoint 全链强度：KAL 0.769→**0.845/0.829（v2）**、HRL 检索 0.938、HCA 召回 0.625、in-context 0.688。
- NIAH：GDN-2 10k 0.207 vs GDN-1 0.177；max_seq=1024 硬限（扩容前）；512/1024 长度 first-token 0.04–0.12（判据过严+状态饱和双因素）。
- Muon vs AdamW：6.523 < 6.868（收敛更好），吞吐开销 4.6%。

## 5. 文档地图（全部中文，事实来源）
- `AGENTS.md`：项目约定（结构/命令/红线/M0–M8 链）。
- `docs/memories/README.md` + 47 份记忆：路线（next-steps-roadmap.md）、**0p5b-training-dual-gpu-dp.md（本轮全细节）**、kal-gdn2-truth-finetune.md、fb1-feedback-verification.md、dataset-research.md、hardware-dual-gpu.md、memlayer-internalization.md 等。
- `docs/`：细致框架设计文档 v2.5（Why）→ 子系统架构规格 v1.0（怎么组合）→ 接口与实现计划 v1.0 + 部件实现详细计划 v1.0（怎么写代码）→ D0_0p1B先导实验报告、0.1B 学术报告 v1.0。
- `article_ref/`：5+2 簇论文笔记（注意力压缩/动态 tokenizer/记忆自编译/元认知安全/神经科学/HRL checkpoint 决策/KAL 数学规范）。

---
*由 Kimi Code CLI 会话整理（2026-07-30 晚）。旧机训练已停、GPU 已释放。祝新机顺利。*
