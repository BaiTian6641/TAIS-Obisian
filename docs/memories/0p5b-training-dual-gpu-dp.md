# 0.5B 训练启动 + 双卡 DP + 3B 数据集（2026-07-30）

> 本轮三任务（用户指示）：①3B tokens 数据集+0.5B 训练（双卡）；②P1 校准；③256K 上下文扩充。本文件记 ① 的落地。

## ① 3B tokens 数据集（data/shards_0p5b，已验收）

- `scripts/prepare_data_0p5b.py`（前轮 Copilot 会话起草，本轮修复+加固+验收）：
  - 修复 1：删除未实现的 `--resume` 占位 flag（隐患，防误导）；
  - 修复 2：`list_parquet` 改**按 config 子目录轮询选片**（原字母序前 N 片会让 cosmopedia 全落同一 config，领域单一）；
  - 加固 3：**断流重试**——首次全量跑因 SSL EOF 四个源全崩（网络瞬断），改为流中断后指数退避（15s×n，上限 300s）重建流续收，max 12 次；重建流从源头重读，引入 <0.1% 重复文档（预训练可接受）。
- 配比（OLMo 课程对齐）：fineweb_edu 70%（2100M）/ NuminaMath-CoT 15%（450M）/ cosmopedia 10%（300M，the-stack-v2 gated 401 替代）/ FineWeb2-HQ cmn_Hani 5%（150M）；val 10M 独立（各源头部取 val_tokens/4）。
- **实测**：3B tokens / 60 train shards + 1 val shard / 47 min（本地 parquet 流 2000k tok/s，fineweb 原生流式）；复用 32k BPE（不重训），EOT 拼文档，uint16 <u2 与 prepare_data.py 同格式；Shards 加载+解码抽验通过。
- 访问要点：fineweb-edu 直连流式可用；其余三源 HTTP GET 直下 parquet → 本地 `load_dataset("parquet", streaming=True)`（hf-mirror 不暴露 auto-converted parquet 分支）。

## ② 0.5B 配置（configs/pilot_0p5b_gdn2.json，已验收）

- **512.81M 参数**（实例化确认）：d_model 1280 / 24 层 = 6×{G2,G2,G2,A}（18 GDN-2 + 6 三级栈）/ mlp 3072 / 头 20q:5kv（注意力）、20v:10qk（GDN GVA 2:1）/ head_dim 64 / vocab 32768 tied / seq 1024 / gdn_decay_g_min=-5（有界）。
- Muon：矩阵组 470.8M（muon_lr 0.02）+ 非矩阵 AdamW 组 42.0M（lr 7e-4）。
- **max_steps 16000 ≈ 3B ÷ 184k tok/step**（DP 全局 batch；原 23000 按单卡 128k 口径已修正，防 1.4 epoch 过训）。
- train.py 新增钩子 `build_model_config(cfg)`：从 JSON 读模型尺寸字段（vocab_size/d_model/n_layer/头数/mlp_hidden/max_seq/check_0p1b_params），缺省回退 0.1B——**train.py 原本不读尺寸字段（硬编码 0.1B），0.5B 必须加钩子**，向后兼容（84 项相关 pytest 全绿）。

## ③ 双卡 DP（scripts/train_dp.py，已验收）

- **Windows 无 NCCL → 单进程手动 DP**（不走 torch.distributed）：master=cuda:1（PRO 4000 24GB）、worker=cuda:0（4070 8GB）；各卡按 token 占比加权 backward → worker 梯度 fp32 经 pinned+side stream D2D 搬回 master 累加 → clip 1.0 → Muon step → 参数广播回 worker（与下一步 master 计算重叠，event 链隔离读写窗）。
- **WorkerNode 线程化 + 时间均衡 worker 独立 accum**（收益关键）：慢卡小 micro×多 accum ≈ 快卡大 micro×少 accum，两卡每步耗时对齐；4070 从 11% 贡献提到 **30%**。
- 实测：单卡 2.5k → 串行 DP 2.1k（负收益）→ **重叠 DP 3.1k tok/s（+24%）**；bench 自动测速定 worker micro/accum（4070 上限 micro 2，峰值 6.83GB 留 >1GB 显示余量，OOM 自动折半重试）；bench 在双卡无收益时会明确建议回退单卡。
- 正确性：`_dp_grad_check.py` 梯度等价 PASS（Δloss 4.8e-07，224 参数最大相对误差 ~1.4e-04，主代理复跑确认）；0.1B 30 步同种子 DP vs 单卡 loss 逐点贴合；50 步稳定性（显存恒定 M 15.6GB/W 6.8GB 无 hang）。
- 用法：`python -u scripts/train_dp.py --config configs/pilot_0p5b_gdn2.json --bench 5`（**勿设 CUDA_VISIBLE_DEVICES**；cuda:0=4070/cuda:1=PRO4000）；checkpoint 只存 master，格式与 train.py 一致（`generate --ckpt` 已验证可加载）。

## ④ Muon×WSD 隐患修复（顺手防御）

- **Bug**：Muon 组读 `muon_lr`/`adamw_lr` 键，train.py 循环只设 `g["lr"]` → Muon 全程恒 lr，WSD warmup/decay 完全失效（0.1B 短跑没暴露，23k 步长跑末段无衰减会显著影响收敛）。
- **修复**：train.py 新增 `set_lr(opt, lr, cfg)`——AdamW 组设 `g["lr"]`，Muon 组按 WSD 比例（lr/peak）同步缩放 `muon_lr`、`adamw_lr` 跟随 lr；train.py 与 train_dp.py 均改用。单测验证（muon 缩放 + adamw 直通）+ test_muon 8 项全绿。

## ⑤ 0.5B 正式训练（进行中）

- 命令：`python -u scripts/train_dp.py --config configs/pilot_0p5b_gdn2.json --bench 5 > logs_train_0p5b.txt 2>&1`（后台任务 bash-qk8vbd0b，2026-07-30 启动）。
- bench 实测 worker accum 28（波动 28~30），global batch 184k tok/step；**16000 步 ≈ 3.08B tokens，ETA ~11 天**（61s/step）；ckpt_every 1000（~17h/个），val_every 500。
- 启动即验证：loss 10.63 正常、gnorm 19.1、双卡显存 M 13.5GB/W 6.8GB。

## ⑥ 2026-07-30 下午续：单卡切换 + Unsloth 评估 + 事故记录

- **P1 校准已达标**（承接 kal-gdn2-truth-finetune.md）：`scripts/kal_truth_finetune_v2.py`（锚集扩充臂 A）双口径 AUROC **0.845（脚本 n200）/ 0.829（测试 n400）**，3 seed 均值≥0.8 最低 0.816；**预测反馈循环（B 臂）诚实负结果**——候选打分式反馈无 OOD 增益（0.843/0.823），按双臂择优回滚保存 A 臂（`checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2/`）；val loss 微调前后 diff=0.0；10 项 pytest 复验全绿。反馈循环收益或在行为层（TIAR 拒答策略）而非探针 AUROC，留作 T1 迭代议题。
- **事故 1（GPU 抢占崩训练）**：双卡 DP 训练跑 82 分钟时，主代理在 4070（显存 6.8/8.6GB 近满）上跑 FP8 基准，直接崩掉训练（exit 1，log 无 traceback——CUDA 上下文级死亡）。**纪律：训练期间严禁任何其他进程碰训练 GPU，尤其近满的 4070**。
- **事故 2（bench 配比失真）**：DP 训练 v2 重启时 P1 校准任务正在 PRO 4000 上跑，bench 测得 master 1.9k（被拉慢）→ worker accum 误判 50（正确 28）→ 全程 worker 失衡慢 ~35%。**纪律：bench 必须在 GPU 空闲时测速**。
- **事故 3（set_lr 重构 NameError）**：set_lr 替换循环里删了 `lr` 变量绑定，但日志行仍引用 → step 1 即 NameError。AST 查不出。**纪律：训练循环改动后必须 3 步冒烟（--max_steps 3）再长跑**。
- **单卡标定**：micro 16 = 2.4k tok/s（峰值 14.5GB）为最优；micro 32 反而 2.0k（显存 22.5GB，占用换不到吞吐）。
- **Unsloth 评估（诚实负结果）**：本机确有 unsloth **2026.3.17**（conda env `llm`，`C:\Users\Tass\.conda\envs\llm`，python 3.11 + torch 2.12.0.dev+cu128 + triton 3.6，CUDA 可用）。但 unsloth 的加速机理是 monkey-patch 特定 HF 模型类的 forward（`unsloth/models/`：llama/llama4/mistral/qwen2/qwen3(+moe)/gemma/gemma2/cohere/granite/falcon_h1/glm4_moe/vision/sentence_transformer）——**自研 TaisObsidianForCausalLM（GDN-2+三级栈+PM-stream）不在其列，结构性不适用**，无法用 unsloth 跑 0.5B 训练。
- **替代加速路线**：FP8 `_scaled_mm` 双卡可用（matmul 基准：4070 52.8 vs bf16 29.8 TFLOPS；PRO 4000 85.9 vs 51.2，~1.7×）；torch.compile（llm 环境 triton 3.6）对自研模型的实测见 `scripts/_bench_torch_compile.py` 结果。
- **torch.compile 实测**：llm 环境（triton 3.6 + torch 2.12 nightly）对 0.5B 模型编译 30 分钟未完成被掐（自研模型图复杂/疑似 hang），未得结果——该路线不优先。
- **当前训练**：已于 2026-07-30 晚主动停止（迁移新机）。0.5B 仅跑早期步数、无 checkpoint 损失，新机从头训。**交接总档见仓库根 `MEMORY.md`，云端脚本见 `scripts/cloud/`（setup_env/sync/resume/README 四件，bash -n 全过）**。双卡 DP 脚本 train_dp.py 保留（3.1k tok/s 已验证可用；用户决定新机先用单大卡排除 PCIe 变量）。

1. **网络瞬断杀全源**：SSL EOF 会让 datasets 流直接 RuntimeError，首次全量跑四源全崩 → 断流重试是长任务必需（12 次×退避）。
2. **train.py 不读模型尺寸字段**：加 build_model_config 钩子（向后兼容）。
3. **Muon 组 WSD 失效**：set_lr 修复（见 ④）。
4. **4070 只装得下 micro 2**（0.5B）：靠时间均衡 accum 而非大 micro 提贡献。
5. **worker 线程必须 set_device + 异常经 Queue 回传主线程**（不静默）；`p.grad is None` 参数跳过累加。
6. 编码：Windows 控制台 GBK，脚本一律 `sys.stdout.reconfigure(encoding="utf-8")` + `PYTHONIOENCODING=utf-8`。

## 待接

- P1 校准 0.769→≥0.8（先在 0.1B _kaltruth 开发，0.5B checkpoint 出来后迁移）；
- RoPE 扩容+NTK（max_seq 1024→256K，渐进扩窗 4K→32K→256K，双卡）；
- 0.5B 训练完成后：val 终值 vs 0.1B 基线、KAL 探针强度复测、记忆层读出训练（统一最优解候选）。

---
*写入自 2026-07-30 会话（Kimi Code CLI）；对应数据准备 bash-ps2xtz5w、训练 bash-qk8vbd0b。*
