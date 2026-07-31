# Colab 1B 训练转向（2026-07-31，当前主线）

> 取代 0.5B 本地双卡路线（0.5B 资产全部保留可复用）。执行载体 = `notebooks/TAIS_1B_Colab.ipynb`；计划详情 = `docs/Colab_1B训练计划.md`。

## 决策与依据
- **硬件**：Google Colab G4 = RTX PRO 6000 Blackwell（96GB，sm_120）[已核实存在，付费档]；torch 必须 **cu128+**（cu126 无 sm_120 内核）。
- **目标**：1B（实测 1017.7M，d1536×32 = 8×{G2G2G2A}，mlp 4096，24q:8kv / 24v:12qk）预训练 9B + 中训练退火 1B = 10B tokens，上传 HF 后直接推理测试。
- **欠训声明**：10B/1B 低于 Chinchilla（20B）一半、远低于当代 1B 实践（4T–11T）[已核实] → 研究 pilot 定位，模型卡如实标注。
- **中训练模式**：Stable（decay_frac=0）→ 独立退火 run（`--init_from` 仅权重、step=0、新优化器、decay_frac=1.0 线性到 0、退火混合 math 40%/code 20%）= SmolLM2 多阶段 WSD / OLMo Dolmino 同构 [已核实]。

## 数据（scripts/prepare_data_1b.py）
- 配比：fineweb_edu **sample-100BT** 73% / math 12%（NuminaMath ~430M + **FineMath-4+** 补足）/ cosmopedia 10% / FineWeb2-HQ cmn 5%。
- 工程：HTTP GET 直下 parquet→本地流式、断流重试、原子写、**断点续跑（_progress.json）**、**max_id 全量扫描 <32768**、Drive 缓存可选。
- 本机冒烟：30M 四源全通、max_id 32767、续跑跳过验证、**10B 全量 ETA 3-5h**。
- 词表坑：tokenizer 32773 > 模型 32768（5 个保留特殊 token id≥32768），generate 已加越界 assert；prompt 勿含其字面量。

## 审阅修复（437 pytest 全绿，主代理复验）
- P0：tokenizer.json 随权重产物 + generate 三级回退；Colab cu128 自检；`scripts/export_final.py`（latest.pt→save_pretrained）。
- P1：OOM 折半不重建优化器（此前静默丢 Muon 动量）；generate 守卫（超长/越界 id/max_seq 触顶）；resume 加固（RNG 切片 + **map_location ByteTensor 修复——GPU resume 此前必炸** + model_cfg 13 字段校验）；build_model_config 补透传（pm_sk_t_max 曾是死旋钮）；attach_kernel 同步标志；reconfigure 守卫 46 处；冒烟脚本死字段；extend_context --resume；train.py --data_dir/--init_from/--micro_batch/--grad_accum；writer try/finally。
- 新回归测试 13 项：set_lr WSD×Muon、resume e2e（逐 bit 接续）、非默认尺寸往返、generate 守卫、export_final 全链。

## Colab 执行（notebook 28 cells 纯 Python，逐 cell AST 校验）
配置（GIT_URL 或 Drive zip + HF_TOKEN secret）→ GPU 自检（非 cu128 换装后**必须重启运行时**）→ 依赖 → 代码 → Drive 持久化（15min ckpt 同步循环）→ 数据（幂等续跑）→ 3 步冒烟 → micro 标定（打印 ETA）→ Phase-1 预训练（断连自动 --resume）→ Phase-2 中训练 → 推理自验 → 模型卡 → upload_folder 上传。断连恢复 Runbook 在 notebook 末尾。

## 遗留（后续工程）
- lm-eval 生态接入：需 auto_map+trust_remote_code 或 PreTrainedModel 化（自研 save_pretrained 格式 HF AutoModel 不可直接加载）。
- RoPE 256K：初版已落地（rope_scaling none/yarn + extend_context.py + 20 测试绿）；渐进扩窗（4K→16K→64K→256K，YaRN 主流已核实）待 1B 之后执行。
- tests/ 旧文件 reconfigure 未加守卫（仅 notebook import 触发）；muon per_head_qkv 语义粗糙（默认关）。

---
*写入自 2026-07-31 会话（Kimi Code CLI）。*
