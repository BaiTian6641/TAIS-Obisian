# AGENT PLAN — E+-7 三级注意力栈（滑窗 + CSA 选择检索 + HCA 重压缩）

> **用途**：交 GLM5.2（GitHub Copilot）同步推进/交叉验证核心算法正确性。本文档自包含。
> **日期**：2026-07-25 ｜ **状态**：实现进行中
> **对应**：《TAIS_Obsidian_细致框架设计文档》§2（v1.3 注意力栈行）、§6（主干）、§17（v1.3 全节）；《从零构建TAIS-Obsidian_总体实施计划.md》§7.5 E+-7。

---

## 1. 目标

把当前作为"CSA 占位"的**全注意力层**（`CSAAttention`：RoPE + GQA + QK-Norm + SDPA 因果）升级为设计文档 v1.3 的**三级注意力栈**（DeepSeek V4 式混合压缩注意力，NSA 谱系），作为 **config 可选开关**（默认全注意力，基线数值路径零改动），并在 0.1B 上消融对照已确立的 hybrid 基线（val loss 3.768 @ 2000 步）。

## 2. 设计依据（必须逐条对齐，禁止凭记忆实现）

1. **NSA（arXiv:2502.11089）**：三分支 = 压缩表示（块压缩 KV，全因果可见）+ top-n 块选择检索 + 滑窗精确注意力，学习门控融合；64K 解码 11.6×。**实现前必须联网核对**：压缩块构造（块长/步长/重叠）、重要性分数来源（压缩注意力分数 vs 独立 indexer）、门控形式、位置编码在压缩分支的处理、训练时选择操作的梯度路径（top-k 不可微的处理——原文是"选择只影响前向的值聚合、分数本身参与压缩注意力训练"还是辅助 loss）。
2. **DeepSeek V4 混合压缩注意力（2026-04-24，设计 §17.1–17.3 已核实）**：CSA + **HCA（128 tokens → 1 KV entry）** + 滑窗三件套；1M 上下文 10% KV cache + 27% 单 token FLOPs。**我们的 CSA ≡ DeepSeek CSA（独立命名收敛，§17.1）**。
3. **设计文档的项目化约束**：
   - 三级与 L0/L1/L2 存储一一对应：滑窗 512（L0 精确）/ CSA（L1 情景，stride-4 压缩 + indexer top-128）/ HCA（L2 gist，128:1）；
   - **HCA 区是知识块注入的原生落点**（§17.3：注入即"读自己写的东西"，前缀偏差从结构上消失）——本原型只在结构上保证"HCA 条目可被外部条目前置拼接注入"（注入函数单独提供，复用 E+-4 的 namespace/fail-closed 纪律），注入训练留后续；
   - GDN 层无 KV cache 的红线不变：三级栈只替换 "A" 层；"G" 层不动。

## 3. 代码现状（已核实的关键接口）

- `src/tais_obsidian/model/attention.py`：`CSAAttention(cfg)`，`forward(x, state=None, offset=0)` → `(out, new_state)`；state = {"k","v"}（[B, n_kv, T, hd]，k 为 RoPE 后）。GQA 12Q/4KV × 64（0.1B 配置），QK-Norm，half-split RoPE（buffer rope_cos/sin，按 offset 索引）。
- `src/tais_obsidian/model/model.py`：`Block`（mixer + SwiGLU MLP，pre-norm RMSNorm）；`ModelConfig.layer_types`（"G"/"A"）；cache = {"pos","layers"}。
- `src/tais_obsidian/config.py`：`ModelConfig` dataclass（vocab 32768、d 768、12 层 GGGAGGGA、max_seq 1024）。
- `src/tais_obsidian/model/blockpath.py`（E+-4）：外部 stride-4 压缩器 + namespace 五元组 + fail-closed——本任务的三级栈让压缩**内生于层**，blockpath 的 namespace/fail-closed 纪律复用。
- 基线：hybrid 2000 步 val 3.768（9.5k tok/s，峰值 7.02GB，seed 42，64k tok/step，配置 `configs/pilot_0p1b.json`）；PM 变体消融进行中（§7.5 E+-5，与本项并行不悖——三级栈与 PM-stream 是正交开关）。

## 4. 实现规范（验收以此为准）

1. **开关**：`ModelConfig` 新增（如 `attn_impl: str = "full"`；`"tri"` = 三级栈）。默认 `"full"`，既有 checkpoint/train/generate/全部测试行为零改动；`attn_only=True` 控制组始终全注意力。`train.py` 按 cfg 透传（参照既有 `pm_stream` 接线）。
2. **新模块**（建议 `src/tais_obsidian/model/tri_attention.py`，不改动 `attention.py` 数值路径）：三分支——
   - **滑窗分支**：最近 w=512 token 的精确注意力（全量 k/v 的尾部窗口；可用 SDPA + 窗口掩码，或与选择分支共用展开的 GQA）；
   - **CSA 分支**：stride-4 学习压缩器（建议复用/对齐 `blockpath.CSACompressor` 结构）把 k/v 压成 T/4 条目；轻量 indexer（per-query 打分，O(L)）对压缩条目打分取 **top-128**（仅因果集合内）；query 对选中压缩条目做注意力；**压缩条目因果性**：块尾位置 j 的压缩条目只对 >j 的 query 可见；
   - **HCA 分支**：128:1 重压缩（每 128 token → 1 KV entry，T=1024 → 8 条），对所有因果内 query 恒可见（gist 性质）；同样学习压缩器；
   - **融合**：学习门控（per-head，自 q 产生，sigmoid， init 均等 1/3——记录选择）合并三分支输出；GQA/QK-Norm/RoPE 纪律与现 CSAAttention 一致（压缩/HCA 条目的位置处理按 NSA 原文核对结果执行并注释）。
3. **注入接口**：`inject_hca_entries(state, entries, namespace)` 形式（前置拼入 HCA 区 + namespace 五元组 fail-closed，复用 blockpath 的 check_namespace/NamespaceMismatchError）；本任务只做结构与校验单测，注入训练不做。
4. **KV cache（生成路径）**：允许原型级实现——cache 保存全量 k/v，每步由全量 cache 现算三个分支（O(L)/token，0.1B/seq 1024 可接受）；**docstring 必须注明**：生产路径是增量维护压缩/HCA cache（V4 的 10% KV 正由此来），原型的现算方式不影响前向数值语义。增量与整段前向一致性测试必须通过（<1e-4，对齐 test_cache 判据）。
5. **测试**（`tests/test_tri_attention.py`，pytest 可收集）：
   a) 形状：三分支各自与融合输出形状正确（T 与 T=1 两种）；
   b) **因果性（红线）**：扰动位置 j 之后的 token，位置 ≤j 的输出逐点不变（三分支分别 + 融合后）；
   c) 选择合法性：top-k 索引全部落在因果压缩集合内；
   d) 滑窗分支等价性：窗口内与"加掩码的全注意力"参考逐点一致；
   e) HCA 注入：namespace 全对放行 / 任一字段不匹配 fail-closed；注入后 HCA 区长度与簿记正确；
   f) 增量 vs 整段一致性（含注入后）；
   g) save/load 往返。
6. **过拟合烟测**：参照 `scripts/smoke_overfit.py` 模式，tiny 三级栈配置（GGGA，"A" 层换三级栈）在固定真实 batch 上 300 步，final loss（末 10 步均值）<0.1。
7. **消融**（主代理执行）：`configs/pilot_0p1b_tri.json`（基线配置 + `attn_impl: "tri"`）2000 步同数据/种子/步数，对照 val 3.768；记录吞吐/显存与 KV 占用对比（全注意力 vs 三级栈——seq 1024 下 KV 优势不显著属预期，1M 才是主战场，如实记录）。
8. **参数增量**：压缩器×2 + indexer + 门控应 <5%（0.1B 配置）；实测填入报告。

## 5. 红线与纪律

- 不得修改 `CSAAttention`/`GDNBlock`/`Block` 单流默认路径的任何数值行为；三级栈只经 config 开关生效。
- NSA/V4 机制细节必须联网核对原文（arXiv:2502.11089 为主；V4 技术报告为辅），引用以注释落到代码；不可得处标"推断实现"+理由。
- top-k 选择的训练梯度处理按 NSA 原文执行（若原文用 straight-through/辅助 loss/无梯度，照抄并注释）；禁止自创梯度路径。
- 压缩/HCA 条目的 RoPE 处理按原文；原文未规定处选择并注明（推断实现）。
- 验证命令（Git Bash，仓库根）：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（torch 在 `.venv`，cu128；PRO 4000 为该视图下 cuda:0；**注意消融 run 可能在占卡，测试模型保持 tiny、秒级**）。

## 6. 交付物

- `src/tais_obsidian/model/tri_attention.py`（+ config/model/train 最小接线）
- `tests/test_tri_attention.py`、`scripts/smoke_overfit_tri.py`（可并入现有 smoke 参数化亦可）
- 消融结果（val 对照 + 吞吐/显存/KV 占用）→ 回填实施计划 §7.5 E+-7 行
