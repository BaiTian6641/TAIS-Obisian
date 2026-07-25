# AGENT PLAN — E+-5 PM-stream（mHC 多流残差）实现与消融

> **用途**：交 GLM5.2（GitHub Copilot）同步推进/交叉验证核心算法正确性。本文档自包含，读者无需其他上下文。
> **日期**：2026-07-24 ｜ **状态**：实现进行中（主代理子代理同步实现，本文档为算法判据）
> **对应**：《TAIS_Obsidian_细致框架设计文档》§12.2 / §13.4；《从零构建TAIS-Obsidian_总体实施计划.md》§7.5 E+-5。

---

## 1. 目标

在自研 0.1B 混合架构 LLM（GDN:CSA = 3:1，纯 PyTorch，Windows 原生）中，把**单流残差**改造为 **mHC 式多流残差（n=5：4 内容流 + 1 感知-记忆流 PM-stream）**，作为**可选的 config 开关**（默认关闭，基线数值路径零改动），并用消融实验证明：相对已确立的 hybrid 基线（val loss 3.768 @ 2000 步）**不劣化且信号放大受控**。

## 2. 设计依据（必须逐条对齐）

1. **mHC（Manifold-Constrained Hyper-Connections，arXiv:2512.24880）**：Hyper-Connections（arXiv:2409.19606）把残差流扩展为 n 条并行流、每层用可学习混合矩阵做跨流读写；无约束时深层信号放大峰值可达 ~3000×；mHC 把混合矩阵用 **Sinkhorn-Knopp 迭代投影到 Birkhoff 多胞形（双随机矩阵）**，放大压到 ~1.6×，27B 规模验证、开销 6.7%。**实现前必须联网核对原文公式**（H_pre/H_post 的参数化、Sinkhorn 迭代次数、约束施加在哪些矩阵上、残差映射是否含非负约束）。
2. **设计文档 §12.2 / §13.4 的项目化约束**：
   - n=5：4 条内容流 + **1 条 PM-stream**（感知-记忆流）；
   - **读**：KAL 各头统一从 **GDN-MemBlock 输出处的 PM-stream** 读取（GDN 状态压缩特性使其输出最适合做"已理解内容"的摘要信号）；
   - **写**：HRL 注入经 H_post 映射写入 **CSA-AttnBlock 残差前的 PM-stream**；人格向量等干预共用此单一写入纪律；
   - 初始化必须**恒等于单流残差**（恒等初始化是消融公平性与训练稳定的前提）；
   - 稳定性指标：沿深度的信号放大 ≤ ~1.6×（设计红线，对齐 mHC 原文）。

## 3. 代码现状（已核实的关键接口）

- `src/tais_obsidian/model/model.py`：`TaisObsidianForCausalLM.forward(input_ids, cache=None, capture_layers=None)`；`Block.forward(x, state=None, offset=0)` 返回 `(x, new_state)`，内部 `x = x + mixer(norm1(x))`，`x = x + mlp(norm2(x))`（pre-norm RMSNorm + 单流残差）。`Block.type ∈ {"G","A"}`。
- `src/tais_obsidian/config.py`：`ModelConfig` dataclass；`layer_types` 展开 block_pattern。
- 训练：`src/tais_obsidian/train.py`（bf16 autocast + fp32 参数、WSD、grad ckpt 逐 Block、`ModelConfig(attn_only=cfg["attn_only"])` 以默认值构建）。
- 推理 cache：`cache = {"pos": int, "layers": [state]}`；增量生成逐 token。
- 基线：hybrid 2000 步 val loss **3.768**（train 3.58，9.5k tok/s，峰值 7.02GB，seed 42，64k tokens/step，FineWeb-Edu 118M tokens，配置 `configs/pilot_0p1b.json`）。

## 4. 实现规范（验收以此为准）

1. **开关**：`ModelConfig` 新增字段（如 `pm_stream: int = 1`；1 = 现状单流，5 = 4 内容 + 1 PM）。默认 1，**既有 checkpoint、train.py、generate.py、全部测试行为零改动**。
2. **结构**：Block 的 mixer/mlp 输出不再直接加回单流，而是：流状态 `S ∈ [B,T,5,d]`；每个子层（mixer 与 mlp 各算一次"块"）按 mHC 方式：读 = H_pre（从 5 流聚合出该子层输入，作用于 norm 后）、写 = H_post（子层输出经双随机约束的混合矩阵分配回 5 流）。具体参数化以 mHC 原文为准并在代码注释逐条引用公式编号。
3. **恒等初始化**：所有混合矩阵初始化为"内容流 0 即原残差、其余流为零"的映射——初始化态的 PM 变体前向必须与同权重单流基线**逐点一致**（<1e-6），这是最强正确性判据。
4. **双随机约束**：Sinkhorn-Knopp 迭代（次数按原文，典型 ~20 次内）作用于跨流混合矩阵；fp32 内计算；约束必须在每次 forward 生效（或按原文的参数化方式保证）。
5. **PM-stream 访问点**：按 §13.4 暴露——`Block` 输出处提供"当前 PM-stream 张量"的可读引用（与 E+-1 `capture_layers` 兼容：捕获内容流的同时可捕获 PM 流）；注入写入点（CSA 残差前）留注释与接口位，HRL 头簇（E+-6）后续接。
6. **兼容性**：grad checkpoint 逐 Block、cache 增量生成（cache 中需含流状态还是仅 pos——选择并记录理由）、`save_pretrained/from_pretrained`、`capture_layers` 全部可用；参数增量 <2%（混合矩阵 5×5×每层每子层量级，很小）。
7. **测试**（`tests/test_pmstream.py`，pytest 可收集）：
   a) 恒等初始化：同种子同基础权重下，PM 变体与单流基线 logits 逐点 <1e-6；
   b) 稳定性探针：随机输入过全部层，测 PM 流与内容流的逐层幅度比，放大 ≤ ~1.6×（或显著低于无约束对照——可做约束开关对照证明 Sinkhorn 有效）；
   c) 反向：loss.backward() 无异常、混合矩阵梯度非零；
   d) save/load 往返；e) 增量生成 3 步形状/簿记正确；f) 捕获 API 兼容。
8. **训练烟测**：复用 `scripts/smoke_overfit.py` 模式（tiny PM 配置过拟合固定 batch 300 步，final loss <0.1）。
9. **消融**（主代理执行）：PM 变体 2000 步、与基线同数据/种子/步数/全局 batch，对比 val loss 曲线（基线 3.768）；记录吞吐/显存开销。

## 5. 红线与纪律

- 不得修改单流默认路径的任何数值行为；`pm_stream=1` 时 forward 逻辑与现状逐行一致。
- mHC 公式必须来自原文核对（arXiv:2512.24880），禁止凭记忆实现核心算法；引用以注释落到代码。
- 明确区分"mHC 原文设计"与"本项目 PM-stream 的独创分配（4+1、读写纪律）"——后者是设计文档的设想，无先例背书，文档与注释中不得表述为已验证。
- 验证命令（Git Bash，仓库根）：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（torch 在 `.venv`，cu128；PRO 4000 为该视图下 cuda:0）。

## 6. 交付物

- `src/tais_obsidian/model/`（PM-stream 实现，文件自定，建议 `pmstream.py` + model.py 最小接线）
- `tests/test_pmstream.py`
- 消融结果（val loss 对照 + 吞吐/显存 + 稳定性数据）→ 回填实施计划 §7.5 E+-5 行与 D-0 报告后续
