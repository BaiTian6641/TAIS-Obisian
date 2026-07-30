# 思考核接入主干前向 + 推理增益（P1，从模块到原生，2026-07-30）

## 产出
`src/tais_obsidian/model/thought_core_integration.py`（151 行，ThoughtCoreIntegration）+ model.py 接入（attach_thought_core + forward(use_thought_core=False 默认关）+ `scripts/thought_core_e2e_eval.py`（247 行）+ `tests/test_thought_core_integration.py`（213 行，10 项全绿）。子代理实现，我验收（report 数据确认+独立重跑+全量 392 绿）。统一 checkpoint 基座。

## 接入方式
- **接入点**：最终 norm_f 前——对主干最末层 hidden [B,T,768] 做 ThoughtCore.think 多 tick 演化，残差加回。
- **维度桥接**：down_proj 768→384 / up_proj 384→768。
- **可选开关**：use_thought_core 默认 False（未挂载也跳过），须 attach_thought_core+True 才走核路径，**向后兼容（默认关 392 测试零改动）**。
- **有界演化**：max_ticks=8 + certainty 早停（核状态范数 sigmoid 占位，正式接 KAL P(IK)）+ **zero-init 输出门（ReZero 同族）**——随机核 gate=0 恒等，门可学离线训练打开。
- **梯度边界**：detach_backbone=True 默认（核输入 detach，思考核梯度不回主干——HRL 梯度隔离红线）；核参数正常反传。

## ⭐ 推理基准增益（真实跑出，达 fb1 原生门槛）
- 基准：多步链式推理 chain（build_chain 2 跳），n_eval=64。
- 无核 **0.828** / 有核（已训）**0.906** → **增益 +0.078**（>1 点，**思考核挣到 FLOPs，达 fb1"哪怕 1-2 点"门槛**）。
- **稳健性**：4 个 seed chain 增益 +0.078/+0.078/+0.078/+0.125（**一致正非样本偶然**）；fact 单事实迁移 +0.000/+0.021（无害，排除模板捷径破坏）。
- **随机/未训核增益恰 0**：zero-init 门恒等是结构保证（dist_core≈dist_no_core 的结构根因修复点）——增益须来自已训核（离线训练打开门），不臆造未训核增益。
- gate 学负值（-0.060）但 loss 降答对率升（tanh 门方向由训练定，非缺陷）。

## 意义
**第二阶段思维能力强化从"通路验证"（dist_core≈dist_no_core 未挣到 FLOPs）到"能力证明"（接入主干前向推理增益 +0.078~+0.125）的关键一跳**——思考核从 pilot 模块迈向原生部件（达 fb1 从模块到原生的唯一门票）。

## 诚实边界
①增益依赖训练（步数/样本/seed）；②基准是 0.1B 小样本（n=64）合成 chain；③certainty 早停是核状态范数占位（未接真实 KAL P(IK)）；④1.5B/真实任务需复验；⑤CTM 语言域零证据（思考核只证"改变动力学+小样本增益"，未证大规模更优）。

## 子代理踩坑（验收记录）
①build_chain 的 Answer 被 tokenizer 与 prefix 合并编码（仅答案区 mask 全空 CE nan）→改全序列 next-token CE（主干 frozen 只有核路径可反传，监督聚焦答案）；②bf16 autocast+深层 tick 反传不稳→训练改 fp32；③zero-init tanh 门（比 detach 更优，门可学是挣 FLOPs 的结构开关）。

## 待接
①certainty 早停接真实 KAL P(IK)（isotonic 校准）；②增益在更大样本/真实任务复验；③思考核接入推理循环（reasoning_loop 用接入的核）；④1.5B 扩展时思考核内生进 28 层主干。

---
*导出自 /memories/repo/thought-core-integration.md（2026-07-30 同步快照）。*
