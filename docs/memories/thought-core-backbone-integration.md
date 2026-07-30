# 思考核接入主干前向 + 推理增益验证（fb1 P1，2026-07-30 落地）

**fb1 硬约束达成**：思考核从 pilot 独立模块（dist_core≈dist_no_core 未挣到 FLOPs）→ 接入 model.py 主干前向，多步推理基准实测正增益。

## 产出
- `src/tais_obsidian/model/thought_core_integration.py`（151 行）：ThoughtCoreIntegration（down_proj 768→384 + ThoughtCore + up_proj 384→768 + zero-init 门 gate_alpha）。
- `model.py`：attach_thought_core() + forward(use_thought_core=False 默认关/thought_core_max_ticks)；接入点=最终 norm_f 前。
- `scripts/thought_core_e2e_eval.py`（247 行）：有核/无核基准对照 demo（含离线训练打开门）。
- `tests/test_thought_core_integration.py`（213 行，10 项全绿）；全量 392 绿（382+10）。
- 报告：`runs/thought_core_e2e/thought_core_e2e_report.json`。

## 关键数据（禁止臆造，实测）
- 基准：多步链式推理 chain（build_teaching_data.build_chain，2 跳）。
- 无核 0.828 / 有核（已训）0.906 → **增益 +0.078**（>fb1 门槛 1-2 点）。
- 稳健性：4 seed chain +0.078~+0.125（一致正）；fact 迁移 +0.000/+0.021（无害，排除模板捷径）。
- 训练：loss 5.84→2.84（200 步），gate zero-init→-0.060（tanh 门可正可负，方向由训练定）。

## 核心设计决策
1. **zero-init 输出门（ReZero 同族）= dist_core≈dist_no_core 结构根因修复**：随机核 gate=0 → 前向恒等（增益恰 0 是结构保证非挣到）；门可学 → 离线训练打开 → 思考增量流入 logits。**未训核增益≈0 是诚实通路验证，增益须来自已训核**。
2. **接入点 = 最终 norm_f 前**：思考核作"最终表征精炼器"（logits 前多 tick 演化）。
3. **有界演化**：max_ticks=8 + certainty 早停（状态范数 sigmoid 占位，正式接 KAL P(IK)）+ tanh 有界门。
4. **梯度隔离双保险**：detach_backbone=True（核输入 detach）+ 训练时主干全 frozen（requires_grad=False）——HRL 隔离红线。

## 踩坑
- **build_chain 的 `Answer: {A}` 被 tokenizer 与 prefix 合并编码**（n_prefix==len(ids)），仅答案区 mask 全空→CE nan。改全序列 next-token CE（主干 frozen 只有核路径可反传，重点自然落答案 token）更稳。
- 训练用 **fp32（关 autocast）**：思考核是 fp32 关键路径，bf16 autocast + 深层 tick 反传不稳。
- 基准选型：chain（2 跳，无核 0.83 有 headroom）优于单事实 fact（失败多为 0.1B 容量/判对口径，非思考可解决）。

## 诚实边界
- 增益依赖训练（步数/样本/seed）；测试只验方向（有核≥无核−0.05），不臆造阈值。
- 未训/随机核增益≈0（zero-init 恒等）——已在 demo 明确标注非"挣到"。
- gate 学负值（-0.060）但 loss 降、答对率升：tanh 门方向由训练定，非问题。

## 关联记忆
/memories/repo/fb1-feedback-verification.md（P1 硬约束）、/memories/repo/thought-core.md（pilot 模块）、/memories/repo/unified-checkpoint.md（统一 checkpoint）。

---
*导出自 /memories/repo/thought-core-backbone-integration.md（2026-07-30 同步快照）。*
