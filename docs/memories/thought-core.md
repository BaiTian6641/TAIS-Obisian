# CTM 式思考核 pilot（第二阶段迭代③，2026-07-28 落地）

## 产出
`src/tais_obsidian/model/thought_core.py`（284 行）+ `tests/test_thought_core.py`（221 行，12 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 201 绿）。**pilot 独立模块，不接 model.py 主干**（迭代④才形式化进推理循环）。

## 核心组件（CTM arXiv:2505.05522 两原理抽取做小核，不全网套用）
- **ChannelGroupHistory**：通道组级活动历史（CTM 原理①降维，非逐神经元）。core_dim=384÷G=8 组（每组 48 维），每组近 H=4 tick 历史 FIFO（左零填充定长，最新在 dim=-2 末尾）。
- **ThoughtTimeRotary**：思考时间相位化（CTM 原理②复用 RoPE）。同 tri_attention half-split NeoX 构造（inv_freq/cos/sin buffer），但**维度语义=思考 tick 非 token 位置**——第 k tick 施加第 k 相位步进，对候选增量（非残差主干）旋转，fp32 关键路径。
- **ThoughtCore.forward_step**：history.update→组历史平坦化→逐组 group_mlp（Linear(H·48→48)+GELU+Linear(48→48)）→候选增量→use_sync 时相位化→残差循环（state+cand）。
- **think**：多 tick 循环。每 tick forward_step→可选 bridge.tick 驱动流形位移写 PM-stream→certainty_fn(state)>stop_threshold(0.9) 早停（CTM 式自适应算力）。history.reset() 每轮前清空。返回（最终状态, 轨迹 list, stop_tick）。

## 关键设计决策
- **自消融开关 use_sync**（默认必选项，诚实边界）：True=思考时间相位化（CTM 同步代理）/False=纯 MLP 残差循环。**CTM 语言域零证据 [降预期]**（论文 §12 自认 future work+民间复现负面）→ 若消融无差异应回检设计修订。test_f 验证同种子同输入两路轨迹分叉。
- **certainty_fn 接口**：pilot 简化（state→float），迭代④接真实 KAL P(IK)。
- **bridge 集成可选**（integrate_bridge，默认 False）：think 内 bridge.tick(state, target, alpha) 驱动流形位移写 PM-stream（detach 由 bridge 内部保证，W1–W2 零梯度快写）。
- **写纪律**：bridge.tick 写 PM-stream 是 steering 有界加法，绝不触碰权重；to_hidden 走离线目标。

## 迭代④ 推理循环形式化（2026-07-28 落地，pilot 独立编排）
`src/tais_obsidian/model/reasoning_loop.py`（311 行）+ `tests/test_reasoning_loop.py`（298 行，11 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 212 绿）。**pilot 独立编排模块，不接 model.py 主干**（正式接入属后续 milestone）。
- **ReasoningLoop**（持有 thought_core/bridge/kernel，复用迭代①③）：reasoning_tick 按 §1.3 五步——① GDN 状态读出（state 即持续状态）→ ② glimpse 观察 [pilot 占位 mean-pool+obs_proj Linear，正式应接 CSA/TriRetrieval 注意力] → ③ hrl_propose [kernel.route_candidates top-k [B,1,k]；无 kernel/candidates 时 None 接口位] → ④ kal_certainty [有 kernel：sense(state).pik_logits→softmax→known 类(类0)概率末 token 均值，no_grad 只读；无 kernel：mock sigmoid(norm)∈[0,1]] → ⑤ thought_core.forward_step → ⑥ bridge.tick 位移写 PM。
- **run**：多 tick 循环，certainty>stop_threshold(0.9) 早停（CTM 式自适应算力，循环层回填 early_stop），thought_core.history.reset() 每轮前清空。返回（最终 state, 轨迹, stop_tick）。
- **ReasoningTickState**（审计）：tick_index/current_coord/disp/certainty/hrl_topk_idx/early_stop/recall_triggered。
- **trajectory_to_recall_tokens**：空白 tick（certainty<recall_threshold=0.3）显式标 `<|recall|>`，非空白标 `<|tick_k|>` 占位——对齐"`<|recall|>` 必须显式出现在 CoT"红线。
- **红线**：监测/执行分置（sense 只读 detach / bridge.tick 只写 PM）；梯度边界（steering detach，仅 group_mlp 反传）。
- **待接**：正式接 CSA 注意力 glimpse + KAL isotonic 校准 P(IK) + HRL CA3 PPR 联想（orchestrator.associative_recall），0.1B pilot 消融。

## 关联记忆
- /memories/repo/thinking-manifold-layer.md（迭代①流形层+桥接）
- /memories/repo/verified-literature-thinking-manifold.md（CTM 语言域零证据/网格码不涌现/HRPO U形）

---
*导出自 /memories/repo/thought-core.md（2026-07-30 同步快照）。*
