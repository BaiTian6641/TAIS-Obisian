# TAIS Obsidian 架构接入状态评估（动态 tokenizer / 思考流形 / 自我学习闭环）

> **版本 v0.1 · 2026-07-29**。回答用户两个检查点：① 动态 tokenizer 启用后是否加入自我学习闭环；② 思考流形（空间流形推理）等是否启用并正确接入模型架构。**诚实标注：区分"已真实接入运行时主干"与"pilot 独立模块（设计预留接口，未接主干）"**。

---

## 1. 总览：三部分接入状态

| 部件 | 实现 | 测试 | **运行时架构接入状态** | pilot 边界 |
|---|---|---|---|---|
| **动态 tokenizer（concept_slot）** | ✅ dyn_vocab.py + kaplan_extract.py | ✅ 6 项全绿 | 🟡 **真实启用**（Kaplan 提取真实 + orchestrator 装配 + concept_slot 注册→HRL 检索→inject 闭环），但**是 orchestrator 可选注入件**，非 model.py 主干默认 | extract_fn 需装配时注入；非主干内生 |
| **思考流形（ThoughtManifold）** | ✅ manifold.py | ✅ 6 项全绿 | 🟡 **pilot 独立模块**——经 manifold_bridge 接 PM-stream（桥接），但**未接 model.py 主干前向**（设计明确 pilot 独立，迭代④才形式化进推理循环） | 独立模块，主干前向不含流形 |
| **CTM 思考核 + 推理循环** | ✅ thought_core.py + reasoning_loop.py | ✅ 12+11 项全绿 | 🟡 **pilot 独立编排模块**——串起 §1.3 tick 动力学，但**不接 model.py 主干**（glimpse/certainty pilot 占位，正式接 CSA/KAL） | 独立编排，certainty/glimpse 占位 |
| **自我学习闭环（主动求知）** | ✅ inquiry_branch + inquiry_executor + inquiry_consolidation | ✅ 15+15+11 项全绿 | 🟡 **闭环逻辑全通**（求知→验证→写入→检索→注入→固化），**运行时经 orchestrator + 真实部件（KAL/HRL/HCA）** | ask_fn/tool_fn pilot mock；实时召回 0.625（扩容门控） |

---

## 2. 动态 tokenizer ↔ 自我学习闭环的连接（检查点①）

**结论：concept_slot 已加入自我学习闭环的知识生态，连接点明确。**

```
自我学习闭环（主动求知）中的 concept_slot：
  KAL 词表摩擦检测（求知触发信号之一）
    → assess_vocab_friction（orchestrator）
    → DynamicVocab.promote：Kaplan 提取 → concept_slot 注册
        ├─ 页表（动态词表 codebook，元数据）
        ├─ BlockStore（向量载荷，可被 Pager 检索）
        └─ HRL route_graph 入图（CA3 PPR 联想检索节点）
    → 后续推理：HRL route_candidates / associative_recall 检索到 concept_slot
    → kernel.inject 向量路径注入（steer 行为/单槽理解）
```

**与求知执行器的关系**：concept_slot 与求知知识块**同存 BlockStore**（知识生态统一），都是 HRL 可检索节点——但**载体不同**（红线）：
- **concept_slot** = 位置不变向量（单槽理解，steer 行为，factual_recall=False）——OOV 概念/专名的"理解"；
- **求知知识块（mem_entry/kv）** = token 寻址载体（事实召回）——求知学到的事实。

**接入状态确认**：concept_slot 经 orchestrator 的 `assess_vocab_friction` + `register_block_to_graph` + `associative_recall` **已接入 HRL 检索生态**（自我学习闭环的检索侧）；真实 Kaplan 提取（kaplan_extract.py）+ 真实装配（dynamic_vocab_real_demo.py）已验证闭环（注册→HRL 入图→CA3 PPR 检索→inject 可用）。

---

## 3. 思考流形/推理循环的架构接入（检查点②）

**结论：思考流形↔PM-stream 桥接真实接通，但思考流形/推理循环是 pilot 独立模块，未接 model.py 主干前向——这是设计明确的 pilot 边界，非遗漏。**

### 已真实接入（桥接）
- **思考流形↔PM-stream 桥接**（manifold_bridge.py）：PM-stream 末位流（思考段载体）↔ 思考流形坐标（几何坐标系）——`ThoughtManifoldBridge.tick`（读 PM 流→流形坐标→位移→反投影→有界写回 PM-stream）**真实接通**（迭代①×③交汇，测试验证位移朝 target）。

### pilot 独立模块（设计预留接口，未接主干）
- **思考流形**（manifold.py）：ThoughtManifoldProjector（共享投影+共形等距）——独立模块，主干前向**不含**流形投影（设计：迭代①pilot 起手，独立训练不触碰主干）。
- **CTM 思考核**（thought_core.py）：通道组历史+RoPE 相位化+certainty 早停——独立模块，**不接 model.py 主干**（设计：迭代③pilot 消融，迭代④才形式化进推理循环）。
- **推理循环**（reasoning_loop.py）：§1.3 五步 tick（GDN 状态→glimpse→HRL 提议→KAL certainty→bridge.tick）——独立编排，**glimpse/certainty 是 pilot 占位**（mean-pool/mock sigmoid），正式应接 CSA 注意力 glimpse + KAL isotonic P(IK)。

### 设计意图（诚实边界）
这些是**第二阶段（思维能力强化）的 pilot 模块**——设计文档 §6 明确"pilot 独立模块不接主干，迭代④/正式 milestone 才形式化进推理循环"。**当前状态符合设计预期**：模块落地+端到端集成验证（thinking_e2e_demo）+真实部件适配（thinking_real_adapter_demo 接真实 GDN/CSA/certainty），但**运行时主干前向仍走 GDN-2+三级栈+PM-stream（第一阶段），思考流形/推理循环是叠加的 pilot 层，非主干内生**。

---

## 4. 接入缺口的诚实清单（不留坑）

| # | 缺口 | 状态 | 影响 | 补救路径 |
|---|---|---|---|---|
| 1 | concept_slot 是 orchestrator 可选注入件，非主干默认 | 🟡 pilot | 需装配时注入 dynamic_vocab+extract_fn | 装配脚本已提供（dynamic_vocab_real_demo.py）；正式可设默认装配 |
| 2 | 思考流形未接 model.py 主干前向 | 🟡 pilot（设计预留） | 主干前向不含流形 | 迭代④形式化进推理循环（设计路线） |
| 3 | 推理循环 glimpse/certainty 是 pilot 占位 | 🟡 pilot | 非真实 CSA glimpse/KAL P(IK) | 真实部件适配已验证通路（thinking_real_adapter_demo），正式接 CSA/KAL isotonic |
| 4 | 求知 ask_fn/tool_fn 是 mock | 🟡 pilot | 未接真实对话/检索工具 | 接真实对话接口/搜索工具 |
| 5 | concept_slot 与求知知识块的检索统一 | 🟡 部分 | 同存 BlockStore 但载体不同（向量 vs token 寻址） | 载体能力边界已标注；检索器按 compiled_kind 路由（已有） |

**关键诚实结论**：**所有"未接主干"都是设计明确的 pilot 边界（独立模块验证概念），非实现遗漏或错误**。各部件闭环逻辑全通（测试验证），且真实部件适配（动态 tokenizer 真实 Kaplan 提取 / 推理循环真实 GDN-CSA-certainty / 求知真实 KAL）已验证通路与真实 KAL 行为。**从 pilot 到主干内生是下一阶段（形式化进推理循环/统一 checkpoint）的工作，非当前缺陷**。

---

## 5. 验证证据（已亲自验收）

- **动态 tokenizer**：test_kaplan_extract.py 6 项全绿（真实 Kaplan 提取，同类相近/不同类相远：electron-photon 0.513 vs electron-democracy 0.217）；dynamic_vocab_real_demo.py 闭环（注册→HRL 入图→CA3 PPR 检索→inject）。
- **思考流形↔PM-stream 桥接**：test_manifold_bridge.py 6 项全绿（tick 位移朝 target，有界写回）。
- **思考核+推理循环**：test_thought_core.py 12 项 + test_reasoning_loop.py 11 项全绿。
- **自我学习闭环**：test_inquiry_branch.py 15 项 + test_inquiry_executor.py 15 项 + test_inquiry_consolidation.py 11 项全绿；active_inquiry_full_chain_demo.py 三阶段端到端（真实 KAL 诚实降级 Decline 6/6）。
- **全量回归**：338 项 pytest 全绿。

---

> **下一步**：① 迭代④ 推理循环形式化进主干（思考流形/推理循环从 pilot 到内生，设计路线）；② 统一 checkpoint（KAL 校准+indexer/扩容门控+内化 SFT+concept_slot 装配合并）；③ ask_fn/tool_fn 真实实现；④ 1.5B 扩展时把 pilot 验证的部件内生进 28 层主干。
