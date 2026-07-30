# 架构接入状态评估（2026-07-29，docs/TAIS_Obsidian_架构接入状态评估.md v0.1）

用户检查点：①动态 tokenizer 是否加入自我学习闭环；②思考流形（空间流形推理）等是否启用并正确接入架构。

## 动态 tokenizer（concept_slot）✅ 真实启用+接入闭环
- **真实启用**：kaplan_extract.py（真实 Kaplan 内词典提取，末 token detokenized hidden，0.1B ℓ3 实测最强——小模型峰值前移，正式 28 层回 ℓ10–14）+ dynamic_vocab_real_demo.py（真实装配 orchestrator+dynamic_vocab+blockstore）。
- **接入自我学习闭环**：concept_slot 经 orchestrator assess_vocab_friction→promote→页表+BlockStore+HRL route_graph 入图→associative_recall 检索→inject 向量路径。**与求知知识块同存 BlockStore（知识生态统一），但载体不同**：concept_slot=位置不变向量（单槽理解 steer，factual_recall=False）vs 求知知识块=token 寻址载体（事实召回）——载体能力边界红线。
- **边界**：是 orchestrator 可选注入件（extract_fn 需装配时注入），非 model.py 主干默认。
- 验证：electron-photon 同类 0.513 vs electron-democracy 不同类 0.217（真实语义，非 mock 常数）。

## 思考流形/推理循环 🟡 pilot 独立模块（设计预留，未接主干）
- **已真实接入（桥接）**：思考流形↔PM-stream 桥接（manifold_bridge.tick：读 PM 流→流形坐标→位移→反投影→有界写回）真实接通。
- **pilot 独立模块（未接 model.py 主干前向）**：思考流形（manifold.py 独立训练不碰主干）+ CTM 思考核（thought_core.py pilot 消融）+ 推理循环（reasoning_loop.py 独立编排，glimpse/certainty pilot 占位）。**这是设计明确的 pilot 边界（§6：迭代④/正式 milestone 才形式化进推理循环），非遗漏**。
- 运行时主干前向仍走 GDN-2+三级栈+PM-stream（第一阶段）；思考流形/推理循环是叠加的 pilot 层非主干内生。

## 自我学习闭环（主动求知）✅ 闭环逻辑全通
certainty 校准→求知分支→求知执行器（交叉验证+写入）→HRL 检索（1.000）→HCA 注入（扩容门控召回 0.625）→实时可用→睡眠固化。真实 KAL 行为验证（虚构事实→Decline 诚实降级）。边界：ask_fn/tool_fn pilot mock。

## 诚实结论
**所有"未接主干"都是设计明确的 pilot 边界（独立模块验证概念），非实现遗漏**。各部件闭环逻辑全通（338 测试全绿），真实部件适配验证通路与真实 KAL 行为。从 pilot 到主干内生是下一阶段（形式化进推理循环/统一 checkpoint/1.5B 扩展）的工作。

---
*导出自 /memories/repo/architecture-integration-status.md（2026-07-30 同步快照）。*
