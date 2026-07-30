# 下一步规划（2026-07-29，统一 checkpoint 能力基线确认后）

## 当前状态（已完成）
- **第一阶段**：GDN-2（门收敛+有界 decay 4×加速）+ KAL 三层（校准 0.769）+ HRL（检索 0.938）+ PM-stream + 知识块 + 睡眠固化。前置①②③收尾。
- **第二阶段（思维能力强化）**：7 迭代 pilot 全落地（流形层/路径积分/思考核/推理循环/CoT投影/可视化）+ 端到端集成+真实部件适配。
- **主动求知闭环**：certainty→求知分支→求知执行器（交叉验证）→知识内化（SFT）→HRL 检索→HCA 召回（扩容门控 0.625）→实时可用→睡眠固化。全链已训强度（统一 checkpoint）。
- **动态 tokenizer**：concept_slot 真实启用（Kaplan 提取），接入自我学习闭环。
- 测试 357 全绿；架构详图 v2.2（Carbon）；数据集选型+知识内化+主动求知+架构接入评估文档。

## 下一步规划（优先级排序）
1. **门控上下文感知自适应**（进行中）：方案 A 解耦双通道已实现（注入召回隔离 0.625，但副作用未消除——natural_gate=已训对 gist 开权重）。**真正的解**：natural_gate 换"对 gist 关"的权重（方案 A 变体 natural=零初始化+重训对 gist 关，或方案 C 正则压制 gist）。
2. **0.1B 学术报告**（✅ 已完成 v1.0，commit 4d26610，IEEE+Carbon+文献核实）。
3. **ask_fn/tool_fn 真实实现**：求知执行器接真实对话接口/检索搜索工具（当前 pilot mock）。
4. **0.1B 真实数据全链**：用真实（非虚构）知识跑主动求知闭环端到端。
5. **1.5B 扩展规划**：基于统一能力基线+数据集选型（OLMo 课程），把 pilot 验证的部件内生进 28 层主干（GDN-2+三级栈+KAL+HRL+PM-stream+思考流形/推理循环内生+concept_slot），规划预训练（30B tokens Dolma 式）+后训练（SFT+三元奖励 RL）。
6. **召回 0.625→0.70 余量**：更多步/更大 hidden/更多事实（依赖门控自适应）。

## fb1.md 反馈（2026-07-30 学术报告评审，共同研究者）
**总评**：报告正面回答了风险登记最高的三个未知项（KAL 探针强度/写入即可用闭环/零梯度栈工程可行性），数据纪律严（0.75945 如实保留、dist_core≈dist_no_core 如实报告）。
**短板排序（critical path to 1.5B）**：
- **P0 门控上下文自适应 + 记忆层条目 A/B**：副作用不消除召回 0.625 与纯文本 0.688 不可兼得；建议把事实条目迁到**记忆层条目**（设计 §25.2 优先级本来就是记忆层>KV 拼接）与门控方案 A/B，赢的进 1.5B。
- **P0 PM-stream 吞吐优化**：3.0k tok/s 进不了预训练（fp32 Sinkhorn/迭代裁剪/kernel 融合三管齐下回 90%+）。
- **P1 思考核接入主干前向 + 推理增益指标**：从模块到原生的唯一门票（推理基准可测量增益 1-2 点）；同时跑 indexer 网格码自发探针。
- **P1 校准 0.769→≥0.8**（TIAR 轨迹知情重加权+锚集扩充，§27.2 药方）：全链最弱环。
- **P2 NIAH 长度扫描（512→4K→32K）+ Muon 决策**（pilot AdamW，§21 优化器一致性 2605.06654 要求 W4 与预训练同优化器）。
- **P2 融合力评测**（跨块综合推理考题，§25.1 预测 1.5B 短板）。
- **P3 真实数据全链**（ask_fn/tool_fn 去 mock）。
**建议沉淀文档 v2.5**：① §29 思维模块落地状态登记（模块级验证，接入门槛写明）；② 新增 §30「0.1B Pilot 验证登记」（逐项结果+开放问题/风险登记修订：§9#1 部分关闭、§25.3 KAL 风险降级、§28.6 内词典⚠️→✅(0.1B)、P0–P3 清单）；③ 把"真值锚校准"和"收敛阶段判别方法学"升格为设计原则。

## 0.5B 模型训练规划（2026-07-30 用户指示）
**三任务**：①准备数据集训练 0.5B 模型；②P1 校准（0.769→≥0.8）；③上下文扩充到 256K（双卡训练）。
**现状评估**：
- 0.1B pilot：120M tokens FineWeb-Edu 自训 32k BPE，seq_len=1024（max_seq 硬限），12 层 d_model=768。
- **0.5B 规模**：按 6:1 tokens:param 需 ~3B tokens（vs 当前 120M，25×）；模型约 18-24 层 d_model~1024-1536（~0.5B 参数）；数据准备是关键瓶颈（FineWeb-Edu 扩大+多领域混合，对齐数据集选型文档：英文 fineweb-edu 主力+数学 NuminaMath-CoT+科学 SCP-116K+代码 the-stack-v2）。
- **上下文扩充 256K**：max_seq=1024 硬限（RoPE 缓存仅 1024 行，NIAH 扫描发现）→需 RoPE 缓存扩容+位置插值/NTK+渐进扩窗（8K→64K→256K，对齐 K3 渐进课程）；256K 训练需显存（0.5B×256K，GDN 递归状态固定+三级注意力滑窗/CSA/HCA 控制复杂度）。
- **双卡训练**：PRO 4000（24GB sm_120）+ RTX 4070（8GB）——DDP/ZeRO 或梯度并行提吞吐；0.5B bf16 ~1GB 权重+激活，单卡 24GB 可行（grad checkpoint），双卡提吞吐。
- **P1 校准**：锚集扩充（更多样真值数据）+预测反馈循环（§27.2），TIAR 是行为层配套（依赖 GRPO 管线，工程前置被低估）。
**步骤**：①数据准备（0.5B 数据集混合+shard）→②0.5B 配置+训练（PRO 4000 主/4070 辅）→③P1 校准→④256K 扩充（RoPE 扩容+渐进课程+双卡）。

## 关键记忆文件索引（/memories/repo/）
- 架构：unified-checkpoint.md、architecture-integration-status.md、gdn-decay-bounded.md
- 第二阶段：thinking-manifold-layer.md、thought-core.md、path-integration.md、cot-projection.md、thought-visualizer.md、thinking-e2e-integration.md、thinking-real-adapter.md
- 主动求知：active-inquiry-design.md、inquiry-branch.md、inquiry-executor.md、inquiry-sleep-consolidation.md、active-inquiry-full-chain.md
- 知识内化：knowledge-internalization.md、teaching-sft.md、internalization-e2e.md、retrieval-recall-training.md、gated-fusion-mlp.md
- KAL：kal-math-spec.md、kal-gdn2-truth-finetune.md
- 动态 tokenizer：architecture-integration-status.md（含）
- 数据：dataset-research.md；硬件：hardware-dual-gpu.md；路线：next-steps-roadmap.md（本文件）

---
*导出自 /memories/repo/next-steps-roadmap.md（2026-07-30 同步快照）。*
