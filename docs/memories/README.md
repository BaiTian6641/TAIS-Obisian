# TAIS Obsidian 记忆导出文档（docs/memories/）

> 本目录是 `/memories/repo/` 工作区记忆的**导出快照**，供 Git 版本控制、文档化、跨会话审计与协作者查阅。
> 导出日期：2026-07-30  ·  对应代码库 commit：95880b2  ·  对应测试基线：412 项 pytest 全绿
>
> 活体记忆仍在 `/memories/repo/`（session 级 + repo 级）；本目录是**冻结快照**，供历史追溯。新会话仍读 `/memories/repo/`。

## 文档结构

本目录按主题分类组织 46 份记忆笔记。所有笔记均为简体中文 Markdown，遵循：
- **诚实优先**：负结果与不确定度如实标注（不臆造阈值、不粉饰数据）；
- **可追溯**：每条结论标注 commit / 测试 / report 来源；
- **可复核**：关键数据附复跑命令与产出文件路径。

## 分类索引

### 1. 路线与规划（总览）
| 文件 | 内容 |
|---|---|
| `next-steps-roadmap.md` | 下一步规划总览（fb1 反馈优先级、0.5B 训练、256K 扩充、P1 校准路线）|
| `fb1-feedback-verification.md` | fb1 学术报告评审反馈交叉验证（A 组文献核实、3 处措辞修正、P0–P3 优先级）|
| `architecture-integration-status.md` | 架构接入状态评估 |
| `unified-checkpoint.md` | 统一 checkpoint（pilot_0p1b_gdn2_10k_unified）合并方案与全链强度 |
| `hardware-dual-gpu.md` | 双卡硬件（PRO 4000 + 4070）分工 |
| `training-efficiency.md` | 训练吞吐与显存效率 |

### 2. 门控副作用根治（fb1 P0，最终方向：记忆层）
| 文件 | 内容 |
|---|---|
| `memlayer-internalization.md` | **记忆层条目迁移根治**（in-context 0.688=基线零干扰，token 寻址可事实召回）|
| `gated-fusion-mlp.md` | GatedFusionMLP 扩容门控（召回 0.625，但有副作用）|
| `decoupled-gate.md` | 解耦双通道门控（方案 A，KV 0.625/ic 副作用权衡）|
| `fully-decoupled-gate.md` | **彻底解耦负结果**（0.1B 注入召回依赖扩容门控整体开权重状态）|
| `niah-length-scan-gate-adaptive.md` | NIAH 长度扫描（max_seq=1024 硬限发现）|
| `context-aware-gating-research.md` | 上下文感知门控研究 |

### 3. 优化与吞吐（fb1 P0）
| 文件 | 内容 |
|---|---|
| `muon-pmstream-optimization.md` | Muon 优化器（6.523<AdamW 6.868）+ PM-stream 吞吐优化（×1.68）|

### 4. 思维能力强化（第二阶段，fb1 P1）
| 文件 | 内容 |
|---|---|
| `thought-core-backbone-integration.md` | **思考核接入主干前向**（增益 +0.078~+0.125 达 fb1 门槛）|
| `thought-core-integration.md` | 思考核集成（早期探索）|
| `thought-core.md` | CTM 式思考核 pilot 模块 |
| `thinking-manifold-layer.md` | 思考流形层（共形等距 + VICReg 去相关）|
| `thinking-e2e-integration.md` | 思考链端到端集成 |
| `thinking-real-adapter.md` | 真实部件适配 |
| `cot-projection.md` | CoT 投影层（投影非计算 + 忠实性审计）|
| `path-integration.md` | 路径积分辅助任务（GridCodeProbe）|
| `thought-visualizer.md` | 可解释性前端（3D 轨迹 + 坏路径四类检测）|

### 5. 主动求知闭环（自我学习）
| 文件 | 内容 |
|---|---|
| `active-inquiry-design.md` | 主动求知闭环架构设计 |
| `active-inquiry-full-chain.md` | 主动求知全链强度 |
| `inquiry-branch.md` | 求知分支（四选一 RPL/LP）|
| `inquiry-executor.md` | 求知执行器（交叉验证、绝不裸自我修正）|
| `inquiry-sleep-consolidation.md` | 求知睡眠固化 |
| `pmstream-kimi-k3.md` | PM-stream（mHC n=5）Kimi K3 交叉验证 |

### 6. 知识内化
| 文件 | 内容 |
|---|---|
| `knowledge-internalization.md` | 知识内化设计 |
| `teaching-sft.md` | teaching_sft 内化行为可训 |
| `internalization-e2e.md` | 内化端到端 |
| `retrieval-recall-training.md` | HRL 检索训练（召回 0.938）|
| `kaplan-extract.md` | Kaplan 内词典提取（ℓ3 最强）|

### 7. KAL 元认知 + GDN-2 + 校准（fb1 P1）
| 文件 | 内容 |
|---|---|
| `kal-math-spec.md` | KAL 三层元认知数学规范 |
| `kal-gdn2-truth-finetune.md` | KAL 真值锚微调（AUROC 0.769）|
| `gdn-decay-bounded.md` | GDN decay 有界化（g_min=-5 sigmoid，4× 加速门收敛）|

### 8. 学术产出
| 文件 | 内容 |
|---|---|
| `academic-report-0p1b.md` | 0.1B 学术报告 v1.0（IEEE + Carbon + 文献核实）|
| `dataset-research.md` | 数据集选型研究 |

### 9. 已核实文献综述（A 组证据链）
| 文件 | 内容 |
|---|---|
| `verified-literature-self-evolution.md` | 自我演化文献综述（48KB，最完整）|
| `verified-literature-active-inquiry-loop.md` | 主动求知循环文献 |
| `verified-literature-memory-compilation.md` | 记忆编译文献 |
| `verified-literature-metacognition.md` | 元认知文献 |
| `verified-literature-metacognition-neuroscience.md` | 元认知神经科学 |
| `verified-literature-dynamic-tokenizer.md` | 动态 tokenizer 文献 |
| `verified-literature-kal-probe.md` | KAL 探针文献 |
| `verified-literature-thinking-manifold.md` | 思考流形文献 |

## 维护

- **更新策略**：每次重大里程碑后（或用户要求同步时）批量导出最新 `/memories/repo/` 到本目录；
- **冲突处理**：本目录与 `/memories/repo/` 不一致时以 `/memories/repo/` 为准（活体）；
- **新增条目**：新主题出现时新增分类章节并更新本 README 表格。

---

*本索引文件由 GitHub Copilot 自动生成，对应 AGENTS.md 2026-07-30 同步状态。*
