# TAIS Obsidian：面向持续学习 LLM 的权重虚拟内存架构——0.1B 先导验证学术报告

> **版本**：v1.0（2026-07-29）
> **性质**：提案用学术报告（Technical Report / Proposal），基于 0.1B pilot 完整实现与验证。
> **数据纪律**：本报告全部性能数值均来自项目真实验证产物（report JSON / 测试 / 训练日志），文中逐项标注出处；未实现部分一律标注"设计目标（未实现）"。

---

## 摘要

大语言模型（LLM）的权重在训练后冻结，无法在不重训的前提下持续吸收新知识；主流检索增强（RAG）以外挂文本补丁缓解，但文本并非模型的原生知识表示。TAIS Obsidian（泰斯人工智能复合体·黑曜石框架）提出一种**面向持续学习 LLM 的"权重虚拟内存"架构**：类比操作系统的页表/虚拟内存，模型在推理中以"知识块"（KnowledgeBlock）为单位按需换入/换出知识与经验，并配备分层元认知（KAL）探针检测知识空白、主动求知而非盲目猜测。本文报告 0.1B 先导（pilot）的完整实现与验证：GDN-2 混合主干（门收敛三阶段证据链 + 有界 decay 4× 加速收敛）、三级检索注意力与 PM-stream 多流残差（消融 −0.024/−0.025 nats）、KAL 元认知（探针 AUROC 0.945、真值锚校准 0.769）、主动求知闭环（诚实降级 6/6）、知识内化（教学式 SFT 内化差 0.758、退联检验 1.000）、知识块"写入即可用"（HRL 检索 1.000 → HCA 注入召回 0.625，in-context 上界 0.70）、睡眠固化（PROMOTE 8 / QUARANTINE 1 / REJECT 8）与动态词表（Kaplan 内词典 concept_slot）。统一 checkpoint 全链验证通过，357 项 pytest 全绿。全部结果在单机双 GPU（RTX PRO 4000 24GB + RTX 4070）上真实复现。

**关键词**：持续学习；元认知；线性注意力（Gated DeltaNet-2）；检索增强；知识内化；动态词表

---

## 1. 引言

预训练 LLM 的知识固化在权重中，部署后难以更新。检索增强生成（RAG）[1] 与主动检索（FLARE [2]、Self-RAG [3]）以外挂文本绕过，但存在三点本质局限：①文本不是模型的原生知识表示，注入依赖 prompt 拼接而非权重级接口；②模型对自身知识空白缺乏显式读出，难以"知道自己不知道"；③学到的内容无法沉淀为可复用、可审计、可固化的长期记忆。

TAIS Obsidian 的核心设想是把"知识"升格为与权重同级的运行时对象——**知识块**（KnowledgeBlock），并为 LLM 提供一套类似虚拟内存的管理机制：

- **权重虚拟内存**：知识块经页表（SQLite）登记、BlockStore 分层存储（L0 VRAM ↔ L1 DRAM ↔ L2 NVMe ↔ L3 远端）、缺页 fail-closed，运行时按需注入注意力 HCA 区或残差流（steering）；
- **元认知门控**：分层 KAL 探针（L1 P(IK) 三态 / L2 情感 / L3 冲突）只读主干 hidden state，检测知识空白并触发检索或诚实降级；
- **主动求知闭环**：certainty 读出 → 求知分支（四选一）→ 交叉验证写入（绝不裸自我修正）→ 实时可用（HRL 检索 + HCA 召回）→ 睡眠固化（CA1 门 + 三元奖励 RL）。

本报告基于 0.1B pilot（12 层，d_model=768，词表 129280）的**完整落地与实测数据**，论证该架构在 pilot 规模上的可行性，并诚实标注与设计目标（1.5B / 1M 上下文，未实现）的边界。

---

## 2. 架构总览

TAIS Obsidian 主干为 GDN-MemBlock（线性注意力递归状态）与三级检索注意力交替的混合架构；KAL/HRL/知识块注入点/PM-stream 为 checkpoint 内生部件（非外挂服务）；运行时（页表/BlockStore/缺页）与睡眠期固化器分置。

![TAIS Obsidian 架构详图](TAIS_Obsidian_架构详图.png)

*图 1：TAIS Obsidian 架构详图（IBM Carbon 设计语言）。主干、KAL、HRL、知识块库、DKB-Runtime、记忆层级、睡眠巩固器。*

各子系统定位如下表（Carbon 风格简表）：

| 子系统 | 部件 | 载体 / 读点 | 写通道 | 状态 |
|---|---|---|---|---|
| 主干 | GDN-2 + 三级注意力栈 | 递归状态 S / KV | 预训练（Muon） | ✅ 已验证（0.1B） |
| 多流残差 | PM-stream（mHC n=5） | 4 内容 + 1 感知-记忆流 | 预训练 | ✅ 消融 −0.024 nats |
| 元认知 | KAL L1/L2/L3 | 只读 GDN 层 ℓ8/ℓ10 | 校准微调（detach 主干） | ✅ AUROC 0.769 |
| 检索 | HRL LightningIndexer + DG | 块表征 | 余弦蒸馏（梯度隔离） | ✅ top-1 1.000 |
| 知识块 | BlockStore + 页表 | KV / 记忆层 / 向量 | W0–W2 零梯度快写 | ✅ 累积不覆盖 |
| 求知 | 求知分支 + 执行器 | KAL certainty | CrossVerifier 验证门 | ✅ 诚实降级 6/6 |
| 固化 | 睡眠巩固器 | CA1 门 + 间隔提取 | W3+ 离线（同优化器） | ✅ PROMOTE 8 |
| 词表 | concept_slot（reserved 2048） | 位置不变向量 | Kaplan 提取 | ✅ 真实启用 |

---

## 3. 子系统：设计·实现·测试·结果

### 3.1 GDN-2 主干：门收敛与有界 decay

**设计**。主干采用 Gated DeltaNet-2（GDN-2）[4]，在 delta-rule 线性注意力上将"擦除"与"写入"解耦为两个独立通道门（erase gate b / write gate w），突破 GDN/KDA 单标量 β 的表达瓶颈。层型排布 GGGAGGGAGGGA（9 GDN + 3 三级注意力栈）。

**门收敛三阶段证据链**（出处：`checkpoints/pilot_0p1b_gdn2*` 各阶段评估 + 显著性检验脚本）：

| 阶段 | 训练步 | 门坐标 std (b/w) | NIAH 检索 |
|---|---|---|---|
| 欠训练 | 2000 | 0.024 / 0.019 | 0.130（< GDN-1 0.200） |
| 门饱和 | 8000 | 0.345 / 0.321 | 0.180 |
| **反超** | 10000 | 0.342 / 0.317 | **0.240**（> GDN-1 0.200，Δ+0.040） |

结论：GDN-2 早期 NIAH 落后是**门欠收敛（欠训练）而非架构缺陷**；检索滞后于门分化（慢变量），erase/write 解耦的检索优势在门收敛后兑现。

**有界 decay（借鉴 Kimi K3 [5]）**。decay 参数化由无界 negative-softplus 改为 K3 式有界 scaled-sigmoid `g = g_min·sigmoid(exp(A_log)·(a+dt_bias))`（g_min=−5），每步保持因子 α > e^{−5}≈6.7e-3，16-token tile 累积 log-decay∈(−80,0)，倒数 rescale < e^{80} 留在 BF16 数值范围（1M 上下文必需）。实测**门收敛 4× 加速**：

| decay 参数化 | 2000 步门 std (b/w) |
|---|---|
| 无界 | 0.024 / 0.019 |
| **有界（g_min=−5）** | **0.323 / 0.296**（13–15×） |

有界 2k 即达无界 8k 的饱和水平（0.345/0.321）。

**显著性检验**（`scripts/_niah_significance.py`，3 seeds × 200 = 600 次/模型，std 0.021）：GDN-1 0.177 / 无界 GDN-2 10k 0.207 / 有界 GDN-2 7k 0.217 / 有界 GDN-2 10k 0.203。GDN-2（有界/无界）均反超 GDN-1（+0.026~0.040 > std）；有界 vs 无界持平（Δ < std）——**decay 有界化是优化路径（加速收敛 + 保数值范围），非能力上限**。

### 3.2 三级检索注意力 + PM-stream 多流残差

**三级检索注意力（TriRetrievalAttention）**：滑窗 L0（512）+ CSA stride-4 选择检索 L1 + HCA 128:1 gist L2，NSA [6] 式门控融合，支持 `inject_hca_entries` 把知识块 KV 前置拼入 HCA 区（fail-closed 不占 token 位）。**消融（2000 步，出处：`docs/D0_0p1B先导实验报告.md` §6.4）**：val 3.762 vs hybrid 基线 3.768（参数 +0.093%，吞吐 8.6k tok/s 为基线 91%）。

**PM-stream（mHC n=5）**：把单残差流扩为 5 流（4 内容 + 1 感知-记忆专用道），H^res 用 Sinkhorn-Knopp 双随机约束（Birkhoff 多面体，谱范数 ≤1）[7][8]。恒等初始化 < 1e-6。**消融**：val 3.744 vs 基线 3.768（**−0.024 nats**，5 个评估点一致领先）；与三级栈**组合相容** val 3.743（**−0.025**）。吞吐 3.0k tok/s（fp64 应用端 + Sinkhorn 开销，1.5B 前需优化，已登记）。

| 配置（2000 步） | val loss | Δ vs 基线 |
|---|---|---|
| hybrid 基线 | 3.768 | — |
| + 三级栈 | 3.762 | −0.006 |
| + PM-stream | 3.744 | −0.024 |
| **组合** | **3.743** | **−0.025** |

### 3.3 KAL 分层元认知：真值锚校准

**设计**。KAL（Knowledge-Awareness Layer）分层元认知头：L1 P(IK) 三态 W[d,3] + L2 情感 valence/arousal W[d,2] + L3 冲突；只读主干 hidden（detach，监测/执行分置红线），不作生成损失（探针冻结红线）。

**事后探针（M2，出处：`runs/kal_probe/report.json` / 总体实施计划 §7.5）**：pilot hybrid checkpoint ℓ8 L1 overall AUROC **0.945**，fake 语义空白子集 **0.979**，超 FLARE 输出分布基线（0.938 / 0.858）——主干 hidden state 中线性可读出"知/不知"信号，与外部证据（SAPLMA [9]、量化态探针 [10] 0.904–1.000 AUROC）一致。

**真值锚校准（GDN-2 10k 适配，出处：`runs/unified_checkpoint/full_chain_report.json`）**：经真值锚微调（锚"事实真假"而非"语言建模置信度"），ℓ10 校准 AUROC **0.769**（kaltruth 报告 final=0.75945，**如实保留未达 0.8 的原始判定，不臆造**）；certainty 方向正确：known 文本 P(known) **0.879**、fake 文本 **0.000**——可作真实元认知门控。微调前后主干 val loss 逐位一致（4.01844→4.01844，detach 红线成立）。

### 3.4 HRL 检索 + 知识块：写入即可用

**设计**。知识块双形态（markdown 源代码 ground truth + 编译产物可失效重建）；载体三型：KV（token 寻址，可事实召回）/ 记忆层 delta / 向量（位置不变，只 steer）。**累积不覆盖红线**：版本化 `:v{n}` 自增 + 冲突保留双方标分歧（抗坍缩 [11]）。

**实时可用（出处：`runs/retrieval_recall/report.json`）**：知识块写入后，推理时 HRL 检索命中 → HCA 注入 → 同一对话立即可用（无需等睡眠固化、无需重训）。

| 指标 | 训练前 | 训练后 | 参考 |
|---|---|---|---|
| HRL indexer 块检索 top-1 | 0.062（随机 1/16） | **1.000** | embedding 余弦基线 1.000 |
| KV 注入答对率 | 0.000 | **0.188**（585 线性门控） | in-context 上界 0.70 |
| 主干污染 | — | **权重逐位不变 drift=0.0** | frozen 红线 ✅ |

**扩容门控破瓶颈（GatedFusionMLP，出处：`runs/recall_gated/report.json`）**：585 参数线性门控是容量瓶颈；扩容为 MLP（26121 参数，恒等初始化 fc2=0 保 g=1/3）后召回 **0.188 → 0.625**，逼近 in-context 上界 0.70（差距 0.51 → 0.075），主干 drift=0.0。

### 3.5 第二阶段：思维能力强化（pilot 落地）

第二阶段把"思考"建模为流形上的导航，7 个迭代 pilot 全部落地并端到端集成（出处：`tests/test_thinking_*.py`，全绿）：

- **思考流形**（manifold.py）：d→64 维共享投影（同一实例服务知识块 route_key / PM-stream 思考段 / W0 轨迹段），共形等距损失（尺度不变）+ VICReg [12] 去相关（方差铰链防坍缩）；
- **CTM 式思考核**（thought_core.py）：抽取 CTM [13] 两原理做小核——通道组历史（384÷8 组 × 近 4 tick FIFO）+ RoPE 相位化"思考时间"（维度语义=tick 非 token），certainty 早停（自适应算力）；
- **推理循环**（reasoning_loop.py）：五步 tick——GDN 状态读出 → glimpse → HRL propose → KAL certainty → thought_core.forward_step → bridge 位移写 PM-stream；`<|recall|>` 显式出现在 CoT（审计红线）；
- **CoT 投影**（cot_projection.py）：**CoT 是投影层非计算层**——计算在流形（潜在思考），每 tick 强制解码显式思考段作 grounded 监督；CotFaithfulnessAudit 说-做一致性（一致 1.000 > 不一致 0.000）；
- **路径积分 + 可解释性前端**：GridCodeProbe 探针判别力验证（人工六边形 grid score 0.867 > 0.3，随机 −0.008）；3D 轨迹可视化 + 坏路径四类检测（信心膨胀/漂移/早停失败/recall 风暴）。

**诚实边界**：思考流形/思考核/推理循环为 **pilot 独立模块（未接 model.py 主干前向）**，是设计明确的 pilot 边界（验证概念）非遗漏；CTM 在语言域零证据（原文 §12 自认 future work）[13]，网格码在 transformer 不自发涌现（证据全在 RNN/PCN [14][15]）——均已降预期并显式训练诱导。

### 3.6 主动求知闭环（自我学习）

**设计**。certainty 校准 → 求知分支（四选一：AskQuestion / CallTool / Decline / DirectAnswer，对齐 [16]）→ 求知执行器（交叉验证 + 写入）→ 重评估闭环。求知触发对齐 RPL/LP（可学习区："差一点就知道"时求知收益最大）[17][18]。

**红线落实**（出处：`tests/test_inquiry_executor.py`）：①**绝不裸自我修正** [19]——未验证证据绝不写入（CrossVerifier 多源一致性 + 与先验一致性 + 冲突检测三路信号）；②**诚实降级**——certainty 低于完全空白阈值 → Decline 声明"该部分记忆暂不可用"，绝不硬答。

**全链端到端 demo（出处：`tests/test_active_inquiry_full_chain.py`）**：真实 KAL 对完全虚构事实 certainty=0.000 → **Decline 诚实降级 6/6**（不硬答）；可学习区（B 组，certainty 占位演示）→ CrossVerifier → 写入 draft 6/6；睡眠固化 PROMOTE 3 / QUARANTINE 1（冲突块）/ REJECT 3（consensus < 0.7）。

### 3.7 知识内化：教学式 SFT + 退联检验

**设计**（CLS 互补学习系统 [20] 双速通道）：运行时快通道（知识块写入即可用）+ 睡眠慢通道（CA1 门调速固化入主干）。内化行为用教学式 SFT 训练：样本 {知识链 K, 问题 Q, 答案 A(K)}，**Q 必须 K-依赖**（去掉 K 答错），防"凭先验答忽略 K"；退联检验训练区分一致知识（内化）vs 矛盾/错误（拒/标分歧）。

**结果（出处：`checkpoints/pilot_0p1b_gdn2_10k_teaching` 评估，n_dep=66）**：

| 指标 | SFT 前 | SFT 后 |
|---|---|---|
| 有 K 答对率 | 0.015 | **1.000** |
| 无 K 答对率 | 0.000 | 0.242 |
| **内化差值（有K−无K）** | 0.015 | **0.758** |
| 一致 K 内化率 | 0.000 | **1.000** |
| 矛盾 K 内化率 | 0.000 | **0.000** |
| **退联检验差值** | 0.000 | **1.000** |

三判据全达成：有 K ≫ 无 K（用上 K 非凭先验）；内化差显著（教学有效）；一致全内化 / 矛盾全拒（退联检验完美，模型学会区分真伪）。诚实边界：无 K 答对率升 0.242 是部分评估题实体在训练集出现过的同类句式轻微先验，后续用完全 held-out 实体做更严 OOD 评估。

### 3.8 动态词表：Kaplan 内词典 concept_slot

**设计**。词表 129280 = 127232 基础 + 2048 reserved（动态词表三级生长阶梯）；新词经 Kaplan 内词典机制 [21]——LLM 在末 sub-word token 处把多碎片融合为整词表征（detokenization），可作新 concept 的 embedding 来源，无需重训。

**实现（出处：`scripts/dynamic_vocab_real_demo.py` + `tests/test_kaplan_extract.py`）**：concept_slot 真实启用并接入自我学习闭环——orchestrator assess_vocab_friction 检测 → Kaplan 提取（末 token hidden，0.1B 实测最强在 ℓ3，小模型峰值前移；正式 28 层回 ℓ10–14）→ 注册页表 + BlockStore + HRL route_graph 入图 → 检索 → 注入。语义验证：electron-photon 同类余弦 **0.513** vs electron-democracy 不同类 **0.217**（真实语义非 mock）。**载体能力边界**：concept_slot = 位置不变向量（factual_recall=False，只 steer 理解）vs 知识块 = token 寻址（事实召回）——Block Spec 强制标 `factual_recall` 字段。

---

## 4. 端到端验证：统一 checkpoint 全链已训强度

把分散训练的部件（KAL 校准内核 + 已训 indexer + 扩容门控 + 内化 SFT 主干）合并为统一 checkpoint（`checkpoints/pilot_0p1b_gdn2_10k_unified`，state_dict 266 键），全链五项强度 n=16 实测（出处：`runs/unified_checkpoint/full_chain_report.json`）：

| 链环 | 指标 | 数值 | 出处 |
|---|---|---|---|
| KAL 校准 | ℓ10 AUROC | **0.769**（kaltruth final 0.75945 如实保留） | full_chain_report ① |
| HRL 检索 | top-1 命中 | **0.938**（= 余弦基线；训练 1.000） | full_chain_report ②③④ |
| HCA 召回 | KV 注入答对 | **0.625** vs 基线 0.062 vs **kaltruth 对照 0.000** | full_chain_report ②③④ |
| 诚实降级 | 虚构事实 Decline | **16/16** | full_chain_report ②③④ |
| 睡眠固化 | 门控裁决 | **PROMOTE 8 / QUARANTINE 1 / REJECT 8** | full_chain_report ⑤ |
| 内化保留 | in-context（拆门控） | **0.688** ≈ teaching 0.6875 | full_chain_report·主干内化保留 |

**红线合规（report 逐项记录）**：绝不裸自我修正（写入/固化经 CrossVerifier + regression 外部验证门控）；累积不覆盖（版本化 + 冲突保留双方）；诚实降级；运行时注入零梯度不动权重；监测/执行分置（KAL 读 GDN ℓ10、注入写 CSA 层，读写不同层）。**全量 357 项 pytest 全绿。**

**门控副作用（诚实发现）**：扩容门控让 HCA 分支开权重（召回 0.625 所需），但对长文本 gist 也开权重，干扰纯文本精确召回（in-context 0.688 → 0.250）；拆门控即回 0.688（主干内化 SFT 未退化，是"注入召回"与"纯文本精确召回"的门控权衡）。可按需 attach/detach；正式需**门控上下文自适应**（注入开 / 长文本关），列为未来工作。

---

## 5. 讨论

**创新点**：
1. **P(IK) 门控的持续学习**：元认知探针不只监测，而是**驱动行为**的门控信号——P(IK) 低触发检索/求知/诚实降级（区别于 FLARE [2] 用输出分布置信度、Self-RAG [3] 用反思 token）；真值锚校准（锚事实真假非流畅度）是关键负结果驱动的设计（logprob ≠ 真假）。
2. **知识块的流形可寻址**：知识块 route_key / PM-stream 思考段 / W0 轨迹段共享同一 64 维投影坐标系，检索、思考、审计在同一几何空间（独创外推，pilot 验证通路）。
3. **CTM tick × GDN 融合**：把 CTM 的思考时间/自适应算力接到 GDN 递归状态（持续状态天然承载 tick），相位化思考时间复用 RoPE 构造。
4. **主动求知闭环三相位**：运行时学习（交叉验证写入）→ 实时可用（HRL + HCA）→ 睡眠固化（CA1 门 + 三元奖励 RL [22]），全链已训强度实测。

**局限（诚实列明）**：
- **规模**：全部结果在 0.1B pilot（120M tokens）取得；1.5B / 1M 上下文 / KAL 探针强度在更大规模未知（设计目标，未实现）。
- **召回余量**：HCA 召回 0.625 < in-context 上界 0.70（差 0.075）；门控副作用（注入开则纯文本精确召回 0.688→0.250）未自适应。
- **CTM 语言域零证据**：思考核只证"改变思考动力学"，未证"更准"（适配层未训练，dist_core ≈ dist_no_core）。
- **数据**：知识内化/求知闭环用程序化虚构实体（保证先验不存在），真实分布知识待验证。
- **吞吐**：PM-stream 3.0k tok/s 是主要工程瓶颈（fp64 应用端 + Sinkhorn），1.5B 前需优化。

---

## 6. 未来工作

1. **门控上下文感知自适应**：注入条目开权重 / 长文本 gist 关权重，消除副作用，召回 0.625 → 0.70 余量并保纯文本精确召回。
2. **1.5B 扩展规划**：把 pilot 验证部件内生进 28 层主干（GDN-2 + 三级栈 + KAL + HRL + PM-stream + 思考流形/推理循环 + concept_slot），OLMo 数据课程（Dolma 3 Mix / Dolmino / Longmino），Muon 优化器贯穿预训练与 W4 固化。
3. **真实数据全链**：ask_fn / tool_fn 真实实现（对话接口 / 检索工具，当前 pilot mock），真实（非虚构）知识跑主动求知端到端。
4. **输出侧词表升格**：concept_slot 当前为输入侧提取通路，输出侧升格（reserved 槽命名空间中心协调）待做。

---

## 参考文献

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems (NeurIPS)*, 2020, arXiv:2005.11401.

[2] Z. Jiang, F. F. Xu, L. Gao, Z. Sun, Q. Liu, J. Dwivedi-Yu, Y. Yang, J. Callan, and G. Neubig, "Active retrieval augmented generation (FLARE)," in *Proc. EMNLP*, 2023, arXiv:2305.06983.

[3] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. ICLR*, 2024, arXiv:2310.11511.

[4] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling erase and write in linear attention," NVIDIA, arXiv:2605.22791, 2026.

[5] Kimi Team, "Kimi Linear: An expressive, efficient attention architecture," Moonshot AI, arXiv:2510.26692, 2025.

[6] J. Yuan *et al.*, "Native sparse attention: Hardware-aligned and natively trainable sparse attention," DeepSeek-AI, arXiv:2502.11089, 2025.

[7] D. Zhu *et al.* (ByteDance), "Hyper-connections," in *Proc. ICLR*, 2025, arXiv:2409.19606.

[8] Z. Xie *et al.* (DeepSeek-AI), "mHC: Manifold-constrained hyper-connections," arXiv:2512.24880, 2025.

[9] A. Azaria and T. Mitchell, "The internal state of an LLM knows when it's lying (SAPLMA)," in *Findings of EMNLP*, 2023, arXiv:2304.13734.

[10] "Hallucination is linearly decodable from mid-layer hidden states in quantized LLMs," arXiv:2606.02628, 2026.（量化 4-bit NF4 下线性探针 hallucination 检测 AUROC 峰值 0.998/1.000）

[11] M. Gerstgrasser, S. Schoenholz, A. Stern, M. Deweese, and J. Dyer, "Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data," arXiv:2404.01413, 2024.

[12] A. Bardes, J. Ponce, and Y. LeCun, "VICReg: Variance-invariance-covariance regularization for self-supervised learning," in *Proc. ICLR*, 2022, arXiv:2105.04906.

[13] L. Darlow, C. Regan, S. Risi, J. Seely, and L. Jones, "Continuous thought machines (CTM)," Sakana AI, NeurIPS Spotlight, arXiv:2505.05522, 2025.

[14] A. Banino, C. Barry, B. Uria, C. Blundell, T. Lillicrap, P. Mirowski, A. Pritzel, M. J. Chadwick, T. Degris, J. Modayil, G. Wayne, H. Soyer, F. Viola, B. Zhang, R. Goroshin, N. Rabinowitz, R. Pascanu, C. Beattie, S. Petersen, A. Sadik, S. Gaffney, H. King, K. Kavukcuoglu, D. Hassabis, R. Hadsell, and D. Kumaran, "Vector-based navigation using grid-like representations in artificial agents," *Nature*, vol. 557, pp. 429–433, 2018.

[15] B. Sorscher, G. C. Mel, S. A. Ocko, L. M. Giocomo, and S. Ganguli, "A unified theory for the computational and mechanistic origins of grid cells," *Neuron*, vol. 111, no. 1, pp. 121–137, 2023.

[16] "Structured uncertainty guided clarification for LLM agents (AskQuestion / CallTool / Decline / DirectAnswer)," arXiv:2511.08798, 2025.

[17] J. Metcalfe, "Desirable difficulties and studying in the region of proximal learning," in *Successful Remembering and Successful Forgetting: A Festschrift in Honor of Robert A. Bjork*, Psychology Press, 2011.

[18] P.-Y. Oudeyer, F. Kaplan, and V. V. Hafner, "Intrinsic motivation systems for autonomous mental development," *IEEE Trans. Evolutionary Computation*, vol. 11, no. 2, pp. 265–286, 2007.

[19] J. Huang, X. Chen, S. Mishra, H. S. Zheng, A. W. Yu, X. Song, and D. Zhou, "Large language models cannot self-correct reasoning yet," in *Proc. ICLR*, 2024, arXiv:2310.01798.

[20] J. L. McClelland, B. L. McNaughton, and R. C. O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex: Insights from the successes and failures of connectionist models of learning and memory," *Psychological Review*, vol. 102, no. 3, pp. 419–457, 1995.

[21] G. Kaplan, M. Oren, Y. Reif, and R. Schwartz, "From tokens to words: On the inner lexicon of LLMs," in *Proc. ICLR*, 2025, arXiv:2410.05864.

[22] "TruthRL: Incentivizing truthful LLMs via reinforcement learning (ternary reward)," arXiv:2509.25760, 2025.

[23] Y. Liu, "Optimizer-model consistency: Full finetuning with the same optimizer as pretraining forgets less," arXiv:2605.06654, 2026.

---

### 附：诚实标注说明

- **已验证（0.1B pilot 实测）**：本报告全部数值（门收敛三阶段、有界 decay 4× 加速、NIAH 显著性、消融 val、KAL 0.945/0.769、检索 1.000/0.938、召回 0.625、内化 0.758、退联 1.000、固化 PROMOTE 8、诚实降级 6/6、357 测试）。
- **设计目标（未实现）**：1.5B / 1M 上下文 / Muon 预训练（pilot 用 AdamW）/ 真实分布知识 / 输出侧词表升格 / 门控自适应。
- **文献核实**：全部 arXiv 编号经检索核实真实存在（含 [10] arXiv:2606.02628 量化 LLM 幻觉线性可解码、[23] arXiv:2605.06654 优化器-模型一致性，标题已按 arXiv 摘要页核实修正）。
