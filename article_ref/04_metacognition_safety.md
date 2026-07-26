# 04 · 元认知 / 边界感知 / 安全 —— 背景论文笔记

> 面向 TAIS **KAL（知识感知层）**内生化设计、`<|recall|>`/`<|blank|>` 原生动作、ITI 干预头蒸馏、§26.2 注入即攻击面安全红线。
> ✅=本会话已读 arXiv 摘要核实；🔍=ID 存在性核实但具体数字待全文；⚠️=二手引用未独立核实。

## 探针 / 内部诚实信号

### SAPLMA — Azaria & Mitchell (CMU), arXiv:2304.13734 ⭐⭐⭐【关键】✅
读/生成陈述时用**隐藏层激活**训二分类器输出"该陈述为真"概率；半真半假测试集达 **71%–83% 准确率（accuracy）**。**【核实修正】原文指标为 accuracy 71–83%，非 AUROC**——设计文档 §8"探针（SAPLMA 71–83%）"应理解为准确率。→ KAL 存在性证据：真实度/知识性信号**确实编码在中层激活**，可线性读出；但 SAPLMA 是**外挂分类器**，TAIS 独占性在于把它**蒸馏为 checkpoint 内生权重 + 学习型投影**。

### Hallucination Is Linearly Decodable from Mid-Layer Hidden States in Quantized LLMs — Aiersilan, arXiv:2606.02628 ⭐⭐⭐【关键】✅
三个 7B–8B 指令模型（Llama-3.1-8B/Mistral-7B/Qwen2.5-7B）**4-bit NF4 量化**下，在 TruthfulQA/HaluEval-QA/FEVER/合成集上：**单中层线性探针 0.904–1.000 AUROC**；采样式检测器（EigenScore/self-consistency/attention entropy）同协议 **≤0.541 AUROC**；信号**近似线性**（MLP 探针极少超线性 +0.01 AUROC）；峰值层一致落在深度 50–90%（Llama/Mistral=block 13–18/32，Qwen=19–25/28）；首块注意力熵知识接地场景 0.866–0.941 AUROC，零额外推理成本；单 8GB GPU 可复现。→ ① **线性探针即足够**（支撑 KAL 用朴素 `nn.Linear W[2048,3]`）；② 峰值层与 KAL ℓ10/14/18 同区间；③ **量化不破坏信号 → 边缘 Q4 元认知可行**；④ 首块注意力熵可作零成本互补侧信道。

### Kadavath et al. "Language Models (Mostly) Know What They Know" — Anthropic, arXiv:2207.05221 ⭐⭐⭐【关键】✅
① 更大模型在 MC/T-F 题校准良好；② 开放式生成可先答再评 `P(True)`；③ **模型可被训练预测 `P(IK)`（"我是否知道"）**——不依赖候选答案；部分跨任务泛化；`P(IK)` 随上下文材料合理上升；**但新任务上校准会漂移**。→ P(IK) 训练范式的**方法学祖本**；漂移 → T2 定期重校准。

### Do I Know This Entity? — Ferrando et al., arXiv:2411.14257, ICLR 2025 ⭐⭐【机会】✅
**稀疏自编码器（SAE）**发现幻觉机制关键是**实体识别**——模型内部有方向能检测"这个实体我是否能回忆事实"；方向**因果相关**（可 steer 已知实体拒答、未知实体诱导幻觉）；**SAE 在 base 上训的方向对 chat 拒答行为也因果作用**——chat 微调复用 base 机制；机制上扰乱下游"把实体属性移到末 token"的注意力头。→ `<|blank|>` 空白态的机理脚注；暗示 P(IK) 信号预训练期就应埋好。

### Inference-Time Intervention (ITI) — Li et al. (Harvard), arXiv:2306.03341, NeurIPS 2023 ⭐⭐⭐【关键】✅
沿少数注意力头的一组**方向**平移推理时激活，提升真实度。Alpaca+TruthfulQA：真实度 **32.5% → 65.1%**；存在真实度-有用性权衡可调强度；**仅需数百样本**定位方向。结论："**LLM 可能在内部表征了某事为真的似然，即便表面输出 falsehood。**"→ ITI 方向**蒸馏为内生干预头**=KAL"执行"通道原型；内部诚实 ≠ 口头诚实（§8.3-2）。

## 自我认知 / 行为自知

### Tell Me About Yourself — Betley et al. (Oxford), arXiv:2501.11120, ICLR 2025 ⭐⭐【关键】✅
研究**行为自知**——模型能描述自己**隐式学到的行为策略**，无需 in-context 示例；并能识别自身被植入的后门。→ **功能性自我模型**直接证据（KAL/人格块自我模型非涌现玄学）；后门识别能力与 §26.2 后门扫描器互补。**【诚实红线】仅指功能性自我建模，绝不支持现象学意识主张**。

### Do LLMs Know What They Are Capable Of? — Barkan, Black, Sourbut, arXiv:2512.24661 ⭐⭐【风险】✅
所有被测 LLM 都**过度自信**，但多数判别力**优于随机**；**更新更大的模型一般并不具备更强判别力**（仅 Claude 系列正相关趋势）；多步 agentic 任务过度自信**随进程加剧**；**推理模型 ≈ 或劣于非推理模型**；给失败 ICL 经验后**部分**模型降低过度自信；**给定估计成功概率，所有模型决策近似理性——问题出在估计本身过于乐观**。→ **核心反面边界证据**：① 规模不自动修复校准/判别 → 元认知**必须显式训练**；② 多步任务 KAL 须全程在线；③ CoT 长度 ≠ 元认知质量；④ KAL 瓶颈在校准而非决策机制 → P(IK) 损失要对准**绝对校准**。

## CoT 忠实性

### Language Models Don't Always Say What They Think — Turpin et al., arXiv:2305.04388, NeurIPS 2023（被引 1685）⭐⭐【风险】✅
CoT 解释**系统性误表**模型预测真实原因；加偏置特征（如重排 few-shot 选项使答案恒为"(A)"），CoT 会**理性化**被诱导答案却**不提及**该偏置；BBH 13 任务准确率最多掉 36%；社会偏见任务解释为刻板答案辩护。→ 直接支撑 §17.2 归因监测头与"说-做分歧"惩罚；**自报置信度不可靠**的根证据——元认知**不能靠模型自己说**，必须靠内部探针。

## 安全 / 注入即攻击面（§26.2 命名防御范式）

### MemoryGraft — Srivastava & He, arXiv:2512.16962 ⭐⭐⭐【关键】✅
针对**依赖长期记忆+RAG 的经验学习 agent**的新型**间接注入攻击**：不靠即时越狱，而是把**恶意"成功经验"植入 agent 长期记忆**——利用 agent **语义模仿启发式**（倾向复制检索到的成功任务模式）。攻击者仅提供 agent 执行时读取的**良性 ingestion 级制品**，诱导构建"少量恶意过程模板+良性经验"混合毒化 RAG 库；后续语义相似任务时 lex/embedding 并集检索浮出 grafted 记忆，agent 采纳不安全模式 → **跨会话持久行为漂移**。MetaGPT DataInterpreter+GPT-4o 验证：少量毒化记录即可占检索经验大比例。**与传统 prompt injection（瞬时）和标准 RAG poisoning（针对事实知识）本质区别：攻击的是"信念/策略"而非"事实"，且时间解耦**。→ §26.2 升级为命名防御范式的直接实证：① 写通道是真实可利用攻击面；② "检测恶意动作"防御失效（攻击腐蚀信念非动作）；③ 时间解耦使在线审查不够，必须离线筛查。我们的 markdown 源代码+块签名+namespace fail-closed+CA1 回归+draft 区隔离 = 微软 Defender 三原语（memory contracts/belief drift detection/context provenance tracking）的实现。

### The Trigger in the Haystack — Bullwinkel et al. (Microsoft AI Red Team), arXiv:2602.03085v1 ⭐⭐🔍/⚠️
实用**睡眠代理式后门扫描器**：① sleeper agent 倾向**记忆投毒数据**可用**记忆提取**泄露后门样本；② 被毒化模型在输入含触发器时**输出分布与注意力头**呈现独特模式。据此开发无需预知触发器/目标、仅需推理操作的扫描方法；不改变模型性能；恢复可用触发器。**【核实修正】摘要未含"87.8% 检出/0 误报"数字——该数字出自设计文档 §26.2 二手引用，本会话标 ⚠️ UNVERIFIED，需查全文**。→ 可接入睡眠固化前 draft 区筛查。

### 元认知 LLM 框架群（Meta-R1/CLEAR/MeCo/Think2/MIND）⭐⚠️
- **Meta-R1** — arXiv:**2508.17291** ✅ 已核实（**【勘误】设计文档/前轮记的 +8% 错；实测超 SOTA 高达 +27.3%**，token 消耗降到 15.7–32.7%、效率 +14.8%；把推理分解为 object-level + meta-level，级联式主动规划/在线调节/自适应早停；可跨数据集与底座迁移）。
- **AutoMeco（"LLMs Have Intrinsic Meta-Cognition, but Need a..."）** — EMNLP 2025 main 171 ✅ 新增。自动化元认知评测：用 hidden state + logits + 概率算步级置信 spred，PRM 标注步正确性，输出 AUROC/AUPR/FPR95。→ **KAL 评测协议参考**（步级、PRM 标注、FPR95）。
- **"LLMs Are Capable of Metacognitive Monitoring and Control of Their Internal Activations"** — NeurIPS Poster ✅ 新增【关键·安全】。神经反馈范式量化 LLM 报告/控制自身激活的能力；发现"元认知空间"维度**远低于**神经空间（LLM 只能监控激活的小子集）→ 支撑 KAL 用低维线性头；**安全警示**：模型可能**混淆(obfuscate)**内部过程以逃避激活式监督 → KAL 探针权重须**冻结、不进模型梯度影响**（与监测/执行分置红线呼应）。
- **Know More, Know Clearer** — arXiv:**2602.12996** (Chen et al., 13 Feb 2026) ✅ 已核实。用内部认知信号把知识空间分为 **mastered/confused/missing** 三区，引导定向知识扩展；**认知一致性机制同步主观确定性与客观准确率（校准）**。→ 与 KAL 三态（知道/不确定/空白）+ 校准目标直接同构。
- **MeCo** — arXiv:**2502.12961** (Li et al., **ACL 2025 camera-ready**) ✅ 已核实（标题《Adaptive Tool Use in LLMs with Meta-Cognition Trigger》）。从**表示空间高级认知信号**量化元认知分数决定何时调用工具，**fine-tuning-free、成本极低**。→ `<|recall|>` 触发 HRL 的同族（学到探针决定何时"调用"外部记忆）。
- **Think²** — arXiv:**2602.18806** (Elenjical, Kavuri, Varma, 21 Feb 2026) ✅ 已核实。把 **Ann Brown 监管周期**（Planning/Monitoring/Evaluation）做成结构化 prompting + 轻量 dual-process MetaController；**自校正成功 3×、580 query 对盲测 84% 信任偏好**。→ KAL 监测-执行分置 + 自我校正训练参考。
- **MIND** — arXiv:**2509.05714** (Fan et al., 6 Sep 2025) ✅ 已核实。多模态元认知知识编辑：构建**元知识记忆**做自我觉察、**博弈论交互监控知识激活**、label refinement 抗噪。→ CA1 仲裁 + 知识激活监控参考。
- **CLEAR** — ❌ **误归属已删除**：arXiv 2412.16112 实为《CLEAR: Conv-Like Linearization Revs Pre-Trained Diffusion Transformers Up》（DiT 论文，非元认知框架）。"entropy-triggered expert expansion / 70-80% 检出" 的元认知 CLEAR 无法定位，**从框架群移除**；该生态位由 Know More Clearer（mastered/confused/missing 三区）+ AutoMeco（步级评测）覆盖。
- **结论**：元认知框架群**已从"待核实"升为 6 篇全核实**（Meta-R1/AutoMeco/Know More Clearer/MeCo/Think²/MIND），证明"元认知可被显式训练且带来增益"是 KAL 的成熟同代平行工作；但**全部外挂/后训练式**，TAIS 差异化在**预训练期即把探针/干预头内生为 checkpoint 权重**。
- **"Entropy Alone is Insufficient for Safe Selective Prediction in LLMs"** ✅ 新增【关键·评测】。entropy-based UQ 不足；须监督式 correctness 探针（Kadavath 式）；部署指标应用 **AURC（risk-coverage 曲线下面积）+ TCE（target calibration error）**，而非仅 AUROC。→ **KAL 评测铁律升级**：不止 AUROC≥0.8，还要报 AURC/TCE。
- **CritiCal（critique-based calibration SFT）** ✅ 新增。LRMs（推理模型）事实校准稳定性优于 LLMs（归因于扩展推理）→ TAIS reasoning-native 有助校准。

---

## 1.5B 内生元认知可行性与最小训练配方

**1.5B 探针强度：未知，但有乐观间接证据**——SAPLMA/2606.02628 均在 7B–8B 测，**无 1.5B 直接数据**（§9 开放问题 #1，T1 首要观测）。三条乐观理由：① 信号**近似线性**（2606.02628：MLP 探针极少超线性 +0.01）——线性特征训练中**廉价且早出**；② `P(IK)` **可被训练**（Kadavath）不依赖巨参；③ Barkan 判别力**优于随机**普遍存在且**不随规模单调增**——"小模型也有信号，只是校准差"。**合理推断：1.5B 的知/不知线性方向大概率存在，瓶颈在校准而非存在性**（KAL 用线性头 W[2048,3] 的依据）。

**监测 vs 控制分离**（PMC9053853⚠️未核，Nelson-Narens 框架已核）：MetaM/MetaC 神经基质部分分离 → 工程映射：**检测**（探针只读 hidden state，零副作用）与**执行**（ITI 蒸馏头写残差/`<|recall|>` 触发 HRL）**必须分置**——共用路径引入反馈耦合（探针读到自己刚写的干预 → 自激）。要求：探针读 GDN 层输出处 PM-stream，干预头写 CSA 层残差前 PM-stream，**读写不同层**。

**为什么能力缩放不修复校准**：Barkan 直接证伪"更大就更准"——更新更大模型判别力一般不增、多步任务过度自信随进程加剧、推理模型 ≈ 或劣于非推理；"决策对估计理性，但估计乐观"——**瓶颈在估计（校准）**。Kadavath：`P(IK)` **必须显式训练**。结论：**元认知是独立训练目标，不能指望规模红利**。

**功能性自我模型 vs 现象学意识**：Betley 证明的是**功能性自我模型**（可测行为自知）；TAIS"自我认知"**仅此一义**，绝不主张现象学/主观体验意识。

**最小训练配方（T1/T2）**：
1. 预训练后期在 ℓ10/14/18 挂 `nn.Linear W[2048,3]` 三态头，以 **P(IK) 式辅助目标**参与训练（Kadavath 范式）；用"已知/未知"事实对构造协议。
2. **线性优先**（2606.02628 证明线性够用）。
3. **监测/执行分置**（探针只读 GDN 输出 PM-stream；执行走 ITI 头+`<|recall|>` 写 CSA 残差前 PM-stream——不同层读写避耦合）。
4. **校准对准绝对值**（Barkan：瓶颈在校准）；T2 定期重校准（Kadavath：新任务漂移）；三元奖励持续压制"自信编造"。
5. **失败经验课程**（T3 用失败 ICL 经验 + KAL 三态作天然过程奖励）。
6. **评测铁律**：探针 AUROC ≥ 0.8 **且**显著优于 token 概率、自报置信度两基线；避开 2606.02628 指出的"配对标签 vs 采样检测器"结构性误配。
