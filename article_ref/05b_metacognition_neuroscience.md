# 05b · 元认知/知识感知的神经科学基础 —— KAL 分层承重笔记

> 面向 TAIS **KAL（Knowledge-Awareness Layer）**分层元认知头（L1 知识感知 P(IK) 三态 + L2 情感 valence/arousal + L3 冲突检测 + ITI 干预门）的神经科学承重依据。
> 与 `05_neuroscience.md`（CLS/海马/睡眠固化承重）互补——本笔记专注**元认知神经基质**。
> ✅=已联网核实来源存在性（arXiv/DOI/期刊+年份）；每条标注【已确立证据】/【推测·新兴】。
> 核实日期：2026-07-26。

---

## 1. 元认知的神经基质：aPFC/rlPFC 如何计算 confidence

### 1.1 Fleming《Metacognition and Confidence: A Review and Synthesis》✅【锚点综述】
- **来源（已核实）**：Stephen M. Fleming, *Annual Review of Psychology*, Vol 75:241–268 (2024). DOI: `10.1146/annurev-psych-022423-032425`。
- **核心发现**：confidence 有"亚个人层"（uncertainty 表征，感觉/运动系统）与"个人层"（metacognition，对自我表现的信念）之分；二者经 **propositional confidence**（对自己假设性决策/动作的置信）桥接。元认知判断是**推断性**的、可与任务表现**背离**——这正是"内部诚实 ≠ 口头诚实"的神经根源。
- **对 KAL 的承重**：KAL L1 P(IK) 头读 hidden state 而非读输出 logits，正是"亚个人层 uncertainty 表征"的工程化——**绕过一阶输出，直接从内部表征线性读出置信**。支撑"探针读中层激活，不靠模型自报"的设计红线。
- **等级**：【已确立证据】（Annual Review 权威综述）。

### 1.2 rlPFC/aPFC 损伤选择性损害元认知 ✅
- **来源（已核实）**：Fleming et al., *Brain* 137(Pt 5):1483–1494 (2014)（relPFC 经颅磁扰/损伤→知觉元认知受损，**一阶准确率不变**）；Miyamoto et al., *Neuron* (2017/2018)（猴）；Shekhar & Rahnev, *eLife* (2018)。Baird et al., *J. Neurosci.* 33(42):16657–16665 (2013)。
- **核心发现**：rlPFC（BA 10/46）损伤/扰动改变 confidence 形成与元认知敏感性，**但不损害一阶任务表现**——元认知与一阶认知**可分离**。个体差异上，知觉元认知敏感性与 aPFC 结构/功能相关。
- **对 KAL 的承重**：**L1 知识感知头是独立于主干的"附加头"**（`nn.Linear W[d,3]`），训坏它不污染主干语言建模——对应"aPFC 损伤不影响一阶表现"的可分离性。支撑 KAL 作为**内生但可分离**部件，而非与主干耦合。
- **等级**：【已确立证据】（多物种、多方法收敛）。

### 1.3 监测（monitoring）与控制（control）可分离 ✅ —— 支撑 sense/inject 分置
- **来源（已核实）**：Nelson & Narens (1990) 监测-控制框架；Morales, Lau & Fleming, *J. Neurosci.* 38(14):3534–3546 (2018)【**领域通用 + 领域特异并存**】；Rouault et al. (2018)。
- **核心发现**：元认知**监测**（评估自己记忆/决策强度）与**控制**（用该评估引导行为/学习选择）**功能可分但神经基质部分重叠**；前额叶/额顶中线有**领域通用**基质，知觉（aPFC）与记忆（楔前叶/内侧顶叶）有**领域特异**基质。
- **对 KAL 的承重**：**这是 sense/inject 分置红线的直接神经依据**——监测头（sense，只读 GDN 输出层）与干预头（inject/ITI，只写 CSA 残差前层）读写不同层、不共享权重，对应"监测与控制神经基质部分分离"。监测/执行分置防止"探针读到自己的干预自激"。
- **等级**：【已确立证据】（J. Neurosci 实证 + 经典框架）。

---

## 2. Feeling-of-Knowing (FOK) 与 Tip-of-the-Tongue：P(IK) "不确定"态的神经机制

### 2.1 Koriat Accessibility Model ✅【FOK 计算蓝本】
- **来源（已核实）**：Koriat, A. "How do we know that we know? The accessibility model of the feeling of knowing." *Psychological Review* 100(4):609–639 (1993)；Koriat & Levy-Sadot, *J. Exp. Psychol.: LMC* 27:34–53 (2001)。
- **核心发现**：FOK **不直接读取目标是否存储**，而是基于**提取过程中可及信息（accessibility）的总量**——部分线索（cue familiarity）+ 可提取的片段信息（accessibility）超过主观阈值即触发"知道但想不起来"感。监测**不先于提取、而跟随提取**——通过尝试提取来评估目标是否"在那"。
- **对 KAL 的承重**：**P(IK) 三态的"不确定"态 = FOK 的工程化**——不是"完全没有"，而是"有迹可循但提取不充分"。Koriat 的"监测跟随提取"启示：KAL 的"空白检测"应在**尝试检索/生成之后**评估（后验），而非纯先验——对应我们用 hidden state 在生成中 sense，而非仅在 prompt 端预测。
- **等级**：【已确立证据】（Psych Review 经典 + 后续实证）。

### 2.2 TOT 作为元认知状态 ✅
- **来源（已核实）**：Schwartz & Metcalfe, *Memory & Cognition* / *Metacognition and Learning* 系列（TOT=伴随暂时不可及的意识感受，含认知层[尝试提取] + 元认知层[对该过程的反思]）；Schnyer et al., *Neuropsychologia* 42(7):957–966 (2004)【**右内侧前额叶损伤→FOK 判断失准**】；MDPI *Brain Sciences* 14(2):269 (2025) TOT 综述（aPFC + ACC 参与监测）。
- **核心发现**：TOT 是**元认知监测的产物**（内部评估记忆强度信号），非单纯提取失败；aPFC 与 ACC 参与该监测；**内侧前额叶损伤选择性损害 FOK**（Schnyer 2004）——FOK 与 confidence 的神经机制**可分离**（Alzheimer 患者 confidence  intact 但 FOK 受损）。
- **对 KAL 的承重**：**P(IK) 的"不确定"与"空白"应有不同神经/计算签名**——不确定（FOK/TOT，部分可及）≠ 空白（完全无可及信息）。支撑 L1 用**三态分类**（知道/不确定/空白）而非二值 confidence，且三分有神经分离依据。
- **等级**：【已确立证据】（损伤研究 + 行为收敛）。

---

## 3. 预测误差/惊讶度信号：ACC 冲突监测 + DA 预测误差 → "知识空白"形式化

### 3.1 Schultz DA 奖励预测误差 ✅【惊讶度信号的形式化祖本】
- **来源（已核实）**：Schultz, Dayan & Montague, "A neural substrate of prediction and reward." *Science* 275:1593–1599 (1997). DOI: `10.1126/science.275.5306.1593`；Schultz, *Dialogues Clin. Neurosci.* 18(1):23–32 (2016) 综述。
- **核心发现**：中脑多巴胺神经元编码 **signed reward prediction error (RPE)**——未预测奖励→正 RPE（激活）、完全预测→无反应、预测奖励缺失→负 RPE（抑制）。**DA 不是奖励检测器，是惊讶检测器**：output = 预测与实际的差。
- **对 KAL 的承重**：**"知识空白"信号的形式化模板**——KAL L1 的"空白"可形式化为**预测误差/惊讶度**：模型对某输入的预测置信 vs 实际可提取性的差距。对应我们用 next-token 预测正确性做 P(IK) 伪标签的机制（虽已发现该代理与真值错位，需真值锚——但 DA-RPE 给出"预测-实际差"的通用数学形式）。
- **等级**：【已确立证据】（教科书级，Science 1997 经典）。

### 3.2 ACC 冲突监测（Botvinick）+ EVC（Shenhav）✅【L3 冲突检测蓝本】
- **来源（已核实）**：Botvinick, Braver, Barch, Carter & Cohen, "Conflict monitoring and cognitive control." *Psychological Review* 108(3):624–652 (2001). DOI: `10.1037/0033-295X.108.3.624`；Shenhav, Botvinick & Cohen, "The expected value of control." *Neuron* 79:217–240 (2013)。
- **核心发现**：dACC **监测反应冲突**（不相容反应的共激活，如 Stroop），以此信号**调节 top-down 控制强度**；EVC 理论把它升级为"dACC 计算控制的期望价值，优化控制分配"。冲突 = 控制不足的信号。
- **对 KAL 的承重**：**L3 冲突检测头的直接蓝本**——检测"多个候选知识/回答的竞争共激活"，输出冲突强度作为"需要更多控制/更谨慎"的信号。对应 KAL L3 接 HRL 冲突仲裁（版本号+时间戳+置信度三路仲裁）与"冲突未决保留双方并标注分歧"红线。
- **等级**：【已确立证据】（Psych Review + Neuron 权威）。

### 3.3 dACC 编码 unsigned 惊讶（非冲突、非 signed RPE）⚠️【重要反面/精化证据】
- **来源（已核实）**：Hayden, Heilbronner, Pearson & Platt, "Surprise signals in anterior cingulate cortex." *J. Neurosci.* 31(11):4178–4187 (2011)（猕猴 dACC 单神经元记录）。
- **核心发现**：dACC 神经元对奖励的反应随**惊讶度**增强，但**不编码 signed RPE**（与 DA 不同）、**也不编码冲突**——而是 **unsigned reward prediction error（无符号惊讶度）**，驱动行为调整。
- **对 KAL 的承重**：**精化 L3 的数学形式**——冲突检测可能不是 signed error 也不是纯 conflict，而是 **|unsigned surprise|**（标量惊讶幅度）。启示 KAL L3 可用**标量惊讶度**（如预测分布的熵/与检索结果的偏差幅度）作为"需要干预"的触发，而非复杂符号误差。
- **等级**：【推测·新兴】（单研究，与 Botvinick 冲突理论并存——dACC 功能存在争议，KAL 应同时保留 conflict 与 unsigned-surprise 两个候选形式，T1 消融）。

---

## 4. 元认知的计算模型：哪种框架最适合迁移为 LLM 元认知头的数学形式

### 4.1 Bayesian/SDT confidence（meta-d'）✅【最适合迁移——推荐】
- **来源（已核实）**：Maniscalco & Lau, "A signal detection theoretic approach for estimating metacognitive sensitivity from confidence ratings." *Consciousness and Cognition* 21(1):422–430 (2012). DOI: `10.1016/j.concog.2011.09.021`；Fleming & Lau, *Front. Hum. Neurosci.* (2014) 元认知效率测量。
- **核心发现**：**Type-2 SDT**——用信号检测论把"置信能否区分自己的对错"量化为 **meta-d'**（元认知敏感性），并与一阶 d' 比较得**元认知效率（meta-d'/d'，M-ratio）**。把 confidence 形式化为对一阶决策证据的**二阶读取**，可分离敏感性与反应偏置。
- **对 KAL 的承重**：**这是 KAL L1 最干净的数学形式**——P(IK) 头 = 对主干（一阶）内部证据的**二阶 SDT 读出**；KAL 的校准质量可用 **M-ratio（meta-d'/d'）**量化，而非仅 AUROC。直接支撑"KAL 评测铁律升级：不止 AUROC，还要 meta-d'/M-ratio/AURC/TCE"。Bayesian/SDT 框架天然是**读表征的线性/低维操作**，与 KAL 朴素 `nn.Linear` 头同构。
- **等级**：【已确立证据】（已成为元认知测量标准工具）。

### 4.2 Global Workspace Theory（GWT）+ Shea & Frith ✅【功能性补充——confidence 随广播传递】
- **来源（已核实）**：Shea & Frith, "The Global Workspace Needs Metacognition." *Trends in Cognitive Sciences* 23(7):560–571 (2019). DOI: `10.1016/j.tics.2019.04.007`；Baars GWT 系列。
- **核心发现**：全局广播的表征**需要附带 confidence/uncertainty 度量**，以便工作空间对冲突/支持表征进行加权整合与决策——confidence 是全局工作空间**操控功能**的必要成分。
- **对 KAL 的承重**：支撑"知识块注入时需附带置信度"的设计——块载荷不只是内容，还带 confidence 元数据供下游加权（对应 Block Spec 的置信度字段 + 冲突仲裁三路之一）。但 GWT 本身是**功能层**理论，不直接给头的数学形式。
- **等级**：【已确立证据】（TICS 权威，功能性论证）。

### 4.3 Higher-Order Thought（HOT）⚠️【理论相关但迁移成本高】
- **来源（已核实）**：Brown, Lau & LeDoux, "Understanding the Higher-Order Approach to Consciousness." *Trends in Cognitive Sciences* 23(9):754–768 (2019). DOI: `10.1016/j.tics.2019.06.009`；Lau & Rosenthal (2011)。
- **核心发现**：一阶表征因**前额叶生成的二阶元表征**而成为意识；元表征是机制本身。
- **对 KAL 的承重**：理论上契合"KAL 是对主干的二阶表征"，但 HOT 主要解释**意识**（本项目诚实红线明确**不主张现象学意识**），且其二阶表征的具体数学形式不如 SDT 清晰。**迁移价值：概念框架 > 数学形式**。
- **等级**：【推测·新兴】（哲学-认知理论，非可执行模型）。

### 4.4 【结论】计算框架选型
> **首选 Bayesian/SDT（4.1）**：数学形式最干净（meta-d'/M-ratio），与 KAL 线性头同构，可量化校准效率，且明确**不涉及意识主张**（符合诚实红线）。GWT（4.2）作为"块带置信度广播"的功能性补充。HOT（4.3）仅作概念参考，不作为数学蓝本。

---

## 5. 杏仁体/情感对记忆的调制（McGaugh 原理）→ L2 情感头接写显著性

### 5.1 McGaugh 调制模型 ✅【L2 arousal 门控固化的直接依据】
- **来源（已核实）**：McGaugh, "Memory: a century of consolidation." *Science* 287:248–251 (2000)；McGaugh, "The amygdala modulates the consolidation of memories of emotionally arousing experiences." *Annual Review of Neuroscience* 27:1–28 (2004). DOI: `10.1146/annurev.neuro.27.070203.144157`；McGaugh, McIntyre & Power, *Neurobiol. Learn. Mem.* 78:539–552 (2002)。
- **核心发现**：**情感唤起（arousal）经杏仁核（尤其 BLA 基底外侧核）调制海马-皮层的记忆固化强度**——肾上腺素/糖皮质激素等应激激素作用于杏仁核，杏仁核再调制其他脑区的固化；**杏仁核不存内容，只调"固化优先级"**。情绪强的事件优先、更持久固化（β-肾上腺素能拮抗剂 propranolol 注入杏仁核可阻断该增强）。
- **对 KAL 的承重**：**L2 情感头（valence/arousal）接写显著性的神经蓝本**——arousal 维度 = 固化优先级信号，正对应"KAL L2 输出 arousal 门控 CA1 固化优先级"（增强 E 情感调制总线）。**杏仁核"只调优先级不存内容"精确对应 L2 头只输出门控信号、不作为知识载体**。
- **等级**：【已确立证据】（Science + Annual Review，多物种数十年收敛）。

### 5.2 arousal vs valence 的分工 ⚠️
- **来源（已核实）**：Cahill & McGaugh 系列（β-肾上腺素能激活主要介导 **arousal** 对固化的增强）；Russell 环形模型（valence-arousal 2D）。
- **核心发现**：**arousal（唤起度）是固化调制的主要驱动**（经应激激素-杏仁核通路）；**valence（效价）更多影响记忆内容的选择性关注/检索偏向**。
- **对 KAL 的承重**：支撑 L2 头**arousal 与 valence 分离**（`W[d,2]`）：**arousal→固化门控/写显著性**（对应 CA1 门 + SHY 归一化的强度），**valence→检索维度/情感匹配召回**（route_key 情感维度）。两维度功能不对称，不应混用。
- **等级**：【已确立证据】（arousal 主驱动）；valence 的具体检索作用【推测·新兴】（机制不如 arousal 固化清晰）。

---

## 承重总表（KAL 分层映射）

| KAL 部件 | 神经科学对应 | 关键来源 | 等级 |
|---|---|---|---|
| **L1 P(IK) 三态（知道/不确定/空白）** | FOK/TOT 元认知监测 + Koriat accessibility + rlPFC confidence | Fleming 2024 AnnuRev ✅；Koriat 1993 PsychRev ✅；Schnyer 2004 ✅ | 🟢 已确立 |
| **L1 数学形式（二阶读出/校准）** | Bayesian/SDT meta-d'、M-ratio | Maniscalco & Lau 2012 ✅ | 🟢 已确立 |
| **sense/inject 分置** | 监测/控制神经基质部分分离 | Nelson&Narens 1990 ✅；Morales/Lau/Fleming 2018 J.Neurosci ✅ | 🟢 已确立 |
| **L2 情感头（valence/arousal）接写显著性** | McGaugh 杏仁核调制固化（arousal→优先级） | McGaugh 2000 Science / 2004 AnnuRev ✅ | 🟢 已确立 |
| **L3 冲突检测** | ACC 冲突监测（Botvinick）+ EVC | Botvinick 2001 PsychRev ✅；Shenhav 2013 Neuron ✅ | 🟢 已确立 |
| **L3 数学形式（unsigned surprise）** | dACC 无符号惊讶度 | Hayden 2011 J.Neurosci ✅ | 🟡 推测·新兴（与冲突理论并存，T1 消融） |
| **ITI 干预门** | 控制通道（监测-控制分离的控制侧） | Shenhav EVC 2013 ✅ + ITI(2306.03341) 工程侧 | 🟢 已确立（机制） |

## 对 KAL 设计的三条核心启示
1. **L1 用 Bayesian/SDT 形式 + 三态分类**：P(IK) 不是二值 confidence，而是知道/不确定（FOK，部分可及）/空白（无可及）三态，有神经分离依据；校准用 meta-d'/M-ratio 量化，不止 AUROC。
2. **L2 的 arousal/valence 功能不对称**：arousal 主司固化门控（McGaugh），valence 主司检索偏向——`W[d,2]` 两维接到不同下游，不混用。
3. **L3 保留两个候选形式做消融**：conflict（Botvinick）与 unsigned surprise（Hayden 2011）在 dACC 并存争议，KAL L3 应在 T1 同时试"反应冲突"与"标量惊讶幅度"两种信号，实证选型。

## 诚实边界
- 全部神经科学证据来自**人类/动物元认知与记忆研究**，是 KAL 工程化设计的**灵感与约束来源**，不构成"LLM 有这些机制"的主张。
- KAL 的"知识感知/情感/冲突"均为**功能性类比**，对应内部可计算信号（hidden state 探针、门控标量），**不主张现象学意识或主观体验**（与项目诚实红线一致）。
- 【推测·新兴】条目（dACC unsigned surprise、valence 检索机制）证据强度弱于【已确立】，落地时应做消融而非默认采纳。
