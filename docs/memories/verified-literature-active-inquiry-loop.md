# TAIS Obsidian：主动求知闭环文献交叉验证报告（2026-07-28，Tavily 联网）

> 任务：为"元认知检测空白→主动发起学习→交叉验证→强化/修正"闭环找文献支撑。三视角（CS/ML、神经科学、认知科学）+ 数学形式。所有 arXiv 编号均真实出现在搜索结果中，标注 [已核实存在]/[未找到/存疑]。与既有记忆 verified-literature-metacognition.md / self-evolution.md / metacognition-neuroscience.md 互补不重复。

---

## A. 元认知触发的主动学习（计算机科学）

### A1. LLM 主动提问/澄清（uncertainty → ask）
- **[已核实] arXiv:2603.26233《Ask or assume? Uncertainty-Aware Clarification-Seeking in Coding Agents》**（Sun/Zhou/Du/Welleck/Neubig/Sap/Yang）——前沿模型具备开箱即用的"监测自身不确定性→主动求澄清"潜能，多智能体脚手架+定制 prompt 即可激发；RLHF 会损害校准致过度自信（引 Kapoor 2024 / Zhou 2024）。**承重启示**：P(IK) 探针驱动的提问能力在强基座上可能是 latent 的，TAIS 0.1B 需显式训练（内生 KAL 已在做），但接口设计可对齐"多智能体求澄清"范式。
- **[已核实] arXiv:2601.22139《Reasoning While Asking: Passive Solvers → Proactive Inquirers》**——把推理型 LLM 从被动求解变主动提问；CLAM（Kuhn 2023 合成模糊 query）、CollabLLM（Wu 2025）为前驱。**承重启示**：reasoning 与 asking 可联合训练，正合 TAIS"思考流形 + 主动回想"耦合。
- **[已核实] arXiv:2512.13159 SpeakRL**——RLVR 端到端训"何时问"，把澄清当控制原语（detect underspecification→ask→execute），25k 合成多轮对话。**承重启示**：RLVR 可训"提问"动作本身，奖励设计=提问后任务成功率，可作 TAIS W0 日志→睡眠期 RL 的范式。
- **[已核实] arXiv:2511.08798《Structured Uncertainty guided Clarification》**——GRPO 训 4 动作策略 {AskQuestion, CallTool, Decline, DirectAnswer}，数据用 When2Call（Ross 2025）。**承重启示**：这与 TAIS 内核 route() 的候选动作集同构——"提问/检索/拒答/直答"四选一可作为 HRL 路由头的动作空间。
- **[已核实] arXiv:2409.00557《Learning to Ask: When LLM Agents Meet Unclear Instruction》**——工具学习场景下主动求助；引 STaR-GATE（Andukuri 2024 self-play 迭代澄清）、Kuhn 2022。**承重启示**：澄清是 tool-use 可靠性的前置，TAIS 调用网络搜索前应先学"何时该查"。

### A2. 自我学习的工具调用（uncertainty-triggered retrieval）
- **[已核实] arXiv:2310.11511 Self-RAG**（Asai 2023）——reflection tokens 内化"何时检索/自评"为生成目标，按需检索可跳过。**承重启示**：`<|recall|>` 显式审计锚点的同构先例，已落设计。
- **[已核实] arXiv:2305.06983 FLARE**（Jiang 2023）——生成中 token 概率低于阈值即触发检索重写。**承重启示**：token 级不确定性门的最小实现；但 TAIS 实证（KAL 负结果）表明**语言建模置信度≠事实真假**，FLARE 的 logprob 门应换为真值锚 P(IK)。
- **[已核实] arXiv:2406.12534《Unified Active Retrieval (UAR)》**——统一四类主动检索判据（FLARE/Self-RAG/SKR/self-aware），分两路：knowledge-aware（指令事实相关性）vs self-aware（模型自觉不知）。**承重启示**：TAIS 的 HRL 检索触发应同时吃两路信号——KAL P(IK)（self-aware）+ 路由层任务类型（knowledge-aware）。
- **[已核实] SKR**（Wang 2023b，UAR 引）——收集模型 self-knowledge（knowns/unknowns）训分类器判"是否知道"，不知则检索。**承重启示**：与 KAL L1 三态直接同构，是"P(IK)→检索门"的已验证管线。
- **[已核实] DRAGIN**（2501.12835 引）——动态检索增强，兼顾效率评估。**承重启示**：检索有成本，触发门须计入效率预算（呼应 UAR"被动检索伤通用性"）。
- **[已核实] Toolformer**（Schick 2023，Learning to Ask 引 Schick 2024）——自监督学工具调用。**承重启示**：工具调用的"何时调"可自监督学，非必须人工标注。

### A3. 持续学习中"知道何时学"的元认知门控
- **[部分核实] arXiv:2406.08391《Large Language Models Must Be Taught to Know What They Don't Know》**——LLM 须经训练才能识别自身知识边界；Kadavath 2207.05221 / Tian 2023 prompting 产校准不确定性；Ulmer 2024 用其生合成数据训辅助置信模型。**承重启示**：核心论点直接支撑 TAIS"元认知必须显式训（KAL），非涌现"；Ulmer 路线=自生成数据训置信头，可借鉴于 KAL 自标注。
- **[已核实] Uncertainty-based continual learning with adaptive regularization**（Ahn/Cha/Lee/Moon, NeurIPS 2019，持续学习综述引）——不确定性驱动的自适应正则门控。**承重启示**：持续学习的"何时学/学多少"可用不确定性加权正则实现，对应 TAIS W2 写的强度门。
- **[未找到/开放]** 专门针对 LLM 的"元认知门控持续学习"（metacognition-gated CL）的成熟方法——现有 CL 多为数据流触发（loss plateau / 新任务到达），**"由模型自身 P(IK) 决定何时发起一次学习"仍是空白**，属 TAIS 独创点。

---

## B. 学习的验证与强化（计算机科学）

### B4. 自我修正实证：何时有效、何时退化
- **[已核实] arXiv:2310.01798《Large Language Models Cannot Self-Correct Reasoning Yet》**（Huang/Chen/Mishra/Zheng/Yu/Song/Zhou，Google DeepMind/UIUC，ICLR 2024，被引 540）——**内在自我修正（无外部反馈）不仅无效且常致性能退化**；既往正面结果多因用了 oracle label 决定何时停止修正（不公平）；多智能体辩论等效采样数下不超 self-consistency（有效来自一致性非真修正）。**承重启示（决定性）**：**TAIS 闭环绝不能依赖模型"自己觉得自己改对了"——修正必须由外部信号（检索证据/用户反馈/校验集）门控**。这直接支撑红线"draft→固化必须验证门"。
- **[已核实] arXiv 2505（ACL2025 Findings, aclanthology 2025.findings-acl.331）《Self-Correction is More than Refinement》**——VLM 上内在自我修正不可靠、敏感于 prompt 表述、可引入 bias（引 Xu 2024）。**承重启示**：自我修正跨模态同样不可靠，加强"外部锚"必要性。
- **对照（正面谱系）**：CRITIC（工具接地 critiquing 真有效）、Reflexion（Shinn 2023）、Self-Refine（Madaan 2023）——**共性：凡有效的"自我修正"都外接了工具/环境反馈**。承重启示：TAIS 修正回路的设计原则 = "Reflexion 的记忆 + CRITIC 的工具验证 + 绝不裸自我修正"。

### B5. 交叉验证新知识的机制 + RL 奖励设计
- **[已核实] arXiv:2509.25760 TruthRL**——GRPO + **三元奖励 {correct +1 / abstain / hallucinate −1}**，减幻觉 28.9%；指出二元 RLVR 把"拒答"与"答错"混为一谈，反而惩罚校准的"我不知道"；truthfulness = w1·Acc + w2·Unc − w3·Hall。**承重启示（最直接）**：**奖励"学到真知识"的设计已成熟——三元奖励区分答对/拒答/幻觉；TAIS 睡眠期 RL 应用三元而非二元奖励**，且 abstain 不应被重罚（否则模型宁可幻觉）。
- **[已核实] arXiv:2601.20126《Rewarding Intellectual Humility》**——RLVR 改奖励激励智识谦逊：答对得奖、部分 abstention 得小奖（r_abs≈−0.25~0.3）、答错受罚；中等 abstain 奖励降幻觉不重伤精度；开放域因探索不足需先 SFT 教 abstain。**承重启示**：abstain 奖励的量级窗口窄（≈0~0.3），过大则模型过度拒答——TAIS 需调此超参；且**先用 SFT 教"会拒答"再上 RL**（两阶段，正合我们真值锚微调先行）。
- **[已核实] arXiv:2203.11171 Self-Consistency**（Wang 2022，ICLR 2023）——多路径采样+多数投票。**承重启示**：自洽性可作零成本一致性校验，但 2310.01798 警示其收益来自"一致性"非"正确性"——只能作辅助信号，不能当真值。
- **[已核实] arXiv:2311.17311 Universal Self-Consistency**——自由形式答案的自洽（LLM 判一致性）。**承重启示**：开放域答案的一致性可测，扩展了自洽适用范围。
- **[已核实] arXiv:2212.09561 Self-Verification**（Weng 2023）——模型对自建正确答案/外部知识源自验证。**承重启示**：验证须挂外部知识源，裸自验证弱。
- **[已核实] arXiv:2505.09031**——CoT+RAG+Self-Consistency+Self-Verification 组合降幻觉。**承重启示**：交叉验证=多机制叠加（检索+自洽+自验证），单一机制不足。
- **RLVR 边界警示 [已核实]**（aipolicytakes 引 2025 论文）——RLVR 不诱发基座新能力，只锐化已有分布。**承重启示**：RL 只能"加强/修剪"已有观念，**全新知识须先经知识块/SFT 注入，再 RL 加固**——顺序不能反。

### B6. 防错误固化/信念漂移
- **[已核实] arXiv:2307.01850《Self-Consuming Generative Models Go MAD》**（Alemohammad/Baraniuk，ICLR 2024）——自消费循环致 MAD（model autophagy disorder）；掺固定真实数据只延迟不阻止退化。**承重启示**：TAIS 自学的产出若回灌训练流，必须掺足量 ground-truth 且**合成占比有上限**——支撑"markdown 源代码形态=最终审计/回滚依据"红线。
- **[已核实] Shumailov et al. Nature 631:755 (2024)《AI models collapse when trained on recursively generated data》**——递归生成数据训练致模型坍缩（尾部分布丢失）。**承重启示**：睡眠固化的训练数据必须是"真实锚 + 有限合成"，且保留真实数据分布的尾部。
- **[已核实] arXiv:2404.01413《Is model collapse inevitable? Accumulating data avoids collapse》**——**数据累积（不替换）时测试误差有有限上界，可避免坍缩**；替换数据则误差随迭代线性增长。**承重启示（关键工程解）**：**TAIS 知识块库应"累积不覆盖"（页表版本化保留旧块），这与设计"冲突不静默覆盖、版本号+时间戳仲裁"红线完全互证**——累积式块存储在理论上抗坍缩。

---

## C. 神经科学与认知科学

### C7. 元认知监测触发主动求知（epistemic curiosity）
- **[已核实] Metcalfe et al.《Epistemic curiosity and the Region of Proximal Learning (RPL)》**（Columbia PDF）——**人并非在"确定不知"时最求知，而是在 TOT（舌尖状态）/RPL 区（接近知道但未及）最好奇**；内省调查仅 25% 猜到 TOT 最好奇（多数人误以为"完全不知"最求知）→ **支持预测误差模型而非"知识空白"模型**；高置信错误的纠正反馈诱发 P3a ERP（惊讶+注意聚焦+增强编码）。**承重启示（反直觉、极重要）**：**TAIS 主动求知的触发点不应是 P(IK) 最低（完全空白），而应是"中等不确定/RPL 区"——完全空白处学习成本过高，系统应优先在"差一点就知道"的边缘区求知**。这给 KAL 触发门一个非平凡的设定：求知优先级 ∝ 可学习性（learning progress），非 ∝ 无知程度。
- **[已核实] bioRxiv 157644《The control of epistemic curiosity in the human brain》**——预测加工启发的 EC 整合模型：信息寻求始于"知识空白觉察"，新信息与先验比较诱发 epistemic surprise 促进编码；好奇心与平均惊讶呈反比，由 **rlPFC** 血氧活动逐试次调控；高好奇问题诱发纹状体（奖赏系统）BOLD（Gruber 2014/Kang 2009）。**承重启示**：① rlPFC 与 KAL 的 PFC 定位一致（Fleming rlPFC 损伤损元认知不损一阶）；② "好奇心=奖赏"（纹状体）支撑把"学到东西"本身当内在奖励信号。
- **[已核实] Maril 2003 NeuroImage（FOK fMRI）**——K>FOK>DK 分级激活（左 PFC/顶叶/ACC），FOK 是"知与不知之间的中间提取态"的神经对应。**承重启示**：KAL 三态 {知道/不确定/空白} 有神经分级激活证据，"不确定"态（FOK）是独立可辨神经状态，非二值的过渡噪声。
- **[已核实] Metcalfe 2012 J.Cogn.Neurosci 24:1571（hypercorrection 神经基础）**——高置信错误被纠正时 ACC（冲突监测+增强编码）+ dlPFC（工作记忆/抑制错误响应）+ 右 TPJ（"他人所信与我矛盾"的 ToM）激活。**承重启示**：错误修正的神经回路 = 冲突检测（ACC）+ 旧响应抑制（dlPFC）+ 外部真值锚（TPJ/他人）——三件套与 TAIS"L3 冲突检测 + erase gate 抑制旧关联 + 检索外部锚"结构同构。

### C8. 错误检测与信念修正（修正 vs 加强）
- **[已核实] Hypercorrection effect（Metcalfe 谱系）**——**高置信错误被纠正后反而记得更牢**（vs 低置信错误）；机制=高置信错误引发更强惊讶/注意→增强编码。**承重启示（直接回答"修正 vs 加强"）**：**模型对"高置信错误"的修正收益最大——TAIS 应优先纠正那些 P(IK) 高但事实错误的块（最危险），而非低置信块**。这给出修正优先级的排序规则：置信度×错误度乘积最高者优先。
- **[已核实] O'Reilly et al.（pupillometry+fMRI，PMC9955423 引）**——瞳孔惊讶可分解为两神经过程：**Shannon 信息（后顶叶，预测误差/惊讶）与 KL 散度（前扣带 ACC，信念修正量）可分离**；KL 散度（先验→后验的不相似度）预测信念修正。**承重启示（数学形式白送）**：**"惊讶"与"信念修正量"是可分离信号，ACC 专司 KL 散度（修正幅度）——TAIS 应分别测 (a) 预测误差（触发检测）与 (b) 先验-后验 KL（决定修正强度）**，且 L3 冲突头对应 ACC 的 KL 角色。
- **[已核实] PMC9955423《Seeing the Error in My "Bayes"》（儿童 pupillary surprise）**——Bayesian 模型（Shannon vs KL）比较：KL 散度（需 accommodate 信息时信念态变化大）比 Shannon 信息更贴信念修正；冲突觉察提高结果信息量的主观价值促进修正。**承重启示**：Piaget 同化/顺应在 Bayesian 框架下的操作化——**顺应（accommodation）= 大 KL 信念更新，同化（assimilation）= 小 KL 吸收**；TAIS 知识块的"并入主干 vs 新建块"可用 KL 阈值仲裁。
- **[已核实] arXiv:2305.10937（Hierarchical Gaussian Filter 变体）**——层级信念更新：节点 belief update = precision-weighted prediction error，volatility child/parent 分层传递。**承重启示**：HGF 给出"何时大改 vs 小改"的形式化——精度（precision）加权决定更新幅度，低精度先验大改、高精度先验小改（呼应 B11 信任度加权）。

### C9. 单次情景学习巩固不污染已有知识
- **[已核实] McClelland/McNaughton/O'Reilly 1995 Psych.Rev（CLS 原始论文）**——海马快学（稀疏模式分离码）存单次情景、避免直接改新皮层致灾难性干扰；海马经 interleaved replay 慢训新皮层，最小干扰地整合。**承重启示**：这是 TAIS"运行时只写知识块（海马）、睡眠期才固化入主干（新皮层）"双速架构的原始理论依据，已落设计。
- **[已核实] Schapiro et al. 2017（PMC5124075，CLS within hippocampus）**——海马内部也是 CLS 微宇宙：DG/CA3（TSP 通路）稀疏高抑制→模式分离存独特情景；CA1 直接 MSP 通路全连接低抑制→重叠表征→跨情景统计规律。**承重启示**：**"防干扰"与"提取共性"在海马内部已分工——TAIS 的 HRL（存情景/模式分离）与 GDN 记忆层（提取统计规律）可借鉴 TSP/MSP 双通路设计**，不必强求单一载体兼顾。
- **[已核实] PNAS 2022（2123432119，hippocampus-neocortex autonomous model + C-HORSE）**——睡眠依赖巩固模型：海马 MSP 快学统计+TSP 快学情景，离线回放训练新皮层；**McClelland 2013 修正：新皮层学习快慢取决于与先验知识一致性——一致则快（同化），不一致则慢（防干扰）**。**承重启示（直接回答"防错误单次经验固化"）**：**与先验一致的新知识可快速并入（低风险），与先验冲突的必须慢速+反复校验（高风险）——TAIS 的 CA1 巩固门应以"与既有知识的一致性"作为固化速度的调控变量**：一致→快固化，冲突→留 draft 区+版本仲裁。单次错误经验因与先验冲突会被天然挡在慢通道。

---

## D. 数学形式

### D10. 主动学习/好奇心框架与 P(IK) 耦合
- **[已核实] Lindley 1956 / Bayesian Experimental Design + EPIG（PMLR v206 bickfordsmith23a《Prediction-Oriented Bayesian Active Learning》）**——BALD=参数的 EIG，**EPIG=预测的 EIG（直接在目标输入分布上减预测不确定性）**；EPIG 优于 BALD 当目标是下游预测。**承重启示**：TAIS 求知的目标不是"减模型参数不确定性"而是"减对未来 query 的预测不确定性"→ **EPIG（非 BALD）才是与 P(IK) 耦合的正确形式**：P(IK) 低标记"何处不知"，EPIG 排"先学哪个最减未来预测不确定性"。
- **[已核实] Oudeyer/Gottlieb/Lopes《Intrinsic motivation, curiosity and learning》（PBR 2016）+ Schmidhuber 1991 + Oudeyer & Kaplan 2007**——**Learning Progress（LP）假说**：内在奖励 ∝ 预测误差的下降率（非误差本身）；有机体对"太易（误差低）或太难（误差高但不可约）"都失去兴趣，聚焦中等可学习区——**与 C7 RPL 互证（同一现象的两面）**。**承重启示（最适合与 P(IK) 耦合）**：**Learning Progress 是最适配 P(IK) 的好奇心形式**——TAIS 求知目标选择 = argmax[E(学习进展)] = 选"P(IK) 中等偏低但可经检索/提问快速提升到高"的区，避开已掌握区与不可学习区。EPIG 管"学哪个值"，LP 管"哪个学得进"，二者应联合。
- **[已核实] arXiv:1802.10546《Computational Theories of Curiosity-Driven Learning》**——好奇心计算理论综述（知识空白/预测误差/学习进度/信息增益分类）。**承重启示**：框架选型的分类学参考。
- **[已核实] MAX / Model-Based Active Exploration（Bayesian DL workshop 2018）**——集成模型 disagreement 作探索效用；Sun et al. 2011 最优 Bayesian 好奇心（EIG 期望可加性 + DP 最大化 IG）。**承重启示**：disagreement-based 效用可作为 EPIG 的廉价近似（多采样分歧≈不确定性）。

### D11. 信任度加权信念修正
- **[已核实] arXiv:2506.16015 BEWA《Bayesian Epistemology-Weighted AI》**——信念更新代数：后验 π(φ,t) 经 source-weighted 证据（作者可信度/领域信任校准/复现史）+ 时间衰减阻尼 + 显式矛盾处理（概率散度+熵阈值调制过时/被驳/异常数据影响）。**承重启示（最贴题）**：**信源可信度加权的 Bayesian 修正已有形式化（BEWA）——TAIS 知识块应带 source_credibility 元数据，修正幅度 ∝ 信源权重×时间衰减**；矛盾证据经"概率矛盾模型"处理而非覆盖（呼应冲突不静默覆盖红线）。
- **[已核实] Stanford Encyclopedia of Philosophy: Bayesian Epistemology**——条件化原理（Ratio Formula）+ Zeroing/Rescaling/Resetting；**证据须以概率 1 持有是理想化，实践中证据本身带 credence（信源可信度）**。**承重启示**：标准 Bayesian 条件化假设证据为真，TAIS 必须用 **Jeffrey 条件化**（证据带概率）处理"可能错的检索结果"——新知识的证据强度 P(E)∈(0,1) 直接进入更新，P(E) 低则弱更新。
- **[已核实] Precision-weighted prediction error（Predictive Processing）**——Δμ = [π_s/(π_p+π_s)]·(x−μ_p)，π=精度（1/方差）。**承重启示（最简可用形式）**：**信任度加权修正的最简数学 = 精度加权——证据精度高（可信源/多源一致）则大权，先验精度高（既有强知识）则抗改**。TAIS 块修正可用此式：信源可信度→π_s，块现有置信度→π_p。

---

## 总判定

### ✅ 已有成熟文献支撑（可直接借鉴）
1. **不确定性触发检索/提问**：Self-RAG/FLARE/SKR/UAR + 主动澄清 RL（SpeakRL/Structured-Uncertainty-GRPO）——"何时问/何时查"的训练管线完整，动作空间 {AskQuestion, CallTool, Decline, DirectAnswer} 可直接用作 HRL 路由头输出。
2. **奖励"真知识"的 RL 设计**：TruthRL 三元奖励 + Rewarding Intellectual Humility 的 abstain 奖励窗口——成熟，直接可用（关键：abstain 不重罚、先 SFT 教拒答再 RL）。
3. **防坍缩/防错误固化**：MAD/Nature 坍缩 + **累积式存储抗坍缩（2404.01413）**——"累积不覆盖"有理论保证，与设计红线互证。
4. **好奇心触发点**：RPL（Metcalfe）+ Learning Progress（Oudeyer/Schmidhuber）——触发区="中等不确定可学习区"非"完全空白"，神经（rlPFC/纹状体/ACC-KL）与计算两侧互证。
5. **信任度加权修正**：BEWA + Jeffrey 条件化 + 精度加权预测误差——数学形式现成。

### ⚠️ 开放问题（TAIS 需独创外推）
1. **LLM 的元认知门控持续学习**——"由 P(IK) 自主决定何时发起一次学习"无成熟先例（CL 多为数据流触发）。
2. **修正 vs 加强的仲裁在 LLM 权重/块上的操作化**——Hypercorrection（高置信错误优先改）是心理学结论，如何在知识块/LoRA 上实现"置信×错误乘积排序修正"无直接文献。
3. **P(IK)×EPIG×LearningProgress 三信号联合的求知目标排序**——各自成熟，但三者在单模型内的耦合公式需自拟。
4. **"顺应（大 KL 新块）vs 同化（小 KL 并入）"的块级仲裁**——CLS/McClelland2013 给原则，LLM 块实现的 KL 阈值需自标定。

### 🔴 最关键 3 个设计约束
1. **防错误固化 = 绝不裸自我修正 + 累积式存储**——2310.01798 证明无外部锚的自我修正会退化；所有修正必须经检索证据/用户反馈/校验集门控，且知识块库累积不覆盖（版本化），单次冲突经验走慢通道（CA1 门按先验一致性调速）。
2. **防过度自信 = 三元奖励 + 真值锚 P(IK)**——Barkan 证规模不自动修校准、RLHF 致过度自信；必须用真值锚（非语言建模置信度）训 P(IK)，且 RL 奖励区分答对/拒答/幻觉（abstain 奖励窗口 ≈0~0.3，先 SFT 后 RL）。
3. **验证成本 = 求知目标排序必须计入可学习性与 EPIG**——好奇心不是"哪里不会学哪里"（RPL：完全空白区学习成本过高）；优先"中等不确定×高学习进展×高 EPIG（减未来预测不确定性）"的区，完全空白区与已掌握区都降权——这把有限的检索/提问/睡眠固化预算花在刀刃上。

---
*导出自 /memories/repo/verified-literature-active-inquiry-loop.md（2026-07-30 同步快照）。*
