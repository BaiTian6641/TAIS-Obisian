# 07 · KAL 数学与工程规范（2026-07-26，文献核实整合）

> 4 个并行子代理 tavily 联网核实（arXiv 编号逐条验证）整合。为 KAL 分层元认知头确立**更稳固的数学形式与工程实现**。配套持久记忆 `/memories/repo/kal-math-spec.md`。
> **诚实红线**：本项目"元认知/自我认知"仅指**功能性自我建模**（P(IK)/情感/冲突的可计算信号），绝不主张 LLM 有神经机制或现象学意识。

## 0. 勘误（覆盖既往笔记）
- **MIND = arXiv:2403.06448**（Su et al. 清华，HELM + token 级实时幻觉检测）；2509.05714 是同名多模态知识编辑 CogEdit，非幻觉检测。
- **Truth is Universal = Bürger/Hamprecht/Nadler 2407.12831**（非 Pacchiardi 2309.15840）。
- **Attention Satisfies = Chuang 2502.13490**；**Still No Lie Detector = Levinstein 2307.00175**。

## 1. 实证闭环（本项目已走通）
M2 事后探针 **0.945**（fake 0.979）→ 在线自标注 **0.433**（伪标签=next-token 正确性，**测流畅度非真假，方向错位**）→ 真值锚微调 **0.998**（fake，跨种子）→ 泛化定界（shuffled 0.576：学的是"语义连贯真实陈述"非"流畅文本"）。
**核心结论**：信号一直在中层线性可解（2606.02628 量化下 0.904–1.000 互证），**成败全在标签协议——锚"事实真假"而非"语言建模置信度"**。

## 2. L1 P(IK) 探针（知识空白检测）

| 维度 | 推荐形式 | 依据（已核实） |
|---|---|---|
| 探针形式 | **线性 diff-in-means**（$w=\Sigma^{-1/2}(\mu_+-\mu_-)$），`nn.Linear(d,3)` 三态 | Marks&Tegmark 2310.06824；2606.02628（MLP 仅 +0.01） |
| 决策面 | **truth 是 2D 子空间**（negation 轴+truth 轴）；三态头天然容纳 ≥2D | Bürger 2407.12831 |
| 标签协议 | **真值锚**（fake=blank/real=known）+ **contrast-pair**（肯定/否定/未知三元组） | 本项目实证；Levinstein 2307.00175（negation 失败） |
| 免真值备选 | **SEP 语义熵回归**（零外部标签，单次前向） | Kossen 2406.15927 |
| 多层融合 | ℓ10/14/18 各挂头，**AUROC 软加权** $w_l=\mathrm{softmax}(\mathrm{AUROC}_l/T)$；ℓ8 注意力熵侧信道 | Wang 2605.26366；Chuang 2502.13490 |
| 校准 | **isotonic regression** → **conformal quantile 拒答阈值**（有限样本覆盖保证） | Kossen 2406.15927；Mohri&Hashimoto 2402.10978 |
| 评测 | AUROC≥0.8 **且 AURC/TCE 达标**（非仅 AUROC） | Su 2603.21172 |
| 防 Goodhart | 探针 `requires_grad=False`、只读、梯度隔离、feature clipping | NeurIPS 2025 激活监控；INSIDE 2402.03744 |

## 3. L2 情感头（valence/arousal）
- **VA 线性子空间已确立**（大模型）：Anthropic 2604.07729（circumplex，v/a 不相关 r=-0.02，线性可提取+steer）；Sun 2604.03147（开源 Russell 环形）。**1.5B 复现 = T1 观测项**。
- **形式**：残差流 → PCA+ridge 学两**正交**轴 → (v,a) 回归；`W[d,2]` 两列**正交化约束**。
- **写显著性门控（功能不对称）**：**arousal a 进写门**（McGaugh 04 巩固增益主驱动）、**valence v 进极性**；多显著并发用 **softmax 竞争**（Mather ABC/GANE 2011：高唤醒选最强压次强）；a 作 W0 巩固标签供睡眠 CA1 门（Payne&Kensinger 2018，与读写不对称同构）。

## 4. L3 冲突头
- **中层残差流线性 logistic 三态**（一致/参数优先/上下文优先）：2410.16090（中层升起，logistic 90%）；分类框架 Xu 2403.08319（三分）。
- **取整体残差非少数头硬路由**（2503.10996：memory/context 头非互斥、superposition）。
- **存争议（T1 消融）**：dACC 编码 unsigned surprise 非 signed RPE（Hayden 2011）——conflict vs unsigned-surprise 两候选。

## 5. ITI 干预门
- **形式**：top-K 头 mass-mean shift $h \leftarrow h+\alpha(\mu^+-\mu^-)$（ITI 2306.03341）；RepE 对比向量（Zou 2310.01405）。
- **红线级警示**（Braun 2505.22637）：强度↑连贯性/忠实性↓、属性一致幻觉、小模型更敏感 → **α 有界（残差模长 ±0.1 级）、仅触发时、单方向**，"steering 后人效不降"纳入退出标准（对齐 M5 Δ+0.0001）。

## 6. 神经科学承重（功能性类比）
- 监测/控制分离（Morales/Lau/Fleming 2018；Nelson&Narens 1990）→ **sense/inject 分置红线直接依据**。
- rlPFC 损元认知不损一阶（Fleming 2014）→ L1 头与主干可分离。
- FOK 基于提取可及性（Koriat 1993）+ 三态神经分离（Schnyer 2004）→ 空白检测应**后验**（生成中 sense）。
- 首选计算模型 **Bayesian/SDT（meta-d'）**（Maniscalco&Lau 2012）——数学最干净、与线性头同构、不涉意识。

## 7. 落地优先级（对当前代码的增量）
1. **L1 校准层**：内生头 logits → isotonic/temperature scaling → conformal 拒答阈值（替代裸 logit 阈值）——让 `<|blank|>`/缺页声明有有限样本覆盖保证。**评测升级为 AUROC+AURC+TCE**。
2. **contrast-pair 训练数据**：真值锚微调加入否定/未知三元组（防 negation 伪相关，Bürger 2407.12831 证明单方向探针在否定句必失败）。
3. **多层融合**：ℓ10/14/18 三挂点 + AUROC 软加权（当前仅 ℓ8 单点）。
4. **L2/L3 头**：L2 VA 正交回归（T1 观测 1.5B 可复现性）；L3 中层 logistic 冲突头（数学形式先做 conflict vs unsigned-surprise 消融）。
5. **ITI 干预头**：top-K 头 mass-mean shift，α 有界仅触发（副作用红线）。

> 全部 arXiv 编号已经 4 子代理逐条联网核实存在性。新增关键引用：2310.06824 / 2407.12831 / 2406.15927 / 2605.26366 / 2502.13490 / 2402.10978 / 2603.21172 / 2307.00175 / 2604.07729 / 2604.03147 / 2410.16090 / 2403.08319 / 2503.10996 / 2505.22637 / 2306.03341 / 2310.01405 / 2403.06448(MIND勘误)。
