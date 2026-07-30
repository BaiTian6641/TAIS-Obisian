# TAIS Obsidian：KAL 元认知探针数学栈已核实文献（2026-07-26 核实，article_ref/04 配套）

> 全部 arXiv 编号已联网核实存在性。核心结论：线性+diff-in-means 为主干，校准用 isotonic+conformal，多层融合用 AUROC 加权 stacking，冻结防 goodhart。

## 关键修正（务必采纳）
- **"Attention Satisfies" 的正确 arXiv = 2502.13490**（Chuang et al.，非 Farquhar；2025-02）。用首层注意力熵/sink 模式做零成本幻觉探针，与 KAL 侧信道互补。
- **"Still No Lie Detector" arXiv = 2307.00175**（Levinstein & Herrmann，2023）。核心：探针在否定句("not")上泛化失败——diff-in-means 也会学到位置/表面伪相关，必须用 contrast-pair 训练+跨格式验证。
- **"Truth is Universal" 是 Bürger, Hamprecht, Nadler**（arXiv 2407.12831，NeurIPS 2024），非 Pacchiardi；2D truth subspace 解释 negation 泛化失败，94% acc 跨格式。Pacchiardi 那篇是 2309.15840（black-box lie detector via follow-up questions）。
- **"Generalization of Truth Directions" 是另一篇**（Bürger, Liu, Pacchiardi, Farquhar, arXiv 2410.22388 待二次核实——本会话搜索结果主要命中 2407.12831 与 Bao et al. ACL2025《Probing the Geometry of Truth》）。truth direction 高度 layer-dependent，跨层泛化弱 → 支撑多层融合必要性。

## 探针形式（Q1）已核实要点
- **线性足够**：2606.02628（4-bit 量化 7-8B，单中层线性 0.904-1.000 AUROC，MLP 极少超线性 +0.01）。支撑 KAL `nn.Linear W[2048,3]`。
- **MLP 无增益**：Belrose et al. 2312.01037（Quirky LM）：LogR / diff-in-means / LDA 三者 AUROC 接近（0.63-0.85 区间，视数据集），MLP 无系统优势。
- **SAE 用于机制发现，不用于在线探针**：Ferrando 2411.14257——SAE 找到"entity recognition"方向可 steer 拒答/幻觉，但 SAE 训练成本高、推理开销大；在线探针仍用线性读 SAE 潜在维度或原始 hidden state。
- **CCS 弱于监督**：Burns 2212.03827 CCS 无监督但易崩（Belrose 表：CCS 在多数据集上 0.5-0.7，远低于 LogR/diff-in-means 0.74-0.85）；contrast-pair 训练（LogR on contrast pair）才是稳定版。
- **mass-mean probing**：Marks & Tegmark 2310.06824——diff-in-means 方向泛化最好且因果介入最强；PCA top-1 也有效但略脆。

## 特征方向（Q2）已核实要点
- **diff-in-means = 最优默认**：μ_true−μ_false 归一化即方向；Marks 2310.06824 证明其跨数据集泛化 > LogR，且因果 steer 效果最强。
- **监督 LogR 仅在样本充足时略优**，且容易过拟合表面特征（Levinstein 2307.00175 否定句失败）。
- **PCA top-1**（无监督）在 geometry-of-truth 数据集上可用，但跨格式不稳。
- **2D truth subspace**：Bürger 2407.12831——truth 不是单方向而是 2D 平面（negation 轴+truth 轴），单方向探针在否定句上必然失败 → KAL 三态头 W[2048,3] 天然容纳 2D 子空间（3 类 logit 张成 ≥2D 决策面）。

## 多层融合（Q3）已核实要点
- **单探针弱于融合**：Automatic Layer Selection（2605.26366，ICML 2026）系统比较——峰值层因数据集/模型而异，固定单层次优；跨层加权/stacking 稳定提升 2-5 AUROC。
- **融合数学**：各层 logit z_l，学习权重 w_l（凸组合）或 stacking（logistic on [z_1..z_L]）；2605.26366 用验证集 AUROC 作软权重 w_l = softmax(AUROC_l/T) 效果接近全 stacking 且更省。
- **Lookback Lens**（Chuang 2407.07071）+ **Attention Satisfies**（2502.13490）：注意力侧信道与 hidden-state 探针互补，融合后跨模型迁移（7B→13B 不重训）。

## 检测→校准（Q4）已核实要点
- **Platt scaling**（sigmoid on logit）最简单但假设 logit 已近似校准——对探针常过自信。
- **Isotonic regression**：Kossen 2406.15927 SEP 用 isotonic 把 probe score → P(correct)，OOD 下仍保校准；是探针校准默认推荐。
- **Temperature scaling**：对探针 logit 单参数 T 拟合验证集 NLL；比 Platt 更稳当样本少。
- **Conformal**：Mohri & Hashimoto 2402.10978（conformal factuality，back-off 到 less specific）+ Su et al. 2603.21172（AURC/TCE 部署指标）——conformal 给出有限样本覆盖保证，是"诚实降级"的数学基础；KAL 缺页/拒答阈值应用 conformal quantile。
- **AURC/TCE 是部署铁律**（Su 2603.21172）：AUROC 掩盖高置信错误，必须报 risk-coverage 曲线下面积 + target calibration error。

## 鲁棒性/防 goodhart（Q5）已核实要点
- **OOD 失败主因=伪相关**：Levinstein 2307.00175（否定句）、Bao et al. ACL2025（跨 logical transformation 不一致）、Hallucination Neurons 2604.19765（跨域 AUROC 0.783→0.563）——探针学到的是"知识回忆过程"而非真值本身。
- **防 goodhart = 冻结+只读**：NeurIPS 2025《LLMs Are Capable of Metacognitive Monitoring and Control》——模型能混淆内部表征逃避探针 → 探针权重绝不进生成损失，只读 hidden state，干预头写不同层（监测/执行分置）。
- **feature clipping**（INSIDE 2402.03744）：测试时截断极端激活值，防模型"过度自信"绕过探针，与 EigenScore 互补。
- **对比训练数据**：Belrose 2312.01037 + Bürger 2407.12831——必须用 contrast pairs（同句 ±not/±答案）训练，强迫探针学语义而非表面。

## 推荐 KAL 探针数学栈（工程落地）
1. **探针形式**：`nn.Linear(d→3)`（知/不确定/空白），contrast-pair 训练（同 statement 配肯定/否定/未知答案三元组）；不用 MLP/SAE 做在线头。
2. **方向初始化**：用 diff-in-means μ_known−μ_unknown 初始化 L1 P(IK) 方向，再真值锚微调（对齐我们 0.998 内生头路径）。
3. **多层融合**：ℓ10/14/18 各挂头 → 每层 logit z_l → softmax(AUROC_l/T) 加权或 3 维 stacking LogR；保留 ℓ8 作侧信道（对齐 0.945 事后探针）。
4. **校准**：isotonic regression 把融合 score → P(correct)；再套 conformal quantile 定"拒答/缺页"阈值（目标 coverage 1−α）；评测报 AURC+TCE 而非仅 AUROC。
5. **鲁棒**：探针权重冻结（requires_grad=False），梯度隔离（HRL 红线）；feature clipping 截断 activation 极端值；训练集强制含 negation/格式变换 contrast pairs；定期用 held-out 格式（长对话、代码、数学）做 OOD 回归。

---
*导出自 /memories/repo/verified-literature-kal-probe.md（2026-07-30 同步快照）。*
