# KAL 元认知/知识探测已核实文献（2026-07-26 联网逐条核实）

> 用途：KAL L1 P(IK) / L2 情感 / L3 冲突的数学形式与工程实现依据。所有 arXiv 编号已逐条核实存在性。

## 编号勘误（对既往记录）
- **MIND（幻觉检测）正确编号 = arXiv:2403.06448**（Su et al.，THU，HELM 基准 + 无监督 token 级实时检测）。既往记忆里 2509.05714 是《Towards Meta-Cognitive Knowledge Editing for MLLMs》（CogEdit/MIND 编辑框架），同名不同文。
- **Know More Clearer 正式标题 =《Know More, Know Clearer: A Meta-Cognitive Framework for Knowledge Augmentation in LLMs》**（arXiv:2602.12996，Chen/He/Che/Sun，AI9Stars）。

## P(IK)/知识探测的数学形式（按稳固性排序）
1. **真值锚探针（最稳固，与我们实证一致）**：SAPLMA（2304.13734）hidden state 线性探针，**原指标是 accuracy 71–83% 非 AUROC**；2606.02628 证实 4-bit 量化下单中层线性探针 0.904–1.000 AUROC、MLP 仅 +0.01（信号近线性）、峰值层 50–90% 深度、采样式检测器≤0.541。→ 支撑我们"中层 ℓ8 线性足够 + 真值锚"。
2. **P(IK) 可学习（Kadavath 2207.05221）**：P(IK)=P(答对)=E[正确性]/采样均值；二分类头训于真值正确性，可跨任务泛化但 **OOD 校准漂移**（须温度缩放 T≈2.5 修复）。
3. **语义熵探针 SEPs（2406.15927）**：**回归 hidden state 预测 semantic entropy**，无需多次采样、零标签（熵本身是标签）、比直接预测正确性更稳。→ 这是"免真值标签"时最优代理，优于我们的 next-token 正确性自标注。
4. **自一致性/能量分数**：self-consistency（2203.11171）、INSIDE EigenScore（2402.03744，响应嵌入协方差特征值=微分熵，采样成本高）；2606.02628 实测采样式≤0.541 AUROC，弱于线性探针。
5. **Kadavath 的 P(True)/P(IK) 区别**：P(True) 针对具体答案、P(IK) 针对问题本身（不依赖答案）——KAL L1 更接近 P(IK)。

## 校准数学（KAL 输出校准置信度的做法）
- **Temperature scaling（Guo 1706.04599）**：单参数 T 缩放 logits，NLL 优化；Kadavath 证实 LLM 校准差时 T≈2.5 即修复。KAL L1 logits 须过 TS 再出 P。
- **Selective prediction / 风险-覆盖（Geifman 1705.08500）**：选择函数 g(x)∈{0,1}，选择性风险 R=ℓ/coverage；AURC 作指标。→ L1 的"blank"态=reject option，按目标覆盖率设阈值。
- **Conformal abstention（2405.01563 / 2502.06884）**：用 conformal 校准拒答阈值，给有限样本覆盖保证。→ KAL 拒答/回想触发门的严格化路径。

## 三态/多态知识分类
- **Know More, Know Clearer（2602.12996）**：知识空间三分 mastered/confused/missing（内部认知信号分区）+ cognitive consistency 对齐主观确定性与客观正确性（ECE 60%→24%）。→ 与 KAL L1 三态（知道/不确定/空白）直接同构，confused 态=L2 情感高 arousal 不确定。
- **TruthRL（2509.25760）**：GRPO + **三元奖励**（correct/hallucinate/abstain）——简单三元比复杂奖励更好，减幻觉 28.9%。→ 三态标签+三元奖励的 RL 化路径。
- **R-Tuning（2311.09677）**：把指令数据按"模型是否已掌握"分 D0(uncertain)/D1(certain)，训拒答未知。→ 在线自标注的对照：它也用预测正确性切分，但目的是拒答而非知识真假，故不冲突。

## 探针→干预闭环
- **ITI（2306.03341）**：truthful 方向（探针学到）+ 推理时把激活沿方向平移 α，TruthfulQA 32.5→65.1%，tradeoff 用 α 调。→ L1 检测→干预的最小闭环范式（方向+强度门）。
- **FLARE（2305.06983）**：生成中遇低置信 token 即触发检索重写。→ "置信低于阈值→触发回想/检索"的 token 级门。
- **MeCo（2502.12961）**：线性探针读 token 激活产 metacognitive score，决定何时调外部工具。→ L1 P(IK)→路由外部知识块的直接先例。
- **CAD（2305.14739）**：对比解码 logit=(1+α)logit_ctx − α·logit_prior，无训练解冲突。→ L3 冲突检测后的解码侧干预。
- **Self-RAG（2310.11511）**：reflection tokens（retrieve/critique）内化"何时检索/自评"为生成目标。→ `<|recall|>` 显式化与审计接口的同构先例。
- **Meta-R1（2508.17291）**：object-level/meta-level 分解，主动规划+在线调节+早停，+27.3%。→ KAL L3 冲突/调节的推理期编排。

## 对 KAL 的总建议（最稳固 P(IK) 形式）
**首选：真值锚三态 CE + 温度缩放校准 + selective/conformal 拒答门**。
- 标签协议：mastered/uncertain/blank 三态（Know More Clearer 式），真值锚（fake=blank/real=known，我们已实证 0.998）；**避免用 next-token 预测正确性作伪标签**（我们实证方向错位 0.433，它测的是流畅度非真假）。
- 免真值标签的备选：SEPs 语义熵回归（2406.15927）作自监督目标。
- 输出：L1 logits → temperature scaling → P(IK)，报 ECE/Brier + AURC；blank 阈值由 selective risk / conformal 按目标覆盖率定。
- 干预：L1 blank → FLARE/MeCo 式检索或 R-Tuning 式拒答；L3 冲突 → CAD 式对比解码或 ITI 式激活干预。

---
*导出自 /memories/repo/verified-literature-metacognition.md（2026-07-30 同步快照）。*
