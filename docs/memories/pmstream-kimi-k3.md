# PM-stream/mHC 层间互联 + Kimi K3 文献核实（2026-07-27，2 子代理 tavily 核实）

> 全部 arXiv 编号逐条联网核实存在性。为"PM-stream(mHC) 加强层间互联"与"Kimi K3 架构对齐"提供依据。

## PM-stream/mHC 层间互联（子代理1）
- **HC 原始（arXiv:2409.19606，ByteDance ICLR2025）**：单残差流扩为 n 流（典型 n=4），H^pre/H^post/H^res 三组映射；深度连接（跨层流混合，可动态重排层序）+宽度连接；解决"跷跷板效应"（梯度消失 vs 表示坍缩）。**我们 n=5=4内容+1感知-记忆流是"宽度扩展+流特化"合理延伸**（HC 原文未对流做功能分工，这是我们差异点）。
- **mHC（arXiv:2512.24880，DeepSeek）**：H^res=Sinkhorn-Knopp 双随机（Birkhoff 多面体，谱范数≤1，对乘法封闭）→ HC 27B 信号增益 3000×→mHC 1.6×；BBH 43.8→51.0（比 HC +2.1）、DROP 47.0→53.9；开销 +6.7%。数字全部核实无误。
- **mHC 是当前最强通用层间互联**：vs DenseFormer(2402.02622 标量加权弱)/LAuReL(2411.07501 互补)/ResFormer(2410.17897 V空间正交可组合)。
- **xHC（2607.14530）**：首个突破 n=4 + 显存优化（73.5C→40C 流量）→ **佐证 n=5 可行**；mHC-lite（2601.05732）少 SK 迭代降开销。
- **mHC-SSM（arXiv:2605.08300，2026）⭐最接近我们的外部验证**：mHC 约束多流残差应用于 SSM LM，"流特化（stream-specialized）容量进一步增强性能"——正是我们 1 条专用感知-记忆流的设计。**机理解释**：混合模型 GDN 递归状态（压缩衰减记忆）与注意力 KV（精确记忆）天然两种信息流，多流残差给二者显式通道分工+受控混合，避免单流互相覆盖；非负约束消除信号对消不稳定（对含门控递归混合主干尤其重要）。**独创性确认**：未见 mHC 用于 GDN/DeltaNet 混合+功能分工流 → TAIS n=5 PM-stream 仍独创延伸，方向与最新文献演进（xHC 扩 n、mHC-SSM 流特化）一致。
- **建议**：① 维持 PM-stream，1.5B T1 沿用；② xHC 大 n 显存优化（40C）供 n=5 工程参考；③ ResFormer value 残差作未来正交加项；④ mHC-lite 少 SK 迭代降 PM-stream 吞吐瓶颈（9.5k→3.0k 是主要痛点）。

## Kimi K3 / KDA / GDN-2 谱系（子代理2）⭐重大架构对齐证据
- **Kimi K3（官方博客 kimi.com/blog/kimi-k3，2026-07-16，权重 2026-07-27 放出；arXiv 完整报告未发布）**：2.8T MoE、16/896 专家激活（~50B 激活）、1M 上下文、原生视觉；骨干=**KDA+Block AttnRes 混合（3:1 KDA:Gated MLA）**；Stable LatentMoE、Per-Head Muon、SiTU、MXFP4 QAT；scaling 效率 ~2.5× vs K2。**层数/hidden/数据量未披露，需等完整报告**。
- **KDA（arXiv:2510.26692《Kimi Linear》）**：GDN 的逐通道遗忘门版 `S_t=(I−βk kᵀ)Diag(α)S+βk vᵀ`（每维独立 decay）；Kimi Linear 48B-A3B 用 **3:1 KDA:MLA 混合**、1.4T tokens、首个公平比较下全面胜 full attention、KV cache −75%、1M 解码 6×。**架构对齐**：我们 GDN-MemBlock 与 KDA 同属 delta-rule 谱系；**3:1 混合比与我们 28 层=7×{3 GDN+1 三级栈}完全一致——独立收敛设计互证**。
- **AttnRes（arXiv:2603.15031《Attention Residuals》，已集成 K3）**：PreNorm 残差"固定单位权重累加"致 hidden 幅值随深度失控/稀释；AttnRes 用深度维 softmax attention（pseudo-query 加权聚合前序层输出）替代，Block AttnRes 分块 <2% 开销；scaling law 1.870·C^-0.058 ≈1.25× 等效算力。**与 PM-stream 对照**：AttnRes（深度维内容寻址检索）vs PM-stream（固定多流+恒等初始化）是"跨层信息路由"两家族——AttnRes 选择性更强但引入层间注意力参数/通信，PM-stream 零额外注意力+恒等稳定。**其"层间幅值/梯度均匀度"可作我们 T1 新诊断指标**。K3 无显式记忆流（跨层记忆靠 AttnRes 检索）——我们 PM-stream 感知-记忆专用道是独有设计。
- **Muon 谱系**：K2 MuonClip→K3 Per-Head Muon，与我们"D-2 起切换 Muon+预训练与 W4 固化同优化器"直接对齐；arXiv:2607.07953 350M 扫参佐证 KDA+Muon 混合栈最优 val loss；KDA 1.4T tokens 门控收敛良好，间接支持"足够训练量让线性注意力门收敛"（K3 未给小模型加训直接证据）。
- **GDN-2 谱系定位**：DeltaNet→GDN(head-wise decay+标量β)→KDA(channel-wise decay+标量β)→**GDN-2(channel-wise decay+erase/write 解耦）**。我们 GDN-2 与 NVIDIA 同名同构（独立实现）；**K3 未采用 GDN-2（仍 KDA），GDN-2 是纯研究件——我们验证 GDN-2 增益属谱系前沿**。

## K3 完整技术报告补充（2026-07-27，docs/update/k3_tech_report.md 全文提取）
- **KDA 下界衰减（对我们 GDN-2 门收敛的关键启示）⭐**：Kimi Linear 用无界 negative-softplus `g=−exp(A_log)·softplus(z)`（g∈(−∞,0)，1/cumulative decay 可溢出 BF16）；**K3 改为有界 `g=g_min·sigmoid(exp(A_log)·z)`（g_min=−5）**——每 retention factor α>e^−5≈6.7×10⁻³，16-token tile 累积 log-decay∈(−80,0)，倒数 rescale<e^80 在 BF16 范围内；**消除 position-pair 对角路径，全 chunk 用 dense Tensor Core**。**对我们的双重启示**：① 数值范围——我们 gdn.py/gdn2 用的正是无界 negative-softplus，1M 长上下文会溢出（目标 1M，必须改有界）；② 门学习——有界 sigmoid 衰减可能更易收敛（对 GDN-2 门 b/w≈0.5 欠收敛是潜在贡献因子）。**建议：gdn.py/gdn2 decay 参数化改 K3 式有界 sigmoid（g_min=−5），列 GDN-2 改进项**。
- **AttnRes 细节**：每块 3 KDA+1 Gated MLA；Block AttnRes 分 8 块（12 层/块），层专属 pseudo-query 对前序块输出+embedding 做深度维 softmax attention（RMSNorm 防大幅值层主导）；O(Ld)→O(Nd) 开销。与 PM-stream 对照已记。
- **KCP（KDA Context Parallelism）**：KDA delta rule 使段效果依赖入态（不能 S=0 求和），KCP 分解为"累积转移作用入态 + 本地从零生成态"两本地量，prefix scan 组合，固定大小 all-gather——**对我们 GDN 递归状态的 1M 长上下文并行训练直接借鉴（GDN 同 delta rule）**。
- **Per-Head Muon**：Q/K/V 投影按头分块各自 Newton-Schulz 正交化（全矩阵正交化让大头主导更新方向）——与 D-2 Muon 切换对齐，逐头变体可选。
- **训练配方**：cosine 衰减（scaling law 独立扫参后优于 WSD）、1% warmup、wd 0.1；8K→64K 预训练→256K→1M cooldown 渐进上下文；NoPE（KDA 门控隐式编码位置，免 RoPE/YaRN）。

## 跨学科思考建模（2026-07-27，子代理 tavily 核实，docs/update/知识经验思考_神经认知建模.md）
六主题全部已确立（DOI/期刊/年份）：① **预测加工**（Rao&Ballard 1999 Nat.Neurosci；Friston 自由能——思考=逐步降低信念vs知识块预测失配，**KAL P(IK)=精度加权预测误差显式读出**，作思考步门控/终止）；② **全局工作空间**（Baars 1988；Dehaene 1998 PNAS/2011 Neuron——思考=容量受限的竞争选播广播区，非线性全或无 ignition 解释思考离散步，对应 <|recall|> 审计；注 COGITATE 2025 Nature 对 GNW 部分预测混合结果）；③ **世界模型/心理模拟**（Tolman 1948 认知地图；Hassabis&Maguire 2007 TiCS/PNAS 海马建构性模拟——记忆是可重组素材，**知识块须 token 寻址载体可事实召回可重组，非 steering 向量**；清醒 SWR 预演=运行时内部模拟，睡眠重放=巩固，同一机制两相位）；④ **重放/SWR**（Wilson&McNaughton 1994 Science；Girardeau 2009 因果抑制）；⑤ **工作记忆/PFC**（Baddeley 1974/2000 情景缓冲器；Miller&Cohen 2001 前额叶执行控制——思考的选择/维持/切换目标）；⑥ **知识-经验双系统**（McClelland 1995 CLS；Tulving 语义vs情景——**缺情景只能模式匹配，缺语义无法泛化**；睡眠期情景→语义迁移=已有 CLS 巩固管线神经原型）。
**子代理综合判定**：六原理收敛的"思考核"骨架中，目标维持/联合检索/选播仲裁/误差读出/重放固化**均已有 TAIS 部件**，真正新增只有 **③建构性重组流形绑定 + ⑤受限工作空间槽**——与 docs/update/CTM相关搜索.md 的思考流形方案完全互洽。**诚实边界**：所有"对 LLM 架构的启示"均为理论外推，无文献直接验证 LLM 思考核；神经原理作归纳偏置来源非已验证蓝图；GWT 仅取功能架构不取意识主张。

## 综合定位

**K3 = KDA（序列维线性注意力）+ AttnRes（深度维检索残差）+ LatentMoE（参数维稀疏）三维信息流改造；TAIS Obsidian = GDN-2（序列维解耦写）+ TriRetrievalAttention（三级检索）+ PM-stream（多流残差+感知记忆专用道）+ KAL 元认知 + 知识块外挂记忆**。架构对齐点：GDN 混合骨干 3:1、Muon、长上下文、线性注意力门需足量训练收敛。差异化：PM-stream vs AttnRes、GDN-2 解耦写、KAL/KB 元认知与外挂记忆（K3 无）。

---
*导出自 /memories/repo/pmstream-kimi-k3.md（2026-07-30 同步快照）。*
