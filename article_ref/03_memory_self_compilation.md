# 03 记忆层 / 注意力自编译（Memory Layers / Attention Self-Compilation）

> 簇主旨：一次前向如何把"一段经验"变成**可复用的记忆块**——无需训练。三类载体：KV 前缀、steering 向量、记忆层条目。
> 核实状态：以下 arXiv 编号均已于 2026-07-26 联网核实。⚠️ 数字来自二手引用未全文确认。

## A. 记忆层条目载体（训练期学习，非运行时自编译，但是 W3+ 离线写入基底）

### Memory Layers at Scale — Berges, Oğuz et al. (Meta FAIR), arXiv:2412.09764, ICML 2025 ⭐⭐⭐【关键】✅
可训练 key-value 查找，**加参数不加 FLOPs**；≤128B memory 参数/1T tokens/≤8B 基座；事实任务 >+100%，追平 2× 计算预算 dense 与同算力同参数 MoE。memory 层替换 FFN：$I=\text{SelectTopk}(Kq), s=\text{Softmax}(K_Iq), y=sV_I$。**keys/values 是可训练参数（非激活）——与注意力的根本区别**。Product-key：两半键集 $K_1,K_2\in\mathbb R^{N\times n/2}$，全集从不实例化。Memory+ 加输入门控 + silu + qk-norm。**sweet-spot ≈3 层居中大间距**（再加层挤掉 dense 参数而退化）。作者自述稀疏更新→更少遗忘/幻觉、利于持续学习。→ 增强 A 工程依据。

### Hopfield Networks is All You Need — Ramsauer et al., arXiv:2008.02217 ⭐⭐⭐【关键】✅
现代连续 Hopfield 更新规则**等价于 transformer 注意力**；存储容量随关联空间维度**指数增长**；一次更新即检索；检索误差指数小。$\xi^{new}=\text{softmax}(\beta X^\top\xi)X \equiv$ attention。→ **理论地基**：CSA+KV 前缀=被查询的现代 Hopfield 记忆；解释"块注入"数学上=检索；指数容量支撑"块库可极大"。

## B. KV 前缀载体（运行时自编译，保留 token 寻址 → 能做事实回忆）

### In-context Autoencoder (ICAE) — Ge et al. (Microsoft), arXiv:2307.06945, ICLR 2024 ⭐⭐⭐【机会】✅
LoRA 编码器把长 context 软提示压缩为少量 memory slot embeddings；约 1% 额外参数；基于 Llama **4× 近无损**。→ **CSA `harvest()` 自编译接口原型**。

### KV-Distill — Chari et al. (Johns Hopkins), arXiv:2503.10337 ⭐⭐⭐【机会】✅
**问题无关** KV 蒸馏；可作预训练模型 PEFT 适配器；worst-case 抽取任务优于其它压缩法，长上下文 QA/摘要逼近未压缩；**域微调可压缩至 99% 长度保下游性能**。→ **工程级 CSA `harvest()`**，问题无关=可离线编译一次多次复用。

### FastGen / "Model Tells You What to Discard" — Ge et al., arXiv:2310.01801, ICLR 2024 ⭐⭐✅
**自适应、即插即用、无需微调** KV cache 压缩；profiling 把注意力头分三类（局部/特殊 token/广注意）按结构留弃。→ 运行时 KV 块"按头剪枝"策略。

### Expected Attention — Devoto, Jeblick, Jégou, arXiv:2510.00636 ⭐⭐✅
**训练无关** KV 压缩，靠**估计每个 KV pair 对未来 query 的价值**排序剪枝；闭式计算期望注意力分数（解决"未来 query 不可得"+Flash-Attention 不物化矩阵）；附 KVPress 库（20+ 方法）。→ 未来价值估计 = 块显著性/arousal 门控的注意力原生版。

## C. 向量载体（运行时自编译，位置不变偏移 → 只 steer 行为，不能事实回忆）

### In-context Vectors (ICV) — Liu, Ye, Xing, Zou (Stanford), arXiv:**2311.06668**（⚠️非 2310.10678=物理论文）, ICML 2024 ⭐⭐⭐【关键+风险】✅
把 ICL 重述为两步隐空间偏移：从 demo 提 ICV 向量，推理时不再贴 demo 而把向量加到全部 latent state。比标准 ICL 与 LoRA 微调都强；支持向量算术组合。**提取配方**：① 每层最后 token latent $h\in\mathbb R^d$，跨 L 层拼成 $\mathbb R^{L\times d}$；配对 demo ICV=差向量 $h(y_i)-h(x_i)$ 的**PCA top-1**（Lemma 1）；② Feature shifting $\tilde h_{t,l}=h_{t,l}+\lambda h_{ICV}^l$ **加到所有层所有 token**（单层消融几乎无效）+ $\ell_2$ 归一化。**评测仅 safety/style/role/format（变换任务），从不主张事实回忆** → 直接坐实"向量载体不能做事实回忆"边界。

### Function Vectors (FV) — Todd et al., arXiv:2310.15213, ICLR 2024 ⭐⭐⭐【风险/边界】✅
LLM 内存在把"输入→输出函数"表示为**单个向量**的机制；因果中介分析发现少量注意力头搬运紧凑 function vector；因果效应集中**中层**；FV 常含"函数输出空间"信息但**仅靠该信息不足以重建 FV**；FV 可相加生成复合任务。→ 复现的是 ICL **函数**（反义/翻译等抽象映射），非事实查找表。

### DeCoVec — Feiyang Li, Yile Wang, arXiv:2604.11129, **ACL 2026 Findings**（确实存在，已核实）⭐⭐【机会】✅
**training-free、非侵入**任务向量框架，在**解码空间（logits）**直接构造；$\Delta z=\text{logits}(\text{few-shot})-\text{logits}(\text{zero-shot})$ 加到解码 logit；7 个 LLM(0.5B–9B) TruthfulQA/Math-500/AQUA-RAT 稳定优于 few-shot，最高 +5.50。→ 最小侵入注入点（logit 层）。

## D. 理论地基

### Learning without training（ICL ≡ 低秩权重更新）— Dherin et al. (Google), arXiv:2507.16003 ⭐⭐⭐【关键】✅
带上下文的一次标准 forward **数学上等价于**不带上下文 forward + 代表该上下文的**最小低秩（rank-1）MLP 权重更新**，损失曲线几乎重合。ICL 效应可投影成 MLP 权重 rank-1 扰动=激活空间一个方向 → 解释 ICV/FV 能用"一个向量"捕获 forward 经验的数学原因。

---

## 自编译机制综合与载体能力边界

**问题**：一次 forward 如何把经验变成可复用记忆块而无须训练？三种配方，三种边界。

**配方 1 — KV 载体（CSA `harvest()`，运行时）**：一次 forward 产生经验 token 的 K,V 激活即块本身，可再压缩/剪枝（ICAE 4×、kv-distill 99%、FastGen 按头、Expected Attention 未来价值）。**保留 token 寻址 → 能做事实回忆**（RAG 有效、kv-distill 保 QA 的原因）。

**配方 2 — 向量载体（steering/task vector，运行时）**：一次 forward 后把 demo 浓缩成**单个常向量**，推理时加到 residual（ICV: 差向量 PCA top-1）或 logits（DeCoVec: few−zero logit 差）。**位置不变（同一偏移加到每 token）→ 只能 steer 行为/风格/函数，不能做事实回忆**。

**配方 3 — 记忆层条目（product-key KV，训练期）**：Memory Layers keys/values 预训练中学习，top-k 稀疏检索；**有可寻址 key → 能存事实关联**。非运行时自编译目标，但是 W3+ 离线"块即专家/块即条目"天然基底。

**设计文档命题核实结论——"向量载体不能做事实回忆"= 已核实且被强化**：
- ICV 评测全部是 safety/style/role/format 变换任务，从不主张事实回忆；其数学（差向量 PCA + 常量偏移）按构造编码**方向/变换**，无 token 索引。
- FV 自述"编码函数输出空间，不足以重建 FV"，复现 ICL **函数**非查找表。
- 事实回忆（"法国首都=巴黎"）是单一 (key→value) 关联，需 token 寻址检索；常量向量偏移无法提供。
- 对照：Memory Layers 因有可寻址 key 能存事实；KV 前缀因保留 token 索引能保抽取式 QA。

**边界精确化（可直接入设计文档）**：**token 寻址载体（KV 前缀/记忆层条目）能做事实回忆；位置不变向量（ICV/FV/DeCoVec）不能，只能 steer 行为/风格/函数。** 故 TAIS Block Spec 须按载体标注"事实召回能力"字段：事实/陈述性块只能用 KV 或 memory-layer 载体；向量载体专用于人格/风格/技能/格式块（与 §7 页保护位"人格块运行时只读、向量可写"一致）。
