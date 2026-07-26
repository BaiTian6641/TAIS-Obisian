# 02 — 动态词表 / 无 token 化（背景论文参考）

> 配合设计文档 §28「动态词表：三级生长阶梯与全谱系自包含」。
> 三级阶梯：**第 0 级**=concept_slot（运行时零梯度，页表=动态词表 codebook）；**第 1 级**=词表升格（睡眠期轻梯度，reserved 槽激活 + 自蒸馏 CPT）；**第 2 级**=架构级溶解（换代，H-Net 式，27B+）。
> 核实方式：arXiv 摘要/全文 + arXiv API（2026-07-26）。⚠️=未独立定位。

### BLT — arXiv:2412.09871, ACL 2025 ⭐⭐⭐【关键】✅
字节编码为动态大小 patch；分割由下一字节熵驱动（复杂处给更多算力）；8B/4T bytes 追平 tokenization 模型，推理训练双高效。**熵 patching 信号=第 0 级「词表摩擦」检测的 token 级对应物**。

### H-Net — arXiv:2507.07955, 2025 ⭐⭐⭐【关键】✅
端到端层级网络替换 tokenize→LM→detokenize；byte 级单层超 BPE、多层匹配 2× 规模 token Transformer；content/context-dependent 分割与模型联合学习；DNA 数据效率 ~4×。**第 2 级目标架构**。

### T-FREE — arXiv:2406.19223, 2024 ⭐⭐【背景】✅
无 subword tokenizer；词经字符三元组稀疏激活嵌入；embedding 参数 −85%；跨语言迁移改善。

### Over-Tokenized Transformer — arXiv:2501.16975, ICML 2025 ⭐⭐⭐【关键】✅
解耦 input/output 词表；**输入词表↔训练 loss 呈 log-linear**（与模型规模无关）；大输入词表=2× 规模基线无额外成本；输出扩张对小模型**有害**（欠拟合），输入扩张**无条件正向**。**「输入宽进/输出窄升」保守序的硬理论根基**。

### zip2zip — arXiv:2506.01084, NeurIPS 2025 ⭐⭐⭐【机会】✅
推理时上下文自适应 tokenization；三组件：① LZW 压缩增量合并 hypertoken；② 动态 embedding(+unembedding) 层现场算表示；③ AR LM 变体；现有 LLM **10 GPU-hours PEFT** 即可 uptrain；I/O token −15–40%。合并驱动是表面频率（非语义）——TAIS 知识块 = 其 hypertoken 的语义版，差的那层（KAL 语义门控 + HRL 注册表）正是现成设计。

### From Tokens to Words（Kaplan）— arXiv:2410.05864, ICLR 2025 ⭐⭐⭐【关键】✅
LLM 维护超出 tokenizer 的**潜在内词典**。**两阶段 detokenization**：① Token Aggregation（早期层 attention 把前缀 sub-word 聚合到末 token）；② Concept Retrieval（FFN 在表示涌现前把整词概念写入残差，85% 词有 FFN 更新，消融使检出 85%→18%）。词 vs 非词 k-NN 探针 89% @ layer 13（Llama2-7B）；OOV 多 token 词检出 64% @ layers 5-7。**免改参数扩表三步**：(1) Patchscopes 在最早成功层 ℓ 提取 r；(2.1) 仅用现有词表训正交 Procrustes 线性映射 T_ℓ,E、T_ℓ,U（RMS 归一化）；(2.2) 对 r 应用得 ê、û；(3) 冻结主干，**仅训两个 d×d 精修矩阵 W_E、W_U**，20M token CPT。token 数减 10.5–14.5%。

### FOCUS — arXiv:2305.14481, **EMNLP 2023**（⚠️勘误：非设计文档 §28.1 的"NAACL 2022"）⭐⭐【机会】✅
新 token = 源/目标词表重叠 token 的 sparsemax 稀疏组合；锚点按辅助静态 embedding 语义相似度；多语言 XLM-R 为源。

### WECHSEL — arXiv:2112.06598, NAACL 2022 ⭐⭐【背景】✅
源（英语）tokenizer 换目标语言后用多语言静态 word embedding 初始化；训练量减最多 64×。

### ZeTT — arXiv:2405.07883, NeurIPS 2024 ⭐⭐【机会】✅
超网吃 tokenizer 直出整副 embedding，零样本迁移；接近原性能，残差 **<1B token** 续训闭合；同一超网可直接迁移到微调变体无需额外训练。

### MOSAIC — ⚠️ 未能独立定位具体论文 ⭐⭐【风险】
"朴素扩表不重训必然劣化"现象由 FOCUS 对照实验 + OMP（见下）佐证；建议设计文档替换为明确出处。

### SuperBPE — arXiv:2503.13423, **COLM 2025**（⚠️勘误 venue）⭐⭐【机会】✅
跨空白 superword；@200k 词表 token 最多减 33%；8B 固定 size/vocab/compute，30 任务平均 +4.0%（MMLU +8.2%），推理算力 −27%。

### DLCM — arXiv:2512.24617, 2025 ⭐⭐【机会】✅
层级 LM 把计算从 token 移到压缩概念空间；首个 compression-aware scaling law；R=4 重配 ~1/3 推理算力到高容量骨干，12 基准平均 +2.69%。

### OMP（新发现，免训练 tokenizer 移植）— arXiv:2506.06607, 2025 ⭐⭐【机会/风险】✅
新 token = 共享锚 token 的 OMP 稀疏线性组合；Llama→Mistral-NeMo 12B / Qwen→Llama 1B 上 zero-shot 最优，优于 WECHSEL/FOCUS/ZeTT。**警告：数值 token 方案不匹配会损害数学推理**。建议纳入 §28.1-C 工具箱。

---

## 动态词表实现配方综合

**① 检测**（零梯度，运行时，KAL 第四信号源「词表摩擦」）：高熵碎片段 + 反复共现多 token 序列 + 低 P(IK) 专名区域（= BLT 熵 patching token 级对应），由写显著性头加标入 W0 日志。

**② 提取**（零梯度，一次前向）：Kaplan 内词典提取——候选概念多 token 序列喂入，取末 token 在最早成功层 ℓ（≈5–15）的 detokenized hidden state r（Patchscopes/logit-lens 能解码出整词即成功）。绕开 zip2zip ~10 GPU-h PEFT 成本。**绕法有效性是 TAIS 最大不确定项，列入 T1 观测**。

**③ 注册/存储**：Block Spec 扩展 `compiled.kind=concept_slot`；route_key=概念文本，payload=输入侧向量 ê + markdown 源代码（保真回退）；DG 稀疏 key 作防碰撞哈希；存 L1 DRAM 概念槽表镜像 L2 NVMe。

**④ 初始化（升格时三选一）**：Kaplan Procrustes T_E/T_U / FOCUS sparsemax 锚组合 / ZeTT 超网（批量首选）。

**⑤ 注入**：经 CSA 压缩区/记忆层条目注入融合表示（Over-Tokenized 保证输入侧零风险）。

**⑥ 输入-输出非对称**：输入侧免费、风险零 → 第 0 级默认开；输出侧 Over-Tokenized 证明对小模型有害，但 Kaplan 证明精选高频多 token 概念 input+output 都可行 → 输出升格必须选择性、每批限数十条、tied embedding 下先验证输入侧提取强度。保守序：**「输入侧宽进、输出侧窄升、tied 合一」**。

**⑦ 升格判据（第 1 级，睡眠期）**：CA1 门（高使用计数 + 本地回归通过 + 共识度）→ 激活预训练预留 2048 reserved 槽之一（噪声占位防 glitch，升格不改矩阵形状）→ 初始化 → **本地自蒸馏 CPT**（原模型当教师 KL 对齐 + 回放 + 谱修剪，仅训 W_E/W_U，20M token 级；残差大时 ZeTT <1B token），复用 W4/Muon 纪律。**SHY 式退场**：长期不用降级回 concept_slot。**跨设备**：reserved 槽命名空间须中心协调，否则块交换撞 ID。

**勘误汇总（须回填设计文档）**：MOSAIC⚠️存疑；FOCUS 实为 EMNLP 2023（非 NAACL 2022）；SuperBPE 实为 COLM 2025。
