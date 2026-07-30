# 动态词表文献核实笔记（2026-07-26 联网独立复核）

> 配合 article_ref/02_dynamic_tokenizer.md 与设计文档 §28。以下为已核实事实 + 对设计文档的勘误。

## 设计文档勘误（venue / 引用）
- **FOCUS**：实际是 **EMNLP 2023**（arXiv:2305.14481，Dobler & de Melo），**不是** §28.1 写的"NAACL 2022"。全名 "Fast Overlapping Token Combinations Using Sparsemax"，机制 = sparsemax 组合 source/target 重叠锚点 token（锚点按辅助静态 embedding 空间语义相似度选取）。
- **WECHSEL**：确为 **NAACL 2022**（arXiv:2112.06598，Minixhofer 等）——§28.1 把 FOCUS 与 WECHSEL 并标 "NAACL 2022" 只对 WECHSEL 成立。
- **SuperBPE**：venue 是 **COLM 2025**（arXiv:2503.13423），不是 ACL/NAACL。
- **MOSAIC**（§28.6 标 ✅ 的"反面证据"）：未能在 arXiv 独立定位到同名 vocab-expansion 论文（同名者均为机器人/视觉论文）。现象本身有支撑（FOCUS 随机初始化失败；OMP arXiv:2506.06607 称"其他 zero-shot 方法显著退化"），但"MOSAIC"这一具体引用 ⚠️ 存疑，待复核或替换为明确出处。

## 已核实关键数字（可直接引用）
- BLT arXiv:2412.09871：byte-level，熵驱动动态 patching，首个 FLOP-controlled 扩展研究到 **8B 参数 / 4T bytes**。
- Over-Tokenized arXiv:2501.16975（ICML 2025）：解耦 input/output 词表，**输入词表大小 ↔ loss 呈 log-linear**，大输入词表达到 2× 规模基线水平（无额外成本）。
- zip2zip arXiv:2506.01084（NeurIPS 2025）：LZW 在线合并 + 动态 embedding/unembedding 层 + AR 变体；现有 LLM **10 GPU-h PEFT** 即可 uptrain；输入+输出 token **减少 15–40%**。
- Kaplan《From Tokens to Words》arXiv:2410.05864（ICLR 2025）：detokenization 早-中层；词 vs 非词 k-NN 探针 **89% @ layer 13**（Llama2-7B）；OOV 多 token 检出 **64% @ layers 5-7**，22.6% 从未解码；3 步免微调扩表（Patchscopes 提取 r @ 最早成功层 ℓ → 正交 Procrustes T_ℓ,E/T_ℓ,U → 冻结主干训 d×d 精修矩阵 W_E/W_U，**20M token CPT**）；token 数减少 10.5/13.5/14.5%。
- H-Net（原版）arXiv:2507.07955（Hwang/Wang/Gu, 2025-07）：动态 chunking + 显式层级网络端到端替换 tokenize-LM-detokenize；byte 级单层 H-Net 超过 BPE-token Transformer；多层匹配 **2× 规模** token Transformer；Chinese/code/DNA ~4× 数据效率。（注："4.5-5 字节/块"出自全文分析，非摘要）
- DLCM arXiv:2512.24617（2025）："Dynamic Large Concept Models"，首个 **compression-aware scaling law**；R=4 重配 ~1/3 推理算力到高容量推理骨干，**12 基准平均 +2.69%**（matched FLOPs）。
- ZeTT arXiv:2405.07883（NeurIPS 2024）= "Zero-Shot Tokenizer Transfer"：超网吃 tokenizer 直出整副 embedding；泛化 XLM-R/Mistral-7B；残差 **<1B token** 续训闭合；同超网可零额外训练迁移到微调变体。
- T-FREE arXiv:2406.19223：字符三元组稀疏激活，无参考语料，embedding 层参数减少 **>85%**，跨语言迁移改善。
- 新增佐证 OMP arXiv:2506.06607（2025）：免训练 tokenizer 移植（Orthogonal Matching Pursuit），新 token = 共享锚 token 稀疏线性组合，在 Llama→Mistral-NeMo 12B / Qwen→Llama 1B 上 zero-shot 最优，优于 WECHSEL/FOCUS/ZeTT；警告：数值 token 方案不匹配会损害数学推理。

## 配方要点（见 article_ref/02_dynamic_tokenizer.md「实现配方综合」）
检测=KAL 词表摩擦（BLT 熵 patching 的 token 级对应物）；提取=Kaplan 内词典一次前向；注册=concept_slot 页表项；初始化=T_ℓ,E/U 或 FOCUS sparsemax 或 ZeTT 超网；注入=input 侧经 CSA/记忆层免费（Over-Tokenized log-linear）；升格=CA1 门 + 本地自蒸馏 CPT 仅训 W_E/W_U；输入侧宽进、输出侧窄升（Over-Tokenized 输出扩张对小模型有害，Kaplan 输出侧仅选择性可行）。

---
*导出自 /memories/repo/verified-literature-dynamic-tokenizer.md（2026-07-30 同步快照）。*
