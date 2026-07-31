# 项目提议：基于"权重虚拟内存"架构的自学习边缘语言模型

**Tianrui Bai**
2026 年 7 月 31 日

## 项目意义（Project Significance）

本提议规划将 TAIS Obsidian——一个以"权重虚拟内存"（weight virtual memory）为核心机制的自学习大语言模型（LLM）架构——从已验证的 0.1B 先导（pilot）扩展到 1B 参数研究模型，并评估其在边缘级硬件上部署后持续学习的能力。

预训练 LLM 的知识固化在冻结的权重中，部署后无法在不重训的前提下吸收新知识。业界主流的绕行方案——检索增强生成（RAG）[1] 及其主动变体（FLARE [2]、Self-RAG [3]）——把检索到的文本拼接进提示词，但存在三点本质局限：其一，纯文本并非模型的原生知识表示，注入依赖 prompt 拼接而非权重级接口；其二，模型对自身知识边界缺乏显式读出，无法可靠地"知道自己不知道"，这是幻觉（hallucination）的根源之一 [4]；其三，交互中学到的内容无法沉淀为可复用、可审计的长期记忆。

这些问题在 LLM 增长最快的方向——边缘计算（edge computing）——上最为尖锐。小语言模型（SLM）市场预计从 2025 年的 9.3 亿美元增长到 2032 年的 54.5 亿美元（CAGR 28.7%），驱动力正是手机、IoT 与嵌入式设备对低延迟、数据控制与能效的需求 [5]。2026 年一篇边缘 LLM 部署综述得出同样结论：靠近数据源部署可降低延迟、增强隐私、节省带宽，但资源约束限制了模型容量 [6]。边缘模型必然小，小模型的参数化知识必然有限——因此**部署后持续学习不是加分项，而是边缘 LLM 的核心需求**。

此外还有两道工程屏障。注意力的 KV 缓存（KV cache）随上下文长度与模型规模线性增长，而注意力计算本身随序列长度平方增长——一条长序列即可占用数 GB 缓存，对边缘设备是禁得起推敲的硬约束 [7]。而朴素的持续微调会导致灾难性遗忘（catastrophic forgetting）——新数据覆盖旧权重；近期研究进一步表明，连优化器的选择都会显著改变遗忘程度 [8]。

TAIS Obsidian 的应对之道，是把"知识"升格为与权重同级的运行时对象——**知识块（KnowledgeBlock）**，并用操作系统式的虚拟内存机制加以管理：页表（SQLite）登记、分层存储（L0 VRAM / L1 DRAM / L2 NVMe / L3 远端）、缺页 fail-closed、读写不对称（运行时仅允许零梯度快写，梯度固化只在离线"睡眠期"进行）[9]。在此之上，分层元认知模块（KAL）读出模型内部状态以检测知识空白，让模型主动检索或提问而非盲目猜测 [4][10]。该架构已在 0.1B 规模完成实现与验证：GDN-2 线性注意力主干 + 三级检索注意力栈（滑窗 + 压缩选择 + 重压缩 gist）使长上下文成本保持近线性 [11][12]；"感知→求知→验证→写入→召回→睡眠固化"的自学习闭环已端到端打通，437 项单元测试全绿。

![TAIS Obsidian 架构详图](assets/architecture_v3.png)

*图 1：TAIS Obsidian 架构详图（v3.0，IBM Carbon 设计语言）：主干、TAIS 内核（KAL/HRL）、主动求知闭环、知识块库与运行时、睡眠固化器。*

## 研究目标（Objectives）

我提议把已验证的 0.1B pilot 扩展为 1B 参数的自学习研究模型并做严格评估。目标如下：

**目标一**：在 10B tokens 多领域语料上预训练 1B 模型（实测 1,017.7M 参数：24 层 GDN-2 线性注意力 + 8 层三级检索注意力），随后进行 1B tokens 的中训练退火（mid-training annealing，高质量上移混合），复刻 SmolLM2 的多阶段 WSD 配方与 OLMo 3 的 Dolmino 中训练实践 [13][14]。

**目标二**：把 pilot 的内生部件迁移到 1B 并复测——KAL 元认知探针（知识空白 AUROC）、HRL 块检索（top-1 命中）、HCA 注入召回、诚实降级行为——检验设计关于"部件强度随规模上升"的预测。

**目标三**：通过 RoPE 缓存扩容 + YaRN 缩放与渐进扩窗课程，把原生上下文从 1,024 扩展到 256K tokens——这是 Qwen2.5-1M 与 Llama 3 长上下文训练验证过的主流路径 [15][17]。

**目标四**：为 1B checkpoint 准备 HuggingFace 发布（含可复现推理链路），并用轻量基准（ARC-Easy/Challenge、HellaSwag、PIQA）对标同尺寸基线，同时如实标注欠训（undertraining）前提。

本提议直接建立在已完成工作之上：0.1B pilot 已完整实现并测量（见配套报告）；1B 配置（d_model 1536 × 32 层，Muon 优化器）已实例化验证；10B tokens 流式数据管线已端到端冒烟；训练到上传的全工具链已通过 437 项回归测试。

## 研究计划草案（Drafted Research Plan）

本项目的文献依据主要来自同行评审的机器学习会议（NeurIPS、ICLR、EMNLP、ACL）、公开技术报告（OLMo、SmolLM、Qwen、DeepSeek），以及本项目已产出的实验产物（训练日志、评估报告、437 项通过测试）。

**针对目标一（1B 预训练）**：数据课程对齐 OLMo 3（预训练 Dolma 3 Mix、中训练 Dolmino 高质量混合）[14] 与 SmolLM2 多阶段 WSD [13]。语料（10B tokens）配比为 FineWeb-Edu 73% / 数学（NuminaMath-CoT + FineMath-4+）12% / 合成教科书（Cosmopedia）10% / 中文网页（FineWeb2-HQ）5%。需要诚实说明：10B tokens 是 1B 模型 Chinchilla 计算最优量（20B）[16] 的一半，也远低于当代 1B 级实践（4T+ [13]）——本次定位是架构验证 pilot，绝对能力将按此口径报告。

**针对目标二（部件迁移）**：复用 pilot 已验证管线——真值锚校准（0.1B 双口径 AUROC 0.845/0.829）、HRL indexer 训练（top-1 = 1.000）、门控融合注入（召回 0.625 vs in-context 上界 0.70）。每项指标在 1B 重新测量，检验规模缩放预测。

**针对目标三（上下文扩展）**：YaRN 式 RoPE 缩放 + 渐进课程（4K→16K→64K→256K），对齐 Llama 3 的六阶段 8K→128K 做法 [17] 与 Qwen2.5-1M 的 YaRN+稀疏注意力路径 [15]。三级检索注意力把精确注意力限定在 512 滑窗内，RoPE 负载只在滑窗分支；压缩选择与 gist 分支按设计就是位置无关的 [12]。

**针对目标四（发布与基准）**：checkpoint 连同 tokenizer 与模型卡打包（如实标注研究性欠训状态），跑 1B 级标准轻量评测；后续工程将增加 auto_map/trust_remote_code 导出路径，使标准工具链可直接加载。

## 项目动机（Project Motivation）

本项目始于一个简单的观察：操作系统很早就用虚拟内存与页表解决了"有限快内存"的问题，但语言模型至今没有等价的知识管理机制。我设计 TAIS Obsidian，就是想检验这个类比能否被字面化——知识在权重空间中被按需换入换出，而模型自己知道它不知道什么。过去数月，我从零用纯 PyTorch 实现了完整架构，在 0.1B pilot 上验证了每个子系统，并诚实记录了正面与负结果。本提议是下一步：证明这些想法能经受住真实模型规模与真实训练预算的考验。

## 参考文献（Reference）

[1] P. Lewis *et al.*, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020, arXiv:2005.11401.

[2] Z. Jiang *et al.*, "Active retrieval augmented generation (FLARE)," in *Proc. EMNLP*, 2023, arXiv:2305.06983.

[3] A. Asai, Z. Wu, Y. Wang, A. Sil, and H. Hajishirzi, "Self-RAG: Learning to retrieve, generate, and critique through self-reflection," in *Proc. ICLR*, 2024, arXiv:2310.11511.

[4] A. Azaria and T. Mitchell, "The internal state of an LLM knows when it's lying (SAPLMA)," in *Findings of EMNLP*, 2023, arXiv:2304.13734.

[5] MarketsandMarkets, "Small language model market report 2025–2032," 2025. [Online]. Available: https://www.marketsandmarkets.com/Market-Reports/small-language-model-market-4008452.html

[6] E. Kristiani, V. K. Verma, and C.-T. Yang, "Deploying LLM transformer on edge computing devices: A survey of strategies, challenges, and future directions," *AI*, vol. 7, no. 1, p. 15, Jan. 2026.

[7] "KV cache optimization strategies for scalable and efficient LLM inference," arXiv:2603.20397, 2026.

[8] Y. Liu, "Optimizer-model consistency: Full finetuning with the same optimizer as pretraining forgets less," arXiv:2605.06654, 2026.

[9] J. L. McClelland, B. L. McNaughton, and R. C. O'Reilly, "Why there are complementary learning systems in the hippocampus and neocortex," *Psychological Review*, vol. 102, no. 3, pp. 419–457, 1995.

[10] "Hallucination is linearly decodable from mid-layer hidden states in quantized LLMs," arXiv:2606.02628, 2026.

[11] A. Hatamizadeh, Y. Choi, and J. Kautz, "Gated DeltaNet-2: Decoupling erase and write in linear attention," NVIDIA, arXiv:2605.22791, 2026.

[12] J. Yuan *et al.*, "Native sparse attention: Hardware-aligned and natively trainable sparse attention," DeepSeek-AI, arXiv:2502.11089, 2025.

[13] L. Ben Allal *et al.*, "SmolLM2: When smol goes big — Data-centric training of a small language model," arXiv:2502.02737, 2025.

[14] OLMo Team, "OLMo 3: Fully open language models," Allen Institute for AI, arXiv:2512.13961, 2025.

[15] Qwen Team, "Qwen2.5-1M technical report," arXiv:2501.15383, 2025.

[16] J. Hoffmann *et al.*, "Training compute-optimal large language models (Chinchilla)," in *Proc. NeurIPS*, 2022, arXiv:2203.15556.

[17] Llama Team, "The Llama 3 herd of models," Meta AI, arXiv:2407.21783, 2024.
