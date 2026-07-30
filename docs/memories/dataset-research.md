# TAIS Obsidian 数据集调研报告（子代理返回，2026-07-29，tavily 联网核实）

> 面向 1.5B 原生 1M 上下文、混合 GDN-2 + 三级检索注意力、思考流形/推理循环/主动求知闭环的预训练+后训练数据集选型。当前 0.1B pilot 用 FineWeb-Edu（120M tokens 自训 32k BPE）。所有 HF ID 已经 tavily_search 核实真实存在，标 [已核实存在]。

## 1. 文学/通用高质量文本（替代/补充 FineWeb-Edu）
- HuggingFaceFW/fineweb-edu [已核实]：1.3T tokens 教育级英文网页（FineWeb 15T 筛选子集）；ODC-By 1.0；pilot 已在用，可直接扩容，教学/解释式内容对齐知识内化。
- HuggingFaceFW/fineweb-2 [已核实]：1000+ 语言，cmn_Hani 中文子集 ~543B 词；ODC-By 1.0；解决 TAIS 中文预训练数据稀缺。
- epfml/FineWeb2-HQ [已核实]：FineWeb2 高质量去重子集，cmn_Hani ~54.2M 文档；ODC-By 1.0；6x 数据效率，中文主力候选。
- opencsg/chinese-fineweb-edu（及 V2.1）[已核实]：~90M 条中文教育网页（~200B tokens，V2 ~180M/420B）；OpenCSG 社区许可（商用需邮件报备，研究可用）；中文高质量补充。

## 2. 数学推理（思考流形/推理循环）
- AI-MO/NuminaMath-CoT [已核实]：860k 题，竞赛到 K12，全部 CoT；Apache 2.0；思考核训练主力。
- nvidia/OpenMathInstruct-2 [已核实]：14M 题解对（600k 独立题），GSM8K/MATH 种子；商业友好许可；SFT 规模化。
- open-r1/OpenR1-Math-220k [已核实]：220k 题，DeepSeek-R1 长推理迹（已验证）；Apache 2.0；长 CoT/推理循环蒸馏。
- nvidia/OpenMathReasoning [已核实]：306k 题，CoT+TIR，AIMO-2 冠军配方；许可见仓库；竞赛级上限。
- GAIR/LIMO [已核实]：817 条精编 CoT（Less-Is-More）；MIT；极小量高质量推理诱导，消融/验证用。

## 3. 物理/科学推理
- EricLu/SCP-116K [已核实]：274k 大学-博士级物理/化学/生物/数学题解对+R1 推理迹；CC-BY-NC-SA-4.0（不可商用，研究可用，需审查）；科学原理推导。
- dvilasuero/natural-science-reasoning [已核实]：自然科学 CoT（<think> 块）；许可见仓库；科学思考流形。
- galaxyMindAiLabs/stem-reasoning-complex [已核实]：生物/数学/物理/化学复杂题 + <think> CoT；许可见仓库；多科推理。
- MegaScience/TextbookReasoning [已核实]：650k 教科书级科学推理；CC-BY-NC-SA-4.0（研究可用，需审查）；知识内化友好。

## 4. 编程能力
- bigcode/the-stack-v2 [已核实]：3B+ 文件、600+ 语言、~900B tokens（训练集）；代码需遵循原始许可（提供溯源）；代码预训练主力。
- OpenCoder-LLM/opc-annealing-corpus [已核实]：24GB 代码退火精调语料；许可已标注于仓库；高质量代码课程尾段。
- OpenCoder-LLM/opc-sft-stage2 [已核实]：375k 条代码 SFT；许可见仓库；代码指令遵循。
- HuggingFaceTB/cosmopedia（合成教科书/故事，含代码教学）[已核实]：30M 文件/25B tokens；Apache 2.0（Mixtral 生成）；知识内化+代码入门。

## 5. 通用推理/指令（后训练 SFT）
- allenai/tulu-3-sft-mixture [已核实]：939k 条高质量指令混合；ODC-BY-1.0（含非商用子集，需按子集审查）；通用 SFT 基线。
- HuggingFaceTB/smoltalk2 [已核实]：SmolLM3 全量 SFT 混合（含 OpenHermes-2.5、OpenThoughts3、IF、多轮）；新增子集 Apache 2.0；1.5B 级模型验证配方，最契合。
- teknium/OpenHermes-2.5 [已核实]：~1M 条 GPT-4 指令；Apache 2.0；通用指令遵循。
- allenai/WildChat / WildChat-nontoxic [已核实]：真实用户多轮对话；许可见仓库（含毒性版本需注意）；多轮对话/真实交互。

## 6. 长上下文（1M 目标预训练适配）
- utter-project/LongBlocks [已核实]：194k 条长上下文 QA（书籍/arXiv/维基/代码）；许可见仓库；长文档后训练。
- Yukang/LongAlpaca-12k [已核实]：12k 条长文档指令（书籍/论文）；Apache 2.0；长上下文 SFT 入门。
- emozilla/pg19 [已核实，经典]：PG-19 长书籍语料（>28k 部长篇）；许可见仓库；超长连续文本预训练（1M 适配底座）。
- togethercomputer/Long-Data-Collections [已核实，经典]：书籍/arXiv 长文档预训练集；Apache 2.0；长上下文预训练。

## 主动求知（"不知道/需澄清"标注）
- rajpurkar/squad_v2 [已核实]：SQuAD 2.0 含 5 万+ 不可回答问题；CC-BY-SA-4.0（需审查）；训练"不知道"拒答。
- Human-CentricAI/chatbot-arena-llm-refusal [已核实]：3.5k 条拒答/技术局限人工标注；Apache 2.0；求知分支/诚实降级。
- SelfAware（Yin et al. 2023，论文已核实，HF 需复核）：1,032 不可答 + 可答对照；研究用；KAL P(IK) 三态监督对齐。
- UA-Bench（论文已核实，HF 需复核）：3.5k 题区分数据不确定 vs 模型不确定；训练澄清提问（求知执行器输入）。

## 知识内化（教学式/解释式）
- HuggingFaceTB/cosmopedia [已核实]：见 §4。
- opencsg/chinese-cosmopedia [已核实]：中文合成教科书（百度百科/知乎/技术博客种子）；OpenCSG 许可；中文知识内化。
- HuggingFaceTB/finemath [已核实，FineMath 4+]：数学教育文本；ODC-By；数学知识链内化。

## 总表
| 领域 | 首选 | 许可 | 规模 |
|---|---|---|---|
| 通用英文 | fineweb-edu | ODC-By | 1.3T tok |
| 通用中文 | FineWeb2-HQ cmn_Hani | ODC-By | 54M doc |
| 数学 | NuminaMath-CoT + OpenR1-Math-220k | Apache 2.0 | 860k/220k |
| 科学 | SCP-116K（研究）+ natural-science-reasoning | CC-BY-NC-SA | 274k |
| 代码 | the-stack-v2 + opc-sft-stage2 | 溯源/见仓库 | 900B tok |
| 通用指令 | smoltalk2 + tulu-3-sft-mixture | Apache/ODC | ~1M |
| 长上下文 | LongBlocks + pg19 | 见仓库 | 194k/长书 |
| 求知 | squad_v2 + chatbot-arena-llm-refusal | CC-BY-SA/Apache | 150k/3.5k |

## 数据混合建议（对齐 OLMo 3 Dolma/Dolmino/Longmino 课程）
预训练（30B tokens 目标，Dolma 式混合）：英文高质量 45%（fineweb-edu）、中文高质量 20%（FineWeb2-HQ cmn + chinese-fineweb-edu）、代码 15%（the-stack-v2 采样）、数学/科学网页 10%（finemath + cosmopedia）、长文档 5%（pg19 + Long-Data-Collections 早期混入）、求知/百科 5%。
退火/中训（Dolmino 式，3–5B tokens）：高质量子集 + OpenMathInstruct-2 + SCP-116K + opc-annealing + cosmopedia 教科书，提升密度。
长上下文适配（Longmino 式，1M 阶段）：pg19 长书 + LongBlocks + LongAlpaca + 长代码仓库（the-stack-v2 repo 级），配合三级检索注意力渐进扩窗。
后训练 SFT（1–2M 条）：smoltalk2 为主干 + OpenThoughts3-1.2M（思考流形 CoT）+ OpenR1-Math-220k（长推理循环）+ opc-sft-stage2（代码）+ squad_v2/拒答集（求知分支 P(IK) 监督）+ WildChat 多轮（真实交互）。许可红线：SCP-116K/TextbookReasoning/Tulu 部分子集 CC-BY-NC-SA 仅研究可用，商用需替换或单独审查；squad_v2 CC-BY-SA 需标注。

---
*导出自 /memories/repo/dataset-research.md（2026-07-30 同步快照）。*
