# TAIS Obsidian 数据集选型（预训练 + 后训练）

> **版本 v0.1 · 2026-07-29**。面向 1.5B 原生 1M 上下文、混合 GDN-2 + 三级检索注意力、思考流形/推理循环/主动求知闭环的数据集选型。当前 0.1B pilot 用 FineWeb-Edu（120M tokens 自训 32k BPE）。
>
> 来源：`/memories/repo/dataset-research.md`（2026-07-29 tavily 联网核实，HF ID 真实存在标 [已核实]，存疑标 [需复核]）。许可红线见文末。

---

## 1. 分领域数据集

### 文学/通用高质量文本
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| HuggingFaceFW/fineweb-edu [已核实] | 1.3T tok | ODC-By 1.0 | 英文主力（pilot 已用，可扩容；教学/解释式对齐知识内化） |
| HuggingFaceFW/fineweb-2 [已核实] | cmn_Hani ~543B 词 | ODC-By 1.0 | 中文预训练（解决中文稀缺） |
| epfml/FineWeb2-HQ [已核实] | cmn_Hani ~54.2M doc | ODC-By 1.0 | 中文高质量去重（6x 数据效率，中文主力） |
| opencsg/chinese-fineweb-edu [已核实] | ~90M 条/~200B tok | OpenCSG（研究可用） | 中文高质量补充 |

### 数学推理（思考流形/推理循环）
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| AI-MO/NuminaMath-CoT [已核实] | 860k 题全 CoT | Apache 2.0 | 思考核训练主力 |
| nvidia/OpenMathInstruct-2 [已核实] | 14M 题解 | 商业友好 | SFT 规模化 |
| open-r1/OpenR1-Math-220k [已核实] | 220k R1 长推理迹 | Apache 2.0 | 长 CoT/推理循环蒸馏 |
| nvidia/OpenMathReasoning [已核实] | 306k CoT+TIR | 见仓库 | 竞赛级上限 |
| GAIR/LIMO [已核实] | 817 精编 CoT | MIT | 极小量高质量（消融/验证） |

### 物理/科学推理
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| EricLu/SCP-116K [已核实] | 274k 大学-博士级 | CC-BY-NC-SA（研究） | 科学原理推导 |
| dvilasuero/natural-science-reasoning [已核实] | 自然科学 CoT | 见仓库 | 科学思考流形 |
| galaxyMindAiLabs/stem-reasoning-complex [已核实] | 多科复杂题+CoT | 见仓库 | 多科推理 |
| MegaScience/TextbookReasoning [已核实] | 650k 教科书级 | CC-BY-NC-SA（研究） | 知识内化友好 |

### 编程能力
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| bigcode/the-stack-v2 [已核实] | 3B+ 文件/~900B tok | 溯源原始许可 | 代码预训练主力 |
| OpenCoder-LLM/opc-annealing-corpus [已核实] | 24GB 退火精调 | 见仓库 | 高质量代码课程尾段 |
| OpenCoder-LLM/opc-sft-stage2 [已核实] | 375k 代码 SFT | 见仓库 | 代码指令遵循 |
| HuggingFaceTB/cosmopedia [已核实] | 30M 文件/25B tok | Apache 2.0 | 知识内化+代码入门 |

### 通用推理/指令（后训练 SFT）
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| allenai/tulu-3-sft-mixture [已核实] | 939k 指令混合 | ODC-BY（部分非商用） | 通用 SFT 基线 |
| HuggingFaceTB/smoltalk2 [已核实] | SmolLM3 全量 SFT | Apache 2.0 | **1.5B 级最契合配方**（含 OpenThoughts3 思考/IF/多轮） |
| teknium/OpenHermes-2.5 [已核实] | ~1M GPT-4 指令 | Apache 2.0 | 通用指令遵循 |
| allenai/WildChat [已核实] | 真实多轮对话 | 见仓库 | 多轮/真实交互 |

### 长上下文（1M 目标预训练适配）
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| utter-project/LongBlocks [已核实] | 194k 长上下文 QA | 见仓库 | 长文档后训练 |
| Yukang/LongAlpaca-12k [已核实] | 12k 长文档指令 | Apache 2.0 | 长上下文 SFT 入门 |
| emozilla/pg19 [已核实] | PG-19 长书籍 | 见仓库 | 超长连续文本预训练（1M 底座） |
| togethercomputer/Long-Data-Collections [已核实] | 书籍/arXiv 长文档 | Apache 2.0 | 长上下文预训练 |

---

## 2. TAIS 特性对齐数据

### 主动求知（"不知道/需澄清"标注，训练求知分支）
| HF ID | 规模 | 许可 | 用途 |
|---|---|---|---|
| rajpurkar/squad_v2 [已核实] | 含 5 万+ 不可答 | CC-BY-SA（需审查） | 训练"不知道"拒答 |
| Human-CentricAI/chatbot-arena-llm-refusal [已核实] | 3.5k 拒答标注 | Apache 2.0 | 求知分支/诚实降级 |
| SelfAware（Yin 2023）[需复核 ID] | 1,032 不可答+可答 | 研究用 | KAL P(IK) 三态监督对齐 |
| UA-Bench [需复核 ID] | 3.5k 题 | 研究用 | 训练澄清提问（求知执行器输入） |

### 知识内化（教学式/解释式，用户给知识链条模型学习）
- HuggingFaceTB/cosmopedia（合成教科书）；opencsg/chinese-cosmopedia（中文教科书）；HuggingFaceTB/finemath（数学教育文本，数学知识链内化）。

---

## 3. 数据混合建议（对齐 OLMo 3 Dolma/Dolmino/Longmino 课程）

**预训练（30B tokens，Dolma 式混合）**：英文高质量 45%（fineweb-edu）、中文高质量 20%（FineWeb2-HQ cmn + chinese-fineweb-edu）、代码 15%（the-stack-v2 采样）、数学/科学网页 10%（finemath + cosmopedia）、长文档 5%（pg19 + Long-Data-Collections 早期混入）、求知/百科 5%。

**退火/中训（Dolmino 式，3–5B tokens）**：高质量子集 + OpenMathInstruct-2 + SCP-116K + opc-annealing + cosmopedia 教科书，提升密度。

**长上下文适配（Longmino 式，1M 阶段）**：pg19 长书 + LongBlocks + LongAlpaca + 长代码仓库（the-stack-v2 repo 级），配合三级检索注意力渐进扩窗。

**后训练 SFT（1–2M 条）**：smoltalk2 为主干 + OpenThoughts3（思考流形 CoT）+ OpenR1-Math-220k（长推理循环）+ opc-sft-stage2（代码）+ squad_v2/拒答集（求知分支 P(IK) 监督）+ WildChat 多轮（真实交互）。

---

## 4. 许可红线

- **CC-BY-NC-SA（仅研究可用，商用需替换/审查）**：SCP-116K、TextbookReasoning、Tulu 部分子集。
- **CC-BY-SA（需标注）**：squad_v2。
- **溯源原始许可**：the-stack-v2（按原始代码许可）。
- **OpenCSG（研究可用，商用需邮件报备）**：chinese-fineweb-edu、chinese-cosmopedia。
- **商用友好**：fineweb-edu/fineweb-2（ODC-By）、NuminaMath-CoT/OpenR1/OpenHermes/cosmopedia/smoltalk2（Apache 2.0）、LIMO（MIT）。

> 后期训练更大模型前，须按目标用途（研究/商用）逐数据集复核许可。
