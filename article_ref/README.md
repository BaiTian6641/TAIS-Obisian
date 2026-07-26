# article_ref —— TAIS Obsidian 背景论文参考库

> 本目录存放 TAIS Obsidian 细致框架设计文档（v2.5）所引用的关键背景论文的结构化笔记。
> 每个文件按"主题簇"组织，每条论文条目含：标题/作者/会议或 arXiv、核心主张（含关键数字/公式）、与 TAIS 的对应、**重要性评级**（⭐⭐⭐ = 设计基石 / ⭐⭐ = 强支撑 / ⭐ = 背景参考）、核实状态。
>
> **重要性标记约定**：子代理阅读时用 `【关键】` 标注必须进入设计决策的发现，用 `【风险】` 标注反面/边界证据，用 `【机会】` 标注可工程化的新机制。

## 簇索引

| 文件 | 主题 | 关键论文 |
|---|---|---|
| `01_attention_compression.md` | 注意力/压缩/测试时学习 | DeepSeek V4 CSA/HCA、NSA、DSA、TTT-E2E、Titans(MAC/MAG/MAL)、Gated DeltaNet |
| `02_dynamic_tokenizer.md` | 动态词表/无 token 化 | BLT、H-Net、T-FREE、Over-Tokenized、zip2zip、Kaplan(内词典)、FOCUS/WECHSEL/ZeTT、SuperBPE、DLCM、MOSAIC |
| `03_memory_self_compilation.md` | 记忆层/注意力自编译 | Memory Layers at Scale、ICAE、kv-distill、ICV、Function Vector、DeCoVec、Hopfield Networks、Expected Attention |
| `04_metacognition_safety.md` | 元认知/边界感知/安全 | SAPLMA、2606.02628(量化态探针)、Kadavath P(IK)、Turpin CoT 忠实、MemoryGraft、MS 后门扫描器、Betley 自我建模、Barkan 校准 |
| `05_neuroscience.md` | 神经科学/认知/神经工程 | CLS、TEM、McGaugh、cSPW-R、SHY、Fleming 元认知、奖励调制 STDP、提取练习、Howard-Kahana 时间上下文 |

## 核心交叉问题（本轮重点）

**CSA/HCA（DeepSeek V4 三级压缩注意力）↔ TTT-E2E（测试时 MLP 权重更新）是否相互干扰？**
→ 见 `01_attention_compression.md` 末尾「交叉干扰分析」节，及设计文档子系统规格的相关章节。
