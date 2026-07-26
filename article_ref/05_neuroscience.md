# 05_neuroscience —— 神经科学 / 认知 / 神经工程 参考簇

> 为设计文档（v2.5 §16.2 情感调制总线、§23 脑区映射、§23.4 睡眠修正、§23.5 推理即总结即训练）所引神经科学/认知学背景。
> `[已核]`=可访问一手/二手确认；`[未核]`=本次未能访问原文（PMC 验证码/未找到），核心主张按教科书级共识转述。

### 1. Complementary Learning Systems（CLS）— McClelland, McNaughton & O'Reilly 1995 Psych Rev；Singh & Schapiro 2026 Phil Trans R Soc B `[未核-具体]` ⭐⭐⭐【关键】`[已核]`
两速率学习系统：海马快速一次成型但易干扰；新皮层缓慢渐进整合。新经验先入海马，离线（睡眠/重放）期小步长蒸馏进新皮层避"灾难性干扰"。→ 正当化**"运行时零梯度快写（W0–W2）+ 睡眠蒸馏慢固化（W3/W4）"**双时间尺度（§24）。最根本承重。

### 2. Hippocampal Memory Indexing Theory — Teyler & DiScenna 1986 Behav Neurosci ⭐⭐⭐【关键】`[已核]`
海马**只存索引**（指向皮层内容活动的指针），皮层**存内容**；回忆=海马用索引重建皮层活动模式。→ §23.1"精确同构"：**页表=海马索引，块载荷=皮层内容**，注入"调用需海马（HRL 寻址），分析需皮层（主干推理）"。HRL→注入拓扑根本依据。

### 3. Tolman–Eichenbaum Machine（TEM）— Whittington et al. Cell 2020 `[已核-概念级]` ⭐⭐【机会】
海马-内嗅系统实现**结构抽象**——同一回路既做空间导航又做关系记忆，把"推断"统一为对抽象关系图的操作。前提（grid cell 六边形编码、O'Keefe 认知地图）2014 诺贝尔级。→ 支撑 HRL "统一内外空间导航+关系推理"定位（§16.3）。**注意：grid cell 几何在 LLM 无对应，可迁移的是"关系抽象"思想，非几何编码**。

### 4. McGaugh 杏仁核–海马调制 — McGaugh 2000 Science / 2002 TINS；McIntyre et al. 2003/2006 ⭐⭐⭐【关键】`[已核]`
情感唤起（arousal）经杏仁核调制海马-皮层固化强度——情绪强的事件优先、更持久固化；杏仁核不存内容只调"固化优先级"。→ 正当化 §16.2 情感调制总线——KAL L2 头输出 valence/arousal 作固化优先级与情感匹配召回权重。

### 5. cSPW-R（complex Sharp-Wave Ripples）— Vöröslakos, Lafferty, Zheng, ..., Buzsáki 2026 ⭐⭐【关键】`[已核]`✅
**标题**："Sharp wave-ripple clusters enhance hippocampal-neocortical engagement for memory consolidation"，bioRxiv doi:10.64898/2026.03.27.714843（**György Buzsáki 是共同作者——自带 Buzsáki 兜底**；设计文档原引 PMC13060152 可能是后续 PMC 版本号，论文真实存在）。**核心主张（已核实）**：海马输出组织成 **SPW-R 簇（cSPW-Rs）**，在 UP 态发生、phase-lock 到纺锤波谷；cSPW-R **增强默认网络(DMN)与体感运动网络(SMN)的功能分离**（= 创造受保护的海马-皮层对话窗口，最小化感觉干扰）；学习后优先重放**空间延展的迷宫轨迹**（把清醒经验拼接成连贯情景序列）；簇末涟漪把网络**驱动回 DOWN 态**终止通信窗口。inter-SPW-R 间隔峰 122.5ms（FWHM 78-182ms），簇/孤立涟漪边界 177ms。→ §23.4 修正一**全部坐实**：睡眠固化按轨迹/时间邻近**分簇**处理（路径块优先成簇）；**固化期锁定、不对外服务**（DOWN 态=合并锁）；cSPW-R 增强网络分离 = 离线时"保护性隔离"。**承重等级从"待降级"升回 🟢 承重**。

### 6. SHY（Synaptic Homeostasis Hypothesis）— Tononi & Cirelli 系列 ⭐⭐⭐【关键】`[已核-概念级]`
睡眠核心功能是**突触归一化下调**（downscaling）防饱和、恢复可学习性；为 **down-selection**（重要突触相对受保护）；需断联状态全面采样。→ §23.4 修正二：淘汰策略从 **LRU 删除** 改为**强度归一化+选择性保护**；**巩固必须离线/断联**（在线巩固被当下输入带偏）。**【风险】SHY 与"主动系统巩固假说"（Diekelmann & Born）是两个并存但不同的睡眠假说，应明确区分——前者管"归一化"，后者管"回放蒸馏"，互补非等价**。

### 7. Fleming 元认知（rlPFC/aPFC）— Fleming et al. 2014 Brain PMC4163038；Rouault et al. 2018 PMC6217996；Morales/Lau/Fleming 2018 J Neurosci PMC5895040 ⭐⭐⭐【关键】`[已核]`
元认知监测有**领域通用**（前额叶/额顶中线）+ **领域特异**（知觉=前部前额叶 aPFC；记忆=楔前叶/内侧顶叶）**并存**神经基质；aPFC 损伤选择性损害知觉元认知（不影响一阶准确率）。→ 正当化 KAL 为**骨干内生部件**（非外挂服务）；分层元认知头 L1/L2 即对"aPFC 类监测基底"的工程化。

### 8. 监测 vs 控制分离（MetaM/MetaC）— Nelson & Narens 1990 框架 `[已核]`；PMC9053853 `[未核]`⚠ ⭐⭐【机会】
监测（判断自己记忆强度）与控制（用判断引导行为/学习选择）功能可分但神经基质部分重叠。→ 指导 KAL **双层结构**——L1 探针（监测）与 L2/三元奖励（控制）分离但耦合。

### 9. 奖励调制 STDP（三因子规则）— Frémaux & Gerstner 2016 Front Neural Circuits PMC4717313；Brzosko et al. 2017 eLife 6:e27756 ⭐⭐⭐【关键】`[已核]`
学习规则 = **Hebbian 共激活（突触局部"资格迹" eligibility trace）× 第三因子（延迟全脑神经调制信号）**。**ACh 偏向抑制**（先在海马诱导 LTD）、**DA 偏向增强**（随后将 LTD 翻转为 LTP）的**顺序调制**是 Brzosko 2017 直接证据。→ 写入门控机制模板——W0 日志先记录（ACh 类"惊讶但暂不固化"），睡眠/任务奖励信号（DA 类）到位后才升级固化（§22 惊讶度门控+三元奖励的生物学蓝本）。**证据最硬、迁移最干净**。

### 10. 提取练习效应（Testing Effect）— Roediger & Karpicke 2006 Psych Sci；Karpicke & Roediger 2008 Science；Rowland 2014 元分析 Psych Bull ⭐⭐【机会】`[已核]`
检索本身是学习——测试比再读显著提升长时保留；与"desirable difficulty（合意难度）"协同；间隔重复放大。`[注]` 设计文档 **d=0.46** 未能定位单一来源（Rowland 2014 元分析 d 因设计而异，大致 0.4–0.6），**标为约值**。→ "自我测试"训练课程——固化前对候选块做校验集回归（验证门），既防投毒又借测试增强学习。

### 11. Howard–Kahana 时间上下文模型（TCM）— Howard & Kahana 2002 J Math Psych `[已核-概念级]` ⭐⭐【机会】
记忆检索由**漂移的"时间上下文"**驱动——"何时"经历的情境作检索线索，相近时间项目聚类回忆。→ §23.2 `score(block)` 的 **temporal 项**——route_key 加 temporal context 维度。

### 12. Kairos / NORA（验证门控 Hebbian）— Singh & Yu (UPenn), NeurIPS 2025 NORA Workshop, CEUR Vol-4162 paper4 ⭐⭐【机会】`[已核]`✅
**真实标题**：《Validation-Gated Hebbian Learning for Adaptive Agent Memory》。**核心主张（已核实）**：KG 边在**验证通过的推理**期间强化（LTP analog）、未用边衰减（LTD analog）、频繁共激活实体形成涌现连接；验证门 = 四维质量评估（logical/grounding/novelty/alignment）；**关键设计原则：novelty 与 correctness 是正交维度，在验证系统中平均会退化**。→ 路径块自强化（§24.3）与 CA1 验证门的直接依据。**【注意】为 workshop proof-of-concept（非主会），三实验为机械/效用/消融验证，无大规模基准；作承重证据时配 Ramsauer Hopfield-attention ✅ 与三因子 STDP ✅ 加固**。

### 13. BCI 解码器再校准（神经漂移）— 领域共识 `[已核-概念级]` ⭐
训练好的（线性）探针随主干分布漂移而失效，需定期再校准。→ KAL 探针管线维护策略——部署后周期性重新拟合 SAPLMA/已知-未知探针。

---

## 映射承载力审计：承重 vs 装饰

**总判**：~10 个脑区映射**并非全部装饰性叙事**。约 **6 个承重**（直接改写工程决策），**4 个装饰**（命名/隐喻，删掉不改设计）。

### 🟢 承重（删掉会改设计）
| 映射 → TAIS 部件 | 承重理由（改写的具体决策）|
|---|---|
| CLS（海马快/皮层慢）→ 双时间尺度架构 | 决定"运行时零梯度快写+睡眠蒸馏慢固化"双层载体本身。最根本承重。|
| 海马索引理论 → 页表=索引/块载荷=皮层内容 | 决定 HRL→注入拓扑："调用需海马，分析需皮层"。|
| SHY → 淘汰策略=强度归一化+选择性保护 | 直接把淘汰算法从 LRU 改为归一化（§23.4 修正二）。|
| 三因子 STDP → 写入门控（ACh 暂不固化/DA 升级固化）| 决定 W0 日志 vs W3+ 固化的分离门控（§22）。|
| Fleming aPFC → KAL 为骨干内生部件 | 决定元认知非外挂服务而是 checkpoint 内生头（§16.2）。|
| 提取练习 → 训练课程加"自我测试/验证门" | 决定固化前必经校验集回归（防投毒+借测试增强学习）。|

### 🟡 中度承重（影响子参数非架构）
- **cSPW-R → 睡眠分簇回放+固化期锁定**：✅ **2026 论文全文已核**（Vöröslakos/Buzsáki），DOWN 态合并锁+簇分批+网络隔离三项全部直接引述，**承重升回 🟢**。
- 监测/控制分离 → KAL 双层（L1 监测/L2 控制）。
- TCM → route_key 加 temporal 维度。
- McGaugh 杏仁核 → affect 权重项。

### 🔴 装饰（命名/隐喻，删掉不改工程）
- 丘脑/纺锤波 → 睡眠调度器（调度由软件时钟驱动，"纺锤波"只命名灵感）。
- TEM 导航 → "路径块导航先验"（真正可迁移是 TEM 的"关系抽象"思想；把推理路径叫"导航"是隐喻，LLM 无 grid 几何）。
- 人格块 → 内侧前额叶/默认网络（纯命名性映射）。
- 印迹集群/BCI 漂移 → 探针再校准（仅类比术语；实际由分布漂移检测驱动）。

### ⭐ TOP 3（最该当真、证据最硬）
1. **三因子 STDP（Frémaux & Gerstner 2016 + Brzosko 2017）**——ACh/DA 顺序调制已被实验坐实，且直接模板化写入门控。证据最硬、迁移最干净。
2. **CLS 双时间尺度**——奠基整个"零梯度快写+睡眠慢固化"架构，与运行时写不对称（W0–W2 vs W3+）安全红线一脉相承。
3. **海马索引理论**——页表/载荷的"精确同构"是 HRL 注入拓扑的唯一理论依据，承重最重。

**建议**：把 §23.1 表里**装饰行**明确标"命名性类比"。**cSPW-R（Vöröslakos/Buzsáki 2026）与 Kairos（NORA 2025）均已本轮全文核实**——cSPW-R 升回 🟢 承重（DOWN 态合并锁直接引述），Kairos 为 workshop PoC（配 Ramsauer/三因子 STDP 加固）。
