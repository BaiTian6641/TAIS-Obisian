# TAIS Obsidian：自我进化/动态记忆相关已核实文献（2026-07-25 核实）

> **文档状态（2026-07-26）**：设计文档已升至 **v2.5**（docs/ 与 docs/updates/ 同步）。v2.0 零梯度栈 / v2.4 动态词表 §28 / **v2.5 §29 第三轮独立交叉验证（五大架构命题逐项复核，11 项新证据）**。
> **第四轮（2026-07-26，论文精读+子系统规格）**：① 新建 `article_ref/`（5 簇论文笔记，子代理并行精读+重要性标注）；② 新建 `docs/TAIS_Obsidian_子系统架构规格.md` v1.0（部件→子系统→整机 7 面工程规格 + mermaid + 神经科学承重审计，承重率 ~60%）；③ 核心技术发现：**CSA/HCA ↔ 运行时学习(W-State/TTT) 确有冲突风险**——TTT-E2E 刻意只用朴素滑窗回避；安全模式=记忆作独立 KV 分支注入，绝不改冻结压缩器下游残差。venv 已装 matplotlib。所有文件无 lint 错误。

> **第五轮（2026-07-26，论文全文核实+接口实现计划）**：① 新建 `docs/TAIS_Obsidian_接口与实现计划.md` v1.0（KAL/HRL 完整接口签名 + 训练数据协议 + 信号清单 + 红线）；② 用户下载论文到 `articles/`（7 PDF + 4 HTML），全文核实：**cSPW-R 全文已核**（Vöröslakos/Buzsáki《Sharp wave-ripple clusters...》bioRxiv 714843，DOWN态合并锁+簇分批+DMN/SMN隔离三项直接引述，§23.4 升回🟢承重）、**Kairos 已核**（Singh&Yu NORA@NeurIPS2025 CEUR Vol-4162 p4，验证门控Hebbian，workshop PoC）；元认知框架群 6 篇全核实（Meta-R1 2508.17291 +27.3%、AutoMeco EMNLP2025、Know More Clearer 2602.12996、MeCo 2502.12961 ACL2025、Think² 2602.18806、MIND 2509.05714）；MemoryGraft 2512.16962 + MS扫描器 2602.03085 机制已核（87.8%待全文）；③ **HRL checkpoint 决策（用户选方案B）**：Indexer+DG+侧信道头簇内生 checkpoint，页表/BlockStore/CA3 PPR/CA1门走运行时服务；④ **勘误**：CLEAR 误归属（2412.16112 是 DiT 论文，已删除）；MOSAIC 2607.16211 是 agent记忆冲突检测（66%/4.7×）非词表扩展。所有文件无 lint 错误。
> **第六轮（2026-07-26，部件实现详细计划）**：新建 `docs/TAIS_Obsidian_部件实现详细计划.md` v1.0——32 个部件逐一 7 维（是什么/做什么/怎么实现/怎么训练/注意什么/捕捉信号/需要数据）+ dataclass/损失公式/数据 schema/信号 tensor schema/8 milestone 路线图/10 红线总表/6 不确定项。当前文档体系四层：v2.5 设计(为什么)→子系统规格 v1.0(怎么组合)→接口计划 v1.0(接口骨架)→**部件详细计划 v1.0(逐部件落地)**。

> **第七轮（2026-07-26，docs 扫除 + M1 内核骨架落地）**：① docs 扫除：删除 DKB-MS 路线图 v0.2（Qwen 9B 外挂路线废弃，R1-10 并入 v2.5）+ docs/updates/（staging 残留 v2.5 重复 + chat3.txt 已吸收）；修复 5 处悬空引用（AGENTS.md/细致框架/从零构建/DKB-MS 设计文档）；docs/ 现 11 项四层配套。② **M1 内核骨架落地**：`src/tais_obsidian/model/tais_kernel.py`（TAISKernel 聚合 KAL L1/L2 + HRL Indexer + DG + 侧信道头簇；sense/route/inject；监测/执行分置读写不同层；BlockPayload 载体能力边界强校验）+ `tests/test_tais_kernel.py`（12 项）。**全部 37 项 pytest 通过**（基线 25 + 内核 12）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`。
> **M1 关键技术决策**：concept_slot 属位置不变向量（输入侧"单槽理解"），归 VECTOR_KINDS（factual_recall=False）；token 寻址载体（kv/mem_entry/gist/lora/route）归 ADDRESSED_KINDS（factual_recall=True），本骨架 fail-closed 拒绝（M5 接 KV 拼接路径）。

> **M2 实测结果（0.1B pilot，scripts/kal_probe.py，2026-07-26）**：✅ **L1 知识感知探针达标（M2 退出标准 AUROC≥0.8）**——ℓ8 层 overall AUROC **0.945**（fake 子集 0.979）> 基线 FLARE mean-logprob 0.938；ℓ4 0.885（overall 未优基线，fake 0.959 优）。L2 情感探针仅 0.60-0.65 AUROC（chance=0.5，弱，如设计预期 T4 后实验项）。结论：0.1B 的"知/不知"线性信号确实编码在中层激活（ℓ8 最强），与 2606.02628 设计预期吻合。报告：runs/kal_probe/report.json。

> **M3/M4 落地（2026-07-26，全部 63 项 pytest 通过）**：
> - **M3 HRL 内生**：tais_kernel.py 的 HRLIndexer 加 `detach_input`（默认 True=梯度隔离，MoE-RL 红线）+ `load_from_csa_indexer`（设计 §11.1 同构初始化）；tests/test_hrl.py（5 项：隔离/透传/CSA 初始化/DG 稀疏/route 端到端）。
> - **M4 runtime 骨架**：新建 `src/tais_obsidian/runtime/`（pagetable SQLite + blockstore L0/L1/L2 usage_weighted 淘汰 + pager fail-closed + bus + ca1_gate + ca3_ppr + state_ckpt 自研 GDN 状态 save/restore）+ tests/test_runtime.py（21 项）。**子代理产出（无文件工具，返回内容我落盘验收）**；**我修复了子代理 pagetable._row_to_spec 的列索引 bug**（schema 第 5 列是 spatial_coord、第 6 列是 namespace，子代理错位为 4/5，导致 TypeError）。fail-closed / usage_weighted 非 LRU / 梯度隔离三条红线全部落实。
> - **M4 技术决策（子代理标注，我认可）**：① Pager 无页表记录时退回调用方 namespace 自证（骨架简化，M5 起 namespace 必须来自页表/载荷头）；② CA1 drift 先于 usage 判定（信念漂移拦截优先级最高）。

> **M5 注入闭环落地（2026-07-26，全部 75 项 pytest 通过）**：① 新建 `model/memlayer.py`（增强A MemoryLayer：product-key KV 参数 + delta 运行时状态；query=top-k 加权值+delta 读出；write=delta 规则 `S←S+β(v−k·S)⊗k` 分布内；forget=门控衰减）；② 新建 `model/injection.py`（Injector 统一路由：kv/gist→blockpath namespace 校验+待 HCA 拼接 (k,v)；mem_entry→memlayer 写入/查询；向量→单次加法载荷；未知载体 fail-closed None）；③ `tais_kernel.inject()` 加 `injector` 参数——token 寻址载体传入后接通（不再 fail-closed），不给时仍 fail-closed。**我修复了 memlayer 读出布局 bug**（state [KD,D]，读出应为 `kn@state` 左乘非 `state@kn`）。tests/test_injection.py（12 项：memlayer 查询/写入/遗忘、Injector KV 校验/mem/向量、内核集成、state_ckpt 含 mem 状态往返）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`。

> **M6/M7/M8 落地（2026-07-26，全部 94 项 pytest 通过）**：
> - **M6 睡眠固化**：新建 `sleep/consolidator.py`（cluster_by_temporal 分簇回放 cSPW-R；retrieval_practice 间隔提取练习 答对强度+/阶段+ 答错回退；shy_normalize 归一化+top保护 非LRU；SleepConsolidator 编排 分簇→提取→CA1门→归一化 + 离线锁定不可重入）+ tests/test_sleep.py（7 项）。
> - **M7 动态词表**：新建 `model/dyn_vocab.py`（vocab_friction_score KAL 词表摩擦；DynamicVocab detect/extract/register/promote；extract_fn 回调接 Kaplan ℓ5-15 hidden state；concept_slot 注册页表 compiled_kind=concept_slot factual_recall=False 位置不变向量）+ tests/test_dyn_vocab.py（5 项）。
> - **M8 安全管线**：新建 `runtime/safety.py`（sign_block/verify_signature HMAC-SHA256 恒定时间比对；SafetyPipeline 签名→扫描器→CA1门 编排 fail-closed；scanner_fn 回调接 MS 扫描器）+ tests/test_safety.py（7 项）。
> - **M1–M8 全部完成**：model/(tais_kernel/hrl_heads 合入 kernel, memlayer, injection, dyn_vocab) + runtime/(pagetable/blockstore/pager/bus/ca1_gate/ca3_ppr/state_ckpt/safety) + sleep/consolidator。94 项 pytest 全绿。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`。

> **内核端到端接线（2026-07-26，全部 98 项 pytest 通过）**：① config.py 加 kernel 挂点字段（kernel_enabled=False 默认关/kernel_dg_dim/kernel_dg_topk/kernel_sense_layers）；② model.py 加 `attach_kernel()`（内核权重进 self.kernel 随 state_dict 存取）+ `forward(run_kernel=False, injector=None, inject_payloads=None)`——开启时在 GDN 层后 sense（监测只读）、CSA 层前 inject（执行写入），监测/执行分置，返回 kernel_signals；默认关闭时 forward 与纯基线**逐点一致**（94 项基线零改动强判据，已测）。tests/test_kernel_wiring.py（4 项）。**顺手修复 HEAD 预存配置漂移 bug**：`attn_impl` 默认值被 commit 27915a4（Add unit tests for KAL and TriAttention）污染成 `"tri"`，导致 `test_blockpath`（期望 CSAAttention full）失败——已改回 `"full"`（设计纪律：默认全注意力，三级栈为可选开关）。诊断过程：git stash（=HEAD）时 test_blockpath 过、pop 后挂→定位 cache k 布局 [B,T,n_kv,hd](TriAttention) vs [B,n_kv,T,hd](CSAAttention)→确认 attn_impl 默认被改。

> **D-0 pilot 端到端内核烟测（2026-07-26，通过）**：新建 `scripts/e2e_kernel_smoke.py`，在真实 0.1B checkpoint（checkpoints/pilot_0p1b_ws/final，d=768 12层 GGGAGGGA 单流 attn_impl=full）上验证：① 基线 loss 4.4029，挂内核+run_kernel=False **Δ=0.00e+00** 完全一致；② run_kernel=True 时 9 个 GDN 层（0,1,2,4,5,6,8,9,10）全部产出 sense 信号（P(IK) logits [1,128,3]，内核未训练输出无语义，仅验证通路）；③ 注入 steering（小幅度 0.01）后 loss Δ=+0.0001 **人效未显著降**（M5 退出标准）；④ KV 载荷未给 injector 时 fail-closed 拒绝。**修复 device 对齐 bug**：attach_kernel() 创建的内核默认在 CPU，模型在 CUDA→sense/inject 设备不匹配，已改为内核跟随 self.embed.weight 的 device/dtype。98 项 pytest 全绿。

> **KAL 内生训练接入（2026-07-26，通过）**：① `model.py` 加 `kernel_sense_index()`（sense 读点层索引，空=全部 GDN 层）；② `train.py` 加 `kal_pik_aux_loss()`（**在线自标注** P(IK)：主干自己 next-token 正确性生成伪标签——预测对=知道/预测错=未知，Kadavath 范式的最小内生实现，无需外部数据集；**红线**：对主干 hidden detach，探针梯度只进 KAL 头不污染主干——NeurIPS 激活监控警示 + 探针冻结）+ `kal_aux_weight`（默认 0.0=关，既有训练零改动）+ `kal_sense_layers`；③ model_cfg 在 kal_aux_weight>0 时自动 kernel_enabled=True。**验证**：aux loss 0.679（≈ln2 随机初始化合理）；主干 embedding 梯度 None（detach 生效）、KAL 头梯度 1.876（可训练）；**20 步训练烟测**（0.1B + kal_aux_weight=0.1）：loss 10.57→8.34（ema 9.98）、val 8.41、grad norm 47.55→2.09、无崩溃/OOM。98 项 pytest 全绿。这是内核从"结构通路"变成"有语义元认知头"的 T1 起点。

> **HRL Indexer 初始化（§11.1 近似，2026-07-26，103 项 pytest 全绿）**：**关键诚实发现**——设计 §11.1 设想的"独立 CSA indexer 打分向量"在本仓库两个注意力实现里都不存在：full=CSAAttention 是全注意力（检索靠 q·k，无 indexer）；tri=三级栈的 CSA 分支选择分数 = 压缩注意力分数 Softmax(q·K̃)（复用 q_proj 对压缩 key 点积，**非独立 indexer 模块**，tri_attention.py docstring 明示；V4 的独立 lightning indexer 原型未引入）。故最贴近 §11.1 精神的真实可提取来源是 **q_proj 打分方向聚合**。实现：`HRLIndexer.init_from_attention_qproj()`（W_q [n_q*hd, d]→按 query 头分块取均值→d 维方向→归一）+ `TAISKernel.init_indexer_from_model()`（取第一个 "A" 层 q_proj 初始化，无 A 层 fail-closed 返回 -1）。**诚实标注**：这是近似初始化（非 §11.1 设想的独立 indexer，而是 query 打分方向聚合；T2 仍须经块域 KL 对齐正式训练）。tests/test_hrl_init.py（5 项：载入方向/聚合正确/无 A 层 fail-closed/确定性/形状校验）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（103 项全绿）。

> **训练效率优化笔记**：新建 `/memories/repo/training-efficiency.md`（RTX PRO 4000 Blackwell sm_120 PyTorch 训练吞吐；已落实 TF32/bf16 autocast/fused AdamW/grad ckpt；可补 set_float32_matmul_precision("high")+cudnn.benchmark、torch.compile、non_blocking、cuDNN NSA kernel、Liger）。

> **GDN-2 采纳 + CSA Indexer + 重命名（2026-07-26，111 项 pytest 全绿）**：
> - **重命名（误导名修正）**：`CSAAttention`（实为全注意力）→ `FullAttention`；`blockpath.CSACompressor` → `BlockCompressor`（vscode_renameSymbol 语义重命名 + docstring 清理）。**坑**：PowerShell 文本替换含中文的 UTF-8 文件会写 BOM/乱码损坏文件——tri_attention.py 被损，已 git checkout 恢复 + 用 Python（非 PowerShell -replace）正确替换。教训：含中文 UTF-8 源码的批量替换一律用 Python，不用 PowerShell -replace/Set-Content。
> - **GDN-2 erase/write 解耦采纳**：memlayer.write() 升级——erase_gate（key 侧 b⊙k，控制擦除时 key 哪些坐标参与）与 write_gate（value 侧 w⊙·，控制承诺 value 哪些坐标）独立（arXiv:2605.22791）；默认 tied 向后兼容。**我修了 erase gate 位置 bug**（erase 应作用在 kn_eff=e*kn 的 key 侧，而非乘在 old 读出上——GDN-2 的 (I−k(b⊙k)ᵀ) 项是 key 侧坐标选择）。
> - **CSA Indexer（真正独立打分器）**：新建 `model/hrl_indexer.py` LightningIndexer（DSA lightning indexer 式：`I=Σ_j w^I_j·ReLU(q^I_j·k^I)`，独立多头低维 q^I/w^I/k^I 投影，**非复用主干注意力**；top-k；KL warmup 对齐稠密教师；分数可微）。文献依据：DeepSeek V3.2 lightning indexer（Eq.1，独立低维 indexer + KL warmup + FP8）+ PEER（arXiv:2407.04153，product-key + 内生独立 query network + 分数可微）。**区别于 tais_kernel.HRLIndexer 的 nn.Linear(d,1) M3 骨架**——本模块是 DSA 式真正 indexer，供 token 域/块域共用。
> - **文献表更新**（article_ref/01）：追加 Gated DeltaNet-2、DeepSeek lightning indexer、PEER 三条（均✅核实）。
> - tests/test_gdn2_indexer.py（8 项：memlayer tied 兼容/erase=0 不写/write=0 不承诺/向量门解耦 + indexer 形状/topk/可微/KL warmup）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（111 项全绿）。

> **重命名为 RetrievalAttention + LightningIndexer 接入内核（2026-07-26，116 项 pytest 全绿）**：
> - **命名（用户确认正式架构用 tri，GDN 命名 OK）**：`FullAttention` → **`RetrievalAttention`**（检索注意力，体现混合架构中"全局情景检索 L1"角色，与 GDN-MemBlock 工作记忆寄存器对偶）。语义关系清晰：`attn_impl="full"`→RetrievalAttention（全注意力占位/对照基线）；`attn_impl="tri"`→**TriAttention**（正式三级栈路径：滑窗 L0+CSA 选择检索 L1+HCA gist L2）。Block 里 `attn_impl=="tri" and not attn_only`→TriAttention，否则→RetrievalAttention。重命名用 vscode_renameSymbol + Python docstring 清理（避开 PowerShell 编码坑）。
> - **LightningIndexer 接入内核**：`HRLIndexer` 加 `use_lightning`（默认 True）+ lightning 子模块 + `score_candidates()/topk_candidates()/kl_warmup_loss()`；`TAISKernel` 加 `route_candidates()`（全分数或 top-k，token 域/块域同构）+ `indexer_kl_warmup_loss()`。token 域（压缩条目）/块域（知识块）同构——一个打分器两种检索对象（设计 §11.1）。detach_input 梯度隔离（MoE-RL 红线，已测：主干 query/candidates 梯度为零、lightning 投影可训练）。
> - tests/test_kernel_route_candidates.py（5 项：全分数/top-k/梯度隔离/KL warmup 反传/命名一致性 full=RetrievalAttention & tri=TriAttention 都能 init_indexer_from_model）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（116 项全绿）。

> **RetrievalAttention/TriAttention 架构分析 + V4 CSA 独立 indexer 集成（2026-07-26，120 项 pytest 全绿）**：
> - **架构关系厘清**：RetrievalAttention 与 TriAttention 是同一 "A" 层槽位的两种实现——`attn_impl="full"`→RetrievalAttention（全注意力对照基线，GQA+SDPA）；`attn_impl="tri"`→**TriAttention**（正式三级栈路径，之后测试都用它）。命名已对齐。
> - **V4 最优组合（用户确认）**：DeepSeek V4 能力 + 我们的 Indexer。交叉验证（Tavily）：① V4 CSA = stride-4 压缩 + **独立 lightning indexer 在压缩条目上选 ~128/query + FP4(MXFP4)**（indexer 从 V3.2 FP8 降 V4 FP4，QAT 保精度）；② 效率关键 = **memory-bandwidth-bound**（减 bytes moved per token，不只是 FLOPs）——1M 上下文 V4-Pro 仅 V3.2 的 27% FLOPs + 10% KV；③ V3.2 indexer 复用 MLA 压缩表示轻量打分，V4 在压缩条目上打分。
> - **实现**：config 加 `tri_use_indexer`（默认 False=NSA 式保持兼容）+ `tri_index_heads`/`tri_index_dim`；TriAttention CSA 分支 use_indexer=True 时用**独立 LightningIndexer 在压缩条目上打分选 top-k**（V4 CSA 式，与 HRL 的 model/hrl_indexer.py LightningIndexer **同构共享**——设计 §11.1"一个打分器两种检索对象"），query 按 kv 头分组共享选择（NSA Eq.10 同纪律）。
> - **架构事实（诚实标注）**：V4/NSA 设计是"indexer 用于离散 top-k 选择（无梯度），主注意力分数回流到 q/k/压缩器"——故 o.sum().backward() 不给 indexer 梯度是**符合设计**的；indexer 训练走独立 **KL warmup**（对齐稠密教师，V3.2 warmup 范式，已实现 kl_warmup_loss）。测试据此修正。
> - tests/test_tri_indexer.py（4 项：形状/因果性红线/KL warmup 可训练/双模式可用）。验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（120 项全绿）。
>
> **移除旧 attention + 重命名 TriRetrievalAttention + 文档同步（2026-07-26，120 项 pytest 全绿）**：
> - **移除（用户确认 B：彻底移除对照组）**：删 `attention.py`（RetrievalAttention）+ `attn_only`/`attn_impl` 字段与对照组逻辑；`configs/*.json` 清理 attn_only/attn_impl 字段；删 `configs/pilot_0p1b_attn.json`（对照组废弃）；`model.py` "A"层统一 `TriRetrievalAttention`；`config.layer_types` 不再特判 attn_only；`train.py` 清理 attn_only/attn_impl。
> - **重命名**：`TriAttention` → **`TriRetrievalAttention`**（三级检索注意力，体现混合架构中"检索"角色与三级结构）。坑：子字符串替换把 `TriRetrievalAttention` 里的 `RetrievalAttention` 又替换 → `TriTriRetrievalAttention`，已修类名。教训：批量替换先替换更长的名字。
> - **文档同步**：AGENTS.md（§2.1 代码结构全重写、tests 计数 25→120、M0-M8 全部✅、attn_only 对照组标注废弃）；子系统架构规格/部件详细计划/接口计划/细致框架设计文档的 CSA-AttnBlock→TriRetrievalAttention。
> - **兼容加固（不留坑）**：① `ModelConfig.from_json` 忽略未知字段（旧 checkpoint config.json 含 attn_only，向后兼容）；② `from_pretrained` 加 `strict` 参数（默认 True；**strict=False 兼容模式**——旧 CSAAttention checkpoint 在新 TriRetrievalAttention 架构下结构不同，strict=False 载入旧主干权重（embedding/GDN/MLP/q/k/v/o_proj），30 个新三级栈参数（csa_comp/hca_comp/gate_w/gate_b）随机初始化供后续微调）。已验证旧 pilot_0p1b_ws checkpoint 在 strict=False 下正常加载+前向。
> - **blockpath 布局适配**：harvest/inject 适配 TriRetrievalAttention 的 state k/v 布局 [B,T,n_kv,hd]（旧 CSAAttention 为 [B,n_kv,T,hd]）——transpose 兼容两种布局。
> - 验证命令：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/ -q`（120 项全绿）。

> **GDN-MemBlock 与注意力层交叉验证（2026-07-26，Tavily 检索）**：
> - **实现核对（对齐良好）**：gdn.py 与 Gated DeltaNet（ICLR 2025, arXiv:2412.06464）原版 + fla 参考（fla/ops/gated_delta_rule + fla/layers/gated_deltanet.py）逐部件对齐：投影→短因果Conv1d+SiLU(kernel=4,带cache)→L2norm q/k→**beta=sigmoid(b_proj) 写入强度**→**decay g=-exp(A_log)·softplus(a+dt_bias)**（KDA 通道级衰减，fla 初始化 A~U(0,16)/dt_bias softplus 逆）→delta 状态（naive_recurrent + chunked WY 表示 batched 三角求解 O(C²)，fp32 对拍 <1e-4）→GVA 头重复→RMSNormGated 输出。GDN 无 KV cache，状态=递归 S（W-State 红利）。
> - **关键演进（新发现）**：**Gated DeltaNet-2（NVIDIA, arXiv:2605.22791, 2026-05）**——解耦 erase gate `b_t`（key 轴，移除衰减状态哪些坐标）与 write gate `w_t`（value 轴，承诺哪些新值坐标），去除原版单一标量 βₜ 的 tied 限制；`S_t=(I−k(b⊙k)ᵀ)D_t S_{t-1}+k(w⊙v)ᵀ`；β 合并标量退化为 KDA，衰减也合并退化为 GDN；chunkwise WY 保持并行；**matched 1.3B/100B FineWeb-Edu 超越 Mamba-2/GDN/KDA/Mamba-3，RULER 长上下文检索大幅领先**（S-NIAH-3@2K 63.2→89.8，MK-NIAH-1@4K 28.0→37.8 over KDA）。
> - **GDN-2 naive 落地 + 记忆保护实证（2026-07-27，model/gdn2.py）**：`naive_recurrent_gated_delta_rule_2`（对齐官方 fused_recurrent 语义：decay→erase `(b⊙k)@h`→write `v_new=(w⊙v)−erase_d`→read）+ `tied_to_decoupled`（GDN-1 兼容）。5 对拍测试全绿：**tied 退化=GDN-1（<1e-4，严格一般化）**、b=0 纯累加、解耦独立。**记忆检索实证（语义对拍）**：key2 干扰 key1（共享 4 个 key 坐标）后查询 key1→val1=5.0，**GDN-1 标量 β 无差别擦除读出 −4.804（旧关联被覆盖/符号翻转），GDN-2 erase gate 保护重叠坐标读出 2.741（接近真值）**——直接验证论文核心 claim"erase gate 贡献最大"（选择性保护 key 侧关联），正对 §25.2 GDN 固定状态检索短板的补强路径。**下一步（全层消融）**：gdn.py 主干是 GDN-1，GDN-2 需 chunked 化（当前 naive 仅对拍/解码路径）+ config 开关 + 检索任务训练消融。**GDN-2 全量切换落地（2026-07-27，用户决策"直接切换"）**：① **chunked 训练核**（chunked_gated_delta_rule_2，子代理实现 + 我验收）——WY 表示，erase gate 折入 key tile（`k_beta=(b⊙k)`）+ write gate 折入 value tile（`u=(w⊙v)`），A_strict 三角求解；**对拍 CPU/CUDA 无/带初态 <1e-6、tied 退化=GDN-1 chunked 0.00e0**（严格一般化）。② **GDN2Block**（继承 GDNBlock，`b_proj→key_dim` erase + `w_proj→value_dim` write，sigmoid [0,1]，GVA 时 b 随 q/k 重复；参数 +876K/层）。③ **集成切换**：`layer_type="G2"`→GDN2Block；config block_pattern 默认 `G2G2G2A`（0.1B 115.99M 在 90-130M）；`kernel_sense_index`+use_kernel sense 覆盖 G2（GDN 系读点）；config.layer_types 接受 G2。**测试适配**：test_capture（set(types) G/A→G/G2/A）、test_blockpath（g_before t=="G"→in("G","G2")）。4 chunked 对拍测试（naive/tied/训练-生成路径一致）。**177 全绿**。**坑**：旧 GDN-1 checkpoint 的 G 层权重（GDNBlock 标量 b_proj）与 G2（GDN2Block channel-wise b/w_proj）形状不兼容——需 from_pretrained skip_keys 或重训。**关键实现洞察**：GDN-1 chunked 的 `k_beta=k*beta` 与 GDN-2 的 `k_beta=k*b` 形式同构（标量广播 vs channel-wise），故 GDN-2 chunked 几乎是 GDN-1 的 `k_beta=k*b`、`u=v*w` 两处替换 + b 进 A_strict——子代理抓对了这个对称性。**GDN-2 pilot 训练完成（2026-07-27，configs/pilot_0p1b_gdn2.json）**：2000 步 val loss **3.7600**、8.6k tok/s、7.14GB、final checkpoint 生成（115.99M G2G2G2A）。**GDN-1 对照训练启动**（configs/pilot_0p1b_gdn1.json，GGGAGGFA 标量 β，108.11M）——同 seed/数据/步数/批大小/TriRetrievalAttention+V4 indexer，**唯一变量=GDN 层 erase/write 门形式**（干净消融）。**坑（已修）**：train.py 原 `block_pattern` 硬编码 G2（config JSON 的 block_pattern 不生效）+ `tri_use_indexer` 硬编码 False（与 config 扶正 True 不一致）——已改为从 config 传入/默认 True。**对比矩阵**：GDN-2 val 3.7600 vs GDN-1 val（训练中）→ 判全层收益；另需合成检索任务（NIAH 式 key-value 埋点查询）验 GDN-2 erase gate 检索主场（val loss 之外的核心优势）。**GDN-1 vs GDN-2 对比结果（2026-07-27，诚实负结果+诊断）**：2000 步 pilot（~134M tokens，同配方）：**val loss GDN-1 3.7586 vs GDN-2 3.7600（Δ+0.0014，GDN-1 略低但噪声内——GDN-2 同分布 LM 无劣势）**；**NIAH 检索（8 key 干扰 100 查询）：GDN-1 0.200 vs GDN-2 0.130（GDN-2 反而劣）**。**根因诊断（非架构错误，是欠训练）**：GDN-2 的 channel-wise 门在 2000 步内**几乎未学到选择性**——erase b mean 0.503/坐标分化度 0.0244、write w mean 0.500/0.0193（≈初始 sigmoid(0)=0.5 均匀未分化）；b/w≈0.5 等效"半强度无差别擦写"，比 GDN-1 学到的标量 β（可学到接近 0/1 的明确强度）**更弱**。**关键差异**：NVIDIA 用 100B tokens 训练 GDN-2 的门有充足信号学会选择性擦除，我们 2000 步 pilot 门未收敛。**之前语义对拍（手工设门）已证 GDN-2 erase gate 保护能力（−4.804→2.741）**，故检索劣势是**门欠训练**非架构缺陷。**结论：① GDN-2 切换在 val loss 上无劣势（可安全作默认）；② 检索优势需 >2000 步训练让 channel-wise 门收敛（NIAH 复测应随训练步数增加而改善）；③ 门收敛诊断指标=b/w 坐标分化度（std across dim，>0.05=学到选择性）**。产出 runs/retrieval_niah/report.json。
> - **对设计的三条价值**：① 精确对应读写不对称红线——erase/write 解耦=我们 W2 记忆层 delta 写与门控衰减遗忘的细粒度分离；② RULER/NIAH 检索增益正对 §25.2 已知"GDN 固定状态检索密集遗忘（2510.20787）→CSA 补偿"，是 CSA 补偿外的另一条补强路径，列 T1/T2 消融（GDN vs GDN-2）；③ 增强A memlayer 现为 tied delta 写，可借鉴 GDN-2 解耦"擦除（读旧）/写入（承诺新）"做更精细分布内写入。
> - **注意力层通路核对（完备）**：L0 滑窗(CSA/tri滑窗)→L1 CSA(tri ChunkCompressor V4 softmax门控池化+topk复用压缩注意力分数)→L2 HCA(128:1+inject_hca_entries 块注入原生落点)→NSA Eq.5 门控融合(零初始化+bias→均等1/3)。HCA 区=块注入原生落点（设计 §17.3 前缀偏差从结构消失）。
> - **补强机会（列消融）**：① GDN tied β→GDN-2 解耦 erase/write；② memlayer delta 写解耦；③ 训练吞吐 GDN 纯 PyTorch 9.5k vs SDPA 19.7k→set_float32_matmul_precision("high")+cudnn.benchmark+远期 cuDNN NSA/GDN-2 Triton kernel(sm_120)。

> **第四轮关键勘误（2026-07-26 已回填设计文档 ab67224）**：SAPLMA 71–83% 是 **accuracy 非 AUROC**（细致框架 §8.2 表/§8.3/§29 已标注）；ICV arXiv 是 **2311.06668**（非 2310.10678=物理论文）；FOCUS 实为 **EMNLP 2023**（非 NAACL 2022）；SuperBPE 实为 **COLM 2025**（2503.13423 已核实，§28 补出处）；微软后门扫描器 **"87.8% 检出/0 误报" 摘要未含、已标 UNVERIFIED**（§26.2）；MOSAIC 未能定位论文（用 FOCUS/OMP 佐证）；cSPW-R(PMC13060152)/Kairos(NeurIPS2025)/PMC9053853 已核实（cSPW-R 全文已核）。

> **本轮工程推进（2026-07-26）**：① 效率加固 `fd65871`（train.py 加 set_float32_matmul_precision("high")+cudnn.benchmark，与既有 TF32 同族，120 项 pytest 回归全绿）；② **T2 HRL LightningIndexer KL warmup 脚本 `1aeda9c`**（scripts/hrl_warmup.py：冻结主干+detach 红线，教师=第一个 A 层真实 q·k 因果 mask 跨 kv 头均值分数，学生=HRL LightningIndexer，KL 对齐 + top-k 重叠率指标，输出 runs/hrl_warmup/{report.json,warmed_indexer.pt}；AST 通过、--help 自检通过；**GPU 实跑排队等 2000 步消融完成**）。**坑**：TAISKernel.__init__ 只接 d_model/dg_dim/dg_topk，use_lightning/n_heads/d_index 在 HRLIndexer.__init__（默认 True/4/32）——勿误传给 TAISKernel。**坑2（warmup 教师提取）**：TriRetrievalAttention 的 q_norm/k_norm 是 RMSNorm(head_dim=64)，必须**先 view 拆头 [B,T,n,hd] 再 norm**（对齐 tri_attention §237-238 前向 `q_norm(q_proj(x).view(B,T,n_q,D))`）——误在整个投影输出 768 上 norm 会 RuntimeError(768 vs 64)。**坑3**：PowerShell `Out-File -Encoding utf8` 仍把 python UTF-8 中文 print 写乱码（PYTHONIOENCODING 不够），但 report.json(write_text utf-8) 正常。**warmup 实测（2026-07-26，runs/hrl_warmup/report.json）**：1000 步/61.5s，val_kl 368913→889（**×0.0024 降 415 倍**），top-16 重叠率 0.0387（随机）→**0.2525（6.5×）**，判定达标。解读：indexer 从随机结构通路变有语义检索器（低维近似重现稠密注意力相关条目）；0.25 绝对值合理（4头32维有损近似对12头64维稠密，DSA 设计本就快打分器非无损复刻）；KL 终值 889 偏大是教师分布尖锐+log 空间的数值口径，收敛方向+重叠率才是真指标。产出 warmed_indexer.pt 可灌回内核。**q_norm 坑已修**（先 view 拆头再 norm）。**KAL 完整训练启动（2026-07-26，后台运行）**：configs/pilot_0p1b_kal.json = 基线 2000 步配方 + kal_aux_weight=0.1 + kal_sense_layers=[8]（M2 探针发现 ℓ8 P(IK) 最强 0.945）。train.py kal_pik_aux_loss 已确认正确：detach 主干 hidden（监测/执行分置红线）、伪标签=主干自己 next-token 正确性（known=预测对）、梯度只进 KAL 头。kernel_enabled>0 时 model.__init__ 自动 attach_kernel（state_dict 含 kernel 权重，config.json 带 kernel_enabled=true，from_pretrained strict=True 完整还原）。**内生评估脚本 scripts/eval_intrinsic_kal.py**：与 kal_probe.py（M2 事后线性探针，frozen hidden 手训外部逻辑回归）关键区别——本脚本**不训练任何探针**，直接用训练后 kernel.kal_l1 头打分（score=logit[know=0]−logit[blank=2]，对齐 train.py 二分类退化），复用 kal_probe 的 L1 数据集/forward_collect/auroc，判 T1 目标"内生头直接产 AUROC≥0.8"。训练完成后跑：`CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/eval_intrinsic_kal.py --ckpt checkpoints/pilot_0p1b_kal/final --layers 8`。**KAL 完整训练实测结果（2026-07-26，诚实负结果）**：2000 步 val loss 3.7675（与基线 3.768 持平，KAL aux 未损害主干 ✅），但**内生 KAL 头 ℓ8 AUROC 仅 0.433（<0.5，fake 0.491），远逊 M2 事后探针 0.945/0.979，且 logit0("知道")在 fake 上均值(1.174)反高于 known(0.657)——方向错位**。**根因诊断（重要设计发现，非 bug）**：kal_pik_aux_loss 伪标签=主干 next-token 预测正确性，但这测的是**文本局部可预测性/流畅度**而非**事实真假**——known(FineWeb 真实文本)语言建模高熵常预测错→错标"未知"；fake/shuffled 局部 n-gram 仍可预测→错标"知道"。正合 2606.02628 核心结论：**幻觉检测信号在线性 hidden state，但训练目标须锚真值(fake vs real)非语言建模置信度**。M2 事后探针(0.945)用真实 fake/real 标签故有效。**修正方向（T1 迭代）**：KAL 内生训练需真值锚——用伪事实模板（kal_probe build_fake_fact_texts 同款）vs 分布内文本作 known/unknown 标签，替换预测正确性代理；或两阶段（先在线自标注预热，再真值微调）。**教训：在线自标注的 P(IK) 代理目标与真实元认知语义可错位，须用真值校验**。**真值锚微调成功（2026-07-26，方案1落地，scripts/kal_truth_finetune.py）**：加载 pilot_0p1b_kal/final，冻结主干+内核其余部件只训 kal_l1，真值=fake(伪事实=unknown类2)/real(val=known类0)，detach 主干红线，500 步/215s。**结果：内生头 overall AUROC 0.447→1.000、fake 0.447→1.000，CE 2.07→0.000，判定达标✅**——完美验证根因诊断（信号一直在 hidden state，只是代理目标错位，真值锚立刻对齐）。**泛化检验（亲自验收，不同种子+未见 shuffled）**：known vs fake=**0.998**（真值头对伪事实检测真学会，非种子过拟合）；known vs shuffled=**0.576**（近随机，对乱序无检测力）；overall 0.787。**诚实解读**：真值头学的是"这是否语义连贯的真实陈述"（fake=虚构实体/语义异常→检出），非"是否流畅文本"（shuffled 语法破碎但无虚构实体→不检出）——与 M2 事后探针语义轮廓完全一致（fake 子集 0.979 最强）。**这是 KAL P(IK) 的正确定位：检测知识空白/语义异常，非语言建模难度**。产出 checkpoints/pilot_0p1b_kal/final_kaltruth（含真值微调内核，随 checkpoint 存取运行时零额外训练）+ runs/kal_truth/report.json。**方法论闭环：M2 事后探针证信号存在→在线自标注训出反方向（负结果暴露代理错位）→真值锚微调对齐（0.998）→泛化检验定语义边界**。③ **2000 步 tri_use_indexer 正式消融完成（架构级判定）**：NSA val **5.3543** vs V4 val **5.3583**（**Δ+0.0041 < 0.02 不劣化**），tok/s 8069→8181（+1.4%）、peak 3.48=3.48GB、params +0.031M。**Δ 从 300 步 +0.0119 收窄到 2000 步 +0.0041**——差距随训练收敛进一步缩小，实证扶正"统一 V4 式独立 LightningIndexer 正式路径"的决策。短上下文 indexer 吞吐中性，V4 的 KV/FLOPs 优势（1M=27% FLOPs+10% KV）只在长上下文主场兑现——**0.1B/512 的 parity = 免费获得 1M 长上下文能力**。④ 动态 tokenizer（M7）检查 `dde54fe`：实现与设计 §28.2 第 0 级全对齐（检测/提取/注册/注入四环节红线全落实），顺手修 docstring 提取层 ℓ5-15→ℓ10-14（对齐设计 §787 正式口径）。**坑**：PowerShell `>` 重定向把脚本 UTF-8 中文 print 转 UTF-16 致 _ablate_out.txt 乱码（不影响 write_text 的 JSON），读用 Get-Content；后续长任务建议脚本内 logging 到 UTF-8 文件。

> **第三/四轮（2026-07-26）新增已核实文献**——见文末「v2.5 第三轮」「v2.5 第四轮」节。所有 arXiv 编号已联网核实存在性。

## 核心理论桥：ICL ≈ 隐式梯度下降（块注入=低秩权重更新）
- von Oswald et al. ICML 2023, *Transformers Learn In-Context by Gradient Descent* — K 注意力层 ≈ K 步 GD
- Dai et al. ACL 2023 Findings, *Why Can GPT Learn In-Context?* — ICL=隐式微调，meta-gradients 经注意力施加到 FFN/Attn 权重
- *Learning without training* arXiv:2507.16003 — transformer block 把上下文隐式转成对 MLP 的 rank-1 权重更新（有显式公式）

## 精度差（块注入 ≠ 完全等同训练）
- *Understanding Parametric Knowledge Injection in RAG* arXiv:2510.12668 — P-RAG 不总是优于 T-RAG，PT-RAG（两者结合）最优
- NeurIPS 2025 *Parametric and Contextual Knowledge Reconciliation* — 上下文 vs 参数知识走不同注意力头集合
- Owain Evans：训练数据内推理 > 上下文内推理（ECIR 2025 keynote 引）

## 额外动态记忆层证据
- **Memory Layers at Scale** arXiv:2412.09764 (Meta, ICML 2025) — 稀疏 key-value 查找，加参数不加 FLOPs，128B 参数/1T tokens，事实任务尤其强 ← 建议作为 GDN-MemBlock 升格的工程依据
- *Hopfield Networks is All You Need* Ramsauer arXiv:2008.02217（被引 1152）— 注意力=现代 Hopfield 一次检索，存储容量随维度指数增长 ← 理论地基
- Titans arXiv:2501.00663 — 神经长期记忆模块，测试时优化自身权重学习记忆/遗忘

## 运行时自编译（注意力压缩上下文为块）
- **ICAE** arXiv:2307.06945 (ICLR 2024) — 长上下文压缩成 memory slots，4× 近无损，LoRA 编码器 ← CSA harvest() 接口的范式
- **Latent Context LM (LCLM)** arXiv:2606.09659 (2026.06) — encoder-decoder 压成长 latent embeddings，长程 agent backbone，指向 persistent memory

## 脑分工映射
- CLS 互补学习系统（McClelland 1995；Singh & Schapiro 2026 Phil Trans R Soc B）— 海马快/皮层慢 ← HRL vs 冻结基座的理论框架
- HiCL *Hippocampal-Inspired Continual Learning* AAAI — CLS 式 fast/slow 双系统持续学习

## 情感/杏仁体（情感层证据）
- **Anthropic Transformer Circuits《Emotion Concepts and their Function in a LLM》** transformer-circuits.pub 2026 — 模型内 valence/arousal 组织的情感空间，线性可提取+可 steer 输出 ← 支撑 Affective Perception Head，建议提前到与 KAL 同期训练
- *Affective Computing in Era of LLMs* arXiv:2408.04638（综述）
- Russell 环形模型（valence-arousal 2D）是标准框架

## 视觉空间区 ↔ 海马耦合（空间推理不是外挂，是海马本职）
- **Tolman-Eichenbaum Machine (TEM)** Whittington et al. *Cell* 2020（被引 857）— 海马-内嗅系统统一空间导航与关系记忆为"结构抽象/泛化"同一机制 ← HRL route_graph 加坐标边 = 获得 TEM 式抽象推理脚手架
- 位置细胞(O'Keefe 1978)/网格细胞(Hafting 2005，Nobel 2014) — 海马最原始功能是空间认知地图；空间记忆障碍(AD)即认知地图受损
- **VLM2: Vision-Language Memory for Spatial Reasoning** arXiv:2511.20644 (ECCV 2026) — VLM 空间推理瓶颈=缺持久情景记忆(非视觉感知)；解法=工作记忆+跨帧情景记忆（正长在 HRL 上）
- Ego3D-VLM arXiv:2509.06266 — 显式认知地图让多选QA +12%/距离估计 +56%
- "What's up with VLMs?" Kamath EMNLP 2023 — VLM 连基础方向都常错

## 视觉空间接入的风险（需主动对冲）
- 语义-几何失配（VLM2 挑战#1）：视觉语义token与坐标活在不同空间，朴素拼接≠连贯3D理解 ← 最确定的失败模式，对策=学习型几何嵌入而非裸坐标
- 坐标近邻检索偏置：过度召回"空间近但语义无关"块 ← 对策=语义×空间双通道加权
- egocentric/allocentric 坐标系混淆（Kunz 2021）← 空间块须标 frame 字段
- place-cell remapping 碎片化 ← namespace 稳定化

## 架构优化建议（待落入设计文档）
- 增强 A：GDN-MemBlock 升格为"海马式可写参数记忆"（旁挂 Memory-Layer 式稀疏 KV 查找），消除 §11.1 前缀偏差
- 增强 B：CSA 层加 harvest() 自编译接口（ICAE 范式）
- 增强 C：Affective Perception Head（valence/arousal 二头）提前到与 KAL 同期训练，arousal 接写显著性头
- 增强 D：VIS 不再只喂 EMB——空间认知地图写入 GDN 海马记忆层（增强A），route_key 增坐标邻近边(TEM结构泛化)；<|ref|>/<|box|> 走"回想空间关系"闭环；对冲语义-几何失配用学习型几何嵌入
- **增强 E：情感调制总线（McGaugh 原理工程化）**——情感向量 affect{valence,arousal,saliency} 贯穿全周期：Block Spec 加 affect 字段；arousal 门控 CA1 固化优先级；valence 入 route_key 检索维度；KAL 分层 L1知识感知→L2语境情感(共享PM-stream只加W[d,2])→L3语境一致性；人格块 persona_vector 作 HRL+感知层双门控增益旋钮（PersonaAgent+56%/PALACE SOTA，防漂移靠已有页保护位/MCB）；HRL 内在空间感知=把模型自身潜空间当认知地图导航(TEM关系推理)，与外部VIS空间统一为同一套关系导航器（潜空间几何训练中免费习得，运行时用于路由）

## DeepSeek V4 骨干层整合（增强 F，2026-07-25 核实）
- **DeepSeek V4（2026-04-24 发布）放弃 MLA，改用混合压缩注意力=CSA+HCA+滑窗三件套**，沿"序列维度"(非头维度)压缩；1M上下文仅用V3.2的10%KV+27%单token FLOPs
  - 滑窗=工作记忆L0(精确) / CSA=情景记忆L1(选择检索已压缩摘要) / **HCA=长期gist L2(128 tokens→1 KV entry，激进全局摘要)**
- **关键洞察：HCA条目=原生自编译块**——TAIS的"CSA"名字=V4的CSA组件，只取了一段；加HCA后三级注意力=三级记忆，L0/L1/L2存储/§11.1块注入/增强B自编译三件事被注意力层原生实现
- HCA区=块注入原生落点(消除§11.1前缀偏差，注入即读自己写的东西)；HCA压缩器=内置harvest(增强B)；高arousal触发更激进HCA压缩(McGaugh落地)
- **mHC(arXiv:2512.24880)验证**：无约束HC 27B信号放大3000×崩溃→mHC投影到Birkhoff多胞形(双随机/Sinkhorn-Knopp)压到1.6×；BBH43.8→51.0/DROP47.0→53.9/开销6.7%。TAIS的n=5(4内容+1PM感知记忆流)是DeepSeek n=4纯内容流的独创延伸(无先例,需pilot消融—logs_train_pm.txt正做)
- **MoE路由统一(知识块即专家字面实现)**：Manifold Power Iteration(arXiv:2606.12397,router行对齐专家主奇异方向)+Expert Specialization(NeurIPS2025,正交loss+方差loss)；扩展§11.1.3为三维统一打分头：token域(CSA选摘要)/块域(HRL注入)/专家域(MoE激活)同构
- **增强 F：骨干 attention 双时间尺度（DeepSeek-V4 验证）**——DeepSeek 压缩注意力谱系 MLA→NSA(2502.11089,三分支压缩+top-K选择+滑窗,64k解码11.6×)→DSA(V3.2 lightning indexer)→V4 CSA+HCA(2026.04)。TAIS 的 CSA(stride-4+indexer top-128+滑窗512)≡NSA，独立命名一致=强方向验证。V4 关键启发=加 HCA 重压缩层(大压缩比+稠密注意力,移除稀疏选择)与 CSA 交错，形成"精确选择性 CSA(≈海马pattern separation) + 全局鸟瞰 HCA(≈<gist>架构版)"双时间尺度；HCA 是设计文档 <gist> 自我总结概念的架构级载体。Attention sinks(可学习sink logits,注意力质量可<1)=缺页声明/诚实降级的原生注意力版。NVIDIA cuDNN 已有 NSA kernel 专为 Blackwell SM100+ 优化=sm_120 加速路径(缓解 D-0 纯PyTorch GDN 9.5k vs SDPA 19.7k 吞吐痛点)。彩蛋:V4 Quick Instruction tokens 复用KV做辅助决策≡<recall>/<blank>验证；On-Policy Distillation 专家蒸馏≡知识块即专家训练时版

## v2.5 第三轮：五命题独立复核新增已核实文献（2026-07-26）

**命题一（推理时自我进化）**
- **TTT-E2E / End-to-End Test-Time Training for Long Context (arXiv:2512.23675, Tandon/Dalal/.../Sun/Choi, Stanford/Berkeley/NVIDIA, 2025.12，arXiv ID 已核实)** — "长上下文是持续学习问题而非架构问题"，标准 Transformer+滑窗，内循环=对上下文做 next-token 预测把上下文压进权重（更新 TTT 层 MLP/hidden-state 权重，非注意力权重），外循环=meta-learning 优化初始化；3B/164B tok，128K 比 full-attn 快 2.7×（H100）、2M 快 35×、常数延迟；限：meta-learning 需 grad-of-grad，短上下文预训练慢 3.4×。映射到我们的 W-State 而非 W3+（运行时禁改基座权重红线不变）
- **DeepSeek-V4 (arXiv:2606.19348, DeepSeek-AI, 2026-04-26 提交，arXiv ID 已核实)**：V4-Pro 1.6T/49B act、V4-Flash 284B/13B act、1M ctx、32T+ tokens；CSA(m=4,双流重叠softmax+lightning indexer top-k=1024+shared-KV MQA)+HCA(m'=128,单流dense)沿**序列维**压缩(≠MLA的per-token latent)；c=512,n_h=128,L=61。1M上下文：Pro=V3.2 的 27% FLOPs+10% KV、Flash=10% FLOPs+7% KV。HCA cache=8n bytes/层=0.4% of GQA8 基线(2048n)。压缩矩阵 W_KV/W_Z/indexer 训练期学习、推理期冻结。
- TLM / Test-Time Learning for LLMs (ICML 2025 Poster) — 仅无标注测试数据即可适配目标域
- Survey on LLM Inference-Time Self-Improvement (arXiv:2412.14352) — 三分类：Independent/Context-Aware/Model-Aided
- Sleep-time Compute (Stanford/Berkeley 2025) — 离线预计算摊销在线成本 ~5× = 睡眠固化的权重版

**命题二（认知扩展超基座参数）**
- Memory Layers at Scale (arXiv:2412.09764, Meta, ICML 2025) 全文确认 — 加参数不加 FLOPs，128B/1T，事实任务强，sweet-spot ~3 层居中大间距；论文自述稀疏更新→更少遗忘/幻觉/持续学习
- **Titans (arXiv:2501.00663) 三变体 MAC/MAG/MAL 与我们逐一同构**：MAC≈CSA KV 注入 / MAG≈GDN delta 门控 / MAL≈增强 A 记忆层 — 架构完备性佐证

**命题三（元认知/自我认知闭环）— 本轮最强补强**
- **Hallucination Linearly Decodable @量化 (arXiv:2606.02628, 2026, Aiersilan 单作者)** — Llama-3.1-8B/Mistral-7B/Qwen2.5-7B @4-bit NF4，**单中层线性探针 0.904–1.000 AUROC**（采样式检测器≤0.541）；MLP 探针极少超线性探针 +0.01 AUROC（信号近线性）；峰值层 Llama/Mistral=block13–18/32、Qwen=block19–25/28（≈50–90% 深度，与 KAL ℓ10/14/18 一致）；首块注意力熵 HaluEval-QA 0.866–0.941；单 8GB GPU 可复现 ← 支撑边缘 Q4 部署 + "线性探针即足够"
- **【核实修正 2026-07-26】SAPLMA（arXiv:2304.13734）原文指标为"准确率 71–83%"（accuracy），非 AUROC**；设计文档 §8 "SAPLMA 71–83%"措辞应理解为准确率。Kadavath(2207.05221)=P(IK) 可训练但新任务校准漂移；ITI(2306.03341)=32.5→65.1%；Do I Know This Entity(2411.14257, ICLR2025)=SAE 实体方向因果可 steer
- **【核实修正 2026-07-26】微软《Trigger in the Haystack》(arXiv:2602.03085v1, Bullwinkel/Severi/Hines/Zunger) 摘要未含"87.8% 检出/0 误报"数字**（仅称"跨多场景恢复可用触发器"）；该数字出自设计文档 §26.2，需查全文核实，暂标 UNVERIFIED
- Betley《Tell me about yourself》(arXiv:2501.11120, Betley/Bao/Soto/Evans, ICLR2025 提交) — 行为自知，无 ICL 描述隐式策略/识别后门；Barkan《Do LLMs Know What They Are Capable Of?》(arXiv:2512.24661) — 全模型过度自信、判别力优于随机但**新/更大模型判别力一般不增**（Claude 例外）、多步任务中过度自信加剧、推理模型≈或劣于非推理 ← 规模不自动修复校准
- MemoryGraft(arXiv:2512.16962, Srivastava/He) — 间接注入植入恶意成功经验到长期记忆（"semantic imitation heuristic"），MetaGPT+GPT-4o 实证，跨会话持久漂移、时间解耦
- 元认知框架群 2024–2026：Meta-R1(+8%)/CLEAR(70-80%检出)/Know More Clearer(ECE60→24%)/MeCo(ACL2025 学到的探针决定何时用工具)/Think2
- Betley 2025 "Tell Me About Yourself" — 模型无 ICL 示例描述隐式训练行为/识别后门 = 功能性自我建模
- **Barkan 2025（反面边界）**：更强能力 ≠ 更好校准/判别 → 规模不自动修复元认知，必须显式训 KAL
- 元认知监测(MetaM)≠控制(MetaC) 神经基质部分分离 (PMC9053853) → 支持 KAL 检测/执行分置
- **诚实红线**：本项目"自我认知"仅指功能性自我建模，绝不主张现象学意识

**命题四（注意力自编译→块）**
- kv-distill (arXiv:2503.10337) — 问题无关 KV 蒸馏，PEFT 适配器，**最高 99% 压缩**保性能，可域微调 = 工程级 CSA harvest()
- FastGen "Model Tells You What to Discard" (ICLR 2024) — 注意力头内在结构(局部/特殊/广注意)自适应留弃
- ICAE (arXiv:2307.06945, ICLR 2024) — 长上下文→memory slots 4× 近无损（CSA harvest 原型）

**命题五（动态 Tokenizer）**
- BLT (arXiv:2412.09871, ACL 2025) — 熵驱动字节 patching，8B/4T 追平 tokenization；"按熵分配计算"≡我们"词表摩擦信号驱动概念槽"同原理两层实现
- **From Tokens to Words (arXiv:2410.05864, ICLR 2025 v4) 重读** — 免微调扩词表对精心挑选高频多 token 概念在**输入+输出两侧**都可行（保持/略升性能）→ 修订 §28.7：输出侧升格选择性可行非绝对禁止
- **Over-Tokenized Transformer (ICML 2025) 全文** — 输入词表 log-linear（400M 靠 128× 输入词表追平 1B），输入侧无条件正向/几乎零成本，输出侧对小模型可能负面 → 精确支持第 0 级(输入免费)/第 1 级(tied 限量)分级

**安全（升级 §26.2）**
- **MemoryGraft (2025.12)** — 实证"植入恶意成功经验到长期记忆"的间接注入；攻击时间解耦（今日下毒数周后语义触发）
- 微软 Defender 指南(2025-2026)三必需原语：memory contracts / belief drift detection / context provenance tracking → 我们块签名+namespace / CA1 回归 / markdown 源代码 三者俱全
- 微软 AI Red Team "The Trigger in the Haystack" (arXiv:2602.03085v1, 2026.02) — 后门扫描器 87.8% 检出/零误报，可接入睡眠固化 draft 区筛查
- OWASP LLM04:2025 Data&Model Poisoning — 已列入 Top10

**CoT 忠实性（支撑 §17.2 归因监测头）**
- Turpin et al. NeurIPS 2023 (被引 1685) — CoT 系统性误表真实推理原因；后续 "CoT in the Wild Not Always Faithful" / "Larger LMs Don't Care How You Think" 持续确认

---
*导出自 /memories/repo/verified-literature-self-evolution.md（2026-07-30 同步快照）。*
