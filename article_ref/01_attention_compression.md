# 01 · 注意力 / 压缩 / 测试时学习 簇

> 簇主题：长上下文注意力的"压缩-选择-精算"范式 + 测试时把上下文压进权重的学习。
> 所有 arXiv ID 均于 2026-07-26 联网核实存在性。DeepSeek-V4 为技术报告（非同行评审）。

---

### DeepSeek-V4（CSA + HCA 混合压缩注意力）— arXiv:2606.19348, 2026-04
- **核心主张**: V4-Pro 1.6T 参数（49B 激活）/ V4-Flash 284B（13B 激活），1M 上下文，32T+ tokens 预训练。放弃 V3 的 MLA，改用 **CSA + HCA + 滑窗** 沿**序列维**压缩。1M 上下文下 V4-Pro 仅用 V3.2 的 **27% 单 token FLOPs + 10% KV cache**；V4-Flash = 10% FLOPs + 7% KV。
- **机制细节**:
  - **CSA（Compressed Sparse Attention）**: m=4（4 token→1 压缩条目）；双 KV 流 + 重叠 2m-softmax（模糊块边界）；lightning indexer 用 **ReLU 打分（FP4）** 选 top-k=1024；shared-KV MQA。
  - **HCA（Heavily Compressed Attention）**: m′=128（128 token→1 条目）；单流、无重叠、**无 indexer、对全部压缩条目稠密注意**。cache = 8n bytes/层 = **0.4% of GQA8 基线**。
  - 三者交错：滑窗=近期精确 / CSA=选择性检索已压缩摘要 / HCA=全局 gist。压缩矩阵**训练期学习、推理期冻结**。
- **与 TAIS 的对应**: ⭐⭐⭐【关键】TAIS 的 CSA 命名直接取自此；HCA=L2 长期 gist 记忆、是设计文档 `<gist>` 的架构级载体、块注入原生落点。三级注意力=三级记忆 L0/L1/L2。
- **核实状态**: ✅已核实（arXiv 摘要 + Raschka gallery + 逐步推导二次确认压缩比/FLOPs）；⚠️技术报告非同行评审。

### DeepSeek 压缩注意力谱系（MLA → NSA → DSA → V4）
- **NSA** arXiv:2502.11089 (Yuan et al., 2025-02): 动态层级稀疏=粗压缩 + 细选择 + 滑窗；natively trainable；64k 序列显著加速（**11.6× 解码加速为 repo 记忆，未本会话二次核实具体倍数**）。
- **DSA (V3.2)**: lightning indexer，低秩压缩 query + 多头 indexer 权重 + ReLU 打分选 top-k；注意力 **FP4** 精度。V4 CSA 建立其上。
- **重要性**: ⭐⭐⭐【机会】NSA 已有 NVIDIA cuDNN kernel（sm_120 加速路径，缓解 D-0 纯 PyTorch GDN 9.5k vs SDPA 19.7k 吞吐痛点）。

### Titans: Learning to Memorize at Test Time — arXiv:2501.00663, Google, 2024-12
- **核心主张**: 神经长期记忆模块作 meta in-context learner，测试时用**梯度下降**更新自身权重学习记忆/遗忘。surprise = 对 associative loss 的梯度；$S_t=\eta_t S_{t-1}-\theta_t \nabla\ell(\mathcal{M}_{t-1};x_t)$（动量=过去 surprise）；forget gate $(1-\alpha_t)$=weight decay；深 MLP 记忆 $L_\mathcal{M}\ge2$。缩放到 **>2M 上下文**，NIAH/BABILong 超基线（甚至超 GPT-4）。定理：Titans 超越 TC0。
- **机制细节**: $\ell(\mathcal{M};x_t)=\|\mathcal{M}(k_t)-v_t\|_2^2$（associative memory）；**三变体** MAC（memory-as-context，检索后拼入 attention 上下文）/ MAG（memory-as-gate，与滑窗门控）/ MAL（memory-as-layer，压在 attention 前）。
- **关键（对干扰分析）**: 测试时**仅神经记忆（MLP）权重更新**；persistent memory 固定；attention 权重固定。消融 +Attn(MAC/MAG/MAL) 优于纯 LMM——"固定 attention + 测试时记忆"**协同非干扰**。
- **重要性**: ⭐⭐⭐【关键】MAC≈CSA KV 注入 / MAG≈GDN delta 门控 / MAL≈增强 A 记忆层，与 TAIS 三条增强逐一同构。

### TTT-E2E / End-to-End Test-Time Training for Long Context — arXiv:2512.23675, Stanford/Berkeley/NVIDIA, 2025-12
- **核心主张**: 把长上下文建模重新表述为**持续学习问题**。仅用**标准 Transformer + 滑窗 attention**，测试时对上下文做 next-token 预测继续学习、把上下文压进权重；训练时 meta-learning 优化"为测试时学习"的初始化。3B/164B tokens：随上下文长度 loss 缩放与 full-attention 相同，128K 快 **2.7×（H100）**、2M 快 **35×**，**常数推理延迟**。
- **机制细节**: 内循环直接优化**网络末端** next-token loss；外循环 meta-learning 优化 TTT 后最终 loss。更新的是 **TTT 层（hidden-state=线性/MLP）权重**，非注意力权重。局限：meta-learning 需梯度之梯度，FlashAttention 不支持，短上下文(8K)预训练慢 3.4×。
- **重要性**: ⭐⭐⭐【关键】映射到 TAIS **W-State**（运行时学习）而非 W3+；"测试时把上下文压进权重"=块固化的逐序列微缩版（不持久化避遗忘）。

### TTT（原版）— arXiv:2407.04620, 2024-07
- **核心主张**: 序列建模层的 hidden state 本身就是一个 ML 模型，更新规则=自监督学习的一步。TTT-Linear / TTT-MLP。125M–1.3B 上随上下文增多持续降 ppl，Mamba 在 16k 后不再改善。
- **重要性**: ⭐⭐【背景】TTT-E2E 母方法；TAIS GDN 海马记忆层 + W-State 范式的理论母体。

### Gated DeltaNet — arXiv:2412.06464, 2024-12
- TAIS GDN 层的直接母体（Mamba2 + delta rule + forget gate）。Titans §App.C 证明其 LMM 是 Gated DeltaNet 的推广（加 momentum + 深记忆 + 非线性递归 + forget gate）。

### Gated DeltaNet-2 — arXiv:2605.22791, NVIDIA（Hatamizadeh, Choi, Kautz），2026-05 ⭐⭐⭐【机会】✅
- **核心主张**：解耦 **erase gate `b_t∈[0,1]^{d_k}`（key 侧，移除衰减状态哪些坐标）** 与 **write gate `w_t∈[0,1]^{d_v}`（value 侧，承诺哪些新值坐标）**，去除原版 GDN/KDA 单一标量 βₜ 的 tied 限制。更新式 `S_t=(I−k_t(b_t⊙k_t)ᵀ)·D_t·S_{t-1}+k_t(w_t⊙v_t)ᵀ`（D_t=Diag(α_t) 通道级衰减）；β 合并为标量退化为 KDA，衰减也合并退化为 Gated DeltaNet。chunkwise WY 形式保持并行训练（gate-aware backward）。
- **实证**：matched 1.3B / 100B FineWeb-Edu，超越 Mamba-2 / Gated DeltaNet / KDA / Mamba-3（recurrent 与 hybrid 双设定）；**RULER 长上下文检索大幅领先**（S-NIAH-3@2K 63.2→89.8，MK-NIAH-1@4K 28.0→37.8 over KDA），吞吐仅小常量开销。
- **与 TAIS 的对应**：① 精确对应读写不对称红线——erase/write 解耦 = W2 记忆层 delta 写与门控衰减遗忘的细粒度分离；② RULER/NIAH 检索增益正对 §25.2"GDN 固定状态检索密集遗忘（2510.20787）→CSA 补偿"，是 CSA 补偿外的另一条补强路径（列 T1/T2 消融 GDN vs GDN-2）；③ 增强A memlayer 已采纳：write() 支持 erase_gate/write_gate 解耦（默认 tied 向后兼容，见 model/memlayer.py）。
- **核实状态**: ✅已核实（arXiv HTML/PDF + MarkTechPost 二次确认）。

### DeepSeek lightning indexer（V3.2 DSA 原型）— DeepSeek V3.2 技术报告 ⭐⭐⭐【关键】✅
- **核心主张**：`I_{t,s}=Σ_{j=1}^{H_I} w^I_{t,j}·ReLU(q^I_{t,j}·k^I_s)`（Eq.1）——**独立的低维 indexer**：自有 q^I（query 侧）、k^I（key 侧）投影，**非复用主干注意力**；indexer 头数少、维度低、ReLU（吞吐）、可 FP8（V4 降 FP4）。复杂度 O(L²)→O(L·k)。
- **warmup**：先冻结主干，用 **KL 散度对齐** indexer 分布到稠密主注意力分布（~1000 步/2.1B tokens 短校准），再开 top-k 稀疏训练。V4 CSA 在**已压缩条目**上打分（先 stride-4 压缩再选）。
- **与 TAIS 的对应**：HRL 检索的**真正独立打分器**范式——区别于复用 q_proj 近似；我们的 `model/hrl_indexer.py` LightningIndexer 照此实现（多头 q^I/w^I/k^I + ReLU + top-k + KL warmup）。
- **核实状态**: ✅已核实（DeepSeek V3.2 技术报告 Eq.1 + SGLang/分析博客二次确认）。

### PEER（Parameter Efficient Expert Retrieval）— arXiv:2407.04153, DeepMind（He）, 2024-07 ⭐⭐⭐【关键】✅
- **核心主张**：product key 检索（分半键集 K1/K2∈R^{√N×d/2}，笛卡尔积，全集不实例化，O(√N) 而非 O(N)）+ **内生独立 query network** 路由到百万级小专家。top-k 离散无梯度，但**分数可微**（梯度流经分数到 query/keys）。
- **与 TAIS 的对应**：证明"内容寻址检索器该独立、可训练、内生"；PEER 的 product-key 与我们增强A memlayer 的 product-key 同源；HRL Indexer 的内生性佐证（用户选方案 B）。
- **核实状态**: ✅已核实（arXiv abstract + HuggingFace paper page）。

---

## 交叉干扰分析：CSA/HCA ↔ TTT-E2E

**核心问题**：V4 的 CSA/HCA（学习型压缩矩阵、128:1 重压缩）与 TTT-E2E（逐序列 MLP 权重更新）组合，会相互干扰吗？

### ✅ 已核实事实
1. **TTT/Titans 测试时只更新 MLP/hidden-state 权重，不更新注意力/压缩权重。** Titans 全文明确：persistent memory 固定、attention 固定（纯 ICL）、仅神经记忆 MLP 测试时更新。TTT-E2E "把上下文压进权重"指 TTT 层 hidden-state 权重。
2. **CSA/HCA 压缩矩阵训练期学习、推理期冻结**，作用对象是残差流/KV。
3. **Titans（组合测试时记忆+attention）未报告干扰**，消融 +Attn 反而优。其规避机制：神经记忆是**独立分支**——输出检索后作 KV/上下文拼入（MAC）或门控（MAG），**不回流改 attention 所压对象**；且 attention 是 full/SWA，**非学习型序列压缩**。
4. **TTT-E2E 刻意选用"标准 Transformer + 滑窗 attention"**（无学习型压缩）——这是作者规避冲突的**强间接证据**：作者认为学习型压缩与逐序列权重更新共存有风险，故退回 SWA。
5. TTT-E2E 内循环"直接优化网络末端 next-token loss"——TTT 影响传播到下游所有层；若下游有冻结学习型压缩器，其输入分布会被 TTT 改变。

### ⚠️ 假设（无任何论文直接测试 HCA+TTT，需 pilot 消融）
- **风险点**：冻结学习型 HCA（128:1 softmax pooling）若位于 TTT 层下游，TTT 逐序列残差漂移会给 HCA 冻结 pooling 权重喂 OOD 输入 → 压缩质量下降。同理 CSA indexer 选块也会偏。
- **信息冲刷**：HCA 128:1 池化极激进；若 TTT 在 HCA 之前把细节写入残差，HCA 仍会池化掉（除非信号落在 HCA softmax 保留方向）。
- **安全组合模式**（风险低→高）：
  1. **"记忆即独立分支"**（Titans MAC/MAG）：TTT/记忆输出作 KV/上下文注入，**不改动 HCA 所压残差** → 干扰最小。✅ 有 Titans 消融间接支持。
  2. **"先压缩后学习"**（CSA/HCA 在前、TTT 读已压缩表示）：TTT 消费压缩器输出而非喂给它 → 次安全。
  3. **"先学习后压缩"**（TTT 在 HCA 之前改残差再被冻结 HCA 压缩）：**最高风险**，需 RMSNorm 兜底 + pilot。
  - 缓解：压缩前 RMSNorm 限制漂移幅度；或将 HCA 压缩统计量纳入测试时轻量校准（昂贵，TTT-E2E 已回避）。

### 对 TAIS 的工程结论
运行时学习（W-State、≤W2）应**从 CSA/HCA 的输出读取、或以独立分支注入 KV**，**而非改动任何冻结学习型压缩器下游所依赖的残差**。既守住"冻结基座权重"红线，又避免 HCA 128:1 把 TTT 信号池化掉——与 Titans MAC 非干扰证据一致。"已核实"部分可直接落入设计；"假设"部分需 D-0 pilot 加一组「TTT/GDN-mem 位置 ∈ {HCA 前 / HCA 后 / 并行 KV 分支}」消融。
