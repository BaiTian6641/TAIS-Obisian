# TAIS Obsidian：记忆层/注意力自编译簇已核实文献（2026-07-26 核实，article_ref/03 配套）

> 全部 10 篇 arXiv 编号已联网核实存在性。详见 `article_ref/03_memory_self_compilation.md`。

## 关键修正（务必采纳）
- **ICV 正确 arXiv 编号 = 2311.06668**（Sheng Liu, Haotian Ye, Lei Xing, James Zou, Stanford），标题《In-context Vectors: Making In Context Learning More Effective and Controllable Through **Latent Space Steering**》。任务简报与旧记忆里的 2310.10678 是错的（那是 Dirac 场物理论文）。v3=2024-02。
- **DeCoVec arXiv:2604.11129 确实存在**（2026-04-13 提交，ACL 2026 Findings，Feiyang Li & Yile Wang）。解码空间 Δz = logits(few-shot) − logits(zero-shot)，加到解码 logits；training-free、非侵入、无额外输入 token、+5.50 acc。
- **Dherin 2507.16003 正式标题** =《Learning without training: The implicit dynamics of in-context learning》（非"ICL=低秩更新"）；context fwd ≡ no-context fwd + 对 MLP 的最小低秩(rank-1)权重更新；v4=2026-06-02。

## ICV 提取配方（精确，全文已读）
- 两阶段：Task summary（造 ICV）+ Feature shifting（加 ICV）。
- 把每个 demo 的 x、y 分别喂 LLM，取**每层最后 token** 的 latent state h∈R^d，跨 L 层拼成 R^(L×d)。
- 配对 demo：ICV = {h(y_i)−h(x_i)} 的**第一主方向（PCA top-1）**（Lemma 1：= 差向量样本协方差的最大特征向量）。
- 非配对：对比损失梯度 h_ICV=Σ_y((1−p_y)h(y) − Σ_x p_x h(x))。
- 施加：h̃_{t,l}=h_{t,l}+λ·h_ICV^l，**加到所有层所有 token**（单层消融基本无效），再 ℓ2 归一化保模长。
- 评测**仅** safety/style/role/format（全是行为/变换任务），**从不主张事实回忆**。

## Memory Layers（2412.09764）机制（精确）
- 用 memory 层替换 FFN：q→product-key top-k 稀疏查找→softmax→value 加权和。keys/values 是**可训练参数**（非激活），这是与注意力的根本区别。
- I=SelectTopk(Kq), s=Softmax(K_I q), y=s·V_I。product key=两个半键集之积（全集从不实例化）。
- Memory+：输入门控 silu: output=(y⊙silu(x^T W1))^T W2；qk-norm 稳定。
- sweet-spot **≈3 层居中大间距**（134m stride4，其余 stride8；表3：[12,16,20] 最佳；加更多层会因挤掉 dense 参数而退化）。memory 池跨层共享（参数量不随层数增）。
- 规模：≤128B memory 参数(64M values)、1T tokens、≤8B 基座；1.3B+64M≈Llama2-7B(10×FLOPs)。事实任务增益>100%。作者自述稀疏更新→更少遗忘/幻觉/持续学习。
- **关键：keys/values 在预训练中学习，非运行时单 forward 提取** → 是"训练期记忆块"载体，不是运行时自编译目标；但可作 W3+ 离线写入的基底。

## 已核实综合（核心结论，可直接入设计文档）
**自编译机制三配方 × 载体能力边界（已逐篇核实）：**
1. **KV 载体（CSA harvest）**：ICAE/kv-distill/FastGen/Expected-Attention 都是"forward→收割/压缩 KV→存前缀"。保留 token 索引→**能做事实回忆**（kv-distill 保抽取式 QA 99% 压缩、ICAE 4× 近无损）。
2. **向量载体（steering/task vector）**：ICV/FV/DeCoVec 都是"forward→提一个常向量→加到 residual/logits"。**位置不变（同一偏移加到每 token）→只能 steer 行为/风格/函数，不能做事实回忆**。
3. **记忆层条目（product-key KV）**：训练期学习，有可寻址 key（top-k 稀疏检索）→能存事实关联（生日/首都）。

**设计文档"向量不能做事实回忆"命题 = 已核实且被强化**：ICV 评测全部是变换任务；FV 自述"编码函数输出空间，不足以重建 FV"、复现的是 ICL 函数（反义/翻译）非查找表；事实回忆需 token 寻址检索，常向量偏移做不到。边界精确化：**token 寻址载体（KV 前缀/记忆层条目）能事实回忆；位置不变向量（ICV/FV/DeCoVec）不能，只 steer 行为。**
理论支撑：Dherin（ICL 效应≡低秩权重扰动→投影到激活空间方向，解释向量为何能捕获 forward 经验）+ Hopfield（注意力≡现代 Hopfield 一次检索、容量随维数指数增长，为 KV 前缀即被查询记忆奠基）。

---
*导出自 /memories/repo/verified-literature-memory-compilation.md（2026-07-30 同步快照）。*
