# 内化-检索-注入端到端 pilot（2026-07-29）

产出：`scripts/internalization_e2e.py`（361 行）+ `tests/test_internalization_e2e.py`（225 行，10 项全绿）。子代理实现，**我亲自验收**（report 数据确认 samples 通而未用+独立重跑+全量 306 绿）。知识内化"实时可用"承诺的端到端验证——**结论诚实：0.1B 通路全通但召回头未训，未兑现答对（训练缺口非代码缺口）**。

## 验收确认的关键诊断
- KV 注入后输出与基线逐字一致（samples 对比）= 通而未用（门控对 HCA≈0.016）；in-context 答对 0.700=知识本会答；余弦检索 100%=表征可用。**两个训练缺口：HRL indexer 未训（检索随机）+ HCA 召回头未训（注入不用）**。
- **"实时可用"不是免费的**：teaching"有 K 答对 1.0"=K 文本在 token 上下文（滑窗召回+prompt 法续答），非运行时 KV 注入召回。需训练 indexer 检索+HCA 召回头才能兑现（对齐知识内化文档三训练目标之"检索监督"，此前未做）。

## 端到端四阶段
①内化（K prefill 收割各 CSA 层 KV→BlockPayload(kv) 写 BlockStore，**运行时零梯度不动权重**，测试断言权重逐位不变）→②检索（HRL route_candidates/LightningIndexer 打分）→③注入（kv 经 injector namespace 校验 fail-closed→inject_hca_entries 前置拼 HCA 区）→④评估（三条件对照）。

## ⭐ 真实评估数据（n=20，teaching 对齐分布，bf16 autocast）
| 条件 | 答对率 | 含义 |
|---|---|---|
| 不注入基线 | 0.000 | 凭先验答不出虚构事实 ✓ |
| KV 注入(token寻址) | 0.000 | **通路通但召回头未训**：条目进 HCA 区(n_hca_inj>0)，门控对 HCA 分支权重≈0.016 → 通而未用 |
| 向量注入(steering) | 0.000 | 只 steer 行为（α=1 即改变生成）不事实召回，红线验证 ✓ |
| **in-context 上界** | **0.700** | K 作 token 上下文（滑窗读 raw token）能答对=**知识本会答** |
| 检索 HRL indexer | 0.000（随机） | **indexer 未训** |
| 检索 embedding 余弦基线 | **1.000** | **表征完全可分**——缺口纯在 indexer 未训 |

## 关键诊断（诚实缺口，禁止臆造）
1. **teaching"有K答对1.0"= K 文本在 token 上下文（滑窗精确召回）+ prompt 法续答**；运行时 KV 注入走 HCA 分支（门控≈0）→不等价。0.1B 未训"经 HCA 注入块做事实召回"。
2. **检索表征可分(余弦100%)但 HRL indexer 随机**——teaching 只训了内化行为，未训 indexer（知识内化文档三训练目标之"检索监督"未做）。
3. **生成法敏感**：必须用 prompt 法（prefill 后从末 logits 直接续答，对齐 teaching）；cache 法（塞 eot 启动）退化成散文。
4. **数值精度敏感**：必须 bf16 autocast（对齐 teaching）；fp32 前向退化。
5. **数据分布敏感**：用 teaching 对齐分布（复合虚构实体+What does X run on 句式）；随意 prompt（How many moons）不触发 SFT 行为。

## 踩坑（已修复）
- inject_hca_entries 需 [B,n_kv,N,hd]（dim1=n_kv），state cache 是 [B,T,n_kv,hd]→须 transpose(1,2)。
- BlockStore L0 容量仅 8，20 块写 L1(64)。
- run_kernel=True 时 forward 返三元组（kernel_signals），解包需 r[0],r[1]。
- 内核权重不随 checkpoint（kernel_enabled:false），用 attach_kernel() 现挂（indexer 随机初始化）。

## 待接（兑现"实时可用"需训练，非代码缺口）
①训练 HRL indexer 块检索（检索监督，对齐 embedding 余弦上界）；②训练 HCA 注入块召回头（让门控对注入条目开权重，E+ 块召回训练目标，对齐 in-context 上界）；③二者达标后 KV 注入答对率应→in-context 上界（实时可用兑现）。

## ✅ 两缺口已训练兑现（2026-07-29，scripts/train_retrieval_recall.py 484 行 + tests/test_retrieval_recall.py 118 行 4 项全绿）
- **缺口① indexer 块检索**：余弦蒸馏（MSE 回归 query×block 余弦相似度矩阵，query 均值池化——实体语义在均值，末 token "?" 语义弱曾致坍缩）。命中率 **0.062→1.000**（对齐余弦基线）。
- **缺口② HCA 召回头**：注入事实块 KV→HCA 区，prompt 法逐 token 前向（logits 可微、cache 携带注入），只训各 A 层 gate_w/gate_b（585 参数），答案段 CE。答对率 **0→0.188**。
- **闭环**：检索 1.000 ∧ 召回 0.188 → 双达成 **0.188**（实时可用）。主干权重逐位不变（frozen 红线✅）。
- **踩坑**：①LightningIndexer 的 ReLU 门控在 InfoNCE/KL 下坍缩到常数（infonce 陷 ln16=2.7726）→ 必须用 MSE 余弦蒸馏；②init_indexer_from_model 的 q 方向聚合对块域引入偏置→随机初始化更稳；③污染检查须排除内核+门控（方案 B 边界），否则误报；④val loss 漂移 0.43 是门控语义变化（训练目标）非污染。
- **诚实边界**：召回 0.188 << in-context 上界 0.70——门控仅 585 参数是容量瓶颈（pilot 级，判"显著改善"非"完美"）。

---
*导出自 /memories/repo/internalization-e2e.md（2026-07-30 同步快照）。*
