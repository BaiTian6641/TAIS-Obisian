# 主动求知闭环全链端到端 demo（三阶段串联，2026-07-29）

## 产出
`scripts/active_inquiry_full_chain_demo.py`（435 行）+ `tests/test_active_inquiry_full_chain.py`（328 行，11 项全绿）。纯新增未改模块。复用 inquiry_branch/inquiry_executor/inquiry_consolidation/blockstore + internalization_e2e 原语（importlib 按路径加载，scripts/ 非包）。**已亲自验收**：report 数据确认（A 组真实 KAL certainty=0.000→Decline 诚实降级 6/6，B 组写入 6/6）+独立重跑+全量 332 绿。

## 验收确认的关键诚实发现
①**真实 KAL 行为正确**：对完全虚构事实，真实 KAL certainty=0.000 判完全空白区→Decline 诚实降级（非硬答）——主动求知闭环的诚实降级在真实 KAL 上工作。②**实时可用强度依赖 checkpoint 训练状态**：已训 1.000/0.188 的 indexer/门控在 teaching checkpoint（train_retrieval_recall 训的），kaltruth 未训——本 demo 用 kaltruth 故阶段2 仅通路验证（0.167/0.000）。③**占位 embed 语义缺失会误判**（字符 hash 同句式 K 误判冲突→写入 0/6，注入 model_embed 后 6/6）——正式须用真实模型 hidden。

## 三阶段串联（单机跑通，kaltruth checkpoint）
- **阶段1 运行时学习** 分两组：
  - A 组真实虚构事实→真实 KAL certainty≈0（完全空白区）→ **Decline 诚实降级 6/6**（红线成立：学习成本过高拒答"该部分记忆暂不可用"，绝不硬答）。
  - B 组可学习区演示（certainty=0.55 占位，RPL/LP）→ CallTool/AskQuestion→CrossVerifier→写入 draft 6/6。**注：真实 KAL 对完全虚构事实全判空白区，无法自然落可学习区；B 组 certainty 是构造的演示占位，注释已标注**。
- **阶段2 实时可用**：draft→KV 收割→HRL route_candidates 检索命中 0.167（1/6，随机 indexer）→HCA 注入答对 0.000=基线 0.000。**通路通非已训强度**（已训 1.000/0.188 的 indexer/门控权重未存 checkpoint）。
- **阶段3 长期固化**：PROMOTE 3 / QUARANTINE 1（冲突块）/ REJECT 3（部分一致块 consensus<0.7）。三元奖励 correct+1/hallucinate−1/abstain+0.15。

## 踩坑（重要）
1. **scripts/ 非 Python 包**：`from scripts.internalization_e2e import` 报 No module named 'scripts' → 用 importlib.util.spec_from_file_location 按路径加载。
2. **CrossVerifier 字符 hash embed 无语义**：默认 `_embed` 用 `torch.manual_seed(abs(hash(text)))` 随机投影，同句式 K（仅实体/燃料不同）被判 conflict（余弦<0.3）→ 后续证据 verified=False 写入 0/6。**修：注入 model_embed（真实模型首 CSA 层均值 hidden）使同句式 K 表征相近→一致→累积写入 6/6**。InquirySleepConsolidation 同样需 embed_fn=model_embed。
3. **CA1 门 min_usage=10**：usage_count=2<10 → 一致块全 REJECT。**修：usage_count=12**（HRL 命中计数累积）。
4. **冲突块 verified=False 则 write 返回 None**（红线未验证不写入）：用 model_embed 后矛盾 K（WATER）与已存 K 同句式表征相近→verified=True，强制 conflict=True 写入→QUARANTINE 保留双方。
5. **teacher_consensus=consistency×(0.5+0.5×credibility)**，模型 embed 下各 K 相似度有差异，部分一致块 consensus<0.7→REJECT（真实合理，非全一致）。

## kernel 加载坑（复验）
kaltruth config.kernel_enabled=False 但存 kernel.* → from_pretrained(strict=True) 报 Unexpected key。**解：attach_kernel()+load_state_dict(strict=True)**（load_model_with_kernel）。test_kernel_load_trap_without_attach 反向验证坑存在（pytest.raises RuntimeError "Unexpected key"）。

## 待接
①已训 indexer/门控权重存 checkpoint（兑现阶段2 已训强度 1.000/0.188）；②可学习区由真实 KAL 对"半熟"问题读出（非占位）；③embed_fn 正式接 capture_layers。

---
*导出自 /memories/repo/active-inquiry-full-chain.md（2026-07-30 同步快照）。*
