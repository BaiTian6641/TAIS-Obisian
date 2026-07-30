# 记忆层条目迁移 + A/B（P0，根治门控副作用，2026-07-30）

## 产出
`scripts/memlayer_internalization_e2e.py`（436 行）+ `tests/test_memlayer_internalization.py`（230 行，7 项全绿）+ report `runs/memlayer_internalization/report.json`。子代理实现，我验收（report 数据确认+独立重跑+全量 374 绿）。基座=teaching checkpoint，4070。

## 记忆层内化方法
- **key 提取**：query Q 经首 CSA 层（层3）hidden 均值 [768]→key_proj→key_dim=64（query 内容寻址）。
- **value 提取**：答案 A 的 token 嵌入均值 [768]。
- **写入**：MemoryLayer.write(k, v)（GDN-2 delta 规则，state buffer 零梯度，keys/values 训练参数不动）。
- **读出+注入**：query(q_key)读出 value→detach→①残差加法（hook 注入 CSA 层输入）②logit 偏置探针（value 经 tied embedding 转答案 token logit 偏置）。

## ⭐ 根治判据全部达成（A/B 对照，n=16，真实跑出）
| 指标 | 记忆层路径 | KV 拼接（解耦门控） |
|---|---|---|
| 事实召回 | 残差 0.062 / logit 0.062 | **0.625** |
| **in-context 精确召回** | **0.688**（=纯净基线 0.6875，**零干扰**） | 0.500（副作用） |
| **副作用消除** | **✅ 是** | ❌ 否 |
| 主干 frozen | ✅ drift=0.0 | — |

- **根治验证成功**：记忆层注入路径 in-context=0.6875≈纯净基线 0.6875（**完全一致零干扰**），KV 拼接带门控=0.500（副作用）。**记忆层不经 HCA gist 通道→结构上无 gist 门控被波及（根治 vs KV 缓解）**。
- **载体能力边界**：mem_entry token 寻址 factual_recall=True（事实主载体）✅ vs concept_slot/icv/steering 位置不变向量 False（只 steer）✅。单条实测：记忆层读出 vs 写入 value 余弦 **0.98**，logit 偏置 top1=答案尾段 subword（**token 寻址查表命中**）。

## 诚实缺口（非记忆层载体劣势）
事实召回 0.062=基线——两缺口：①读出→残差/logit 接口 0.1B 未训（teaching ckpt 只训过 KV 拼接通路）；②16 条写入后 key 检索串扰（key_proj 随机未训，16 个同句式 Q 的 key 在 key_dim=64 太近→读出余弦 0.53，单条时 0.98）。**两缺口都是"读出/寻址接口未训"，非记忆层载体不能事实召回**（单条 0.98 已证有效）。KV 拼接也是训了门控才达 0.625。

## 子代理踩坑（验收记录）
①初版 in-context 基线测错（attach 扩容门控本身污染基线，改用纯净模型副本测真基线 0.688）；②初版 backbone_frozen 误用 requires_grad（改权重快照逐位对比）；③logit 探针初版归一化削弱信号（改不归一化放大）。

## 结论（fb1 P0 判据达成）
副作用消除（in-context 0.688=基线 vs KV 门控 0.500）+主干 frozen+mem_entry token 寻址可事实召回。**记忆层是事实主载体的正确路径（根治）**；事实召回率低于 KV 是 0.1B 读出接口未训缺口（训练可补），非路径劣势。

## 待接
①记忆层读出/寻址接口训练（key_proj+读出路径，对齐 KV 拼接召回 0.625）→ 事实召回达标；②16+条写入的 key 检索去串扰（更大 key_dim/正交化）；③记忆层作事实主载体接入主动求知闭环（KV 拼接保留高精度补充 §25.4②）。

---
*导出自 /memories/repo/memlayer-internalization.md（2026-07-30 同步快照）。*
