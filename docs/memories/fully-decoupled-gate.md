# 彻底解耦门控（FullyDecoupledGate，2026-07-30）——注入召回走独立 csa 通道的诚实负结果

## 产出
`src/tais_obsidian/model/tri_attention_fully_decoupled.py`（351 行）+ `scripts/train_fully_decoupled.py`（360 行）+ `tests/test_fully_decoupled.py`（257 行，9 项全绿）+ report `runs/fully_decoupled/report.json`。原 tri_attention.py/tri_attention_decoupled.py/tri_attention_gated.py **零改动**（注入式 attach/detach，红线合规）。**已亲自验收**：report 数据确认（负结果 both_targets_met=false 如实标注）+独立重跑+全量 412 绿。

## 我抓到的 5 个 NIAH 测试回归（子代理漏报，亲自验收价值）
子代理批量执行时 3 次修脚本（截断/facts_end 重定位/RoPE 越界优雅降级），测试预期未同步→5 个 test_niah_length_scan 失败。逐个诊断修复：
① **cache pos 预期错**（2 处）：`assert cache["pos"]==query_prefix_len` 在 qpl 非 chunk 对齐时失败（末段 chunk 越入查询 ids）→ seg 截到 qpl。
② **val_ids=None 填充缺陷**：脚本 val_ids=None 时 filler 仅合成句 1/5 且无补足→n_tokens 远小于 target→加合成句补足路径（n_sents 保守 ~10 token/句+截到精确 need）。
③ **off-by-one**（2 处）：`_MockModel` prefill 分支"位置 t 预测 forced[t]"，查询判定位 qpl-1 的 next-token 应对应 forced[qpl-1]，但 forced=[0]*qpl+qv_ids（qv_ids[0] 在 forced[qpl]）→ 改 forced=[0]*(qpl-1)+qv_ids（Case A/B 同）。
**教训：子代理修脚本后必须同步重跑相关测试；测试与实现的语义对齐（cache pos/forced off-by-one/填充边界）需逐个核实**。

## 门控副作用根治最终结论
**记忆层路径（已验证根治，in-context 0.688=基线零干扰）是正确方向**；KV 拼接彻底解耦在 0.1B 召回载体不足（负结果）。出路：①记忆层读出/寻址接口训练（让记忆层"无副作用+召回强"兼得）；②1.5B 扩展后重测彻底解耦（o_inj 表征或足够）。

## 结构（FullyDecoupledGate）
- **natural_gate**（GatedFusionMLP，3 维 win/csa/hca）：门控自然通路（滑窗/自然 csa/gist），可重训"对 gist 关"恢复 in-context。
- **inject_csa_gate**（_Gate4，4 维 win/csa/hca/inject）：门控注入通路（滑窗/独立 csa/hca/注入条目）。
- 融合（注入场景）：o = g_inj_win·o_win + g_inj_csa·o_csa + g_inj_hca·o_nat + g_inj_inj·o_inj（全走 inject_csa_gate，natural_gate 不参与注入前向）；无注入退化 natural 单门控。
- 两路参数完全独立（isdisjoint 验证）；恒等初始化 g=1/3（inject 位可选 -3.0 起点 0.05）。

## ⚠️ 核心诚实发现（关键负结果，禁止臆造）
**两目标（ic 0.688 + KV 0.625）未能同达**，tradeoff_eliminated=false 如实标注：

| 配置 | in-context | KV 注入召回 |
|---|---|---|
| natural=重训对 gist 关（任意 inject_csa） | **0.750~1.000** ✅ | 0.000~0.125 |
| inject_csa 复合初始化（扩容前3位+inject路由）frozen | 0.750 | 0.000 |
| 4 位同训 1000 步 | 0.750 | 0.125 |
| **对照：natural=扩容 + inject_csa=已训 inject_gate** | （未测 ic，方案A副作用0.250） | **0.500** |
| 方案 A（natural=扩容+inject_gate，副作用） | 0.250 | 0.625 |

**根因（诊断链）**：
1. inject 位强制开 0.9 → KV 仍 0（HCA 注入条目响应本身缺失，恒等起点）；
2. 复合初始化 inject_csa 前 3 位=扩容（`test 前3位一致=True` 等效方案A natural）+ 第 4 位=已训 inject 路由，融合全走 inject_csa → KV 仍 0.0625；
3. 唯一达 0.5 的配置是 **natural_gate=扩容门控**（非 inject_csa 复刻）——证明 0.1B 注入召回本质依赖 **扩容门控对注入场景的整体开权重状态**（win 位也参与），该状态无法拆进独立通道复刻而不伤 in-context；
4. KV 召回训练 loss 停滞 2.2~2.3 平台：模型从 Q 自身 win/csa 已能答 ~75%，剩余 25% 需从注入条目精确读取，0.1B frozen 主干下 o_inj 路径表征不足。

**结论**：ic/KV 结构性权衡**比方案 A 更深**——不仅是 win/csa 门控共享，而是召回信号与扩容门控整体状态耦合。彻底解耦结构上成立（两路独立、inject_csa 独立通道、来源路由），但 0.1B 尺度召回载体（o_inj 表征）不足，独立通道无法复现 0.625。**这是有价值的负结果**：提示需更大模型/更强 o_inj 表征（HCA 注入条目专用训练）才能让彻底解耦生效。

## 已验证（红线）
- in-context 精确召回 0.438→**0.750**（≥0.688 达标，方案 A 副作用 0.250 消除）；
- 主干逐位不变（backbone_unchanged=true，w_drift=0.0）；
- inject_csa_gate frozen 时逐位不变（freeze_inject 模式验证相等=True）；
- 两路独立训练互不干扰（test_natural_training_isolated_from_inject_csa 通过）。

## 用法
- 训练（freeze_inject 推荐）：`CUDA_VISIBLE_DEVICES=1 python scripts/train_fully_decoupled.py --freeze_inject --inject_init runs/recall_gated/trained_gate_mlp.pt --steps 500`
- 测试：`CUDA_VISIBLE_DEVICES=0 python -m pytest tests/test_fully_decoupled.py -q`（9 绿）

## 待接
① 0.1B→1.5B 扩展后重测彻底解耦（o_inj 表征或足够）；② HCA 注入条目专用训练（增强 o_inj 表征，非纯门控）；③ 接受 ic 0.750/KV 0.5 折中（natural=扩容+inject_csa=已训 inject_gate）作过渡；④ 根因深挖：方案 A 0.625 的精确机制（win 位贡献 vs o_inj 表征）。

---
*导出自 /memories/repo/fully-decoupled-gate.md（2026-07-30 同步快照）。*
