# 解耦双通道门控（方案 A，门控上下文感知自适应，2026-07-29）

## 产出
`src/tais_obsidian/model/tri_attention_decoupled.py`（331 行）+ `scripts/train_recall_decoupled.py`（401 行）+ `tests/test_decoupled_gate.py`（335 行，10 项全绿）+ report `runs/recall_decoupled/report.json`。子代理实现，我验收（report 数据确认 side_effect_fixed=false 如实标注+独立重跑+全量 367 绿）。原 tri_attention.py/tri_attention_gated.py 未改（注入式 attach/detach）。

## 结构（DecoupledHcaGate 双通道）
- **natural_gate**：win/csa + 自然 gist 的 HCA 门控（复用/包装原 GatedFusionMLP）。
- **inject_gate**：独立零初始化通道（fc2=0+bias=-ln2 起点 g=1/3，fc1 随机破对称），仅对注入条目激活（TokenMem arXiv:2607.22625 零初始化独立通道先例）。
- HCA 注意力拆两路独立 softmax：注入条目→inject_gate，gist→natural_gate。无注入退化为 natural 单门控。
- **来源路由（结构化非学习 embedding）**：inject_hca_entries 拼入的条目排 HCA 区前 n_inj 个（namespace 五元组校验）=True 走 inject_gate，压缩器 gist=False 走 natural_gate。

## ⚠️ 诚实权衡发现（关键，子代理如实上报）
| 配置 | KV 注入召回 | in-context 精确召回 |
|---|---|---|
| 585 线性 | 0.188 | 0.688 |
| 扩容单门控 | 0.625 | 0.250（副作用） |
| **解耦 natural=已训** | **0.625** ✅ | **0.250** ⚠️（副作用仍在） |
| 解耦 natural=恒等 | 0.062（inject 训不动） | 0.438 |
| 拆门控 | — | 0.688（满恢复） |

- **注入召回保留 ✅**：natural=已训时只训 inject_gate 零初始化→召回 0.250→**0.625**（=扩容单门控峰值），natural_gate/主干 frozen 逐位不变。
- **副作用未消除 ⚠️**：natural_gate=已训扩容门控**本身对 gist 开了权重**（召回训练让它对 HCA 开权重波及 gist）→in-context 仍 0.250。
- **结论**：解耦结构成功**隔离注入召回**（inject_gate 独立），但 natural_gate 权重选择是关键——已训扩容门控对 gist 开权重不能直接当 natural_gate。

## 真正的解（下一迭代）
要让 in-context 满恢复 0.688 且注入召回 0.625 同时成立：natural_gate 换"对 gist 关"的权重——
① **方案 A 变体**：natural_gate=零初始化线性门控 + 重训它对 gist 关（保留原行为）；或
② **方案 C 正则**：召头部训练时对 gist 条目加"门控权重趋原值/趋 0"正则（压制 gist 门控）。
两目标（in-context 恢复 vs 注入召回）在当前 natural_gate 取值下存在权衡。

## 子代理踩坑（验收记录）
①梯度 NaN 假象（print None 显示 nan，实际梯度干净）；②natural=恒等时 inject_gate 训不动（natural g=1/3 时 gist 弱泄漏抢梯度）；③任务前提与实测偏差（natural=已训期望恢复 0.688，实测对 gist 开权重仍 0.250）如实上报。

## 待接
①方案 A 变体（natural_gate 重训对 gist 关）或方案 C 正则→同时达成 0.688+0.625；②召回 0.625→0.70 余量。

## 方案 A 变体实测（2026-07-30，scripts/train_natural_gate_gist_off.py + runs/natural_gate_gist_off/report.json）
natural_gate 恒等起点重训"对 gist 关"（in-context 任务，loss 只进 natural_gate，inject_gate 载入已训 frozen）：
- **in-context 精确召回 0.438→1.000**（超目标 0.688，副作用彻底消除；主干/inject_gate 逐位不变）；
- **但 KV 注入召回 0.062→0.125（崩）**——根因：natural_gate 同时控制 win/csa 门控，**KV 注入召回依赖
  natural_gate 对 csa 的开权重**（诊断：natural=已训扩容→KV 0.625/ic 0.250；natural=恒等→KV 0.062/ic 0.438；
  natural=重训对 gist 关→KV 0.062/ic 1.000）。inject_gate 只门控注入条目的 HCA 位，不改变 win/csa 平衡。
- **结论**：两目标（in-context vs KV 注入召回）在 natural_gate 的 win/csa 取向上**根本权衡**——
  对 gist 关（win 主导）必然压 csa，而注入召回要走 csa/HCA。**真正的解需 inject 召回与 natural 的
  win/csa 解耦更深**（如注入召回也走独立 csa 通道，或联合训练 natural_gate 时加 KV 召回锚定损失）。
- side_effect_fixed=false（KV 召回未达标）；产物 runs/natural_gate_gist_off/trained_natural_gate.pt。

## KV 锚定联合训练（2026-07-30，--kv_anchor 破解权衡，runs/natural_gate_gist_off_kvanchor[2]）
在 natural_gate 重训中交替混入"KV 注入样本"答案损失（偶数步注入锚定召回 + 奇数步 in-context 对 gist 关）：
- kv_anchor=1.0 与 2.0 结果相同：**in-context 0.438→0.812（>0.688 达标✅）+ KV 注入召回 0.062→0.438**
  （从崩盘 0.062 大幅回升，但未回 0.625）。backbone/inject_gate 逐位不变✅。
- **结论**：KV 锚定**部分破解**两目标权衡（ic 达标 + 召回回升），但 natural_gate 对 csa 的压制仍限制
  注入召回回不到 0.625——**结构性权衡仍在**（对 gist 关=win 主导必然压 csa，而注入召回走 csa/HCA）。
  side_effect_fixed=false（KV 0.438<0.5 阈值）。下一迭代方向：注入召回也走独立 csa 通道（彻底解耦），
  或 kv_anchor 调度 + 更长步数，或接受 ic 0.812/KV 0.438 折中。

---
*导出自 /memories/repo/decoupled-gate.md（2026-07-30 同步快照）。*
