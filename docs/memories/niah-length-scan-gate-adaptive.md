# P1 NIAH 长度扫描 + 门控上下文自适应（2026-07-30）

## 产出
子代理批量执行，我验收（report 数据确认+独立重跑）。
- NIAH：`scripts/eval_niah_length_scan.py` 批量扫描（长度×key数×判据）+ report `runs/niah_length_scan/report.json`。
- 门控：`scripts/train_natural_gate_gist_off.py`（方案 A 变体+KV 锚定）+ report `runs/natural_gate_gist_off*/report.json`。

## ① NIAH 长度扫描结果（fb1 最有价值批评，升 P1）
**配置**：GDN-1 vs GDN-2 10k 有界，长度 512/1024/2048/4096 × keys 8/32 × 50 queries，first-token+full-VALUE 双判据，4070。
| cell | gdn1_first | gdn2_first | Δ |
|---|---|---|---|
| L512_k8 | 0.100 | **0.120** | +0.020 |
| L512_k32 | 0.060 | 0.040 | −0.020 |
| L1024_k8 | 0.080 | **0.100** | +0.020 |
| L1024_k32 | 0.040 | 0.060 | +0.020 |
| L2048/4096 | 0.000 | 0.000 | 0（外推截断） |

**关键诚实发现**：
- **max_seq=1024 是真实架构硬限**：tri_attention 的 k_rope=_rope(k,0) 对全部 cache key 从 0 重算 RoPE，cos/sin 缓存仅 1024 行——prefill 到 1024 后生成第 1025 token 即越界（RuntimeError）。**>1024 全长外推无法实测，需 max_seq 扩展（RoPE 缓存扩容+位置插值/NTK）**。
- **GDN-2 vs GDN-1 衰减曲线**：512/1024 短-中长度 GDN-2 略优（+0.020）；32-key 干扰下两者均大幅衰减（k8→k32 降约一半）。
- **0.217 低值定位**：first-token 判据本身即低（0.04–0.12），full-VALUE 全 0——**低值=GDN 状态饱和+first-token 判据过严双重**，埋点级精确首 token 检索在 0.1B 本就极弱。

## ② 门控上下文自适应（方案 A 变体+KV 锚定）
| 配置 | in-context | KV 注入召回 | 判定 |
|---|---|---|---|
| natural=已训扩容（原基线） | 0.250（副作用） | 0.625 | 副作用未消除 |
| natural=恒等（起点） | 0.438 | 0.062 | — |
| 重训对 gist 关（无锚定） | **1.000**（超恢复） | 0.125（KV 崩） | — |
| **+ KV 锚定 kv_anchor=2.0** | **0.812** ✅ | **0.438** | ic 达标+召回回升 |

- **in-context 彻底消除副作用**：0.250→**0.812**（KV 锚定，>0.688 达标）/ 1.000（无锚定超恢复）。
- **KV 召回回升但未回 0.625**：inject_gate 逐位 frozen✅，但 natural_gate 的 win/csa 门控连带影响——KV 锚定联合训练回升到 0.438。
- **⚠️ 关键诚实发现（结构性权衡）**：两目标（ic vs KV 召回）在 natural_gate 的 win/csa 取向上存在**结构性权衡**——对 gist 关（win 主导）必然压 csa，而注入召回走 csa/HCA。**真正的解是注入召回走独立 csa 通道（彻底解耦 win/csa）**，natural_gate 只门控 win/gist。side_effect_fixed=false 如实标注。backbone/inject_gate 逐位不变（红线合规）。

## ③ 关键判读
① max_seq=1024 是真实架构硬限（RoPE 缓存），>1024 需扩容才能实测全长外推（1M 目标的必经工程）；② 0.217 低值=状态饱和+判据过严双重，0.1B 埋点级精确检索本就极弱；③ 门控副作用根治需更深解耦（注入召回走独立 csa 通道，非仅 HCA 双通道）。

## 待接
①**max_seq 扩展**（RoPE 缓存扩容+位置插值/NTK，1M 目标必经）；②**彻底解耦**（注入召回走独立 csa 通道，消除 ic/KV 结构性权衡→同时 0.688+0.625）；③NIAH 长上下文复测（扩容后）。

---
*导出自 /memories/repo/niah-length-scan-gate-adaptive.md（2026-07-30 同步快照）。*
