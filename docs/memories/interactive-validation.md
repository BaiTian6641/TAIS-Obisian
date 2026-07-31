# 交互式全链验证（2026-07-31，统一 checkpoint 已训强度）

> `scripts/interactive_chat.py`（REPL）+ `scripts/interactive_validation_demo.py`（确定性四阶段剧本）+ `tests/test_interactive_validation.py`（6 项）。产物：`runs/interactive_validation/{report.json, validation_panel.png, session_log.jsonl}`。子代理实现，主代理独立复跑全部确认。

## 用法
```bash
CUDA_VISIBLE_DEVICES=1 python scripts/interactive_chat.py            # REPL：对话/teach/quiz/probe/blocks/sleep
CUDA_VISIBLE_DEVICES=1 python -u scripts/interactive_validation_demo.py  # 四阶段剧本 → report + 面板
```
REPL `/teach` 支持 `K | Q | A` 三段式显式锚点（中文事实召回双侧失败——0.1B 无中文生成能力，用英文事实走定量路径）。

## demo 四阶段数值（seed=0，n=6；主代理复跑一致）
- **Phase A 空白区**：虚构实体 certainty 全 0.000，**Decline 6/6 = 1.000** ✅（复现 16/16 口径）
- **Phase B 教学即时召回**：写入率 1.00；检索 top-1 = **0.833**（5/6）；baseline = 0.000 vs **KV 注入 = 0.500**（3/6；n=16 判据 0.625，小样本同量级）
- **Phase C CoT/流形探针**：certainty 轨迹 ℓ10 读点（math 1→1 / 虚构 0→0 方向正确）；grid_score 均值 **−0.052**（<0.3，网格码不成立=预期：统一 ckpt 未挂路径积分训练）；3D 轨迹有序性比值 0.68–2.36
- **Phase D 睡眠固化**：**PROMOTE 3 / QUARANTINE 1 / REJECT 3**，逐块理由齐全（复现统一 checkpoint 报告的 8/1/8 模式）

## 偏差（如实标注于 report.json honest_notes）
- 检索 0.833 < 0.9 阈：n=6 小样本（0.938=15/16；1 个 miss 在 n=6 掉 0.167），协议与训练同款（均值池化）非回归。
- KV 召回 0.500 vs 0.625：同为小样本方差。

## ⚠️ 结构性新发现（值得反哺设计）→ 已落地修复（v1.1，2026-07-31）
**CA1 门与信源可信度耦合的边缘效应**：CallTool/doc 源块（credibility 0.7）teacher_consensus=0.68 恰低于 0.7 阈 → 全部 REJECT；user 源（0.9）consensus 0.76 → PROMOTE。**6 条已教事实只有 3 条能固化**——工具来源的知识在默认参数下系统性进不了长期记忆。处置方向：① 工具源的 consensus 计算纳入检索证据强度而非仅信源可信度；② 阈值-可信度边缘带引入"补验证重试"而非直接 REJECT；③ 1B 复测时调参记录。（demo 与 REPL 各自独立复现，非偶发。）

**v1.1 已按①②落地（CA1 门自适应）**：`runtime/ca1_gate.py` 加 RE_VERIFY 边缘带 [0.62,0.7)（`max_reverify=1` 上限 + fail-closed）、`evidence_aware_consensus` 证据加权（0.85·静态主项+0.10·usage/20+0.05·验证通过率，EvidenceWeights 可配）、`SourceCredibilityTracker` 信源可信度 EMA（α=0.2，截断 [0.3,0.95]，initial 映射不动）；`SleepConsolidator` 增 `reverify_fn` 编排（CrossVerifier 二次复核 + 有界加成 0.05）+ `report.verdicts/reverify_log`；`InquirySleepConsolidation` 裁决后回写 tracker（PROMOTE↑/共识侧 REJECT↓/QUARANTINE 与 usage·回归门不更新）。红线回归：drift>0.5 仍 QUARANTINE；<0.62 弱证据进不了补验证带；复核失败摊薄验证通过率不放行（tests/test_ca1_adaptive.py 17 项）。复跑 Phase D：PROMOTE 6/QUARANTINE 1/REJECT 0（doc 源全部经 RE_VERIFY 固化，web 冲突块仍隔离）。③ 1B 复测调参记录仍待做。

## 坑（已解决）
/teach certainty 口径（对事实 K 读数，非对元问题）；~~CA1 逐块理由需用 ca1_gate 纯函数同参数复算（报告只有计数）~~（v1.1 起 `ConsolidateReport.verdicts/reverify_log` 直接给逐块裁决与补验证日志，`sleep_consolidate` 不再复算）；Windows 管道中文 stdin reconfigure；Python hash 盐化（实体 slug 用 sha256）；面板零值柱需数值标注。

---
*写入自 2026-07-31 会话（Kimi Code CLI）。*
