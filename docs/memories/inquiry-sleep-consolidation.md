# 求知知识块→睡眠固化端到端（主动求知闭环最后一块，2026-07-29）

## 产出
`src/tais_obsidian/sleep/inquiry_consolidation.py`（334 行）+ `tests/test_inquiry_consolidation.py`（202 行，11 项全绿）+ sleep/__init__ 导出。子代理实现，**我亲自验收时抓到并修复 1 个 flaky**（详见下）。全量 321 绿。

## 核心组件
- **InquiryW0Adapter**：求知知识块（BlockStore BlockPayload）→W0Item。content/saliency/credibility/conflict 保留；先验一致性（embed 余弦近似，与 CrossVerifier 同策略）；teacher_consensus=一致性×(0.5+0.5×credibility)（Jeffrey 信任度加权）；conflict→belief_drift=0.9（CA1 门拦截到 QUARANTINE）。
- **PriorConsistencyGate**：assess→(consistency, fast_track)。一致>0.6→fast_track（同化快固化）；冲突→慢通道（顺应，挡单次错误经验）。**调速经 teacher_consensus/belief_drift 两既有输入间接进入，保持 ca1_gate 独立裁决权（novelty ⊥ correctness 红线不绕过）**。
- **TriRewardRL**：correct+1/hallucinate−1/abstain 0~0.3（默认 0.15，超窗报错）。先 SFT 教拒答后 RL（arXiv:2601.20126 注释）。
- **InquirySleepConsolidation**：consolidate_inquiry_blocks→W0Item→调速→SleepConsolidator.consolidate（CA1 门+间隔提取）→三元奖励→ConsolidateReport。

## 红线落实（验收确认）
①防错误固化：regression_ok=payload.verified AND 外部回归（双重验证门）；②累积不覆盖：冲突块 QUARANTINE 后仍存 BlockStore（conflict/dispute_note 保留）；③冲突保留双方标分歧（belief_drift=0.9→QUARANTINE 非静默覆盖）；④绝不裸自我修正：固化经外部验证（regression+teacher_consensus）；⑤三元奖励 abstain 不重罚。

## ⚠️ 我抓到的 flaky（子代理漏报，亲自验收的价值）
- **现象**：test_gate_conflict_slow_track 单独跑过、全文件跑失败。
- **根因**：`_embed` 用 `torch.manual_seed(abs(hash(text)))`——Python 内建 `hash()` 对 str 在 PYTHONHASHSEED 未固定时**每进程随机**（DoS 防护）→跨进程/测试顺序结果不同。
- **修复**：改 `hashlib.sha256(text)`（**确定性**）替代内建 hash()。
- **教训（重要）**：**内建 hash() 不可用于需要跨进程确定性的种子/键**（str/bytes 随机化）；须用 hashlib（sha256/md5）确定性 hash。子代理报告全绿但我重跑抓到——亲自验收（尤其重跑测试）不可替代。

## 主动求知闭环三阶段全部落地
运行时学习（求知执行器写知识块）→ 实时可用（HRL 检索+HCA 注入）→ **长期固化（CA1 门调速+间隔提取+三元奖励 RL+防错误固化）**。

## 待接
①三元奖励 RL 实际接入固化蒸馏（当前规则版 reward 信号）；②先验一致性用真实模型 hidden（当前 hash 投影近似）；③0.1B 全链端到端（求知→实时→固化）。

---
*导出自 /memories/repo/inquiry-sleep-consolidation.md（2026-07-30 同步快照）。*
