# 扩展交互测试 S1–S5 + CA1 自适应 v1.1 + 流形训练（2026-07-31）

> 承接 interactive-validation.md。本轮三件套：CA1 门自适应、流形投影器训练+预览、五场景扩展测试（`scripts/extended_validation.py`，产物 `runs/extended_validation/`，113 轮 JSONL 日志，判据 14/15 PASS 1 负）。

## ① CA1 巩固门自适应（v1.0→v1.1，根治信源边缘效应）
- **机制**：① 边缘带 RE_VERIFY（consensus∈[0.62,0.7) 不直接 REJECT，CrossVerifier 复核+有界加成≤0.05 二次入门，上限 1 次）；② 证据感知共识（0.85·base + 0.10·usage + 0.05·验证通过率）；③ SourceCredibilityTracker（EMA α=0.2 在线学习，clip [0.3,0.95]；PROMOTE→1.0、共识 REJECT→0.0、QUARANTINE 不惩罚）。
- **实证**：demo Phase D 旧 P3/Q1/R3（doc 全灭）→ 新 **P6/Q1/R0**（doc 块 0.688→RE_VERIFY→0.743→PROMOTE）；矛盾块仍 QUARANTINE。
- **抗放水**（三重验证）：劣质块 <0.62 直接拒（不进带）；复核恒败摊薄不洗白；三连败 credibility 0.7→0.36 跌出带失去重试资格。残留风险：usage 由调用方传入（权重仅 0.10，正式版应来自页表权威计数）。
- 测试：`tests/test_ca1_adaptive.py` 17 项 + 旧测试期望更新（注明新旧行为）。

## ② 流形投影器：从未训练 → 已训（sidecar）
- **判定**：unified ckpt 266 键零 manifold 键；所有 demo 现场 seed42 新建投影器；权重统计=PyTorch 默认初始化（std 0.02086≈理论 0.02083）。**必须训**。
- **训练**（`scripts/train_manifold_projector.py`，1500 步 73s，冻结主干逐位一致 ✅）：复用 manifold.py 损失（共形等距+VICReg），多层 ℓ4/7/10 hidden 段表征，语义步长自监督。
- **数值**：聚簇对比度 1.558→**1.989**；等距 Pearson 0.882→**0.977**；**语义块邻近性**（最强证据）：数学 prompt 轨迹最近 4 块恰为全部数学块（prime 4.32<integration 5.22<deriv 5.73<pythagorean 6.14 < roman 8.32）。
- **预览**（`scripts/manifold_preview.py`）：逐生成步 ℓ10→project_3d→3D 轨迹渲染 + 知识块叠加 + 坏路径四类检测 + compare 对照图。
- 诚实标注：0.1B 主干表征限制轨迹可解释性上限（直线度仅 0.177）；坏路径检测绝对阈值是 tick 尺度 pilot 经验值，对生成步尺度系统性误报（T1 标定项）。

## ③ 五场景扩展测试（113 轮日志，14/15 PASS）
- **S1 已有知识推理 A→B→C**：逐点检索命中 geo 0.67 / astro 1.00 ✅；但常识点 certainty/作答多为 0（0.1B/120M 规模边界，如实记录）；链式长问题检索弱（0.1B ℓ3 表征）。
- **S2 多轮教学+失败重试**：召回曲线 0.500/0.500/0.500（单调不降 ✅）；检索 0.875；**版本证据 v1/v2/v3 共存（累积不覆盖 ✅）**；曲线平坦=同内容重教确定性重建，版本自增才是重试产物（已注明）。
- **S3 桥接（核心场景）**：基线推不出 D（certainty 0.000）✅ → 只教中间知识 B'(Zorblax-xenon)、C'(xenon-krypton) → **注入答出 'krypton' ✅ / 不注入仍失败 ✅**；流形邻近 C' min 6.12→5.54 改善；前后对比图（s3_bridge_before/after.png）。消融诚实标注：C' 单块也能答出（文本层近端复制），B' 必要性由几何证据补充。
- **S4 动态词表**：concept_slot 注册 ✅（页表+BlockStore+route_graph）；语义邻居 metal/silver 类 0.256 > 无关词 0.168 ✅（banana 0.261 噪声如实保留）；新词块检索 top-1 ✅；**注入召回 FAIL（诚实负结果）**——OOV 新词作主语时 KV 注入召回不工作（5 种句式变体全失败）。
- **S5 睡眠增强**：RE_VERIFY→PROMOTE ×19 + QUARANTINE ×1（矛盾地块被拦 ✅）；tracker doc 0.70→0.95；固化后召回不变（0.500=固化前，固化不破坏召回 ✅）。

## ④ 载体能力边界（本轮最重要发现，反哺设计）
**KV 注入召回仅在 teaching 训练分布（引擎事实+_FUEL 答案词）上有效**：自定义句式事实实测检索/召回仅 0.25 且 CrossVerifier 冲突拒写；OOV 造词答案（florn/Xylos 等）注入后一律回退先验燃料词。**含义**：0.1B 的"写入即可用"是分布内能力，召回泛化需要 ① teaching SFT 分布多样化（多句式/多答案域）② 或 1B 规模自然改善——列为 1B 复测首要观测项（与 KAL 探针强度并列）。

## 待接（1B 复测清单）
① KAL 探针强度；② 注入召回分布泛化（本发现）；③ 坏路径阈值标定（生成步尺度）；④ usage 权威计数接页表；⑤ CA1 v1.1 在真实知识分布的裁决统计。

---
*写入自 2026-07-31 会话（Kimi Code CLI）。*
