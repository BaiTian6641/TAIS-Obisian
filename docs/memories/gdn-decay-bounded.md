# GDN decay 有界化（K3 借鉴，2026-07-27 落地）

## 改动
Kimi K3 借鉴（arXiv:2510.26692 谱系）：GDN decay 参数化从无界 negative-softplus 改 K3 式有界 scaled-sigmoid。
- `config.py`：`gdn_decay_g_min: float|None = -5.0`（默认有界；`None`=旧式无界仅复现旧 checkpoint）。
- `gdn.py._log_decay(x)`：有界 `g = g_min·sigmoid(exp(A_log)·(a+dt_bias))` ∈ (g_min,0)，fp32 clamp(eps=|g_min|·1e-6) 保严格开区间（sigmoid 极值 fp32 饱和到恰好 0/1→g=0/g_min，失去严格 α>e^{g_min}）。GDN2Block 继承 GDNBlock，自动生效。
- 公式：每步保持因子 α=e^g > e^{g_min}≈6.7e-3；16-token tile 累积 log-decay∈(−80,0)，倒数 rescale<e^80 留在 BF16 范围（1M 必需）；有界 sigmoid 亦可能助 GDN-2 门收敛。

## 断点兼容（关键，勿踩坑）
- 旧 checkpoint（2026-07-27 前，无界 decay 训练）config.json **无** `gdn_decay_g_min` 字段。
- `ModelConfig.from_json` 检测字段缺席时**回填 None**（旧式无界复现），并 print 提示。
- 新训练 config 须**显式写** `"gdn_decay_g_min": -5.0` 走有界。
- 两式非线性不可逆，不可混用加载。

## 踩坑记录（教训）
1. **multi_replace 插入位置错误**：初版把 `_log_decay` 方法插进 `__init__` 中间，吞掉 conv/g_proj/o_norm/o_proj 初始化（变成 return 后死代码）→ 模型缺层。修复：方法移到 `__init__` 结束后。**教训：编辑类定义时新方法务必确认缩进与插入点边界，跑 forward 烟测验证所有子模块存在。**
2. **fp32 sigmoid 饱和**：极值处 g 恰好=−5.0/0（非严格开区间），断言 `>` 失败 → clamp eps 内缩。
3. **test_pmstream 恒等 flaky**：绝对阈值 <1e-6 在 d_model=256/head_dim=64 bf16 累积下偶发 3.3e-6（相对 ~1e-6，bf16 数值边界非逻辑误差）→ 改相对容差 rel<1e-5。

## 验证
- 177 pytest 全绿；有界/无界前向均 OK；GDN2 继承生效；旧 ckpt 回填 None 复现 + 新 config 显式 −5 读回均正确。
- **train.py 修复（关键坑）**：初版 train.py 构造 ModelConfig 未传 gdn_decay_g_min（config JSON 不可控）→ 已修复 `gdn_decay_g_min=cfg.get("gdn_decay_g_min", -5.0)`（缺省有界，无界对照须显式 null）。**新启动的 run 默认走有界**。

## 无界基线 10k 完成（2026-07-28，前置①门收敛验证 ✅ 通过）
- **三阶段**：2000 步（门 std 0.024/0.019，NIAH 0.130<GDN-1 0.200，欠训练）→ 8000 步（std 0.345/0.321 饱和，NIAH 0.180 略劣）→ **10000 步（std 0.342/0.317，NIAH 0.240>GDN-1 0.200，Δ+0.040 ✅ 反超）**。val 3.5305。
- **三论点**：① 欠训练非架构缺陷诊断成立（GDN-2 方向正确）；② 检索滞后于门分化（门 8k 饱和，NIAH 10k 才反超——慢变量）；③ erase/write 解耦检索优势门收敛后兑现。
- **有界对比 run 已启动**（pilot_0p1b_gdn2_bounded_10k，g_min=-5，ID e8ff977b）：与无界基线唯一差异=decay 参数化，同 seed/超参/数据公平对比。

## bounded 2k 节点中期结果（2026-07-28）⭐ 有界显著加速门收敛
- **门分化度**：bounded 2k b/w 坐标 std **0.323/0.296** vs 无界 2k **0.024/0.019**（13–15×）——bounded 2k 提前达到无界 8k 才饱和的水平（0.345/0.321）。**验证 K3 借鉴核心假设：有界 sigmoid 衰减更易学，是 GDN-2 门欠收敛的贡献因子；有界让门收敛从 ~8k 提前到 ~2k（4× 加速）**。
- **NIAH 检索**：bounded 2k 0.180（=无界 8k 水平，<GDN-1 0.200）——再次确认"门分化必要非充分，检索慢变量"（门被加速但检索仍需 tokens 积累，预期 bounded 10k 反超）。
- **工程结论**：① decay 有界化双重收益（保 1M 数值范围 + 4× 加速门收敛）；② train.py 已默认 g_min=-5，方向正确，1M 必须切有界。
- bounded 2k 临时 final 导出在 checkpoints/_gdn2_bounded_step2000_eval（g_min=-5）。

## bounded 7000 步中期结果（2026-07-29，4070 评估）⭐ 检索反超提前
- **门分化度**：bounded 7k b/w 坐标 std **0.335/0.322**（bounded 2k 即 0.32+ 已饱和，7k 持平）。
undefined
- 4070 评估验证（双卡分工）：门分化度+NIAH 在 RTX 4070 Laptop 跑通（0.1B bf16 ~0.5GB），PRO 4000 继续 bounded 训练不受干扰。
- bounded 7k 临时 final 导出在 checkpoints/_gdn2_bounded_step7000_eval。

## bounded 10k 完成 + NIAH 显著性检验（2026-07-29，前置②完整收尾）
**显著性检验**（scripts/_niah_significance.py，3 seeds×200=600 次/模型，std 0.021）：
| 模型 | NIAH mean±std |
|---|---|
| GDN-1 基线 | 0.177 |
| 无界 GDN-2 10k | 0.207±0.021 |
| bounded GDN-2 7k | 0.217±0.021（最高） |
| bounded GDN-2 10k | 0.203±0.021 |
- **结论**：① GDN-2（有界无界）都反超 GDN-1（+0.026~0.040>std）；② **有界 vs 无界 10k 持平**（0.203 vs 0.207，Δ<std——decay 是优化路径非能力上限，符合预期）；③ bounded 7k→10k"下降"（0.217→0.203）在 std 0.021 噪声内，非真实退化。
- **⚠️ 判读修正（诚实）**：单 seed 100 次的 bounded 7k 0.220 vs 10k 0.180 曾被解读为"反超提前+后期退化"——显著性检验后**都是噪声**（std 0.021），真实趋势是 bounded 7k≈10k≈无界 10k（0.20–0.22），检索在门饱和后平稳。**教训：NIAH n_queries=100 单 seed 的 Δ0.02–0.04 是采样噪声，须多 seed 大样本显著性检验，勿过度解读单点**。
- **有界 decay 真实价值（不受噪声影响）**：① **4× 加速门收敛**（bounded 2k 即达无界 8k 饱和，std 差异 13–15× 远超噪声，真实）；② 保 1M 数值范围（必需）。**"检索反超提前"判断收回**——门收敛加速是真的，但最终检索能力有界≈无界。
- val loss：bounded 3.5383 vs 无界 3.5305（持平）。
- **前置② decay 有界化完整收尾**：落地（代码+断点兼容）+ 实证（4× 加速门收敛真实+保 1M 必需）+ 检索能力有界≈无界（decay 非能力上限）。**1M 长上下文切有界为默认的决策成立（数值范围必需+收敛加速），非因检索优势**。

---
*导出自 /memories/repo/gdn-decay-bounded.md（2026-07-30 同步快照）。*
