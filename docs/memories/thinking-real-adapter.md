# 第二阶段真实部件适配 + 0.1B 消融（2026-07-28 落地）

## 产出
`scripts/thinking_real_adapter_demo.py`（313 行）+ `tests/test_thinking_real_adapter.py`（169 行，8 项全绿）。把 e2e demo 全 mock 替换为真实 0.1B 部件（checkpoints/pilot_0p1b_gdn2_10k/final，GDN-2 门已收敛）。输出 runs/thinking_real/（gitignore）。**已亲自验收**：读码（RealThoughtAdapter 维度桥接）+ 独立重跑 demo（真实 GDN/glimpse/certainty 读出+有核无核消融跑通）+ 全量 251 绿。

## 意义与待接
第二阶段从"协同骨架（全 mock）"到"真实部件通路"的跃迁——真实 GDN 状态/CSA glimpse/真实 certainty 读出贯通，核确实改变思考动力学。**待接**：①KAL 真值锚微调（certainty 才能作真实元认知门控——当前未校准不可判据）；②适配层离线训练；③0.1B 基准准确率消融（核 vs 无核真实增益）。

## 关键决策
- **checkpoint kernel_enabled=False**：from_pretrained 后须显式 `attach_kernel()`（幂等）——内核头（kal_l1/l2/hrl/side）随机初始化未微调，**KAL 未校准**：certainty 仅演示通路非可靠元认知（实测 known 概率 0.001~0.99 漂移，种子敏感）。
- **维度桥接**：`RealThoughtAdapter` 持 down_proj 768→384 + up_proj 384→768（随机初始化未训练，离线才允许训练）；真实 GDN 层10 输出 capture → down_proj → [B,T,384] 作思考核 state。
- **真实 glimpse**：CSA 层3 输出 capture → down_proj（pilot 简化近似，非真正"往哪看"）。
- **消融**：有核=ThoughtCore 8 tick 演化+bridge 位移；无核=同一真实 GDN 状态单步投影。**dist_core/dist_no_core 均用 `sp.project(state)` 真实投影坐标→target 距离**——严禁用 `current_coord+disp`（disp=target−current 恒等 target，dist 恒 0 是坑）。
- **certainty 缺省 mock 0.2 跑满 tick**（真实 KAL 未校准不作早停依据），`mock_certainty=False` 可切真实通路演示。

## 红线保持
监测/执行分置：model.forward+sense 全 no_grad 只读；bridge.tick detach；共享 projector 单实例+复用 core.bridge；纯新增不改现有模块。

## 缝隙/坑
- sense 读点：pm_stream=1 单流时 sense 读**内容流**（层输出残差流），非 PM-stream（kernel_sense_index 只认 G/G2 层）。
- 末 tick current+disp=target 恒等式陷阱（见上）。
- 适配层随机初始化 → dist_core(90.03)≈dist_no_core(89.74) 量级一致属预期（桥接未训练，核只证"改变动力学"非"更准"）。

---
*导出自 /memories/repo/thinking-real-adapter.md（2026-07-30 同步快照）。*
