# Kaplan 内词典提取（动态词表 concept_slot 真实启用）

- 实现：`src/tais_obsidian/model/kaplan_extract.py` `make_kaplan_extract_fn(model, layer, tokenizer)`；demo `scripts/dynamic_vocab_real_demo.py`；测试 `tests/test_kaplan_extract.py`（6 绿）。
- **提取层选择（关键坑）**：设计 §28.2 口径 ℓ10–14@28层（36–50% 深度）≙ 0.1B 12 层约 ℓ4–6，但**实测 pilot detokenize 最强在 ℓ3**（同类/不同类余弦 gap：L3=0.196 > L4=0.167 > L5=0.163 > L6+=递减）。小模型峰值前移（Kaplan 原文 OOV 检出峰也在 5-7/32 偏早）。`DEFAULT_KAPLAN_LAYER=3`；正式 1.5B 28 层回 ℓ10–14，可传 layer 扫描。
- 数据源：`model.forward(capture_layers=[i])`；**pm_stream=1 时 captures[i] 是裸 hidden [B,T,d]**（不是 dict！pm_stream>1 才是 {"content","pm"}）。gdn2_10k checkpoint pm_stream=1。
- 取末 token（[:, -1, :]）= 碎片融合 detokenized 向量，no_grad 只读，float32 [d_model]。
- 语义验证勿用逐对 min>max 硬断言（pilot 0.1B 部分对会反转，如 Tokyo/Paris 0.29 < dog/bicycle 0.37）；用均值 margin（sim_mean > diff_mean）。虚构专名（Qeltharion/Zorblax）无语义先验，语义测试要用常见同类对（electron/photon、dog/cat、neutron/proton、graviton/neutrino）。
- 装配：`make_dynamic_vocab(pt, NS, extract_fn, blockstore=bs)` → `make_orchestrator(model.kernel, bus, dynamic_vocab=dyn)`；`orch.assess_vocab_friction(text, p_ik, entropy, cooccur)` 一步完成 检测→提取→注册(页表+BlockStore)→route_graph 入图。concept_slot factual_recall=False（VECTOR_KINDS，向量加法 steer）。
- 运行：`$env:CUDA_VISIBLE_DEVICES="0"`（RTX 4070）；checkpoint `checkpoints/pilot_0p1b_gdn2_10k/final`（d_model=768，kernel_enabled=False 需 attach_kernel 作注入路径）。

---
*导出自 /memories/repo/kaplan-extract.md（2026-07-30 同步快照）。*
