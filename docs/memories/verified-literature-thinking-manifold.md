# 思考流形/CTM 补充文献核实（2026-07-27，子代理检索+tavily 联网）

来源：/memories/session/thinking_manifold_lit_verify.md。每条均经联网核实 arXiv 存在性。

## 关键结论
- **CTM (arXiv:2505.05522, NeurIPS 2025 Spotlight)**：语言域**空白**（论文 §12 明写 future work）；民间 38M GPT-2+CTM 复现 PPL 劣化（未同行评审，负面信号）。→ 不能把 CTM 当已验证地基；同步表征 vs 残差循环须自消融。
- **潜在推理 vs CoT**：HRPO arXiv:2505.18454（混合呈 **U 形**——知识任务偏 explicit，STEM 偏纯 latent/explicit，中间态次优）；SwiReasoning/ThinkRouter 置信度路由。→ "潜在思考 + <|recall|> 显式审计"方向加固，但每步须明确 latent 或 explicit，忌糊中间。
- **网格码涌现**：transformer 中**不会自发涌现**六边形网格码（现有证据全在 RNN/PCN/tPCN；arXiv:2502.16690 中层有 2D 位置抽象但非网格；Sorscher 2022 仅 ~10% RNN 涌现且依赖非负/共形约束）。→ 流形导航结构必须**显式训练诱导**，降预期。
- **自适应算力**：GateSkip arXiv:2510.13876（连续可微门，后装稳定，省 15–50% 算力）+ CALM 置信度阈值——最适配 KAL P(IK) 耦合。P(IK) 当早停/检索/终止三合一门控信号**无人做过，TAIS 先发**。
- **lower-bounded decay**：channel-wise 有下界衰减已被 GLA(arXiv:2312.06635)→GDN(2412.06464)→KDA(2510.26692)→GDN-2(2605.22791) **四代独立复现加固**。K3 引用 [97,27,91] 未确认（报告未发布，语义最可能=GLA/Mamba/GatedDeltaNet）。

## 总判定
- 加固：混合架构方向、P(IK) 门控可嫁接 GateSkip/CALM、GDN/KDA 载体四代验证。
- 挑战：CTM 语言零证据；网格码须显式诱导；HRPO U 形陷阱。
- 独创外推：P(IK) 三合一门控、知识块作流形可寻址节点、CTM tick×GDN 状态融合。

---
*导出自 /memories/repo/verified-literature-thinking-manifold.md（2026-07-30 同步快照）。*
