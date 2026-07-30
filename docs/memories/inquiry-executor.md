# 求知执行器（主动求知闭环最后一块：执行+学习，2026-07-29）

## 产出
`src/tais_obsidian/model/inquiry_executor.py`（383 行）+ `tests/test_inquiry_executor.py`（304 行，15 项全绿）+ __init__ 导出。子代理实现，我验收（读码+独立重跑+全量 285 绿）。纯新增未改模块。

## 核心组件（接 ActiveInquiryLoop 的 inquiry_executor 接口）
- **Evidence**：新证据（content/source/credibility/timestamp/verified）。credibility 默认 user0.9>doc0.7>web0.5（信任度加权 §4.1）。
- **CrossVerifier**（绝不裸自我修正红线）：verify→(verified, consistency, conflict)。三路信号：①多源一致性（+0.15 提分）；②与既有知识一致性（pilot 余弦相似度近似，CA1 门"与先验一致性"几何读出，标注"正式应多源检索+外部锚"）；③冲突检测（余弦<0.3→conflict）。verified=一致性>0.6 且无未决冲突。
- **KnowledgeBlockWriter**（累积不覆盖红线）：block_id={ns}/{hash}:v{n} 版本化自增——同内容重复写入产生 :v1/:v2 两版本旧版保留（抗坍缩 arXiv:2404.01413）；冲突时新版仍写但 conflict=True+dispute_note 标分歧双方共存。**未验证证据绝不写入**（write 首行 if not verified: return None）。
- **InquiryExecutor**：__call__(decision)->bool。AskQuestion→ask_fn（用户回答）→Evidence(source=user)→验证→写入→True（闭环）；CallTool→tool_fn（查文档）→同理；Decline/DirectAnswer→False 不执行。True 仅当求知成功且验证通过。
- **ActiveInquiryPipeline**：InquiryExecutor+ActiveInquiryLoop 集成。低 certainty→求知→执行→验证写入→重评估 certainty（P(IK) 升高则闭环退出）。

## 红线落实（验收确认）
①**绝不裸自我修正**（arXiv:2310.01798）：未验证证据绝不写入，仅 verified=True 才闭环；②**累积不覆盖**：版本化 :v{n}+冲突保留双方标分歧；③**运行时写有界**：steering 零梯度快写，写 BlockStore 非梯度更新；④**诚实降级**：未获证据/未验证→不写入+False，Decline 不执行；⑤**监测/执行分置**：CrossVerifier 只读 detach。

## 子代理踩坑（已修复）
首条证据 verified 判定：初版无先验基线 0.475<0.6 致首条永不 verified → 改 0.8×(0.5+0.5×0.9)=0.76>0.6（CA1 门"无先验冲突→可快速同化"，McClelland CLS §4.2）。

## 主动求知闭环完整链（全部落地）
certainty 校准（KAL 真值锚）→ 求知分支（路由四选一 RPL/LP）→ 求知执行器（Ask/CallTool 执行+交叉验证+写入）→ 重评估闭环（P(IK) 升高）。

## 待接
①ask_fn/tool_fn 真实实现（对话接口/检索搜索工具，当前 pilot mock 接口）；②求知后写入的知识块→睡眠固化（CA1 门调速+三元奖励 RL）；③0.1B 基准消融。

---
*导出自 /memories/repo/inquiry-executor.md（2026-07-30 同步快照）。*
