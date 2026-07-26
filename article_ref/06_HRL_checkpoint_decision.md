# 06 · HRL 内生化决策：checkpoint 权重 vs 运行时服务（架构决策记录）

> 回应用户问题："HRL 层是否有必要也加入模型 checkpoint 然后保留接口（通过 TAIS 内核）来访问运行时？"
> 结论先行：**是——但只把"学习型路由/打分头簇"内生进 checkpoint，把"索引数据/块内容/图算法/门逻辑"留作运行时服务。** 二者经"TAIS 内核"接口衔接。

## 一、判定依据（三条独立证据）

1. **PEER（Mixture of a Million Experts, arXiv:2407.04153）**——router 用 product-key 检索、**是模型的一部分（checkpoint 内学习权重）**，专家（内容）被检索。这正是"路由器内生、内容外挂"范式。TAIS 的 HRL indexer 与 PEER router 同构（product-key top-k）→ **应内生**。

2. **MoE router 永远在 checkpoint**（DeepSeek/Mixtral 等）；**R3 / Rollout Routing Replay（arXiv:2510.11370）**证明训练-推理 router 一致性是 RL 稳定性的根本——router 若外挂/不一致会导致 RL 崩溃。TAIS 的 HRL router 要在 T3 做 RL（路由一致性约束）→ **必须在 checkpoint，否则 train-inference router 差异直接破坏 T3**。

3. **延迟**：§26.1 热切换红线要求毫秒级。内生 router 读 PM-stream ≈ 0 延迟；外挂 RPC 服务加不可控延迟。KAL/HRL 头要在**同一次前向**内完成"感知→路由决策"——只有内生才能做到。

## 二、内生 vs 运行时的精确切分

| 部件 | 形态 | 理由 |
|---|---|---|
| **🟢 内生 checkpoint 权重（"TAIS 内核"）** | | |
| KAL L1/L2/L3 头（W[2048,3]/[2048,2]/冲突）| 学习权重 | 监测，须与主干同分布共训练；冻结后推理读 |
| ITI 干预头（学习型投影）| 学习权重 | 执行，写 PM-stream |
| HRL **Indexer 打分头**（product-key top-k）| 学习权重 | = PEER router / MoE router，T3 RL 须一致性 |
| DG key 投影（稀疏化）| 学习权重 | 模式分离，与 indexer 同源 |
| 侧信道头簇 ×5（预取/写显著/冲突/归因/联想触发）| 学习权重 | 各 <1% 参数，旁路抽头 |
| CSA stride-4 压缩器 / HCA 压缩矩阵 | 学习权重 | 主干一部分（本就冻结）|
| **🟡 运行时服务 / 数据（"TAIS Memory Bus / DKB-Runtime"）** | | |
| 页表（Block Spec 索引数据）| SQLite + 向量库 | 数据非权重；可离线/在线更新 |
| BlockStore（块载荷 KV/向量/LoRA/源代码）| 文件存储 | 内容，非权重 |
| CA3 PPR 图算法 | 算法（图遍历）| 用内生 Indexer 头的分数做种子扩散 |
| CA1 巩固门逻辑 | 规则+回归测试 | 验证逻辑非权重 |
| 苏醒序列调度 | 调度器 | 软件时钟 |

## 三、"TAIS 内核"接口（建议）

TAIS 内核 = checkpoint 内全部学习型头（KAL + HRL 头簇）的统一前向接口。主干在每层调用内核读 PM-stream 出信号；内核决定路由后**经 Memory Bus 调用运行时服务取块**；取回的块经注入中间件回填主干。这样：
- 信用分配端到端（RL 直接奖励"感知→路由→注入→答对"整链）——✅ v2.5 §8.1 原生化第四条；
- 内核权重可随 checkpoint 保存/加载（与现有 `model/save_pretrained` 一致）；
- 运行时服务可独立升级（页表扩容、块库增长）不动 checkpoint。

## 四、对现有代码的影响（src/tais_obsidian/）

- `model/kal.py`（已有 L1/L2 头）✅ 已内生；
- 需新增 `model/hrl_heads.py`：Indexer 打分头 + DG 投影 + 侧信道头簇，全部 `nn.Module`，进 `state_dict`；
- `forward(capture_layers=...)` 已支持 hidden-state 捕获挂点 → 内核读挂点；
- 运行时服务（页表/BlockStore/PPR/CA1）单独成包 `runtime/`（不在 checkpoint，经 Memory Bus 与内核交互）。
