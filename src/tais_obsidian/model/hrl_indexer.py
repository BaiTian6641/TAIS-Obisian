"""CSA Indexer（HRL 检索打分器）：DSA lightning indexer 式独立轻量打分模块。

设计依据（本轮文献交叉验证，禁止凭记忆实现）：

1. **DeepSeek lightning indexer（V3.2 DSA 原型，技术报告 Eq.1）**：
   `I_{t,s} = Σ_{j=1}^{H_I} w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )`
   - **独立的低维 indexer**：自有 q^I（query 侧）、k^I（key 侧）投影，**非复用主干注意力**；
     indexer 头数 H_I 少、维度 d_I 低、激活 ReLU（吞吐考虑）、可 FP8 运行——
     "快打分器 + 贵注意力"分离，复杂度从 O(L²) 降到 O(L·k)。
   - warmup：先冻结主干，用 **KL 散度对齐** indexer 分布到稠密主注意力分布（V3.2
     稀疏训练阶段：~1000 步/2.1B tokens 的短校准），再开 top-k 稀疏训练。
   - V4 CSA：indexer 在**已压缩条目**上打分（先 stride-4 压缩再选），FP8→FP4。

2. **PEER（arXiv:2407.04153，DeepMind）**：product key 检索（分半键集 K1/K2，笛卡尔积，
   全集不实例化，O(√N) 而非 O(N)）+ **内生独立 query network**——内容寻址检索器
   该独立、可训练、内生。top-k 离散无梯度，但**分数可微**（梯度流经分数到 query/keys）。

3. **设计 §11.1 / 接口计划 §4.1**：HRL 块索引器与 token 索引器同构（一个打分器两种检索
   对象）；FP8 分块归并 top-k（不物化全分数张量，StreamIndex 红线）；T2 KL 蒸馏 warmup
   （对齐稠密教师）；辅助损失梯度只进 Indexer（MoE-RL 红线）。

本模块定位：HRL 检索的**真正独立打分器**（区别于 tais_kernel.HRLIndexer 的单层
`nn.Linear(d,1)` 骨架——那是 M3 的最小占位；本模块是 DSA 式多头低维 indexer，
供 token 域（压缩条目）与块域（知识块）共用）。与内核解耦，便于独立 warmup/消融。

纪律：
- top-k 离散无梯度（选择只影响前向值聚合）；indexer 分数本身可微（照 DSA/PEER 原文）。
- warmup 用 KL 对齐稠密教师（T2）；辅助损失梯度隔离（detach 主干 query，复用
  tais_kernel 的 detach_input 纪律）。
- 纯 PyTorch，Windows 原生，秒级 CPU 测试。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightningIndexer(nn.Module):
    """DSA lightning indexer 式独立轻量多头打分器（HRL 检索）。

    `I_{t,s} = Σ_j w^I_{t,j} · ReLU( q^I_{t,j} · k^I_s )`（DSA 原型 Eq.1）。
    query 侧：x_q [B,Tq,d] → 多头 q^I [B,Tq,H_I,d_I] + 头标量权重 w^I [B,Tq,H_I]；
    key 侧：x_k [B,Tk,d] → k^I [B,Tk,d_I]（共享于各头）。
    输出每个 query 对每个 key 的 index 分数 [B,Tq,Tk]（按需 top-k，不物化时由调用方截断）。
    """

    def __init__(self, d_model: int, n_heads: int = 4, d_index: int = 32):
        super().__init__()
        self.n_heads = n_heads
        self.d_index = d_index
        # query 侧：多头 q^I 投影（d → H_I·d_I）+ 头标量权重 w^I（d → H_I）
        self.q_index = nn.Linear(d_model, n_heads * d_index, bias=False)
        self.w_index = nn.Linear(d_model, n_heads, bias=False)
        # key 侧：共享 k^I 投影（d → d_I）
        self.k_index = nn.Linear(d_model, d_index, bias=False)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        """x_q [B,Tq,d]，x_k [B,Tk,d] → index 分数 [B,Tq,Tk]。

        复用 DSA Eq.1：对每头 j，ReLU(q^I_j · k^I) 得 [B,Tq,Tk]，乘头权重 w^I_j 后跨头求和。
        """
        B, Tq, _ = x_q.shape
        Tk = x_k.shape[1]
        q = self.q_index(x_q).view(B, Tq, self.n_heads, self.d_index)  # [B,Tq,H,di]
        w = self.w_index(x_q)                                          # [B,Tq,H]
        k = self.k_index(x_k)                                          # [B,Tk,di]
        # ReLU(q^I_j · k^I) per head：q [B,Tq,H,di] × k [B,Tk,di] → [B,Tq,H,Tk]
        rel = torch.relu(torch.einsum("bqhd,bkd->bqhk", q, k))
        # 头加权和：w [B,Tq,H] × rel [B,Tq,H,Tk] → [B,Tq,Tk]
        return torch.einsum("bqh,bqhk->bqk", w, rel)

    def topk_indices(self, x_q: torch.Tensor, x_k: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """打分 + top-k（离散，无梯度）。返回 (top_scores [B,Tq,k], top_idx [B,Tq,k])。"""
        scores = self.forward(x_q, x_k)
        k = min(k, scores.shape[-1])
        return scores.topk(k, dim=-1)

    def kl_warmup_loss(self, x_q: torch.Tensor, x_k: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        """warmup：KL 散度对齐 indexer 分布到稠密教师分布（DSA 稀疏训练阶段）。

        teacher_scores [B,Tq,Tk]：稠密主注意力分数（或全块枚举打分），detach 后作目标。
        返回 KL(student || teacher)（student=indexer softmax 分布）。
        """
        student = F.log_softmax(self.forward(x_q, x_k).float(), dim=-1)
        teacher = F.log_softmax(teacher_scores.detach().float(), dim=-1)
        # KL(teacher || student)：以教师分布为真，让学生逼近（V3.2 对齐方向）
        return F.kl_div(student, teacher.exp(), reduction="batchmean", log_target=False)


def make_lightning_indexer(d_model: int, n_heads: int = 4, d_index: int = 32) -> LightningIndexer:
    """工厂函数。"""
    return LightningIndexer(d_model, n_heads, d_index)
