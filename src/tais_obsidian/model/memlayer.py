"""增强 A 记忆层（GDN 旁挂可写 product-key KV，接口计划 §D2 / 部件详细计划 Part D2）。

设计依据（必须逐条对齐）：
- Memory Layers at Scale（arXiv:2412.09764，Meta ICML 2025）：稀疏 key-value 查找，
  **加参数不加 FLOPs**，事实任务强；product-key（两半键集 K1/K2，全集不实例化，
  半键集先搜 top-k 再合并）；keys/values 是**训练所得参数**（与注意力的根本区别）。
- 部件详细计划 Part D2 / 设计 §15.2：陈述性块**优先写入此原生记忆层**（非拼接 CSA KV 区）；
  **写入规则与 GDN delta 同构** `S ← S + β(v − v̄)⊗k`（先擦除旧关联再写入），
  使运行时写入**由构造保证在分布内**（Fast Weight Programmer 视角）；容量管理复用
  门控衰减（可整段遗忘）。
- 🧠 海马 DG/CA3（DG 模式分离 + CA3 自动联想）。
- 🔧 载体能力边界：记忆层条目为 **token 寻址**（key→value），**能事实召回**（接口计划 §6）。

纪律：
- 本原型只做结构与对拍单测（查询命中、delta 写入、门控衰减）；keys/values 训练、
  与 GDN 的旁挂接线、写入 RL 均在后续 milestone。
- 纯 PyTorch，Windows 原生，秒级 CPU 测试。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MemoryLayer(nn.Module):
    """product-key 稀疏 KV 记忆层（增强 A）。

    参数：
      keys   [n_slots, key_dim]：训练所得键（token 寻址 → 能事实召回）；
      values [n_slots, value_dim]：训练所得值；
      状态 S [key_dim, value_dim]：delta 规则运行时写入的工作记忆（与 GDN 同构）。

    查询：query → top-k 键检索（product-key 近似：此处 n_slots 小、直接全打分，
    大容量时再切两半键集）→ 加权值 + delta 状态读出。
    """

    def __init__(self, n_slots: int, key_dim: int, value_dim: int):
        super().__init__()
        self.n_slots = n_slots
        self.key_dim = key_dim
        self.value_dim = value_dim
        # 训练所得键值（参数，随 checkpoint 存取）
        self.keys = nn.Parameter(torch.randn(n_slots, key_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_slots, value_dim) * 0.02)
        # delta 运行时状态（非参数，不进 state_dict 的常规梯度路径；注册为 buffer 便于持久化）
        self.register_buffer("state", torch.zeros(key_dim, value_dim))
        self.register_buffer("gate", torch.tensor(1.0))  # 门控衰减系数

    def query(self, q: torch.Tensor, topk: int = 4) -> torch.Tensor:
        """q [..., key_dim] → 记忆读出 [..., value_dim]。

        = 训练键值 top-k 加权值（事实召回路径）+ delta 状态读出（运行时写入路径）。
        """
        k = nn.functional.normalize(self.keys, dim=-1)
        qn = nn.functional.normalize(q, dim=-1)
        scores = qn @ k.t()                      # [..., n_slots]
        vals, idx = scores.topk(min(topk, self.n_slots), dim=-1)
        weights = torch.softmax(vals, dim=-1)    # [..., topk]
        mem = (weights.unsqueeze(-1) * self.values[idx]).sum(-2)  # [..., value_dim]
        # delta 状态读出（与键值路径相加，单流读出）：state [KD, D]，kn/ qn [..., KD]
        delta = self.gate * (qn @ self.state)    # [..., value_dim]
        return mem + delta

    def write(self, k: torch.Tensor, v: torch.Tensor, beta: float = 1.0) -> None:
        """delta 规则写入（与 GDN 同构）：S ← S + β(v − k·S)⊗k。

        先擦除旧关联（k·S 的当前读出 v̄，注意 state [KD, D]、读出 = k 左乘）再写入新关联
        → 由构造保证运行时写入在分布内。k [key_dim]，v [value_dim]；无梯度（W2 通道）。
        """
        with torch.no_grad():
            kn = nn.functional.normalize(k, dim=-1)
            old = kn @ self.state                  # 当前对 k 的读出 v̄ [value_dim]
            self.state += beta * torch.outer(kn, v - old)  # [KD, D]

    def forget(self, gate: float) -> None:
        """门控衰减遗忘（容量管理，无需删除逻辑）：S ← gate · S，gate∈[0,1]。"""
        with torch.no_grad():
            self.state *= gate

    def reset_state(self) -> None:
        """清空 delta 工作记忆状态（不碰训练所得键值）。"""
        with torch.no_grad():
            self.state.zero_()


def make_memory_layer(n_slots: int = 256, key_dim: int = 64, value_dim: int = 64) -> MemoryLayer:
    """工厂函数。"""
    return MemoryLayer(n_slots, key_dim, value_dim)
