"""增强 A 记忆层（GDN 旁挂可写 product-key KV，接口计划 §D2 / 部件详细计划 Part D2）。

设计依据（必须逐条对齐）：
- Memory Layers at Scale（arXiv:2412.09764，Meta ICML 2025）：稀疏 key-value 查找，
  **加参数不加 FLOPs**，事实任务强；product-key（两半键集 K1/K2，全集不实例化，
  半键集先搜 top-k 再合并）；keys/values 是**训练所得参数**（与注意力的根本区别）。
- 部件详细计划 Part D2 / 设计 §15.2：陈述性块**优先写入此原生记忆层**（非拼接 CSA KV 区）；
  **写入规则与 GDN delta 同构** `S ← S + β(v − v̄)⊗k`（先擦除旧关联再写入），
  使运行时写入**由构造保证在分布内**（Fast Weight Programmer 视角）；容量管理复用
  门控衰减（可整段遗忘）。
- **Gated DeltaNet-2（arXiv:2605.22791，NVIDIA 2026-05）**：erase/write 解耦——
  erase gate（key 侧，移除衰减状态哪些坐标）与 write gate（value 侧，承诺哪些新值坐标）
  独立，去除原版单一标量 β 的 tied 限制；`S_t=(I−k(b⊙k)ᵀ)D_t S_{t-1}+k(w⊙v)ᵀ`。
  本层 write() 支持 erase_gate/write_gate 解耦（默认 tied 向后兼容）。
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

    def write(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: float = 1.0,
        erase_gate: torch.Tensor | float | None = None,
        write_gate: torch.Tensor | float | None = None,
    ) -> None:
        """delta 规则写入（与 GDN 同构；GDN-2 erase/write 解耦扩展）。

        原版（tied，Gated DeltaNet）：`S ← S + β(v − k·S)⊗k`——标量 β 同时控制
        "擦多少旧读出"和"写多少新值"（建模限制）。

        GDN-2 解耦（arXiv:2605.22791，NVIDIA 2026-05）：erase gate（key 侧，移除衰减状态
        哪些坐标）与 write gate（value 侧，承诺哪些新值坐标）独立：
        `S ← S + β·outer(k, w ⊙ (v − e ⊙ (k·S)))`，其中 e=erase_gate（[key_dim] 或标量），
        w=write_gate（[value_dim] 或标量）；e=w=β 时退化为 tied 原版（向后兼容）。
        先擦除旧关联（e ⊙ (k·S)）再写入新关联（w ⊙ 残差）→ 由构造保证运行时写入分布内。
        无梯度（运行时零梯度快写，W2 通道）。
        """
        with torch.no_grad():
            kn = nn.functional.normalize(k, dim=-1)
            e = self._as_gate(erase_gate, self.key_dim, kn.device, kn.dtype)     # [key_dim]，key 侧
            w = self._as_gate(write_gate, self.value_dim, kn.device, kn.dtype)   # [value_dim]，value 侧
            # erase（key 侧，GDN-2 b⊙k）：控制擦除时 key 的哪些坐标参与读出/外积
            kn_eff = e * kn
            old = kn_eff @ self.state              # 旧读出 v̄ [value_dim]
            # write（value 侧，GDN-2 w⊙·）：控制承诺 value 的哪些坐标
            residual = w * (v - old)
            self.state += beta * torch.outer(kn_eff, residual)  # [key_dim, value_dim]

    @staticmethod
    def _as_gate(gate, dim: int, device, dtype) -> torch.Tensor:
        """门参数规整为 [dim] 向量；None→1（全通），标量→广播，向量→校验维度。"""
        if gate is None:
            return torch.ones(dim, device=device, dtype=dtype)
        if isinstance(gate, (int, float)):
            return torch.full((dim,), float(gate), device=device, dtype=dtype)
        g = gate.to(device=device, dtype=dtype)
        assert g.shape[-1] == dim, f"门维度 {g.shape[-1]} ≠ {dim}"
        return g.reshape(-1) if g.dim() > 1 else g

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
