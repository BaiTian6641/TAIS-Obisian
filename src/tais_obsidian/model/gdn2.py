"""Gated DeltaNet-2（GDN-2，arXiv:2605.22791，NVIDIA 2026-05）：erase/write 解耦递归。

设计依据（逐条对齐官方 NVlabs/GatedDeltaNet-2 与论文，禁止凭记忆实现）：

**GDN-2 递归**（对矩阵状态 S ∈ R^{dk×dv} 逐 token）：
    S_t = (I − k_t (b_t ⊙ k_t)ᵀ) D_t S_{t−1} + k_t (w_t ⊙ v_t)ᵀ
- **erase gate b_t ∈ [0,1]^dk**（key 侧，替代 KDA 标量 β）——选择性保护/修订 key 侧
  坐标关联；**消融确认 b_t 贡献最大**（论文 §Results）。
- **write gate w_t ∈ [0,1]^dv**（value 侧，GDN-2 新增）——选择性承诺新值坐标。
- **channel-wise decay D_t = Diag(α_t)**（继承 KDA，key 轴逐通道衰减）。
- `b = w = β`（共享标量广播）退化 KDA；进一步 decay 退化 Gated DeltaNet（严格一般化）。

**与本仓库 gdn.py（GDN-1）的关系**：gdn.py 是 GDN-1（单一标量 beta，tied erase/write）。
本模块提供 GDN-2 解耦变体，供全层消融（config 开关选 GDN-1/GDN-2，不动 gdn.py 主干
——168 测试 + 既有 checkpoint 不受影响）。

**实现纪律（对齐官方 fused_recurrent_gdn2.py 语义）**：
逐 token：① decay（h *= exp(g)）；② erase 读出 `(b⊙k)ᵀ @ h`（key 侧坐标选择）；
③ write：`v_new = (w⊙v) − erase_d`，外积 `h += k ⊗ v_new`；④ 读出 `o = hᵀ q`。
纯 PyTorch，Windows 原生，fp32 内部（与 gdn.py naive 同纪律），CPU 秒级对拍。
"""
from __future__ import annotations

import torch


def naive_recurrent_gated_delta_rule_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GDN-2 逐步循环参考实现（erase/write 解耦）。

    q,k: [B,T,H,K]（key 侧）；v,w: [B,T,H,V]（value 侧）；
    b: [B,T,H,K]（erase gate，key 侧）；g: [B,T,H]（对数衰减，channel-wise 经广播）。
    返回 o: [B,T,H,V] 与 final_state: [B,H,K,V]（可选），内部 fp32。

    tied 退化：b = β·ones[K]、w = β·ones[V] 时退化为 GDN-1（与 gdn.py 对拍一致）。
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.autocast(device_type="cuda", enabled=False):
        q, k, v, b, w, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, b, w, g))
        B, H, T, K = k.shape
        V = v.shape[-1]
        o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
        if initial_state is not None:
            h = initial_state.float()
        q = q * scale
        for i in range(T):
            b_q = q[:, :, i]      # [B,H,K]
            b_k = k[:, :, i]      # [B,H,K]
            b_b = b[:, :, i]      # [B,H,K] erase gate（key 侧）
            b_w = w[:, :, i]      # [B,H,V] write gate（value 侧）
            # ① decay（channel-wise，key 轴广播到 V）
            h = h * g[:, :, i].exp()[..., None, None]
            # ② erase 读出：(b⊙k) 投影状态 → [B,H,V]（key 侧坐标选择哪些被读出/移除）
            erase_d = (h * (b_b * b_k)[..., None]).sum(-2)  # [B,H,V]
            # ③ write：v_new = (w⊙v) − erase_d；外积 h += k ⊗ v_new（value 侧坐标承诺）
            v_new = (b_w * v[:, :, i]) - erase_d            # [B,H,V]
            h = h + b_k.unsqueeze(-1) * v_new.unsqueeze(-2)  # [B,H,K,V]
            # ④ 读出 o = hᵀ q
            o[:, :, i] = torch.einsum("bhk,bhkv->bhv", b_q, h)
        o = o.transpose(1, 2).contiguous()
    return o, (h if output_final_state else None)


def tied_to_decoupled(beta: torch.Tensor, K: int, V: int) -> tuple[torch.Tensor, torch.Tensor]:
    """GDN-1 标量 beta → GDN-2 tied 退化门（b=β·1[K], w=β·1[V]），供对拍验证。

    beta [B,T,H] → (b [B,T,H,K], w [B,T,H,V])。tied 时 GDN-2 应与 GDN-1 输出一致。
    """
    b = beta.unsqueeze(-1).expand(*beta.shape, K).contiguous()
    w = beta.unsqueeze(-1).expand(*beta.shape, V).contiguous()
    return b, w
