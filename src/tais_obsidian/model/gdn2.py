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
import torch.nn.functional as F


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


# ---------------------------------------------------------------------------
# chunked 训练路径（WY 表示，子代理实现 + 主代理验收对拍 <1e-4）
# ---------------------------------------------------------------------------


def chunked_gated_delta_rule_2(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """GDN-2 分块并行实现（WY 表示法）：块内并行、块间递推，decay 用块内累积对数。

    与 gdn.py 的 chunked_gated_delta_rule（GDN-1，标量 beta tied erase/write）同构，
    差异集中在 WY 辅助量的门注入（对齐官方 NVlabs/GatedDeltaNet-2 chunk_gdn2.py）：
      KDA/GDN-1:  u    = Λ @ (β * v)；w_wy = Λ @ (β * exp(D) * k)（β 标量广播）
      GDN-2:      u    = Λ @ (w ⊙ v)（write gate value 轴）；w_wy = Λ @ (b ⊙ exp(D) ⊙ k)
                  （erase gate key 轴）
    且严格下三角矩阵 A_strict 的 key tile 为 k_beta = (b ⊙ k)——**erase gate 折入 key
    tile，这是 GDN-2 相对 gated delta rule 的唯一结构性改变**；输出核（L⊙QKᵀ 下三角）
    与段间状态递归（尾衰减 e^{D_C}、asymmetric erase factor e^{D_C−D_i}）与 KDA 共享。

    形状约定同 naive_recurrent_gated_delta_rule_2；不足 chunk 整数倍时右侧零填充
    （pad 行 g=0→exp(0)=1 恒等衰减、门控为 0，不污染状态）。fp32 内部，对拍 <1e-4。

    数学（块内局部坐标，D_i = Σ_{r≤i} g_r，L_ij = exp(D_i − D_j)·[i≥j]）：
      Λ = (I + A_strict)^{-1}，A_strict = (L ⊙ (b⊙k)kᵀ) 的严格下三角
      v̂ = Λ @ (w⊙v) − (Λ @ ((b⊙k)⊙e^D)) @ S
      o_i = (q_i⊙e^{D_i}) S + [(L ⊙ qkᵀ) 下三角] @ v̂
      S ← e^{D_C} S + (k ⊙ e^{D_C−D})ᵀ @ v̂
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.autocast(device_type="cuda", enabled=False):
        q, k, v, b, w, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, b, w, g))
        B, H, T, K = k.shape
        V = v.shape[-1]
        BT = chunk_size
        pad = (BT - T % BT) % BT
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            b = F.pad(b, (0, 0, 0, pad))
            w = F.pad(w, (0, 0, 0, pad))
            g = F.pad(g, (0, pad))
        N = q.shape[2] // BT

        q = q * scale
        # GDN-2 门注入点①：write gate 折入 value tile（u = w ⊙ v）
        u = v * w
        # GDN-2 门注入点②：erase gate 折入 key tile（k_beta = b ⊙ k）
        k_beta = k * b
        # 切成块 [B,H,N,C,*]
        q = q.reshape(B, H, N, BT, K)
        k = k.reshape(B, H, N, BT, K)
        u = u.reshape(B, H, N, BT, V)
        k_beta = k_beta.reshape(B, H, N, BT, K)
        decay = g.reshape(B, H, N, BT).cumsum(-1)  # 块内对数衰减累积 D [B,H,N,C]
        decay_exp = decay.exp()[..., None]         # e^{D_i}（key 轴广播）
        # L_mask[i,j] = exp(D_i − D_j)（i>=j），块内衰减权重
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()

        # WY 表示：Λ = (I + A_strict)^{-1}，A_strict = ((b⊙k) @ kᵀ) * L_mask 严格下三角
        # 用 batched 三角求解（单位下三角）替代显式求逆/逐行前代，显存 O(C²)。
        mask_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
        A_strict = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask_diag, 0)
        eye = torch.eye(BT, dtype=torch.float32, device=q.device)
        M = A_strict + eye  # 单位下三角
        attn = torch.linalg.solve_triangular(
            M, eye.expand(B, H, N, BT, BT), upper=False, unitriangular=True
        )
        k_cumsum = attn @ u                          # WY 的 u：Λ @ (w⊙v)
        k_cumdecay = attn @ (k_beta * decay_exp)     # WY 的 w：Λ @ ((b⊙k)⊙e^D)
        v = k_cumsum

        S = (
            torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state.float()
        )
        o = torch.zeros_like(v)
        mask_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
        for i in range(N):  # 块间递推（与 GDN-1/KDA 共享：输出核 + 段间状态更新）
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
            attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask_upper, 0)
            v_new = v_i - k_cumdecay[:, :, i] @ S    # v̂ = Λu − Λ((b⊙k)e^D)·S
            o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S   # 块初状态直通读出 (q⊙e^D)S
            o[:, :, i] = o_inter + attn_i @ v_new
            # S ← e^{D_C}S + Σ_i k_iᵀ e^{D_C−D_i} v̂_i（channel-wise decay 吸收进 asymmetric factor）
            S = S * decay[:, :, i, -1, None, None].exp() + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2) @ v_new
        o = o.reshape(B, H, N * BT, V)[:, :, :T]
        o = o.transpose(1, 2).contiguous()
    return o, (S if output_final_state else None)


# ---------------------------------------------------------------------------
# GDN2Block：GDN-2 mixer 块（切换 GDN→GDN-2 的正式层，config 开关 layer_type="G2"）
# ---------------------------------------------------------------------------

import math  # noqa: E402

import torch.nn as nn  # noqa: E402

from ..config import ModelConfig  # noqa: E402
from .common import RMSNormGated  # noqa: E402
from .gdn import GDNBlock, l2norm  # noqa: E402


class GDN2Block(GDNBlock):
    """GDN-2 mixer 块（erase/write 解耦；继承 GDNBlock 的 conv/decay/g_proj/o_norm/o_proj）。

    与 GDNBlock（GDN-1）的唯一区别：标量写入强度 beta（b_proj: d→n_v_heads 标量）
    替换为**两个 channel-wise 门**——erase gate b（key 侧，d→key_dim）与 write gate w
    （value 侧，d→value_dim），前向调 GDN-2 递归核（chunked 训练 / naive 生成）。
    对齐官方 NVlabs/GatedDeltaNet-2 lit_gpt/gdn2.py 的 b_proj/w_proj 设计（均 sigmoid
    压到 [0,1]；allow_neg_eigval 时 b×2，本实现默认关）。

    tied 退化：初始化时若 b/w 投影学到相同输出，则 b=w≡β 等价 GDN-1（严格一般化）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(cfg)
        d = cfg.d_model
        # 覆盖 GDNBlock 的标量 b_proj（d→n_v_heads）为 channel-wise erase（d→key_dim）
        # + 新增 write gate（d→value_dim）——GDN-2 解耦（key 侧 / value 侧不同轴）。
        self.b_proj = nn.Linear(d, self.key_dim, bias=False)   # erase gate（key 侧）
        self.w_proj = nn.Linear(d, self.value_dim, bias=False)  # write gate（value 侧）

    def forward(self, x: torch.Tensor, state: dict | None = None) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape
        conv_q = state["conv_q"] if state else None
        conv_k = state["conv_k"] if state else None
        conv_v = state["conv_v"] if state else None
        q, new_conv_q = self._causal_conv(self.q_conv, self.q_proj(x), conv_q)
        k, new_conv_k = self._causal_conv(self.k_conv, self.k_proj(x), conv_k)
        v, new_conv_v = self._causal_conv(self.v_conv, self.v_proj(x), conv_v)

        q = q.view(B, T, self.n_qk_heads, self.head_dim)
        k = k.view(B, T, self.n_qk_heads, self.head_dim)
        v = v.view(B, T, self.n_v_heads, self.head_dim)
        # GDN-2 门（channel-wise，sigmoid 压 [0,1]）：erase b（key 侧）/ write w（value 侧）
        b = self.b_proj(x).view(B, T, self.n_qk_heads, self.head_dim).sigmoid()
        w = self.w_proj(x).view(B, T, self.n_v_heads, self.head_dim).sigmoid()
        q, k = l2norm(q), l2norm(k)
        if self.n_qk_heads != self.n_v_heads:  # GVA：qk 头与 b（key 侧）一起重复到 v 头数
            rep = self.n_v_heads // self.n_qk_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
            b = b.repeat_interleave(rep, dim=2)

        g = self._log_decay(x)  # 对数衰减 [B,T,Hv]（继承 GDNBlock：K3 式有界 / 旧式无界）

        rec = state["recurrent"] if state else None
        if T == 1 and not self.training:
            o, new_rec = naive_recurrent_gated_delta_rule_2(
                q, k, v, b, w, g, initial_state=rec, output_final_state=True
            )
        else:
            o, new_rec = chunked_gated_delta_rule_2(
                q, k, v, b, w, g, initial_state=rec, output_final_state=True
            )
        z = self.g_proj(x).view(B, T, self.n_v_heads, self.head_dim)
        o = self.o_norm(o, z)  # 门控 RMSNorm（fp32 出）
        o = self.o_proj(o.reshape(B, T, self.value_dim).to(x.dtype))
        new_state = {
            "recurrent": new_rec,
            "conv_q": new_conv_q,
            "conv_k": new_conv_k,
            "conv_v": new_conv_v,
        }
        return o, new_state
