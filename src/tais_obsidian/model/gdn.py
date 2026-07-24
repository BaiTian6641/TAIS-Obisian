"""GDN（Gated DeltaNet）块：纯 PyTorch 实现（本机无 Triton）。

参照 fla 库参考实现：
- fla/ops/gated_delta_rule/naive.py 的 naive_recurrent / naive_chunk（WY 表示法分块）
- fla/layers/gated_deltanet.py 的层封装（投影 → 短因果 Conv1d+SiLU → L2 norm →
  beta=sigmoid 写入强度 → decay g = -exp(A_log)*softplus(a+dt_bias) → 输出 sigmoid 门控 RMSNorm）

两条核心路径：
- naive_recurrent：逐步循环，参考实现；
- chunked：块内并行 + 块间递推（WY 表示法），训练用。
fp32 下随机输入 seq=128 两路径输出 max abs diff < 1e-4（见 tests/test_gdn.py）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNormGated


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """fp32 L2 归一化后回到原 dtype（对齐 fla l2norm）。"""
    return F.normalize(x.float(), p=2, dim=-1, eps=eps).type_as(x)


def naive_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """逐步循环参考实现。

    q,k: [B,T,H,K]；v: [B,T,H,V]；beta,g: [B,T,H]（g 为对数衰减）。
    返回 o: [B,T,H,V] 与 final_state: [B,H,K,V]（可选），内部 fp32。
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.autocast(device_type="cuda", enabled=False):
        q, k, v, beta, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g))
        B, H, T, K = k.shape
        V = v.shape[-1]
        o = torch.zeros(B, H, T, V, dtype=torch.float32, device=v.device)
        h = torch.zeros(B, H, K, V, dtype=torch.float32, device=v.device)
        if initial_state is not None:
            h = initial_state.float()
        q = q * scale
        for i in range(T):
            b_q = q[:, :, i]
            b_k = k[:, :, i]
            b_beta = beta[:, :, i]
            # 状态衰减 → 读出旧值 → delta 规则写入
            h = h * g[:, :, i].exp()[..., None, None]
            b_v = v[:, :, i] - (h * b_k[..., None]).sum(-2)
            b_v = b_v * b_beta[..., None]
            h = h + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
            o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
        o = o.transpose(1, 2).contiguous()
    return o, (h if output_final_state else None)


def chunked_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    chunk_size: int = 64,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """分块并行实现（WY 表示法）：块内并行、块间递推，decay 用块内累积对数。

    形状约定同 naive_recurrent_gated_delta_rule；不足 chunk 整数倍时右侧零填充。
    """
    if scale is None:
        scale = q.shape[-1] ** -0.5
    with torch.autocast(device_type="cuda", enabled=False):
        q, k, v, beta, g = (x.transpose(1, 2).contiguous().float() for x in (q, k, v, beta, g))
        B, H, T, K = k.shape
        V = v.shape[-1]
        BT = chunk_size
        pad = (BT - T % BT) % BT
        if pad:
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            beta = F.pad(beta, (0, pad))
            g = F.pad(g, (0, pad))
        N = q.shape[2] // BT

        q = q * scale
        v = v * beta[..., None]
        k_beta = k * beta[..., None]
        # 切成块 [B,H,N,C,*]
        q = q.reshape(B, H, N, BT, K)
        k = k.reshape(B, H, N, BT, K)
        v = v.reshape(B, H, N, BT, V)
        k_beta = k_beta.reshape(B, H, N, BT, K)
        decay = g.reshape(B, H, N, BT).cumsum(-1)  # 块内对数衰减累积 [B,H,N,C]
        decay_exp = decay.exp()[..., None]
        # L_mask[i,j] = exp(D_i - D_j)（i>=j），块内衰减权重
        L_mask = (decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp()
        # WY 表示：attn = (I + A_strict)^{-1}，A_strict = k_beta @ k^T * L_mask（严格下三角）。
        # 用 batched 三角求解替代逐行前代循环（同 fla 的 solve_tril），显存 O(C²) 而非 O(C³)。
        mask_diag = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
        A_strict = ((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask_diag, 0)
        eye = torch.eye(BT, dtype=torch.float32, device=q.device)
        M = A_strict + eye  # 单位下三角
        attn = torch.linalg.solve_triangular(
            M, eye.expand(B, H, N, BT, BT), upper=False, unitriangular=True
        )
        k_cumsum = attn @ v  # WY 的 u（新 v）
        k_cumdecay = attn @ (k_beta * decay_exp)  # WY 的 w
        v = k_cumsum

        S = (
            torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
            if initial_state is None
            else initial_state.float()
        )
        o = torch.zeros_like(v)
        mask_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
        for i in range(N):  # 块间递推
            q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
            attn_i = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask_upper, 0)
            v_new = v_i - k_cumdecay[:, :, i] @ S
            o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
            o[:, :, i] = o_inter + attn_i @ v_new
            S = S * decay[:, :, i, -1, None, None].exp() + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2) @ v_new
        o = o.reshape(B, H, N * BT, V)[:, :, :T]
        o = o.transpose(1, 2).contiguous()
    return o, (S if output_final_state else None)


class GDNBlock(nn.Module):
    """GDN mixer 块（pre-norm 与 residual 在模型层处理）。

    forward(x, state) → (out, new_state)；state 含 recurrent state 与三个卷积 cache，
    用于推理时逐 token 生成。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_v_heads = cfg.n_v_heads
        self.n_qk_heads = cfg.n_qk_heads
        self.head_dim = cfg.head_dim
        self.key_dim = self.n_qk_heads * self.head_dim
        self.value_dim = self.n_v_heads * self.head_dim
        self.conv_kernel = cfg.conv_kernel
        assert self.n_v_heads % self.n_qk_heads == 0, "GVA 要求 n_v_heads 整除 n_qk_heads"

        self.q_proj = nn.Linear(d, self.key_dim, bias=False)
        self.k_proj = nn.Linear(d, self.key_dim, bias=False)
        self.v_proj = nn.Linear(d, self.value_dim, bias=False)
        self.a_proj = nn.Linear(d, self.n_v_heads, bias=False)
        self.b_proj = nn.Linear(d, self.n_v_heads, bias=False)
        # decay 参数化（对齐 fla / Qwen3-Next 初始化）
        A = torch.empty(self.n_v_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        dt = torch.exp(
            torch.rand(self.n_v_heads) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        dt = torch.clamp(dt, min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # softplus 逆
        # 短因果 depthwise Conv1d（kernel=4，无 bias，SiLU）
        self.q_conv = nn.Conv1d(self.key_dim, self.key_dim, self.conv_kernel, groups=self.key_dim, bias=False)
        self.k_conv = nn.Conv1d(self.key_dim, self.key_dim, self.conv_kernel, groups=self.key_dim, bias=False)
        self.v_conv = nn.Conv1d(self.value_dim, self.value_dim, self.conv_kernel, groups=self.value_dim, bias=False)
        self.g_proj = nn.Linear(d, self.value_dim, bias=False)
        self.o_norm = RMSNormGated(self.head_dim, eps=cfg.rms_eps)
        self.o_proj = nn.Linear(self.value_dim, d, bias=False)

    def _causal_conv(
        self, conv: nn.Conv1d, x: torch.Tensor, cache: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """短因果卷积 + SiLU；cache 为 [B, C, W-1] 的原始输入历史。"""
        xt = x.transpose(1, 2)  # [B,C,T]
        if cache is None:
            xt = F.pad(xt, (self.conv_kernel - 1, 0))
        else:
            xt = torch.cat([cache, xt], dim=-1)
        new_cache = xt[..., -(self.conv_kernel - 1) :].contiguous()
        y = F.conv1d(xt, conv.weight, groups=conv.weight.shape[0])
        return F.silu(y.transpose(1, 2)), new_cache

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
        q, k = l2norm(q), l2norm(k)
        if self.n_qk_heads != self.n_v_heads:  # GVA：qk 头重复到 v 头数
            rep = self.n_v_heads // self.n_qk_heads
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)

        beta = self.b_proj(x).sigmoid()  # 写入强度 [B,T,Hv]
        a = self.a_proj(x)
        g = -self.A_log.exp() * F.softplus(a.float() + self.dt_bias)  # 对数衰减 [B,T,Hv]

        rec = state["recurrent"] if state else None
        if T == 1 and not self.training:
            o, new_rec = naive_recurrent_gated_delta_rule(
                q, k, v, beta, g, initial_state=rec, output_final_state=True
            )
        else:
            o, new_rec = chunked_gated_delta_rule(
                q, k, v, beta, g, initial_state=rec, output_final_state=True
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
