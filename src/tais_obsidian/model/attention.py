"""CSA 全注意力块：RoPE + GQA + QK-Norm + PyTorch SDPA causal，无 bias，pre-norm 在模型层。

forward 支持 KV cache：state={"k","v"} 为 [B, n_kv, T_past, head_dim]（RoPE 后的 k）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .common import RMSNorm


class CSAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_q = cfg.n_q_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        assert self.n_q % self.n_kv == 0
        self.q_proj = nn.Linear(d, self.n_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.head_dim, d, bias=False)
        # QK-Norm：按 head_dim 归一
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        # RoPE 缓存 [max_seq, head_dim/2]
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        t = torch.arange(cfg.max_seq).float()
        freqs = torch.outer(t, inv_freq)
        self.register_buffer("rope_cos", freqs.cos(), persistent=False)
        self.register_buffer("rope_sin", freqs.sin(), persistent=False)

    def _rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        # x: [B, T, H, D]，half-split（NeoX 风格）旋转
        T = x.shape[1]
        cos = self.rope_cos[offset : offset + T]  # [T, D/2]
        sin = self.rope_sin[offset : offset + T]
        cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
        sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * cos + rot * sin

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        offset: int = 0,
    ) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_q, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv, self.head_dim)
        q = self._rope(self.q_norm(q), offset)
        k = self._rope(self.k_norm(k), offset)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # [B, H, T, D]
        if state is not None:
            k = torch.cat([state["k"], k], dim=2)
            v = torch.cat([state["v"], v], dim=2)
        new_state = {"k": k, "v": v}
        # GQA：展开 kv 头到 q 头数
        rep = self.n_q // self.n_kv
        k_e = k.repeat_interleave(rep, dim=1)
        v_e = v.repeat_interleave(rep, dim=1)
        Tq, Tk = q.shape[2], k_e.shape[2]
        if Tq == Tk:
            o = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=True)
        elif Tq == 1:
            o = F.scaled_dot_product_attention(q, k_e, v_e, is_causal=False)
        else:
            # 带 cache 的多 token 前向：构造相对新段右对齐的因果 mask
            i = torch.arange(Tq, device=x.device)[:, None]
            j = torch.arange(Tk, device=x.device)[None, :]
            mask = j <= i + (Tk - Tq)
            o = F.scaled_dot_product_attention(q, k_e, v_e, attn_mask=mask)
        o = o.transpose(1, 2).reshape(B, T, self.n_q * self.head_dim)
        return self.o_proj(o), new_state
