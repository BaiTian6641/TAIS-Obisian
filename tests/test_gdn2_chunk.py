"""GDN-2 chunked（WY 训练路径）对拍测试：chunked vs naive + tied 退化 GDN-1。

判据（gdn2.chunked_gated_delta_rule_2 / 官方 NVlabs chunk_gdn2）：
- chunked 与 naive 逐 token 输出/末状态一致（fp32 <1e-4，含非整块 T 与初始状态）；
- tied 退化（b=w=β）时 GDN-2 chunked 与 GDN-1 chunked 输出一致（严格一般化）；
- 生成路径（T=1 naive）与训练路径（chunked）一致（推理/训练同一递归）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tais_obsidian.model.gdn import chunked_gated_delta_rule
from tais_obsidian.model.gdn2 import (
    chunked_gated_delta_rule_2,
    naive_recurrent_gated_delta_rule_2,
    tied_to_decoupled,
)

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _mk(T, B=2, H=2, K=16, V=16, seed=0):
    gen = torch.Generator("cpu").manual_seed(seed)
    q = torch.randn(B, T, H, K, generator=gen)
    k = F.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    v = torch.randn(B, T, H, V, generator=gen)
    b = torch.rand(B, T, H, K, generator=gen).sigmoid()
    w = torch.rand(B, T, H, V, generator=gen)
    g = -torch.rand(B, T, H, generator=gen) * 0.5
    return (x.to(DEV) for x in (q, k, v, b, w, g))


def test_chunked_matches_naive_no_state() -> None:
    q, k, v, b, w, g = _mk(130)  # 非整块
    o1, s1 = naive_recurrent_gated_delta_rule_2(q, k, v, b, w, g, output_final_state=True)
    o2, s2 = chunked_gated_delta_rule_2(q, k, v, b, w, g, output_final_state=True)
    assert (o1 - o2).abs().max() < 1e-4, f"out diff {(o1-o2).abs().max():.2e}"
    assert (s1 - s2).abs().max() < 1e-4, f"state diff {(s1-s2).abs().max():.2e}"


def test_chunked_matches_naive_with_state() -> None:
    q, k, v, b, w, g = _mk(96)
    st = (torch.randn(2, 2, 16, 16) * 0.1).to(DEV)
    o1, s1 = naive_recurrent_gated_delta_rule_2(q, k, v, b, w, g, initial_state=st, output_final_state=True)
    o2, s2 = chunked_gated_delta_rule_2(q, k, v, b, w, g, initial_state=st, output_final_state=True)
    assert (o1 - o2).abs().max() < 1e-4
    assert (s1 - s2).abs().max() < 1e-4


def test_tied_degenerates_to_gdn1_chunked() -> None:
    q, k, v, b, w, g = _mk(130)
    beta = torch.rand(2, 130, 2, generator=torch.Generator("cpu").manual_seed(3)).sigmoid().to(DEV)
    bt, wt = tied_to_decoupled(beta, 16, 16)
    o1, _ = chunked_gated_delta_rule(q, k, v, beta, g)
    o2, _ = chunked_gated_delta_rule_2(q, k, v, bt, wt, g)
    assert (o1 - o2).abs().max() < 1e-4, "tied 退化应等于 GDN-1 chunked"


def test_generation_path_matches_training_path() -> None:
    # 逐 token naive（生成路径，T=1 循环）与一次性 chunked（训练路径）一致
    q, k, v, b, w, g = _mk(64)
    o_full, _ = chunked_gated_delta_rule_2(q, k, v, b, w, g)
    # 逐 token 喂 naive（模拟自回归生成）
    S = None
    outs = []
    for t in range(64):
        o_t, S = naive_recurrent_gated_delta_rule_2(
            q[:, t:t+1], k[:, t:t+1], v[:, t:t+1], b[:, t:t+1], w[:, t:t+1], g[:, t:t+1],
            initial_state=S, output_final_state=True)
        outs.append(o_t)
    o_seq = torch.cat(outs, dim=1)
    assert (o_full - o_seq).abs().max() < 1e-4, "训练(chunked)与生成(naive 逐 token)路径应一致"
