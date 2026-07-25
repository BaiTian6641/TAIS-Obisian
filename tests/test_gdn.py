"""GDN 核心算子单元测试：naive_recurrent vs chunked 对拍（fp32，max abs diff < 1e-4）。

用法：python tests/test_gdn.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tais_obsidian.model.gdn import chunked_gated_delta_rule, naive_recurrent_gated_delta_rule


def make_inputs(device: str, T: int = 128, seed: int = 0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    B, H, K, V = 2, 12, 64, 64
    q = torch.randn(B, T, H, K, generator=g)
    k = torch.nn.functional.normalize(torch.randn(B, T, H, K, generator=g), dim=-1)
    v = torch.randn(B, T, H, V, generator=g)
    beta = torch.rand(B, T, H, generator=g).sigmoid()
    # 对数衰减：负值，模拟 -exp(A_log)*softplus(...) 的量级
    g_log = -torch.rand(B, T, H, generator=g) * 0.5
    state = torch.randn(B, H, K, V, generator=g) * 0.1
    return (x.to(device) for x in (q, k, v, beta, g_log, state))


def run_case(device: str) -> None:
    q, k, v, beta, g_log, state = make_inputs(device)
    o1, s1 = naive_recurrent_gated_delta_rule(q, k, v, beta, g_log, output_final_state=True)
    o2, s2 = chunked_gated_delta_rule(q, k, v, beta, g_log, output_final_state=True)
    d_out = (o1 - o2).abs().max().item()
    d_state = (s1 - s2).abs().max().item()
    print(f"[{device}] 无初始状态: out diff {d_out:.2e}, state diff {d_state:.2e}")
    assert d_out < 1e-4 and d_state < 1e-4

    # 带初始状态
    o1, s1 = naive_recurrent_gated_delta_rule(q, k, v, beta, g_log, initial_state=state, output_final_state=True)
    o2, s2 = chunked_gated_delta_rule(q, k, v, beta, g_log, initial_state=state, output_final_state=True)
    d_out = (o1 - o2).abs().max().item()
    d_state = (s1 - s2).abs().max().item()
    print(f"[{device}] 带初始状态: out diff {d_out:.2e}, state diff {d_state:.2e}")
    assert d_out < 1e-4 and d_state < 1e-4

    # 状态连续性：整段 vs 两段（第二段以第一段 final state 为初始状态）
    o_full, s_full = chunked_gated_delta_rule(q, k, v, beta, g_log, output_final_state=True)
    h = 64
    oa, sa = chunked_gated_delta_rule(q[:, :h], k[:, :h], v[:, :h], beta[:, :h], g_log[:, :h], output_final_state=True)
    ob, sb = chunked_gated_delta_rule(
        q[:, h:], k[:, h:], v[:, h:], beta[:, h:], g_log[:, h:], initial_state=sa, output_final_state=True
    )
    o_cat = torch.cat([oa, ob], dim=1)
    d_out = (o_cat - o_full).abs().max().item()
    d_state = (sb - s_full).abs().max().item()
    print(f"[{device}] 两段拼接 vs 整段: out diff {d_out:.2e}, state diff {d_state:.2e}")
    assert d_out < 1e-4 and d_state < 1e-4
    # 非整 chunk 长度（T=100，验证 padding 路径）
    q3, k3, v3, b3, g3, _ = make_inputs(device, T=100, seed=7)
    o1, s1 = naive_recurrent_gated_delta_rule(q3, k3, v3, b3, g3, output_final_state=True)
    o2, s2 = chunked_gated_delta_rule(q3, k3, v3, b3, g3, output_final_state=True)
    d = max((o1 - o2).abs().max().item(), (s1 - s2).abs().max().item())
    print(f"[{device}] T=100 非整 chunk: diff {d:.2e}")
    assert d < 1e-4


def main() -> None:
    run_case("cpu")
    if torch.cuda.is_available():
        run_case("cuda")
    print("test_gdn 全部通过。")


def test_gated_delta_rule_parity() -> None:
    """pytest 收集入口：与 main() 等价。"""
    main()


if __name__ == "__main__":
    main()
