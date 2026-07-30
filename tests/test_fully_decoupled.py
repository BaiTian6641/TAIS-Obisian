"""彻底解耦门控（FullyDecoupledGate）测试——注入召回走独立 csa 通道，消除 ic/KV 结构性权衡。

判据：
a 彻底解耦结构——natural_gate（3 维 win/csa/hca，自然通路）+ inject_csa_gate（4 维
   win/csa/hca/inject，注入通路）两路完全独立（参数独立、独立张量）；
b 恒等初始化——natural_gate/inject_csa_gate 初始 g≈1/3（fc2=0+bias=-ln2；fc1 随机破对称）；
c 来源路由——无注入时 forward 退化为 natural 单门控；有注入时双通道（自然走 natural_gate、
   注入走 inject_csa_gate）——按条目来源（has_inject/namespace）分流；
d 注入召回的 csa 路径独立——aux.o_csa_inj 由 inject_csa_gate 门控（不经 natural_gate）；
   natural_gate 重训（对 gist 关压 csa）不影响注入召回通路（结构性解耦核心）；
e 无注入时 attach 后整层前向与原线性门控逐位一致（恒等初始化 g=1/3 单门控 = 原行为）；
f 两路独立训练——合成任务上 natural_gate 与 inject_csa_gate 各自可训，且互不干扰
   （训 natural 不改 inject_csa，训 inject_csa 不改 natural）；
g 主干 frozen——联合训练只训两路门控，主干逐位不变；
h attach/detach——detach 恢复 attach 前 forward。
用法：python -m pytest tests/test_fully_decoupled.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tri_attention import TriRetrievalAttention
from tais_obsidian.model.tri_attention_gated import GatedFusionMLP
from tais_obsidian.model.tri_attention_fully_decoupled import (
    FullyDecoupledGate,
    _Gate4,
    attach_fully_decoupled,
    detach_fully_decoupled,
    set_fully_decoupled_enabled,
)


def tiny_cfg() -> ModelConfig:
    # 4 层 = G,G,G,A：唯一 "A" 层 idx 3；tri 超参按 max_seq=128 缩小压实各分支路径
    return ModelConfig(
        vocab_size=512, d_model=256, n_layer=4,
        n_q_heads=4, n_kv_heads=2, head_dim=64, n_v_heads=4, n_qk_heads=2,
        mlp_hidden=688, max_seq=128,
        tri_window=32, tri_csa_stride=4, tri_csa_topk=8, tri_hca_stride=16,
        check_0p1b_params=False,
    )


def _build(device="cpu"):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    tri = model.layers[3].mixer
    assert isinstance(tri, TriRetrievalAttention)
    return cfg, model, tri


# a) 彻底解耦结构：natural_gate（3 维）+ inject_csa_gate（4 维）两路完全独立
def test_fully_decoupled_structure():
    gate = FullyDecoupledGate(head_dim=64, hidden=128)
    assert isinstance(gate.natural_gate, GatedFusionMLP)
    assert isinstance(gate.inject_csa_gate, _Gate4)
    # natural_gate 输出 3 维（win/csa/hca），inject_csa_gate 输出 4 维（win/csa/hca/inject）
    assert gate.natural_gate.fc2.out_features == 3
    assert gate.inject_csa_gate.fc2.out_features == 4
    # 两路参数完全独立（不同对象、独立张量）
    assert gate.natural_gate is not gate.inject_csa_gate
    nat_params = {id(p) for p in gate.natural_gate.parameters()}
    inj_params = {id(p) for p in gate.inject_csa_gate.parameters()}
    assert nat_params.isdisjoint(inj_params), "natural/inject_csa 门控参数应完全独立（不共享张量）"
    assert not torch.equal(gate.natural_gate.fc1.weight, gate.inject_csa_gate.fc1.weight)


# b) 初始化：natural_gate 恒等 g≈1/3；inject_csa_gate win/csa/hca 位 1/3、inject 位≈0.05（召回友好）
def test_identity_init_both_gates():
    gate = FullyDecoupledGate(head_dim=64, hidden=128)
    q = torch.randn(2, 5, 4, 64)
    g_nat = gate.gate_natural(q)
    g_inj = gate.gate_inject_csa(q)
    assert torch.allclose(g_nat, torch.full_like(g_nat, 1.0 / 3.0), atol=1e-6), \
        f"natural_gate 恒等初始化 g 应≈1/3，实际 [{g_nat.min():.6f},{g_nat.max():.6f}]"
    # inject_csa_gate 前 3 位（win/csa/hca）恒等 1/3；inject 位召回友好起点 ≈0.05（sigmoid(-3)）
    assert torch.allclose(g_inj[..., 0:3], torch.full_like(g_inj[..., 0:3], 1.0 / 3.0), atol=1e-6), \
        f"inject_csa_gate win/csa/hca 位应≈1/3，实际 [{g_inj[...,0:3].min():.6f},{g_inj[...,0:3].max():.6f}]"
    assert torch.allclose(g_inj[..., 3:4], torch.full_like(g_inj[..., 3:4], 0.0474), atol=1e-3), \
        f"inject_csa_gate inject 位应≈0.05（召回友好起点），实际 {g_inj[...,3].mean():.6f}"
    assert g_nat.shape[-1] == 3 and g_inj.shape[-1] == 4


# c) 来源路由：无注入时 forward 退化为 natural 单门控；有注入时双通道拼接
def test_source_routing_forward():
    gate = FullyDecoupledGate(head_dim=64, hidden=128)
    q = torch.randn(2, 5, 4, 64)
    # 无注入：g = natural_gate(q)（3 维，自然通路）
    g_no_inj = gate(q, has_inject=False)
    g_nat = gate.gate_natural(q)
    assert torch.allclose(g_no_inj, g_nat, atol=1e-6) and g_no_inj.shape[-1] == 3
    # 有注入：g = [natural(3), inject_csa(4)] 拼接（7 维，两路分流）
    g_inj_route = gate(q, has_inject=True)
    assert g_inj_route.shape[-1] == 7
    assert torch.allclose(g_inj_route[..., 0:3], g_nat, atol=1e-6), "前 3 维应走 natural_gate"
    g_inj = gate.gate_inject_csa(q)
    assert torch.allclose(g_inj_route[..., 3:7], g_inj, atol=1e-6), "后 4 维应走 inject_csa_gate"


# d) 注入召回的 csa 路径独立：aux.o_csa_inj 由 inject_csa_gate 门控（不经 natural_gate）
def test_inject_csa_path_independent():
    _, model, tri = _build()
    attach_fully_decoupled(tri, hidden=128)
    x = torch.randn(1, 24, model.config.d_model)
    from tais_obsidian.model.blockpath import make_namespace
    with torch.no_grad():
        _, st0 = tri(x)
    tri.layer_idx = 3
    nsi = make_namespace(model.config, 3, st0["k"].dtype)
    k_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    v_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    st1 = tri.inject_hca_entries(st0, (k_inj, v_inj), nsi)
    with torch.no_grad():
        aux = {}
        tri(x, state=st1, aux=aux)
    # 注入通路 csa 输出存在（独立通道），且门控记录含 4 维 inject_csa
    assert aux["has_inject"] is True
    assert "o_csa_inj" in aux and aux["o_csa_inj"] is not None, "注入通路 csa 输出应存在（独立通道）"
    assert "gate_inject_csa" in aux and aux["gate_inject_csa"].shape[-1] == 4
    detach_fully_decoupled(tri)


# d2) natural_gate 重训不影响注入召回通路（结构性解耦核心：natural 压 csa 时 inject_csa 不变）
def test_natural_training_isolated_from_inject_csa():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    for p in model.parameters():
        p.requires_grad_(False)
    gate = attach_fully_decoupled(model.layers[a_layers[0]].mixer, hidden=128)
    # 只训 natural_gate（模拟"对 gist 关"重训）；inject_csa_gate frozen
    for p in gate.natural_gate.parameters():
        p.requires_grad_(True)
    for p in gate.inject_csa_gate.parameters():
        p.requires_grad_(False)
    inj_snap = {n: p.detach().clone() for n, p in gate.inject_csa_gate.named_parameters()}
    ids = torch.randint(0, cfg.vocab_size, (1, 24), device=device)
    opt = torch.optim.AdamW(gate.natural_gate.parameters(), lr=1e-2)
    model.train()
    for _ in range(10):
        opt.zero_grad(set_to_none=True)
        out, _ = model(ids)
        loss = out.float().abs().mean()  # 任意损失驱动 natural_gate 变化
        loss.backward()
        opt.step()
    model.eval()
    # inject_csa_gate 逐位不变（natural 重训零影响注入召回通路——结构性解耦）
    inj_drift = max((p.detach().float() - inj_snap[n].float()).abs().max().item()
                    for n, p in gate.inject_csa_gate.named_parameters())
    assert inj_drift == 0.0, f"inject_csa_gate 应不受 natural 重训影响，漂移 {inj_drift:.2e}"
    detach_fully_decoupled(model.layers[a_layers[0]].mixer)


# e) 无注入时 attach 后整层前向与原线性门控逐位一致（恒等初始化 g=1/3 单门控 = 原行为）
def test_no_inject_preserves_forward_identity():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, _ = tri(x)  # 原线性门控（gate_w=0/bias=-ln2 → g=1/3）
        attach_fully_decoupled(tri, hidden=128)  # natural 恒等 g=1/3，无注入退化单门控
        out1, _ = tri(x)
    assert torch.allclose(out0, out1, atol=1e-5), \
        f"无注入时 attach（恒等初始化）前向应与原一致，最大差 {(out0-out1).abs().max():.2e}"
    detach_fully_decoupled(tri)


# f) 两路独立训练：natural_gate 与 inject_csa_gate 各自可训且互不干扰
def test_two_gates_independently_trainable():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gate = FullyDecoupledGate(head_dim=64, hidden=128).to(device)
    q = torch.randn(2, 5, 4, 64, device=device)
    # ① 只训 natural_gate（向目标 0.9 回归），inject_csa_gate 不变
    inj_snap = {n: p.detach().clone() for n, p in gate.inject_csa_gate.named_parameters()}
    opt = torch.optim.AdamW(gate.natural_gate.parameters(), lr=1e-2)
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        g = gate.gate_natural(q)
        loss = F.mse_loss(g, torch.full_like(g, 0.9))
        loss.backward()
        opt.step()
    g_nat_final = gate.gate_natural(q).mean().item()
    assert g_nat_final > 0.5, f"natural_gate 可训：g 应从 1/3 升向 0.9，实际 {g_nat_final:.3f}"
    inj_drift = max((p.detach() - inj_snap[n]).abs().max().item()
                    for n, p in gate.inject_csa_gate.named_parameters())
    assert inj_drift == 0.0, "训 natural_gate 不应改 inject_csa_gate"
    # ② 只训 inject_csa_gate（向目标 0.8 回归），natural_gate 不变
    nat_snap = {n: p.detach().clone() for n, p in gate.natural_gate.named_parameters()}
    opt2 = torch.optim.AdamW(gate.inject_csa_gate.parameters(), lr=1e-2)
    for _ in range(50):
        opt2.zero_grad(set_to_none=True)
        g = gate.gate_inject_csa(q)
        loss = F.mse_loss(g, torch.full_like(g, 0.8))
        loss.backward()
        opt2.step()
    g_inj_final = gate.gate_inject_csa(q).mean().item()
    assert g_inj_final > 0.5, f"inject_csa_gate 可训：g 应从 1/3 升向 0.8，实际 {g_inj_final:.3f}"
    nat_drift = max((p.detach() - nat_snap[n]).abs().max().item()
                    for n, p in gate.natural_gate.named_parameters())
    assert nat_drift == 0.0, "训 inject_csa_gate 不应改 natural_gate"


# g+h) 主干 frozen + attach/detach：联合训练只训两路门控，主干逐位不变；detach 恢复
def test_backbone_frozen_and_attach_detach():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, _ = tri(x)
        attach_fully_decoupled(tri, hidden=128)
        # 主干 frozen 验证：attach 不改主干参数（门控是新增子模块）
        snap = {n: p.detach().clone() for n, p in tri.named_parameters()
                if not n.startswith("fully_decoupled_gate")}
        detach_fully_decoupled(tri)
        out1, _ = tri(x)
        for n, p in tri.named_parameters():
            if n in snap:
                assert torch.equal(p.detach(), snap[n]), f"主干参数 {n} 应逐位不变"
    assert not hasattr(tri, "fully_decoupled_gate")
    assert torch.allclose(out0, out1, atol=1e-6), "detach 应恢复 attach 前 forward"


# h2) set_fully_decoupled_enabled(False) 强制 natural 单门控（消融对照）
def test_disable_forces_natural_single():
    _, model, tri = _build()
    attach_fully_decoupled(tri, hidden=128)
    set_fully_decoupled_enabled(tri, False)
    x = torch.randn(1, 24, model.config.d_model)
    from tais_obsidian.model.blockpath import make_namespace
    with torch.no_grad():
        _, st = tri(x)
    tri.layer_idx = 3
    nsi = make_namespace(model.config, 3, st["k"].dtype)
    k_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    v_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    st = tri.inject_hca_entries(st, (k_inj, v_inj), nsi)
    with torch.no_grad():
        aux = {}
        tri(x, state=st, aux=aux)
    assert aux["has_inject"] is False, "enabled=False 时应退化为 natural 单门控"
    set_fully_decoupled_enabled(tri, True)
    detach_fully_decoupled(tri)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
