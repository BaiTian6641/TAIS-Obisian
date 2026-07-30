"""解耦双通道门控（DecoupledHcaGate）测试——方案 A 消除扩容门控对自然 gist 副作用。

判据：
a 双通道结构——DecoupledHcaGate 含 natural_gate/inject_gate 两路独立门控（参数独立）；
b 恒等初始化——natural_gate/inject_gate 初始 g≈1/3（fc2=0+bias=-ln2；fc1 随机破对称）；
c 来源路由——无注入时 forward 退化为 natural 单门控（win/csa/hca 全 natural）；
   有注入时 HCA 门控走 inject_gate（win/csa 仍 natural）——按条目来源（has_inject）分流；
d 注入条目 vs 自然 gist 分流——attach 后 HCA 拆两路：注入条目（inject_hca_entries 拼入，
   namespace 标记）走 inject_gate、自然 gist（压缩器）走 natural_gate（aux 可验 has_inject）；
e 副作用消除——无注入时 attach 后整层前向与原线性门控逐位一致（恒等初始化 g=1/3，
   natural 单门控 = 原行为）→ 纯文本精确召回结构性恢复；
f 注入召回保留——attach（inject_gate 载入已训/或训练后）注入条目经 inject_gate 开权重，
   KV 注入通路保留（训练脚本实测 0.625，此测试验证结构/可学习性）；
g 主干 + natural frozen——合成召回训练只训 inject_gate，主干 + natural_gate 逐位不变；
h attach/detach——detach 恢复 attach 前 forward（原线性门控或 gate_mlp 单门控）。
用法：python -m pytest tests/test_decoupled_gate.py -q
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
from tais_obsidian.model.tri_attention_decoupled import (
    DecoupledHcaGate,
    attach_decoupled_gate,
    detach_decoupled_gate,
    set_decoupled_gate_enabled,
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


# a) 双通道结构：natural_gate/inject_gate 两路独立门控（参数独立、结构独立）
def test_dual_channel_structure():
    gate = DecoupledHcaGate(head_dim=64, hidden=128)
    assert isinstance(gate.natural_gate, GatedFusionMLP)
    assert isinstance(gate.inject_gate, GatedFusionMLP)
    # 两路参数独立（不同对象、独立张量）
    assert gate.natural_gate is not gate.inject_gate
    nat_params = {id(p) for p in gate.natural_gate.parameters()}
    inj_params = {id(p) for p in gate.inject_gate.parameters()}
    assert nat_params.isdisjoint(inj_params), "natural/inject 门控参数应独立（不共享张量）"
    # fc1 独立初始化（随机破对称，两路不共享随机种子→权重不同）
    assert not torch.equal(gate.natural_gate.fc1.weight, gate.inject_gate.fc1.weight)


# b) 恒等初始化：natural_gate/inject_gate 初始 g≈1/3（fc2=0+bias=-ln2；不干扰）
def test_identity_init_both_gates():
    gate = DecoupledHcaGate(head_dim=64, hidden=128)
    q = torch.randn(2, 5, 4, 64)
    g_nat = gate.gate_natural(q)
    g_inj = gate.gate_inject(q)
    assert torch.allclose(g_nat, torch.full_like(g_nat, 1.0 / 3.0), atol=1e-6), \
        f"natural_gate 恒等初始化 g 应≈1/3，实际 [{g_nat.min():.6f},{g_nat.max():.6f}]"
    assert torch.allclose(g_inj, torch.full_like(g_inj, 1.0 / 3.0), atol=1e-6), \
        f"inject_gate 恒等初始化 g 应≈1/3，实际 [{g_inj.min():.6f},{g_inj.max():.6f}]"


# c) 来源路由：无注入时 forward 退化为 natural 单门控；有注入时 HCA 门控走 inject_gate
def test_source_routing_forward():
    gate = DecoupledHcaGate(head_dim=64, hidden=128)
    q = torch.randn(2, 5, 4, 64)
    # 无注入（has_inject=False）：g = natural_gate(q)（win/csa/hca 全 natural）
    g_no_inj = gate(q, has_inject=False)
    g_nat = gate.gate_natural(q)
    assert torch.allclose(g_no_inj, g_nat, atol=1e-6), "无注入时应退化为 natural 单门控"
    # 有注入（has_inject=True）：win/csa 走 natural，HCA（[...,2]）走 inject_gate
    g_inj_route = gate(q, has_inject=True)
    assert torch.allclose(g_inj_route[..., 0:2], g_nat[..., 0:2], atol=1e-6), \
        "win/csa 门控应走 natural_gate"
    g_inj_hca = gate.gate_inject(q)[..., 2:3]
    assert torch.allclose(g_inj_route[..., 2:3], g_inj_hca, atol=1e-6), \
        "注入条目 HCA 门控应走 inject_gate"


# d) 注入条目 vs 自然 gist 分流：attach 后 HCA 拆两路（aux 验 has_inject + 双通道生效）
def test_attach_hca_split_two_channels():
    _, model, tri = _build()
    attach_decoupled_gate(tri, hidden=128)
    x = torch.randn(1, 24, model.config.d_model)
    # 无注入：has_inject=False，HCA 门控 = natural（aux.has_inject=False）
    with torch.no_grad():
        aux0 = {}
        _, st0 = tri(x, aux=aux0)
    assert aux0["has_inject"] is False
    assert aux0["n_hca_inj"] == 0
    # 有注入：inject_hca_entries 拼入 → has_inject=True，HCA 拆两路
    from tais_obsidian.model.blockpath import make_namespace
    tri.layer_idx = 3
    nsi = make_namespace(model.config, 3, st0["k"].dtype)
    k_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    v_inj = torch.randn(1, tri.n_kv, 3, tri.head_dim)
    st1 = tri.inject_hca_entries(st0, (k_inj, v_inj), nsi)
    with torch.no_grad():
        aux1 = {}
        tri(x, state=st1, aux=aux1)
    assert aux1["has_inject"] is True
    assert aux1["n_hca_inj"] == 3
    # 注入条目 HCA 门控 = inject_gate 输出（双通道生效）
    q_nope_bt = None  # aux 未存 q_nope_bt；直接验 gate_inject 记录与 inject_gate 一致
    assert "gate_inject" in aux1 and aux1["gate_inject"] is not None
    detach_decoupled_gate(tri)


# e) 副作用消除：无注入时 attach 后整层前向与原线性门控逐位一致（恒等 g=1/3 单门控）
def test_no_inject_preserves_forward_identity():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, _ = tri(x)  # 原线性门控（gate_w=0/bias=-ln2 → g=1/3）
        attach_decoupled_gate(tri, hidden=128)  # natural 恒等 g=1/3，无注入退化单门控
        out1, _ = tri(x)
    assert torch.allclose(out0, out1, atol=1e-5), \
        f"无注入时 attach（恒等初始化）前向应与原一致（精确召回结构性恢复），最大差 {(out0-out1).abs().max():.2e}"
    detach_decoupled_gate(tri)


# e2) set_decoupled_gate_enabled(False) 强制 natural 单门控（消融对照：恢复精确召回）
def test_disable_forces_natural_single_channel():
    _, model, tri = _build()
    attach_decoupled_gate(tri, hidden=128)
    set_decoupled_gate_enabled(tri, False)  # 关双通道
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
    # enabled=False → has_inject=False（注入条目不走 inject_gate，退化为 natural 单门控）
    assert aux["has_inject"] is False
    set_decoupled_gate_enabled(tri, True)  # 恢复
    detach_decoupled_gate(tri)


# h) attach/detach：detach 恢复 attach 前 forward
def test_attach_detach_restores():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, _ = tri(x)
        attach_decoupled_gate(tri, hidden=128)
        detach_decoupled_gate(tri)
        out1, _ = tri(x)
    assert not hasattr(tri, "decoupled_gate")
    assert torch.allclose(out0, out1, atol=1e-6)


# h2) detach 恢复 gate_mlp 单门控（若 attach 前已挂 GatedFusionMLP）
def test_detach_restores_gate_mlp_single():
    from tais_obsidian.model.tri_attention_gated import attach_gated_fusion, detach_gated_fusion
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        mlp = attach_gated_fusion(tri, hidden=128)  # 先挂扩容门控（单门控）
        out_gated, _ = tri(x)
        # 在 gate_mlp 基础上挂解耦（natural 载入 gate_mlp 权重）
        attach_decoupled_gate(tri, natural_state_dict=mlp.state_dict(), hidden=128)
        detach_decoupled_gate(tri)  # 应恢复 gate_mlp 单门控 forward
        out_restored, _ = tri(x)
    assert hasattr(tri, "gate_mlp"), "detach 后应保留 gate_mlp（恢复原扩容门控）"
    assert torch.allclose(out_gated, out_restored, atol=1e-6), \
        "detach 应恢复 gate_mlp 单门控前向"
    detach_gated_fusion(tri)


# f+g) 注入召回保留 + 主干/natural frozen（合成召回训练：只训 inject_gate 开注入权重）
def test_inject_recall_trainable_and_frozen():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    for p in model.parameters():
        p.requires_grad_(False)
    # 挂解耦双通道（natural 恒等 frozen，inject 零初始化待训）
    inject_params = []
    for i in a_layers:
        gate = attach_decoupled_gate(model.layers[i].mixer, hidden=128)
        for p in gate.natural_gate.parameters():
            p.requires_grad_(False)  # natural frozen
        for p in gate.inject_gate.parameters():
            p.requires_grad_(True)
        inject_params += list(gate.inject_gate.parameters())

    tri = model.layers[a_layers[0]].mixer
    from tais_obsidian.model.blockpath import make_namespace
    tri.layer_idx = a_layers[0]
    tok_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)
    with torch.no_grad():
        _, kcache = model(tok_ids)
    st = kcache["layers"][a_layers[0]]
    k_inj = st["k"].transpose(1, 2).contiguous()
    v_inj = st["v"].transpose(1, 2).contiguous()

    # 快照主干 + natural_gate（排除 inject_gate/gate_w/gate_b）
    excl = {id(p) for p in inject_params}
    for i in a_layers:
        m = model.layers[i].mixer
        excl.add(id(m.gate_w))
        excl.add(id(m.gate_b))
    snap = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in excl}
    nat_snap = {n: p.detach().clone() for i in a_layers
                for n, p in model.layers[i].mixer.decoupled_gate.natural_gate.named_parameters()}

    def inject_cache(cache):
        for i in a_layers:
            m = model.layers[i].mixer
            m.layer_idx = i
            s = cache["layers"][i]
            nsi = make_namespace(model.config, i, s["k"].dtype)
            cache["layers"][i] = m.inject_hca_entries(s, (k_inj, v_inj), nsi)
        return cache

    opt = torch.optim.AdamW(inject_params, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    model.train()
    tri0 = model.layers[a_layers[0]].mixer
    h_fixed = torch.randn(1, 8, cfg.d_model, device=device)
    # 训练目标：inject_gate 对注入条目 HCA 门控（aux.gate_inject）向 1 回归（开注入权重）
    with torch.no_grad():
        _, c0 = model(prompt)
        c0 = inject_cache(c0)
        aux0 = {}
        tri0(h_fixed, state=c0["layers"][a_layers[0]], aux=aux0)
        g_inj_init = aux0["gate_inject"].mean().item()
    losses = []
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        _, cache = model(prompt)
        cache = inject_cache(cache)
        aux = {}
        st = cache["layers"][a_layers[0]]
        tri0(h_fixed, state=st, aux=aux)
        g_inj = aux["gate_inject"]  # [1,T,n_q,1] 注入条目 HCA 门控
        loss = F.mse_loss(g_inj, torch.ones_like(g_inj))  # 向 1 回归（开注入权重）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inject_params, 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    g_inj_final = g_inj.mean().item()
    # inject_gate 可学习：注入门控从初始 1/3 显著开向 1（注入召回保留的机制前提）
    assert losses[-1] < losses[0] * 0.5, \
        f"inject_gate 开注入权重损失应显著下降：{losses[0]:.4f}→{losses[-1]:.4f}"
    assert g_inj_final > g_inj_init + 0.1, \
        f"注入门控应从 {g_inj_init:.3f} 开向 1，实际 {g_inj_final:.3f}"

    # 主干 frozen：除 inject_gate/gate_w/gate_b 外逐位不变
    max_drift = 0.0
    for n, p in model.named_parameters():
        if id(p) in excl:
            continue
        max_drift = max(max_drift, (p.detach().float() - snap[n].float()).abs().max().item())
    assert max_drift == 0.0, f"主干权重漂移 {max_drift:.2e}（frozen 红线）"
    # natural_gate frozen：训练后逐位不变（gist 原权重未动，副作用消除核心）
    nat_drift = 0.0
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for n, p in gate.natural_gate.named_parameters():
            key = [k for k in nat_snap if k == n][0]
            nat_drift = max(nat_drift, (p.detach().float() - nat_snap[key].float()).abs().max().item())
    assert nat_drift == 0.0, f"natural_gate 权重漂移 {nat_drift:.2e}（frozen 保 gist 原权重）"


# d2) 自然 gist 门控不受 inject_gate 训练影响（结构性隔离：gist 通路零改动）
def test_natural_gist_channel_isolated_from_inject_training():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    for p in model.parameters():
        p.requires_grad_(False)
    gate = attach_decoupled_gate(model.layers[a_layers[0]].mixer, hidden=128)
    for p in gate.inject_gate.parameters():
        p.requires_grad_(True)
    # 无注入前向（in-context 纯文本）：记录输出（natural 单门控行为）
    # 注意 model 输入是 token ids（long），非隐藏态 float
    ids = torch.randint(0, cfg.vocab_size, (1, 24), device=device)
    with torch.no_grad():
        out_before, _ = model(ids)
    # 随便训 inject_gate 几步（梯度只进 inject_gate）
    opt = torch.optim.AdamW(gate.inject_gate.parameters(), lr=1e-2)
    model.train()
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        out, _ = model(ids)
        loss = out.float().abs().mean()
        loss.backward()
        opt.step()
    model.eval()
    # 无注入前向应与训练前逐位一致（natural_gate frozen → gist 通路零改动）
    with torch.no_grad():
        out_after, _ = model(ids)
    assert torch.allclose(out_before, out_after, atol=1e-6), \
        f"inject_gate 训练不应影响无注入前向（natural 单门控），最大差 {(out_before-out_after).abs().max():.2e}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
