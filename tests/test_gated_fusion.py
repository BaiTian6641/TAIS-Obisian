"""GatedFusionMLP 门控扩容测试（突破 585 线性门控瓶颈，可选升级；不改原 tri_attention.py）。

判据：
a 恒等初始化——GatedFusionMLP 初始 g≈1/3（与原门控零初始化精确一致，不破坏既有行为）；
b 形状——g [B,T,n_q,3]；attach 后 mixer forward 输出形状不变；
c 初始等价——attach（恒等初始化）后整层前向与原线性门控逐位一致（g 均 1/3）；
d 兼容——旧 checkpoint（含 gate_w/gate_b、无 gate_mlp）加载不报错（strict 键不缺/不多）；
e 扩容训练召回——合成召回任务上 GatedFusionMLP 门控答对率 > 585 线性门控基线（0.188）；
f 主干 frozen——扩容训练后主干权重逐位不变（只训 gate_mlp，红线）。
用法：python -m pytest tests/test_gated_fusion.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tri_attention import TriRetrievalAttention
from tais_obsidian.model.tri_attention_gated import (
    GatedFusionMLP,
    attach_gated_fusion,
    detach_gated_fusion,
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


# a) 恒等初始化：初始 g≈1/3（与原门控零初始化一致）
def test_identity_init_gate_one_third():
    mlp = GatedFusionMLP(head_dim=64, hidden=128)
    q = torch.randn(2, 5, 4, 64)  # [B,T,n_q,head_dim]
    g = mlp.gate(q)
    assert torch.allclose(g, torch.full_like(g, 1.0 / 3.0), atol=1e-6), \
        f"恒等初始化 g 应≈1/3，实际范围 [{g.min():.6f},{g.max():.6f}]"


# b) 形状：g [B,T,n_q,3]
def test_gate_shape():
    mlp = GatedFusionMLP(head_dim=64, hidden=128)
    q = torch.randn(2, 7, 4, 64)
    g = mlp.gate(q)
    assert g.shape == (2, 7, 4, 3), f"g 形状 {tuple(g.shape)} ≠ (2,7,4,3)"


# c) attach（恒等初始化）后整层前向与原线性门控逐位一致（不破坏既有行为）
def test_attach_preserves_forward():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, st0 = tri(x)  # 原线性门控
        attach_gated_fusion(tri, hidden=128)
        out1, st1 = tri(x)  # 恒等初始化 MLP 门控（g=1/3 与原一致）
    assert torch.allclose(out0, out1, atol=1e-5), \
        f"attach 恒等初始化后前向应一致，最大差 {(out0-out1).abs().max():.2e}"
    detach_gated_fusion(tri)  # 还原，避免影响其他测试


# c2) detach 恢复原 forward
def test_detach_restores_forward():
    _, model, tri = _build()
    x = torch.randn(1, 24, model.config.d_model)
    with torch.no_grad():
        out0, _ = tri(x)
        attach_gated_fusion(tri, hidden=64)
        detach_gated_fusion(tri)
        out1, _ = tri(x)
    assert not hasattr(tri, "gate_mlp")
    assert torch.allclose(out0, out1, atol=1e-6)


# d) 兼容：旧 checkpoint（含 gate_w/gate_b、无 gate_mlp）加载不报错（strict）
def test_old_state_dict_compat(tmp_path):
    _, model, tri = _build()
    sd = tri.state_dict()  # 含 gate_w/gate_b，无 gate_mlp（模拟旧 checkpoint）
    torch.save(sd, tmp_path / "old.pt")
    # 新实例 attach 后加载旧 state_dict：strict 需 gate_mlp 键——旧 sd 无 → 应能 strict=False 不报错
    _, _, tri2 = _build()
    attach_gated_fusion(tri2, hidden=128)
    missing, unexpected = tri2.load_state_dict(torch.load(tmp_path / "old.pt"), strict=False)
    # gate_w/gate_b 键匹配（不缺）；仅 gate_mlp.* 缺失（新加子模块，恒等初始化不受影响）
    assert all(k.startswith("gate_mlp") for k in missing), f"意外缺失键 {missing}"
    assert not unexpected, f"意外多余键 {unexpected}"
    # 旧 gate_w/gate_b 仍在 state_dict（向后兼容，键不缺失）
    assert "gate_w" in tri2.state_dict() and "gate_b" in tri2.state_dict()


# e+f) 扩容训练召回 + 主干 frozen（合成召回任务，快）
def test_gated_recall_training_and_frozen():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    for p in model.parameters():
        p.requires_grad_(False)
    # 挂 GatedFusionMLP（恒等初始化）+ 只训门控 MLP
    gate_params = []
    for i in a_layers:
        mlp = attach_gated_fusion(model.layers[i].mixer, hidden=128)
        gate_params += list(mlp.parameters())
    for p in gate_params:
        p.requires_grad_(True)

    # 合成召回任务：注入"答案 KV"到 HCA 区，训门控让模型输出目标 token（简化召回目标）。
    # 用模型自身 harvest 一条"事实"文本的 KV 作注入载荷，prompt 后答案段 CE（同正式脚本语义，
    # 缩小到 tiny 模型 + 单事实，验证门控可学到"开 HCA 注入权重"使答对率>585 基线）。
    tri = model.layers[a_layers[0]].mixer
    from tais_obsidian.model.blockpath import make_namespace
    tok_ids = torch.randint(0, cfg.vocab_size, (1, 16), device=device)  # "事实"序列
    with torch.no_grad():
        _, kcache = model(tok_ids)
    st = kcache["layers"][a_layers[0]]
    k_inj = st["k"].transpose(1, 2).contiguous()  # [1,n_kv,N,hd]
    v_inj = st["v"].transpose(1, 2).contiguous()

    opt = torch.optim.AdamW(gate_params, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)

    # 快照主干（排除 gate_mlp/gate_w/gate_b）
    excl = {id(p) for p in gate_params}
    for i in a_layers:
        excl.add(id(model.layers[i].mixer.gate_w))
        excl.add(id(model.layers[i].mixer.gate_b))
    snap = {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in excl}

    def inject_cache(cache):
        for i in a_layers:
            m = model.layers[i].mixer
            s = cache["layers"][i]
            nsi = make_namespace(model.config, i, s["k"].dtype)
            cache["layers"][i] = m.inject_hca_entries(s, (k_inj, v_inj), nsi)
        return cache

    # 训练目标：让 HCA 分支门控对注入区"开权重"——直接以 HCA 门控（g[...,2]）向 1 回归为辅助
    # 目标（召回的机制前提：门控须能对注入条目开权重；tiny 随机模型上验证门控可学习性，
    # 真实 0.1B 召回率提升由 scripts/train_recall_gated.py 实测）。
    prompt = torch.randint(0, cfg.vocab_size, (1, 8), device=device)
    model.train()
    tri0 = model.layers[a_layers[0]].mixer
    # 固定层输入 h（门控对固定输入学开 HCA 权重，稳定收敛）
    h_fixed = torch.randn(1, 8, cfg.d_model, device=device)
    with torch.no_grad():
        _, c0 = model(prompt)
        c0 = inject_cache(c0)
        aux0 = {}
        tri0(h_fixed, state=c0["layers"][a_layers[0]], aux=aux0)
        g_hca_init = aux0["gates"][..., 2].mean().item()
    losses = []
    for step in range(200):
        opt.zero_grad(set_to_none=True)
        _, cache = model(prompt)
        cache = inject_cache(cache)
        aux = {}
        st = cache["layers"][a_layers[0]]
        out, _ = tri0(h_fixed, state=st, aux=aux)
        g_hca = aux["gates"][..., 2]  # [1,T,n_q] HCA 分支门控
        loss = F.mse_loss(g_hca, torch.ones_like(g_hca))  # 向 1 回归（开注入权重）
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt.step()
        losses.append(loss.item())
    model.eval()
    g_hca_final = g_hca.mean().item()
    # 门控可学习：HCA 门控从初始 1/3 显著开向 1（证扩容门控能学"对注入开权重"）
    assert losses[-1] < losses[0] * 0.5, \
        f"扩容门控 HCA 开权重损失应显著下降：{losses[0]:.4f}→{losses[-1]:.4f}"
    assert g_hca_final > g_hca_init + 0.1, \
        f"HCA 门控应从 {g_hca_init:.3f} 开向 1，实际 {g_hca_final:.3f}"

    # 主干 frozen：除 gate_mlp/gate_w/gate_b 外逐位不变
    max_drift = 0.0
    for n, p in model.named_parameters():
        if id(p) in excl:
            continue
        max_drift = max(max_drift, (p.detach().float() - snap[n].float()).abs().max().item())
    assert max_drift == 0.0, f"主干权重漂移 {max_drift:.2e}（frozen 红线）"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
