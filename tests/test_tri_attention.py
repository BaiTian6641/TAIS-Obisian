"""三级注意力栈测试（E+-7，设计 §17；AGENT_PLAN_E+-7 §4.5 判据 a–g）。

a 形状 / b 因果性红线 / c top-k ⊆ 因果压缩集 / d 滑窗 vs 掩码全注意力参考 /
e HCA 注入放行与 fail-closed / f 增量 vs 整段（含注入后）/ g save-load 往返。
用法：python tests/test_tri_attention.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.attention import FullAttention
from tais_obsidian.model.blockpath import (
    NamespaceMismatchError,
    make_namespace,
)
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tri_attention import TriAttention


def tiny_cfg(attn_impl: str = "tri", attn_only: bool = False) -> ModelConfig:
    # 4 层 = G,G,G,A：唯一 "A" 层为 idx 3；tri 超参按 max_seq=128 缩小以压实各分支路径
    return ModelConfig(
        vocab_size=512,
        d_model=256,
        n_layer=4,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        n_v_heads=4,
        n_qk_heads=2,
        mlp_hidden=688,
        max_seq=128,
        attn_only=attn_only,
        attn_impl=attn_impl,
        tri_window=32,
        tri_csa_stride=4,
        tri_csa_topk=8,
        tri_hca_stride=16,
        check_0p1b_params=False,
    )


def _build(device: str, **kw):
    torch.manual_seed(0)
    cfg = tiny_cfg(**kw)
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    tri = model.layers[3].mixer
    assert isinstance(tri, TriAttention)
    return cfg, model, tri


def check_shapes(device: str) -> None:
    """a) 形状：三分支各自与融合输出形状正确（整段 T 与增量 T=1 两种）。"""
    cfg, model, tri = _build(device)
    B, T = 2, 48
    x = torch.randn(B, T, cfg.d_model, device=device)
    with torch.no_grad():
        aux: dict = {}
        out, st = tri(x, None, 0, aux)
        assert out.shape == (B, T, cfg.d_model), out.shape
        for key in ("o_win", "o_csa", "o_hca"):
            assert aux[key].shape == (B, cfg.n_q_heads, T, cfg.head_dim), (key, aux[key].shape)
        assert aux["gates"].shape == (B, T, cfg.n_q_heads, 3), aux["gates"].shape
        assert aux["n_csa"] == T // 4 and aux["n_hca"] == T // 16
        assert st["k"].shape == (B, T, cfg.n_kv_heads, cfg.head_dim), st["k"].shape
        # init 门控精确均等 1/3（零初始化权重 + bias=-ln2，记录见 tri_attention docstring）
        assert (aux["gates"] - 1 / 3).abs().max().item() < 1e-6
        # 增量 T=1
        aux1: dict = {}
        out1, st1 = tri(x[:, :1], st, T, aux1)
        assert out1.shape == (B, 1, cfg.d_model)
        for key in ("o_win", "o_csa", "o_hca"):
            assert aux1[key].shape == (B, cfg.n_q_heads, 1, cfg.head_dim)
        assert st1["k"].shape[1] == T + 1
    # 纪律：默认 attn_impl="full" 走 CSAAttention；attn_only=True + "tri" 仍全注意力
    torch.manual_seed(0)
    m_def = TaisObsidianForCausalLM(tiny_cfg(attn_impl="full")).to(device)
    assert isinstance(m_def.layers[3].mixer, FullAttention)
    torch.manual_seed(0)
    m_ao = TaisObsidianForCausalLM(tiny_cfg(attn_impl="tri", attn_only=True)).to(device)
    assert all(isinstance(b.mixer, FullAttention) for b in m_ao.layers)
    print("[a] 形状（整段/增量、分支/门控/融合）与 attn_only 纪律通过")


def check_causality(device: str) -> None:
    """b) 因果性红线：扰动 j 之后的 token，位置 ≤j 的输出逐点不变（三分支 + 门控 + 融合）。"""
    cfg, model, tri = _build(device)
    B, T, j = 2, 48, 29
    torch.manual_seed(1)
    x = torch.randn(B, T, cfg.d_model, device=device)
    x2 = x.clone()
    x2[:, j + 1 :] = torch.randn(B, T - j - 1, cfg.d_model, device=device)
    with torch.no_grad():
        aux1, aux2 = {}, {}
        out1, _ = tri(x, None, 0, aux1)
        out2, _ = tri(x2, None, 0, aux2)
    sl = slice(0, j + 1)
    for key in ("o_win", "o_csa", "o_hca"):
        d = (aux1[key][:, :, sl] - aux2[key][:, :, sl]).abs().max().item()
        assert d == 0.0, (key, d)  # 同形状同代码路径，fp32 应逐点一致
    d_g = (aux1["gates"][:, sl] - aux2["gates"][:, sl]).abs().max().item()
    d_o = (out1[:, sl] - out2[:, sl]).abs().max().item()
    assert d_g == 0.0 and d_o == 0.0, (d_g, d_o)
    # 被扰动位置之后的输出应当真的变了（防止空转通过）
    assert (out1[:, j + 1 :] - out2[:, j + 1 :]).abs().max().item() > 1e-3
    print(f"[b] 因果性红线通过（j={j}，≤j 三分支/门控/融合逐点不变，>j 确实变化）")


def check_selection_legality(device: str) -> None:
    """c) 选择合法性：top-k 索引全部落在因果压缩集合内；满 topk 时恰好取满。"""
    cfg, model, tri = _build(device)
    B, T = 2, 48
    x = torch.randn(B, T, cfg.d_model, device=device)
    with torch.no_grad():
        aux: dict = {}
        tri(x, None, 0, aux)
    keep, i_abs = aux["sel_keep"], aux["i_abs"]  # [B,n_kv,T,S], [T]
    S = keep.shape[-1]
    m = cfg.tri_csa_stride
    tail = m * (torch.arange(S, device=device) + 1) - 1
    vis = tail[None, :] < i_abs[:, None]  # [T,S]
    illegal = keep & ~vis[None, None]
    assert not illegal.any(), f"选中非法（非因果）条目 {illegal.sum()} 处"
    n_sel = keep.sum(dim=-1)  # [B,n_kv,T]
    n_vis = vis.sum(dim=-1)  # [T]
    want = torch.minimum(n_vis, torch.tensor(cfg.tri_csa_topk, device=device))
    assert (n_sel == want[None, None, :].expand_as(n_sel)).all(), (n_sel[0, 0], want)
    print(f"[c] top-{cfg.tri_csa_topk} 选择全部 ⊆ 因果压缩集，逐位置数量正确")


def check_window_reference(device: str) -> None:
    """d) 滑窗分支等价性：与"加掩码的全注意力"参考逐点一致（fp32 手算参考）。"""
    cfg, model, tri = _build(device)
    B, T = 2, 48
    torch.manual_seed(2)
    x = torch.randn(B, T, cfg.d_model, device=device)
    with torch.no_grad():
        aux: dict = {}
        tri(x, None, 0, aux)
    q, k, v = aux["q_rope"], aux["k_rope"], aux["v"]  # [B,n_q,T,D], [B,n_kv,T,D]×2
    rep = cfg.n_q_heads // cfg.n_kv_heads
    k_e = k.repeat_interleave(rep, dim=1)
    v_e = v.repeat_interleave(rep, dim=1)
    logits = (q @ k_e.transpose(-1, -2)) / math.sqrt(cfg.head_dim)
    i = torch.arange(T, device=device)[:, None]
    jj = torch.arange(T, device=device)[None, :]
    mask = (jj <= i) & (jj > i - cfg.tri_window)
    logits = logits.masked_fill(~mask[None, None], float("-inf"))
    ref = torch.softmax(logits, dim=-1) @ v_e
    d = (aux["o_win"] - ref).abs().max().item()
    print(f"[d] 滑窗 vs 掩码全注意力参考: max diff {d:.2e}")
    assert d < 1e-5, d


def check_hca_inject(device: str) -> None:
    """e) HCA 注入：namespace 全对放行 / 任一字段不匹配 fail-closed；注入区簿记正确。"""
    cfg, model, tri = _build(device)
    ids = torch.randint(0, 512, (1, 40), device=device)
    torch.manual_seed(3)
    entries = (torch.randn(1, cfg.n_kv_heads, 3, cfg.head_dim, device=device),) * 2
    ns = make_namespace(cfg, 3, entries[0].dtype)
    with torch.no_grad():
        _, cache = model(ids[:, :24])
        st = cache["layers"][3]
        pos_before = cache["pos"]
        st2 = tri.inject_hca_entries(st, entries, ns)  # 全对：放行
        assert st2["hca_inj_k"].shape == (1, cfg.n_kv_heads, 3, cfg.head_dim)
        assert "hca_inj_k" not in st, "inject 不得原地修改入参"
        # 再次注入：新条目前置，注入区长度累加
        st3 = tri.inject_hca_entries(st2, entries, ns)
        assert st3["hca_inj_k"].shape[2] == 6
        # 注入后前向：HCA 区长度 = 注入 6 + 压缩 floor(25/16)=1；pos 不变
        cache["layers"][3] = st3
        aux: dict = {}
        logits, cache = model(ids[:, 24:25], cache)
        assert logits.shape == (1, 1, cfg.vocab_size)
        assert cache["pos"] == pos_before + 1, "HCA 注入不占 token 位置槽"
        tri_aux: dict = {}
        tri(torch.randn(1, 5, cfg.d_model, device=device), st3, 25, tri_aux)
        assert tri_aux["n_hca_inj"] == 6
        # 逐字段篡改：fail-closed
        tampers = {
            "model_id": "d999-L9-h9-kv9-V9",
            "layer_idx": 0,
            "compressor_version": "csa-comp-v0.0",
            "dtype": "torch.float16",
            "rope_theta": 500000.0,
        }
        for field, bad in tampers.items():
            ns_bad = dict(ns)
            ns_bad[field] = bad
            try:
                tri.inject_hca_entries(st, entries, ns_bad)
            except NamespaceMismatchError:
                pass
            else:
                raise AssertionError(f"字段 {field} 不匹配未触发 fail-closed")
        # 形状不匹配同样 fail-closed
        bad_entries = (torch.randn(1, cfg.n_kv_heads + 1, 3, cfg.head_dim, device=device),) * 2
        try:
            tri.inject_hca_entries(st, bad_entries, ns)
        except NamespaceMismatchError:
            pass
        else:
            raise AssertionError("头数不匹配未触发 fail-closed")
    print("[e] HCA 注入：五元组全对放行、逐字段篡改/形状错误均 fail-closed、注入区簿记正确")


def check_incremental(device: str) -> None:
    """f) 增量 vs 整段一致性（<1e-4），含 HCA 注入后的多 token vs 逐 token。"""
    cfg, model, tri = _build(device)
    torch.manual_seed(4)
    ids = torch.randint(0, 512, (2, 40), device=device)
    with torch.no_grad():
        logits_full, _ = model(ids)
        logits_pre, cache = model(ids[:, :17])
        steps = [logits_pre]
        for t in range(17, 40):
            lg, cache = model(ids[:, t : t + 1], cache)
            steps.append(lg)
        d = (logits_full - torch.cat(steps, dim=1)).abs().max().item()
        print(f"[f] 整段 vs 增量: max diff {d:.2e}")
        assert d < 1e-4, d
        # 注入后：prefill 16 → 注入 → 余下 24 token 一段前向 vs 逐 token 前向
        entries = (torch.randn(2, cfg.n_kv_heads, 2, cfg.head_dim, device=device),) * 2
        ns = make_namespace(cfg, 3, entries[0].dtype)
        _, c1 = model(ids[:, :16])
        c1["layers"][3] = tri.inject_hca_entries(c1["layers"][3], entries, ns)
        import copy

        c2 = copy.deepcopy(c1)
        lg_multi, _ = model(ids[:, 16:], c1)  # 多 token 带 cache 前向
        steps = []
        c = c2
        for t in range(16, 40):
            lg, c = model(ids[:, t : t + 1], c)
            steps.append(lg)
        d2 = (lg_multi - torch.cat(steps, dim=1)).abs().max().item()
        print(f"[f] 注入后 多token vs 逐token: max diff {d2:.2e}")
        assert d2 < 1e-4, d2


def check_save_load(device: str) -> None:
    """g) save/load 往返：config 开关持久化 + bf16 存储相对误差内一致。"""
    cfg, model, tri = _build(device)
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        model2 = TaisObsidianForCausalLM.from_pretrained(tmp, device)
    assert model2.config.attn_impl == "tri"
    assert isinstance(model2.layers[3].mixer, TriAttention)
    assert model2.layers[3].mixer.layer_idx == 3
    torch.manual_seed(5)
    ids = torch.randint(0, 512, (1, 24), device=device)
    with torch.no_grad():
        o1 = model(ids)[0]
        o2 = model2(ids)[0]
    d = (o1 - o2).abs().max().item()
    rel = d / o1.abs().max().item()
    print(f"[g] save/load 往返: max diff {d:.2e}, 相对 {rel:.2e}")
    assert rel < 1e-2, rel  # bf16 存储的相对误差量级（对齐 test_cache 判据）


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_shapes(device)
    check_causality(device)
    check_selection_legality(device)
    check_window_reference(device)
    check_hca_inject(device)
    check_incremental(device)
    check_save_load(device)
    print("test_tri_attention 全部通过。")


def test_shapes() -> None:
    check_shapes("cuda" if torch.cuda.is_available() else "cpu")


def test_causality() -> None:
    check_causality("cuda" if torch.cuda.is_available() else "cpu")


def test_selection_legality() -> None:
    check_selection_legality("cuda" if torch.cuda.is_available() else "cpu")


def test_window_reference() -> None:
    check_window_reference("cuda" if torch.cuda.is_available() else "cpu")


def test_hca_inject() -> None:
    check_hca_inject("cuda" if torch.cuda.is_available() else "cpu")


def test_incremental() -> None:
    check_incremental("cuda" if torch.cuda.is_available() else "cpu")


def test_save_load() -> None:
    check_save_load("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
