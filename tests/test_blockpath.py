"""CSA 原生块通路原型测试（设计 §11.1）：压缩形状、收割范围、namespace fail-closed、注入簿记。

用法：python tests/test_blockpath.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.blockpath import (
    COMPRESSOR_VERSION,
    BlockCompressor,
    NamespaceMismatchError,
    harvest_block_kv,
    inject_block_kv,
)
from tais_obsidian.model.model import TaisObsidianForCausalLM


def tiny_cfg() -> ModelConfig:
    # 8 层 = 2×{G,G,G,A}："A" 层为 idx 3、7
    return ModelConfig(
        vocab_size=512,
        d_model=256,
        n_layer=8,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        n_v_heads=4,
        n_qk_heads=2,
        mlp_hidden=688,
        max_seq=128,
        check_0p1b_params=False,
    )


def _build(device: str):
    torch.manual_seed(0)
    cfg = tiny_cfg()
    model = TaisObsidianForCausalLM(cfg).to(device).eval()
    a_layers = [i for i, t in enumerate(cfg.layer_types) if t == "A"]
    comps = {i: BlockCompressor(cfg.head_dim, stride=4).to(device).eval() for i in a_layers}
    return cfg, model, comps, a_layers


def check_compress_shape(device: str) -> None:
    """a) 压缩形状：T → floor(T/4)，尾部不足 stride 丢弃。"""
    comp = BlockCompressor(64, stride=4).to(device)
    for T, want in ((16, 4), (17, 4), (19, 4), (3, 0)):
        k = torch.randn(2, 2, T, 64, device=device)
        v = torch.randn(2, 2, T, 64, device=device)
        kc, vc = comp(k, v)
        assert kc.shape == (2, 2, want, 64), (T, kc.shape)
        assert vc.shape == (2, 2, want, 64), (T, vc.shape)
    print("[a] 压缩形状 floor(T/4)（尾部丢弃）通过")


def check_harvest(device: str) -> None:
    """b) harvest 只对 "A" 层产出条目，压缩长度正确。"""
    cfg, model, comps, a_layers = _build(device)
    ids = torch.randint(0, 512, (2, 33), device=device)
    blk = harvest_block_kv(model, ids, comps, device)
    assert set(blk["layers"].keys()) == set(a_layers), blk["layers"].keys()
    for i in a_layers:
        kc, vc = blk["layers"][i]
        assert kc.shape == (2, cfg.n_kv_heads, 33 // 4, cfg.head_dim), kc.shape
        assert vc.shape == kc.shape
        ns = blk["namespace"][i]
        assert ns["layer_idx"] == i and ns["compressor_version"] == COMPRESSOR_VERSION
    print(f"[b] harvest 仅覆盖 A 层 {a_layers}，长度 {33 // 4} 通过")


def check_namespace_failclosed(device: str) -> None:
    """c) namespace 全对放行；五元组任一字段不匹配 → fail-closed 抛错。"""
    cfg, model, comps, a_layers = _build(device)
    ids = torch.randint(0, 512, (1, 24), device=device)
    blk = harvest_block_kv(model, ids, comps, device)
    _, cache = model(ids[:, :8])  # 目标 cache（prefill 8 token）
    # 全对：放行
    inject_block_kv(cache, blk, cfg)
    # 逐字段篡改：fail-closed
    tampers = {
        "model_id": "d999-L9-h9-kv9-V9",
        "layer_idx": 0,
        "compressor_version": "csa-comp-v0.0",
        "dtype": "torch.float16",
        "rope_theta": 500000.0,
    }
    for field, bad in tampers.items():
        blk_bad = {"namespace": {i: dict(ns) for i, ns in blk["namespace"].items()}, "layers": blk["layers"]}
        blk_bad["namespace"][a_layers[0]][field] = bad
        try:
            inject_block_kv(cache, blk_bad, cfg)
        except NamespaceMismatchError:
            pass
        else:
            raise AssertionError(f"字段 {field} 不匹配未触发 fail-closed")
    print("[c] namespace 五元组全对放行、逐字段篡改均 fail-closed 通过")


def check_inject_incremental(device: str) -> None:
    """d) 注入后逐 token 增量前向 3 步：无异常、形状/pos 簿记正确、G 层 state 逐点不变。"""
    cfg, model, comps, a_layers = _build(device)
    ids = torch.randint(0, 512, (1, 24), device=device)
    blk = harvest_block_kv(model, ids, comps, device)
    n_inj = next(iter(blk["layers"].values()))[0].shape[2]
    with torch.no_grad():
        _, cache = model(ids[:, :8])
        g_before = {
            i: {k: v.clone() for k, v in cache["layers"][i].items()}
            for i, t in enumerate(cfg.layer_types)
            if t == "G"
        }
        pos_before = cache["pos"]
        cache2 = inject_block_kv(cache, blk, cfg)
        # pos 增加量 = 注入长度
        assert cache2["pos"] == pos_before + n_inj, (cache2["pos"], pos_before, n_inj)
        # A 层 k/v 长度 = 注入 + 原有；G 层 state 逐点相等
        for i, t in enumerate(cfg.layer_types):
            if t == "A":
                assert cache2["layers"][i]["k"].shape[2] == 8 + n_inj
                assert cache2["layers"][i]["v"].shape[2] == 8 + n_inj
            else:
                for key, v in g_before[i].items():
                    assert torch.equal(cache2["layers"][i][key], v), (i, key)
        # 逐 token 增量前向 3 步
        for s in range(3):
            logits, cache2 = model(ids[:, 8 + s : 9 + s], cache2)
            assert logits.shape == (1, 1, cfg.vocab_size), logits.shape
        assert cache2["pos"] == pos_before + n_inj + 3
        for i in a_layers:
            assert cache2["layers"][i]["k"].shape[2] == 8 + n_inj + 3
    print(f"[d] 注入 {n_inj} 条 + 增量 3 步：pos/形状簿记正确，G 层 state 逐点不变")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    check_compress_shape(device)
    check_harvest(device)
    check_namespace_failclosed(device)
    check_inject_incremental(device)
    print("test_blockpath 全部通过。")


def test_compress_shape() -> None:
    check_compress_shape("cuda" if torch.cuda.is_available() else "cpu")


def test_harvest() -> None:
    check_harvest("cuda" if torch.cuda.is_available() else "cpu")


def test_namespace_failclosed() -> None:
    check_namespace_failclosed("cuda" if torch.cuda.is_available() else "cpu")


def test_inject_incremental() -> None:
    check_inject_incremental("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
