"""KAL 分层元认知原型（E+-3）测试：头形状、管线 smoke、report schema。

用法：python tests/test_kal.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.kal import KALHead, make_l1_head, make_l2_head, read_point
from tais_obsidian.model.model import TaisObsidianForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tiny_cfg() -> ModelConfig:
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
        check_0p1b_params=False,
    )


def test_head_shapes() -> None:
    """两型头形状：L1=W[d,3]、L2=W[d,2]；[B,T,d] 与 [B,d] 输入均兼容。"""
    d = 256
    l1, l2 = make_l1_head(d), make_l2_head(d)
    assert l1.proj.weight.shape == (3, d) and l2.proj.weight.shape == (2, d)
    h3 = torch.randn(2, 8, d)
    h2 = torch.randn(2, d)
    assert l1(h3).shape == (2, 8, 3) and l1(h2).shape == (2, 3)
    assert l2(h3).shape == (2, 8, 2) and l2(h2).shape == (2, 2)
    assert l1.predict_proba(h2).shape == (2, 3)
    # state_dict 存取（checkpoint 内生权重的结构对齐）
    l1b = make_l1_head(d)
    l1b.load_state_dict(l1.state_dict())
    assert torch.equal(l1b(h2), l1(h2))
    print("[head] L1 W[d,3] / L2 W[d,2] 形状与 state_dict 往返正确。")


def test_capture_compat() -> None:
    """头与捕获兼容：tiny 模型 G/A 层（层 0=G，层 3=A）捕获张量可直接喂头。"""
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(DEVICE).eval()
    ids = torch.randint(0, 512, (2, 16), device=DEVICE)
    with torch.no_grad():
        _, _, caps = model(ids, capture_layers=[0, 3])
    d = model.config.d_model
    l1, l2 = make_l1_head(d).to(DEVICE), make_l2_head(d).to(DEVICE)
    for i in (0, 3):
        h = read_point(caps, i)
        assert h.shape == (2, 16, d), (i, h.shape)
        assert l1(h).shape == (2, 16, 3) and l2(h[:, -1]).shape == (2, 2)
    # read_point 的 PM 分支：dict 形式取 "content"/"pm"
    fake_caps = {0: {"content": torch.randn(1, 4, d), "pm": torch.randn(1, 4, d)}}
    assert read_point(fake_caps, 0, "content") is fake_caps[0]["content"]
    assert read_point(fake_caps, 0, "pm") is fake_caps[0]["pm"]
    print("[capture] G/A 层捕获兼容两种 KAL 头；read_point PM 分支正确。")


def test_pipeline_smoke() -> None:
    """管线 smoke：tiny 随机模型 + 合成 id 序列过 forward_collect + 探针训练评估全链路。"""
    import kal_probe

    torch.manual_seed(1)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(DEVICE).eval()
    rng = np.random.default_rng(0)
    # 合成"已知/未知"：known=低 id 区间，unknown=高 id 区间（随机权重下信号无保证，仅验链路）
    id_list = [rng.integers(0, 256, size=12).tolist() for _ in range(8)]
    id_list += [rng.integers(256, 512, size=12).tolist() for _ in range(8)]
    labels = np.array([1] * 8 + [0] * 8)
    subset = np.array(["known"] * 8 + ["fake"] * 4 + ["shuffled"] * 4)
    feats, base = kal_probe.forward_collect(model, id_list, [0, 3], DEVICE, batch_size=4)
    assert feats[0].shape == (16, 256) and feats[3].shape == (16, 256)
    assert base.shape == (16,) and np.isfinite(base).all()
    res = kal_probe.run_l1_experiment(feats, labels, subset, base, seed=0, test_ratio=0.25)
    assert set(res["layers"]) == {"0", "3"}
    for r in res["layers"].values():
        for k in ("overall", "fake", "shuffled"):
            assert 0.0 <= r["auroc"][k] <= 1.0 or np.isnan(r["auroc"][k])
    # L2 链路
    yv = np.array([1, 0] * 8)
    ya = np.array([0, 1] * 8)
    res2 = kal_probe.run_l2_experiment(feats, yv, ya, seed=0, test_ratio=0.25)
    for r in res2["layers"].values():
        assert 0.0 <= r["valence"]["accuracy"] <= 1.0
        assert 0.0 <= r["arousal"]["auroc"] <= 1.0 or np.isnan(r["arousal"]["auroc"])
    print("[smoke] forward_collect + L1/L2 探针训练评估全链路无异常。")


def test_report_schema() -> None:
    """report schema：合成特征跑 run_l1/run_l2/make_report，断言关键字段存在。"""
    import kal_probe

    rng = np.random.default_rng(42)
    feats = {4: rng.normal(size=(40, 256)).astype(np.float32),
             8: rng.normal(size=(40, 256)).astype(np.float32)}
    labels = np.array([1] * 20 + [0] * 20)
    subset = np.array(["known"] * 20 + ["fake"] * 10 + ["shuffled"] * 10)
    base = rng.normal(size=40).astype(np.float32)
    l1 = kal_probe.run_l1_experiment(feats, labels, subset, base, seed=0)
    yv, ya = np.array([1, 0] * 20), np.array([0, 1] * 20)
    l2 = kal_probe.run_l2_experiment(feats, yv, ya, seed=0)
    report = kal_probe.make_report({"layers": [4, 8], "pooling": "last"}, l1, l2)
    for k in ("experiment", "timestamp_utc", "config", "l1", "l2"):
        assert k in report, f"report 缺字段 {k}"
    assert set(report["l1"]["layers"]) == {"4", "8"}
    assert "baseline_flare_mean_logprob" in report["l1"]
    assert set(report["l1"]["baseline_flare_mean_logprob"]) == {"overall", "fake", "shuffled"}
    for r in report["l2"]["layers"].values():
        assert set(r) == {"valence", "arousal"}
        assert set(r["valence"]) == {"accuracy", "auroc"}
    # 工具函数单测：AUROC 完美可分=1、反向=0、随机≈0.5
    assert kal_probe.auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert kal_probe.auroc(np.array([0.1, 0.2, 0.9, 0.8]), np.array([1, 1, 0, 0])) == 0.0
    print("[schema] report.json schema 与 auroc 边界行为正确。")


if __name__ == "__main__":
    test_head_shapes()
    test_capture_compat()
    test_pipeline_smoke()
    test_report_schema()
    print("test_kal 全部通过。")
