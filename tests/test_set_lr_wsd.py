"""set_lr 对 Muon 组按 WSD 比例缩放 muon_lr/adamw_lr 的回归测试。

对应已修复 bug：Muon 组读 g["muon_lr"]/g["adamw_lr"]（非 g["lr"]，见 optim/muon.py），
若 set_lr 只写 g["lr"] 则 Muon 全程恒 lr（WSD warmup/decay 静默失效）。防再断。
用法：python -m pytest tests/test_set_lr_wsd.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import build_optimizer, lr_at, set_lr

# 训练 cfg：peak lr=1e-3，warmup 10 步，100 步总程，末 20% 衰减；Muon 组 peak 0.02
CFG = dict(lr=1e-3, warmup=10, max_steps=100, decay_frac=0.2, weight_decay=0.1,
           optimizer="muon", muon_lr=0.02)


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


def test_wsd_schedule_shape() -> None:
    """lr_at 的 WSD 形状：warmup 线性升 → 稳定段恒定 peak → 末段线性降到 0。"""
    assert lr_at(0, CFG) == pytest.approx(CFG["lr"] * 1 / CFG["warmup"])
    assert lr_at(CFG["warmup"] - 1, CFG) == pytest.approx(CFG["lr"])
    assert lr_at(50, CFG) == pytest.approx(CFG["lr"])  # 稳定段
    decay_start = int(CFG["max_steps"] * (1 - CFG["decay_frac"]))  # 80
    assert lr_at(decay_start, CFG) == pytest.approx(CFG["lr"])
    assert lr_at(CFG["max_steps"] - 1, CFG) == pytest.approx(CFG["lr"] / 20)  # 末步降到 ~0
    # warmup 段单调递增、decay 段单调递减
    ups = [lr_at(s, CFG) for s in range(CFG["warmup"])]
    downs = [lr_at(s, CFG) for s in range(decay_start, CFG["max_steps"])]
    assert all(a < b for a, b in zip(ups, ups[1:]))
    assert all(a > b for a, b in zip(downs, downs[1:]))


def test_set_lr_scales_muon_groups_with_wsd() -> None:
    """核心回归：set_lr 后 Muon 组 muon_lr = muon_peak×(lr/peak)，adamw_lr = lr（全程随 WSD 变化）。"""
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(tiny_cfg())  # CPU 即可（不跑前向）
    opt = build_optimizer(model, CFG)
    peak, muon_peak = CFG["lr"], CFG["muon_lr"]
    seen_muon_lr: set[float] = set()
    for step in (0, 5, 10, 50, 80, 95):  # warmup/稳定/decay 三段各取点
        lr = lr_at(step, CFG)
        set_lr(opt, lr, CFG)
        scale = lr / peak
        for g in opt.param_groups:
            assert g["lr"] == pytest.approx(lr)
            assert "muon_lr" in g and "adamw_lr" in g, "Muon 优化器各组必须带 muon_lr/adamw_lr 键"
            assert g["muon_lr"] == pytest.approx(muon_peak * scale), (
                f"step={step}: muon_lr 应按 WSD 比例缩放（{muon_peak}×{scale:.3f}），实际 {g['muon_lr']}"
            )
            assert g["adamw_lr"] == pytest.approx(lr)
        seen_muon_lr.add(round(opt.param_groups[0]["muon_lr"], 12))
    # 回归判据：muon_lr 必须随 WSD 变化（bug 时全程恒 0.02，集合只有 1 个值）
    assert len(seen_muon_lr) > 2, f"muon_lr 未随 WSD 变化（疑似旧 bug 复发）: {seen_muon_lr}"
    print(f"[set_lr] muon_lr 随 WSD 取值 {sorted(seen_muon_lr)}")


def test_set_lr_adamw_plain_groups() -> None:
    """AdamW（默认）路径：set_lr 只写 g['lr']，各组一致，无 muon_lr/adamw_lr 键。"""
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(tiny_cfg())
    opt = build_optimizer(model, {"lr": CFG["lr"], "weight_decay": 0.1})  # 缺省 = adamw
    lr = lr_at(50, CFG)
    set_lr(opt, lr, CFG)
    assert len(opt.param_groups) == 2  # decay / no_decay 两组
    for g in opt.param_groups:
        assert g["lr"] == pytest.approx(lr)
        assert "muon_lr" not in g and "adamw_lr" not in g
    print("[set_lr] AdamW 组 lr 写入正确")


def main() -> None:
    test_wsd_schedule_shape()
    test_set_lr_scales_muon_groups_with_wsd()
    test_set_lr_adamw_plain_groups()
    print("test_set_lr_wsd 全部通过。")


if __name__ == "__main__":
    main()
