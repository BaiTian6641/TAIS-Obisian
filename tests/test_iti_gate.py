"""ITI 条件触发门单元测试（双刃剑门控，规范 §5 + 2026-07-27 文献修正）。

判据（ITIGate / 诚实降级红线）：
- L1 空白 → abstain（**绝不 steer 成 know**，hidden 不变——拒答由编排层处理）；
- L3 冲突超阈 → steer_truth（沿真值方向有界 steer）；
- L2 高唤醒超阈 → steer_truth；
- 低信号/无信号 → noop（不干预，hidden 不变）；
- 优先级：空白 > 冲突 > 唤醒（空白即使冲突也 abstain，防 steer 造假）；
- fail-closed：信号 None 时不盲目 steer。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tais_obsidian.model.iti_head import (
    ITI_ABSTAIN,
    ITI_NOOP,
    ITI_STEER_TRUTH,
    make_iti_gate,
)
from tais_obsidian.model.tais_kernel import TAISKernel

D = 64


def _gate():
    kern = TAISKernel(D)
    return make_iti_gate(kern, max_alpha_frac=0.2, conflict_thresh=0.0,
                         arousal_thresh=0.5, truth_alpha_frac=0.1)


def test_blank_abstains_no_steer() -> None:
    g = _gate()
    h = torch.randn(1, 4, D)
    out, action = g.apply(h, is_blank=True)
    assert action == ITI_ABSTAIN
    assert torch.equal(out, h), "空白时 hidden 不变（绝不 steer 成 know，拒答由编排层处理）"


def test_conflict_triggers_steer_truth() -> None:
    g = _gate()
    h = torch.randn(1, 4, D)
    out, action = g.apply(h, is_blank=False, conflict_score=1.0)
    assert action == ITI_STEER_TRUTH
    assert not torch.equal(out, h), "冲突时应沿真值方向 steer"
    # steer 方向应与 iti.direction 正相关（对最后一维 D 求点积）
    shift = (out - h)  # [1,4,D]
    direction = g.iti.direction.to(shift.device, shift.dtype)  # [D]
    proj = torch.einsum("btd,d->bt", shift, direction)
    assert (proj > 0).all(), "所有位置的 shift 应沿 +direction"


def test_high_arousal_triggers_steer() -> None:
    g = _gate()
    h = torch.randn(1, 4, D)
    out, action = g.apply(h, is_blank=False, arousal=0.9)
    assert action == ITI_STEER_TRUTH


def test_low_signal_noop() -> None:
    g = _gate()
    h = torch.randn(1, 4, D)
    # 冲突低于阈值（负）+ 唤醒低于阈值
    out, action = g.apply(h, is_blank=False, conflict_score=-1.0, arousal=0.2)
    assert action == ITI_NOOP
    assert torch.equal(out, h), "低信号应 noop 不干预"


def test_blank_overrides_conflict() -> None:
    # 空白 + 冲突同时 → 仍 abstain（空白优先，防 steer 造假）
    g = _gate()
    h = torch.randn(1, 4, D)
    out, action = g.apply(h, is_blank=True, conflict_score=1.0, arousal=0.9)
    assert action == ITI_ABSTAIN
    assert torch.equal(out, h)


def test_none_signal_fail_closed() -> None:
    # 全部信号 None（非空白）→ noop（不盲目 steer）
    g = _gate()
    h = torch.randn(1, 4, D)
    out, action = g.apply(h, is_blank=False, conflict_score=None, arousal=None)
    assert action == ITI_NOOP
    assert torch.equal(out, h)


def test_decide_pure() -> None:
    g = _gate()
    assert g.decide(is_blank=True) == ITI_ABSTAIN
    assert g.decide(is_blank=False, conflict_score=1.0) == ITI_STEER_TRUTH
    assert g.decide(is_blank=False, arousal=0.9) == ITI_STEER_TRUTH
    assert g.decide(is_blank=False) == ITI_NOOP
