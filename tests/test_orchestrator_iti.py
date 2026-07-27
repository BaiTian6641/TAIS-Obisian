"""ITIGate 集成编排闭环单元测试（监测→门控→干预，三层信号驱动）。

判据（kernel_orchestrator + ITIGate 集成）：
- 空白（is_blank）→ iti_action=abstain，不 route 不 inject 不 steer；
- 非空白 + L3 冲突高超阈 → iti_action=steer_truth，pm_pre 被沿真值方向 steer；
- 非空白 + 低信号 → iti_action=noop，pm_pre 不变；
- RecallDecision 携带 conflict_score/arousal（三层感知供门控）；
- 无 iti_gate 时 iti_action=noop（向后兼容）。
"""
from __future__ import annotations

import torch

from tais_obsidian.model.iti_head import make_iti_gate
from tais_obsidian.model.tais_kernel import TAISKernel
from tais_obsidian.runtime import (
    BlockSpec,
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    make_orchestrator,
)

D = 32
NS = ("m1", 0, 1, "bf16", 10000.0)


def _kernel_with_signals(know_logit, blank_logit, conflict_logit, arousal_logit):
    """构造内核并设 KAL L1（know/blank）+ L3 conflict + L2 arousal 头的固定输出。"""
    k = TAISKernel(D)
    with torch.no_grad():
        # L1
        k.kal_l1.proj.weight.zero_(); k.kal_l1.proj.bias.zero_()
        k.kal_l1.proj.bias[0] = know_logit; k.kal_l1.proj.bias[2] = blank_logit
        # L2（dim1=arousal）
        k.kal_l2.proj.weight.zero_(); k.kal_l2.proj.bias.zero_()
        k.kal_l2.proj.bias[1] = arousal_logit
        # L3 conflict（三态，取 mean 作 score；设 bias 使 conflict logit 固定）
        k.side_heads.conflict.weight.zero_(); k.side_heads.conflict.bias.zero_()
        k.side_heads.conflict.bias[0] = conflict_logit  # 一致态 logit
    return k


def _bus():
    pt = PageTable(); bs = BlockStore()
    return MemoryBus(pt, bs, Pager(bs, pt))


def test_blank_gives_abstain() -> None:
    k = _kernel_with_signals(know_logit=-2, blank_logit=2, conflict_logit=5, arousal_logit=5)
    gate = make_iti_gate(k, conflict_thresh=0.0, arousal_thresh=0.5)
    orch = make_orchestrator(k, _bus(), iti_gate=gate)
    pm = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm.clone())
    assert out.iti_action == "abstain"
    assert out.decision.is_blank is True
    assert out.n_injected == 0


def test_conflict_drives_steer_truth() -> None:
    # 非空白 + 高冲突（conflict_logit 大 → sigmoid 高 → 超阈）→ steer_truth
    k = _kernel_with_signals(know_logit=3, blank_logit=-3, conflict_logit=5.0, arousal_logit=-5.0)
    gate = make_iti_gate(k, conflict_thresh=0.5, arousal_thresh=0.9, truth_alpha_frac=0.1)
    orch = make_orchestrator(k, _bus(), iti_gate=gate)
    pm = torch.zeros(1, 4, D)
    pm_pre = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm_pre, query=None)  # 无候选，仅 sense+ITI 门
    assert out.decision.is_blank is False
    assert out.decision.conflict_score > 0.5, "conflict_score 应高（sigmoid(5)≈0.99）"
    assert out.iti_action == "steer_truth"


def test_low_signal_noop() -> None:
    # 非空白 + 低冲突 + 低唤醒 → noop
    k = _kernel_with_signals(know_logit=3, blank_logit=-3, conflict_logit=-5.0, arousal_logit=-5.0)
    gate = make_iti_gate(k, conflict_thresh=0.5, arousal_thresh=0.5)
    orch = make_orchestrator(k, _bus(), iti_gate=gate)
    pm = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm.clone())
    assert out.decision.is_blank is False
    assert out.iti_action == "noop"


def test_decision_carries_three_layer_signals() -> None:
    k = _kernel_with_signals(know_logit=3, blank_logit=-3, conflict_logit=2.0, arousal_logit=1.0)
    orch = make_orchestrator(k, _bus())
    dec = orch.sense_gate(torch.zeros(1, 4, D))
    assert hasattr(dec, "conflict_score") and hasattr(dec, "arousal")
    assert 0.0 <= dec.conflict_score <= 1.0
    assert 0.0 <= dec.arousal <= 1.0


def test_no_iti_gate_backward_compat() -> None:
    k = _kernel_with_signals(know_logit=3, blank_logit=-3, conflict_logit=5, arousal_logit=5)
    orch = make_orchestrator(k, _bus())  # 无 iti_gate
    pm = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm.clone())
    assert out.iti_action == "noop"  # 无门时 noop（向后兼容）
