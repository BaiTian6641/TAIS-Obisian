"""KAL 词表摩擦 → concept_slot 注册集成测试（动态 tokenizer 如架构，KAL 感知已学内容）。

判据（kernel_orchestrator.assess_vocab_friction / dyn_vocab / 设计 §28.2）：
- 高摩擦（高熵 + 高共现 + 低 P(IK)）→ 触发 concept_slot 注册（KAL 感知"已学概念"）；
- 低摩擦 → 不触发；
- fail-closed：未挂 dynamic_vocab / extract_fn 未注入 → 返回 False（不静默）；
- 注册后页表含 concept_slot 块（compiled_kind=concept_slot，factual_recall=False）。
"""
from __future__ import annotations

import torch

from tais_obsidian.model.dyn_vocab import make_dynamic_vocab
from tais_obsidian.model.tais_kernel import TAISKernel
from tais_obsidian.runtime import (
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    make_orchestrator,
)

D = 32
NS = ("m1", 0, 1, "bf16", 10000.0)


def _orch(with_dv: bool = True, extract_fn=None):
    k = TAISKernel(D)
    pt = PageTable()
    bs = BlockStore()
    bus = MemoryBus(pt, bs, Pager(bs, pt))
    dv = make_dynamic_vocab(pt, NS, extract_fn=extract_fn) if with_dv else None
    orch = make_orchestrator(k, bus, dynamic_vocab=dv)
    return orch, pt


def test_high_friction_triggers_concept_slot() -> None:
    # 高熵 + 高共现 + 低 P(IK) → 高摩擦 → 注册 concept_slot（KAL 感知已学概念）
    extract = lambda text: torch.ones(D)  # 假 Kaplan 提取（返回单位向量）
    orch, pt = _orch(extract_fn=extract)
    ok = orch.assess_vocab_friction("Zorblax", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9)
    assert ok is True
    spec = pt.get("concept/Zorblax")
    assert spec is not None, "页表应含注册的 concept_slot"
    assert spec.compiled_kind == "concept_slot"
    assert spec.factual_recall is False  # 位置不变向量，非事实查表（载体能力边界）


def test_low_friction_no_trigger() -> None:
    extract = lambda text: torch.ones(D)
    orch, pt = _orch(extract_fn=extract)
    ok = orch.assess_vocab_friction("cat", p_ik=0.95, next_token_entropy=0.1, repeat_cooccur=0.1)
    assert ok is False
    assert pt.get("concept/cat") is None


def test_no_dynamic_vocab_fail_closed() -> None:
    orch, _ = _orch(with_dv=False)
    ok = orch.assess_vocab_friction("Zorblax", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9)
    assert ok is False  # 未挂 dynamic_vocab → fail-closed 不静默


def test_no_extract_fn_fail_closed() -> None:
    # 挂了 dynamic_vocab 但 extract_fn 未注入 → promote 抛 RuntimeError → fail-closed False
    orch, _ = _orch(extract_fn=None)
    ok = orch.assess_vocab_friction("Zorblax", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9)
    assert ok is False


def test_friction_boundary() -> None:
    # 摩擦分恰好低于阈值（0.6）→ 不触发；高于 → 触发
    extract = lambda text: torch.ones(D)
    orch, pt = _orch(extract_fn=extract)
    # 0.5*0.5 + 0.3*0.5 + 0.2*(1-0.5) = 0.25+0.15+0.1 = 0.5 < 0.6 → 不触发
    assert orch.assess_vocab_friction("A", p_ik=0.5, next_token_entropy=0.5, repeat_cooccur=0.5) is False
    # 0.5*0.9 + 0.3*0.9 + 0.2*(1-0.1) = 0.45+0.27+0.18 = 0.9 ≥ 0.6 → 触发
    assert orch.assess_vocab_friction("B", p_ik=0.1, next_token_entropy=0.9, repeat_cooccur=0.9) is True
