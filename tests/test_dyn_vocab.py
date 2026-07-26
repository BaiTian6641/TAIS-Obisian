"""M7 动态词表（concept_slot）单元测试：检测→提取→注册→注入。

判据（部件实现详细计划 Part G1 / 设计 §28.2 第 0 级）：
- vocab_friction_score 超阈检测；
- extract 走注入的 extract_fn（Kaplan 回调）；未注入时 fail-closed（RuntimeError）；
- register 注册 concept_slot 到页表（compiled_kind=concept_slot，factual_recall=False）；
- promote 一步到位；注册失败 fail-closed。
"""
from __future__ import annotations

import pytest
import torch

from tais_obsidian.model.dyn_vocab import make_dynamic_vocab, vocab_friction_score
from tais_obsidian.runtime import PageTable

NS = ("m1", 0, 1, "bf16", 10000.0)
D = 32


def _dv(extract_fn=None):
    pt = PageTable()
    return make_dynamic_vocab(pt, NS, extract_fn=extract_fn), pt


def test_vocab_friction_score() -> None:
    # 高熵+高共现+低 P(IK) → 高摩擦
    assert vocab_friction_score(entropy=0.9, p_ik=0.1, repeat_cooccur=0.9) > 0.6
    # 低熵+低共现+高 P(IK) → 低摩擦
    assert vocab_friction_score(entropy=0.1, p_ik=0.9, repeat_cooccur=0.1) < 0.3


def test_detect_threshold() -> None:
    dv, _ = _dv()
    assert dv.detect(entropy=0.9, p_ik=0.1, repeat_cooccur=0.9) is True
    assert dv.detect(entropy=0.1, p_ik=0.9, repeat_cooccur=0.1) is False


def test_extract_requires_fn_fail_closed() -> None:
    dv, _ = _dv(extract_fn=None)
    with pytest.raises(RuntimeError):
        dv.extract("some concept")


def test_extract_and_register_and_promote() -> None:
    torch.manual_seed(0)
    dv, pt = _dv(extract_fn=lambda text: torch.ones(D) * len(text))
    vec = dv.extract("超导量子比特")
    assert vec.shape == (D,)
    spec = dv.register("超导量子比特", vec)
    assert spec.compiled_kind == "concept_slot"
    assert spec.factual_recall is False  # 位置不变向量，非事实查表
    assert pt.get("concept/超导量子比特") is not None
    # promote 一步到位
    spec2 = dv.promote("拓扑绝缘体")
    assert pt.get("concept/拓扑绝缘体") is not None
    assert spec2.route_key == "拓扑绝缘体"


def test_concept_slot_is_vector_not_fact() -> None:
    """concept_slot 属位置不变向量（factual_recall=False），与内核 VECTOR_KINDS 一致。"""
    dv, _ = _dv(extract_fn=lambda t: torch.zeros(D))
    spec = dv.promote("x")
    assert spec.factual_recall is False
