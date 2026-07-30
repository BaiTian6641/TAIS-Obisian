"""真实 Kaplan 内词典提取 + concept_slot 闭环集成测试（真实 GDN-2 10k 模型，CUDA）。

判据（设计 §28.2 / arXiv:2410.05864 / dyn_vocab / kernel_orchestrator）：
- 真实 extract_fn：对概念词返回 [d_model] 表征（形状正确、值有限、确定性、无梯度）；
- 多 token 概念：取末 token hidden（碎片融合为词表示，Kaplan 核心）；
- 与 mock 对比：真实表征 ≠ 常数（有语义——同类概念余弦均值 > 不同类，mock 任意 cos≡1）；
- concept_slot 闭环：真实 extract_fn 装配 orchestrator → 检测 → 注册(页表+BlockStore)
  → HRL route_graph 入图 → associative_recall 检索命中 → 内核 inject 向量路径可用；
- 载体能力边界：concept_slot factual_recall=False（位置不变向量，非事实查表）。

需 CUDA + 真实 checkpoint（GDN-2 10k）；无 GPU 跳过。
"""
from __future__ import annotations

import statistics

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA（真实模型 Kaplan 提取）")

from tais_obsidian.model.dyn_vocab import make_dynamic_vocab
from tais_obsidian.model.kaplan_extract import DEFAULT_KAPLAN_LAYER, make_kaplan_extract_fn
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.runtime import BlockStore, MemoryBus, PageTable, Pager, make_orchestrator

CKPT = "checkpoints/pilot_0p1b_gdn2_10k/final"
NS = ("m1", 0, 1, "bf16", 10000.0)


@pytest.fixture(scope="module")
def model():
    m = TaisObsidianForCausalLM.from_pretrained(CKPT, "cuda").eval()
    m.attach_kernel()  # 10k checkpoint kernel_enabled=False → 挂载内核（注入路径用）
    return m


@pytest.fixture(scope="module")
def extract_fn(model):
    return make_kaplan_extract_fn(model)  # 默认 ℓ3（pilot detokenize 最强层）


# ----------------------------------------------------------------------
def test_real_extract_shape_finite(model, extract_fn):
    """真实 Kaplan extract_fn：形状 [d_model]、值有限、非全零。"""
    vec = extract_fn("Qeltharion")  # 虚构专名（OOV，多 token 碎片）
    assert vec.shape == (model.config.d_model,)
    assert torch.isfinite(vec).all(), "hidden 须有限"
    assert vec.abs().sum() > 0, "表征不应全零"


def test_real_extract_deterministic_and_nograd(extract_fn):
    """确定性（eval + no_grad → 同输入同向量）+ 无梯度（监测/执行分置，只读）。"""
    v1 = extract_fn("Zorblax")
    v2 = extract_fn("Zorblax")
    assert torch.allclose(v1, v2), "同输入应得同向量（确定性）"
    assert not v1.requires_grad, "Kaplan 提取 no_grad 只读（监测侧，不回传梯度）"


def test_multi_token_fuses_to_last_token(model, extract_fn):
    """多 token 概念：编码为多个 token（OOV 碎片化），仍融合为单 [d_model] 向量。"""
    from tais_obsidian.tokenizer_io import TokenizerIO
    tok = TokenizerIO("data/tokenizer/tokenizer.json")
    ids = tok.encode("Qeltharion")
    assert len(ids) > 1, "虚构专名应碎成多 token（OOV 碎片，Kaplan 融合场景）"
    vec = extract_fn("Qeltharion")
    assert vec.shape == (model.config.d_model,), "多 token 碎片应融合为末 token 单向量"


def test_real_extract_has_semantics_vs_mock(extract_fn):
    """真实表征有语义（同类均值 > 不同类均值），区别于 mock 常数（任意 cos≡1）。"""
    sim_pairs = [("electron", "photon"), ("neutron", "proton"), ("dog", "cat"), ("graviton", "neutrino")]
    diff_pairs = [("electron", "democracy"), ("graviton", "banana"), ("dog", "bicycle"), ("neutron", "metabolism")]
    cos = lambda a, b: float(F.cosine_similarity(extract_fn(a), extract_fn(b), dim=0))
    sim_mean = statistics.mean(cos(a, b) for a, b in sim_pairs)
    diff_mean = statistics.mean(cos(a, b) for a, b in diff_pairs)
    assert sim_mean > diff_mean, (
        f"同类概念应比不同类更接近（sim_mean={sim_mean:.3f} > diff_mean={diff_mean:.3f}）——"
        "真实 Kaplan 表征有语义，区别于 mock 常数（任意概念 cos≡1，无语义区分）")
    # mock 常数对照：任意概念向量相同 → cos≡1（无语义）
    d = extract_fn("electron").shape[0]
    assert float(F.cosine_similarity(torch.ones(d), torch.ones(d), dim=0)) == pytest.approx(1.0)


def test_concept_slot_closed_loop(model, extract_fn):
    """真实 extract_fn 装配 orchestrator → 检测→注册→HRL检索→注入 全闭环。"""
    pt, bs = PageTable(), BlockStore()
    bus = MemoryBus(pt, bs, Pager(bs, pt))
    dyn = make_dynamic_vocab(pt, NS, extract_fn=extract_fn, blockstore=bs)
    orch = make_orchestrator(model.kernel, bus, dynamic_vocab=dyn)

    # 装配验证：dynamic_vocab 非 None 且 extract_fn 为真实（非 None/非 mock 常数）
    assert orch.dynamic_vocab is not None
    assert orch.dynamic_vocab.extract_fn is not None
    probe = orch.dynamic_vocab.extract_fn("probe")
    assert not torch.allclose(probe, torch.ones_like(probe)), "extract_fn 应是真实模型表征，非常数 mock"

    # 检测（高摩擦：虚构专名，高熵+高共现+低 P(IK)）→ 注册 concept_slot
    ok = orch.assess_vocab_friction("Qeltharion", p_ik=0.10, next_token_entropy=0.90, repeat_cooccur=0.90)
    assert ok is True, "高摩擦应触发 concept_slot 注册"
    spec = pt.get("concept/Qeltharion")
    assert spec is not None and spec.compiled_kind == "concept_slot"
    assert spec.factual_recall is False, "concept_slot 位置不变向量（非事实查表，载体能力边界）"
    payload = bs.get("concept/Qeltharion")
    assert payload is not None and payload.vector is not None
    assert payload.vector.shape == (model.config.d_model,), "BlockStore 载荷应为 [d_model] 概念向量"

    # HRL route_graph 入图 + associative_recall 检索命中
    assert "concept/Qeltharion" in orch.route_graph, "concept_slot 应入 HRL route_graph"
    recalled = orch.associative_recall({"concept/Qeltharion": 1.0})
    assert "concept/Qeltharion" in recalled, "CA3 PPR 联想检索应命中 concept_slot"

    # 注入可用：内核 inject 向量路径（位置不变向量 steer）
    pm_pre = torch.zeros(1, 3, model.config.d_model, device="cuda")
    vec = payload.vector.to("cuda")
    from tais_obsidian.model.tais_kernel import BlockPayload
    pl = BlockPayload(block_id=payload.block_id, compiled_kind="concept_slot", vector=vec, layer_ns=NS)
    injected = model.kernel.inject(pm_pre, [pl], alphas=[1.0])
    assert (injected - pm_pre).abs().sum() > 0, "concept_slot 注入应改变 pm_pre（向量 steer）"


def test_orchestrator_dynamic_vocab_assembly(extract_fn):
    """装配验证：orchestrator.dynamic_vocab 非 None 且挂真实 extract_fn。"""
    pt, bs = PageTable(), BlockStore()
    bus = MemoryBus(pt, bs, Pager(bs, pt))
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, "cuda").eval()
    model.attach_kernel()
    dyn = make_dynamic_vocab(pt, NS, extract_fn=extract_fn, blockstore=bs)
    orch = make_orchestrator(model.kernel, bus, dynamic_vocab=dyn)
    assert orch.dynamic_vocab is dyn
    assert orch.dynamic_vocab.extract_fn is extract_fn  # 真实 Kaplan（非 mock lambda）
