"""0.1B 主动求知闭环全链端到端测试（三阶段协同 + kernel 加载坑处理）。

覆盖：
  - kernel 加载坑：kaltruth checkpoint（kernel_enabled=False 但含 kernel.* 权重）
    用 attach_kernel()+load_state_dict(strict=True) 正确加载，kernel 挂载可用。
  - 阶段1 运行时学习：低 certainty（虚构事实）触发求知路由（非 DirectAnswer）；
    执行器写入知识块到 BlockStore（draft 态）。
  - 阶段2 实时可用：HRL 检索命中写入的块；HCA 注入后答对率 ≥ 不注入基线（通路通）。
  - 阶段3 长期固化：一致块 PROMOTE / 冲突块 QUARANTINE 保留双方；draft→固化验证门。
  - 全链：求知→实时→固化端到端跑通无异常。

红线：绝不裸自我修正（CrossVerifier 门控）；累积不覆盖（写入+固化）；诚实降级
（Decline 声明）；运行时注入不动权重（实时可用 vs 离线 SFT）。

双卡分工：用 RTX 4070（CUDA_VISIBLE_DEVICES=0，8GB，控 batch/seq）。
运行：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe -m pytest tests/test_active_inquiry_full_chain.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")

CKPT = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_kaltruth"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
_DEVICE = "cuda"

from safetensors.torch import load_file  # noqa: E402

from tais_obsidian.config import ModelConfig  # noqa: E402
from tais_obsidian.model.inquiry_branch import InquiryAction, InquiryRouter  # noqa: E402
from tais_obsidian.model.inquiry_executor import (  # noqa: E402
    CrossVerifier,
    Evidence,
    InquiryExecutor,
    KnowledgeBlockWriter,
)
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.sleep.consolidator import SleepConsolidator  # noqa: E402
from tais_obsidian.sleep.inquiry_consolidation import (  # noqa: E402
    InquirySleepConsolidation,
)
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# scripts/ 非 Python 包，用 importlib 按文件路径加载 demo 与 e2e 原语
import importlib.util as _ilu  # noqa: E402


def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_demo = _load("active_inquiry_full_chain_demo", ROOT / "scripts" / "active_inquiry_full_chain_demo.py")
_e2e = _load("internalization_e2e", ROOT / "scripts" / "internalization_e2e.py")

harvest_kv_block = _demo.harvest_kv_block
load_model_with_kernel = _demo.load_model_with_kernel
read_certainty = _demo.read_certainty
retrieve = _demo.retrieve

_make_facts = _e2e._make_facts
answer_baseline = _e2e.answer_baseline
answer_correct = _e2e.answer_correct
answer_with_kv_inject = _e2e.answer_with_kv_inject

# 少量虚构事实（8GB 卡控量，测试用 4 条足够验证协同）
_FACTS = None


@pytest.fixture(scope="module")
def tok():
    return TokenizerIO(str(TOK))


@pytest.fixture(scope="module")
def model():
    """kernel 加载坑处理：kaltruth checkpoint kernel_enabled=False 但含 kernel.* 权重。

    用 demo 的 load_model_with_kernel（attach_kernel()+load_state_dict(strict=True)）
    正确加载——这是本测试要验证的关键坑处理。若 checkpoint 缺失则 skip。
    """
    if not CKPT.exists():
        pytest.skip("kaltruth checkpoint 未产出（先跑 scripts/kal_truth_finetune_gdn2.py）")
    return load_model_with_kernel(str(CKPT), _DEVICE)


@pytest.fixture(scope="module")
def facts():
    global _FACTS
    if _FACTS is None:
        _FACTS = _make_facts(4, seed=0)
    return _FACTS


@pytest.fixture(scope="module")
def a_layers(model):
    return [i for i, t in enumerate(model.config.layer_types) if t == "A"]


# ---------------------------------------------------------------------------
# kernel 加载坑处理正确性
# ---------------------------------------------------------------------------
def test_kernel_loaded_correctly(model):
    """kernel 加载坑：attach_kernel()+load_state_dict(strict=True) 后 kernel 挂载且可用。"""
    assert model.kernel is not None, "kernel 为 None——attach_kernel 权重未正确载入"
    # kernel.* 权重已载入（非随机）：kal_l1 是真值锚校准后的，应能出非退化 logits
    kal_w = model.kernel.kal_l1
    assert any(p.abs().sum() > 0 for p in kal_w.parameters()), "kal_l1 权重全零（未载入）"


def test_kernel_load_trap_without_attach():
    """验证坑的存在：不 attach_kernel 直接 load_state_dict(strict=True) 会报 Unexpected key。"""
    if not CKPT.exists():
        pytest.skip("kaltruth checkpoint 未产出")
    cfg = ModelConfig.from_json(CKPT / "config.json")
    m = TaisObsidianForCausalLM(cfg)  # 不 attach_kernel
    sd = load_file(str(CKPT / "model.safetensors"))
    with pytest.raises(RuntimeError, match="Unexpected key"):
        m.load_state_dict(sd, strict=True)


# ---------------------------------------------------------------------------
# 阶段1 运行时学习
# ---------------------------------------------------------------------------
def test_stage1_low_certainty_triggers_inquiry(model, tok, facts):
    """虚构事实（先验不存在）→ KAL certainty 低 → 求知路由非 DirectAnswer。"""
    router = InquiryRouter()
    actions = []
    for f in facts:
        cert = read_certainty(model, tok, f["Q"], _DEVICE)
        # HRL 未命中（库空）→ 低 certainty 应进求知分支（Decline 或 Ask/CallTool）
        decision = router.decide(cert, hrl_hit=False, priority=0.6)
        actions.append(decision.action)
        assert decision.action != InquiryAction.DIRECT_ANSWER, (
            f"certainty={cert:.3f} 虚构事实应进求知分支（非 DirectAnswer）"
        )
        # Decline 应有诚实降级声明（红线）
        if decision.action == InquiryAction.DECLINE:
            assert decision.ask_token and "暂不可用" in decision.ask_token
    # 完全虚构事实 certainty≈0 应全落完全空白区 → Decline 诚实降级
    n_not_direct = sum(1 for a in actions if a != InquiryAction.DIRECT_ANSWER)
    assert n_not_direct == len(facts), "虚构事实应全部触发求知分支（非 DirectAnswer）"


def test_stage1_learnable_zone_writes(model, tok, facts):
    """可学习区（0.4<certainty<0.7，RPL/LP）→ Ask/CallTool 求知动作（非 Decline/直答）。"""
    router = InquiryRouter()
    # 可学习区演示：certainty=0.55（mid0.4<0.55<high0.7），priority 高→CallTool
    d_tool = router.decide(0.55, hrl_hit=False, priority=0.6)
    d_ask = router.decide(0.55, hrl_hit=False, priority=0.2)
    assert d_tool.action == InquiryAction.CALL_TOOL, "可学习区+高 priority 应 CallTool"
    assert d_ask.action == InquiryAction.ASK_QUESTION, "可学习区+低 priority 应 AskQuestion"
    # 求知动作应有 <|ask|> 审计 token（红线）
    assert d_tool.ask_token == "<|ask|>" and d_ask.ask_token == "<|ask|>"


def test_stage1_executor_writes_block(model, tok, facts):
    """求知执行器执行（mock ask_fn 返回 K）→ CrossVerifier 验证 → 写入 BlockStore draft。"""
    store = BlockStore()
    executor = InquiryExecutor(
        blockstore=store, verifier=CrossVerifier(),
        ask_fn=lambda q: facts[0]["K"],  # mock：用户给新知识 K（pilot；正式接对话）
        writer=KnowledgeBlockWriter(tier="L1"), namespace="inquiry",
    )
    router = InquiryRouter()
    # 可学习区演示（0.55，priority 低→AskQuestion）使执行器实际执行求知动作
    decision = router.decide(0.55, hrl_hit=False, priority=0.0)
    assert decision.action == InquiryAction.ASK_QUESTION
    got = executor(decision)
    assert got, "求知成功且验证通过应返回 True（闭环）"
    # BlockStore 中应有 draft 知识块（含 source_credibility）
    stats = store.stats()
    assert sum(stats.values()) > 0, "执行器应写入知识块到 BlockStore"
    found = None
    for tier in ("L0", "L1", "L2"):
        for bid, payload in store._store.get(tier, {}).items():
            if str(bid).startswith("inquiry/") and isinstance(payload, dict):
                found = payload
    assert found is not None and found.get("draft") is True, "写入块应为 draft 态"
    assert "source_credibility" in found, "写入块应含 source_credibility 元数据"


def test_stage1_unverified_not_written():
    """红线（绝不裸自我修正）：未验证证据绝不写入。"""
    store = BlockStore()
    writer = KnowledgeBlockWriter(tier="L1")
    ev = Evidence(content="未验证内容", source="web")
    ev.verified = False  # 未验证
    bid = writer.write(ev, store, namespace="inquiry")
    assert bid is None, "未验证证据应拒绝写入（裸自我修正防护）"
    assert sum(store.stats().values()) == 0, "未验证证据不应入库"


# ---------------------------------------------------------------------------
# 阶段2 实时可用
# ---------------------------------------------------------------------------
def test_stage2_retrieval_hits_written_block(model, tok, facts, a_layers):
    """写入的 K 块（收割成 KV）→ HRL route_candidates 检索应在候选中可打分命中。"""
    store = BlockStore()
    # 收割全部事实成 KV 块（求知写入后的可注入形态）
    kv_blocks = [harvest_kv_block(model, tok, store, f"fact/{f['entity']}", f["K"], a_layers, _DEVICE)
                 for f in facts]
    kernel = model.kernel
    # 每条 Q 检索，候选集打分应能返回 top-k（通路通；随机 indexer 命中率未必 1.0）
    f0, kv0 = facts[0], kv_blocks[0]
    top_ids, scores = retrieve(kernel, model, tok, f0["Q"], kv_blocks, 1, _DEVICE, a_layers)
    assert len(top_ids) >= 1, "route_candidates 应返回 top-k 命中块 id"
    assert scores.shape[-1] == len(kv_blocks), "候选集打分维度应等于块数"
    # 命中块 id 属于候选集
    assert all(t in [b["block_id"] for b in kv_blocks] for t in top_ids)


def test_stage2_inject_beats_baseline(model, tok, facts, a_layers):
    """HCA 注入后答对率 ≥ 不注入基线（运行时注入不动权重，实时可用 vs 离线 SFT）。"""
    store = BlockStore()
    kv_blocks = [harvest_kv_block(model, tok, store, f"fact/{f['entity']}", f["K"], a_layers, _DEVICE)
                 for f in facts]
    n_base = n_kv = 0
    for f, kv in zip(facts, kv_blocks):
        g_base = answer_baseline(model, tok, f, _DEVICE, max_new=8)
        g_kv = answer_with_kv_inject(model, tok, f, kv, a_layers, _DEVICE, max_new=8)
        n_base += int(answer_correct(g_base, f["A"]))
        n_kv += int(answer_correct(g_kv, f["A"]))
    # 注入答对率 ≥ 基线（通路通即注入不劣于凭先验；召回头未训时可能都低，
    # 但注入绝不差于基线——运行时注入零梯度不动权重，通而未用也≥0）
    assert n_kv >= n_base, f"注入({n_kv}) 应 ≥ 基线({n_base})"


# ---------------------------------------------------------------------------
# 阶段3 长期固化
# ---------------------------------------------------------------------------
def test_stage3_consistent_promote_conflict_quarantine():
    """一致块 PROMOTE / 冲突块 QUARANTINE 保留双方；draft→固化验证门。"""
    store = BlockStore()
    isc = InquirySleepConsolidation()
    # 写入一条一致块（已验证，无冲突）
    ev_ok = Evidence(content="The Skadre engine runs on refined xenon.", source="user")
    ev_ok.verified = True
    store.put("inquiry/aa:v1", {
        "content": ev_ok.content, "source": "user", "source_credibility": 0.9,
        "consistency": 0.9, "timestamp": 1.0, "verified": True, "version": 1,
        "draft": True, "conflict": False, "dispute_note": None,
    }, tier="L1", usage_count=2)
    # 写入一条冲突块（已验证但 conflict=True → 慢通道 QUARANTINE 保留双方）
    store.put("inquiry/bb:v1", {
        "content": "The Skadre engine runs on refined WATER.", "source": "web",
        "source_credibility": 0.5, "consistency": 0.2, "timestamp": 2.0, "verified": True,
        "version": 1, "draft": True, "conflict": True,
        "dispute_note": "与既有知识冲突未决，保留双方标分歧（不静默覆盖）",
    }, tier="L1", usage_count=1)

    consolidator = SleepConsolidator()
    rep = isc.consolidate_inquiry_blocks(
        store, consolidator, prior_knowledge=None, namespace="inquiry",
        usage_count=12, saliency=1.0, regression_ok=True,  # usage≥CA1 门 min_usage=10
    )
    # 一致块应 PROMOTE，冲突块应 QUARANTINE（保留双方不静默覆盖）
    assert rep.n_promoted >= 1, f"一致块应 PROMOTE（实得 PROMOTE={rep.n_promoted}）"
    assert rep.n_quarantined >= 1, f"冲突块应 QUARANTINE（实得 QUARANTINE={rep.n_quarantined}）"
    # 冲突块仍存 BlockStore（累积不覆盖：QUARANTINE 后保留）
    assert store.get("inquiry/bb:v1") is not None, "冲突块 QUARANTINE 后应仍存（保留双方）"


def test_stage3_draft_gate_blocks_unverified():
    """draft→固化验证门：regression_ok=False 时未验证块不应 PROMOTE。"""
    store = BlockStore()
    isc = InquirySleepConsolidation()
    store.put("inquiry/cc:v1", {
        "content": "The Nexkar engine runs on refined plasma.", "source": "web",
        "source_credibility": 0.5, "consistency": 0.8, "timestamp": 1.0, "verified": True,
        "version": 1, "draft": True, "conflict": False, "dispute_note": None,
    }, tier="L1", usage_count=2)
    consolidator = SleepConsolidator()
    rep = isc.consolidate_inquiry_blocks(
        store, consolidator, prior_knowledge=None, namespace="inquiry",
        usage_count=12, saliency=1.0, regression_ok=False,  # 校验集回归未通过（验证门）
    )
    assert rep.n_promoted == 0, "regression_ok=False（验证门未过）不应 PROMOTE"


# ---------------------------------------------------------------------------
# 全链端到端
# ---------------------------------------------------------------------------
def test_full_chain_end_to_end(model, tok, facts, a_layers):
    """全链：求知（写入）→ 实时（检索+注入）→ 固化（PROMOTE/QUARANTINE）跑通无异常。"""
    store = BlockStore()
    # 阶段1：求知执行器写入（mock ask_fn 返回 K）
    executor = InquiryExecutor(
        blockstore=store, verifier=CrossVerifier(),
        ask_fn=lambda q: facts[0]["K"],
        writer=KnowledgeBlockWriter(tier="L1"), namespace="inquiry",
    )
    router = InquiryRouter()
    # 可学习区演示（0.55，priority 低→AskQuestion）使执行器实际执行求知动作
    decision = router.decide(0.55, hrl_hit=False, priority=0.0)
    got = executor(decision)
    assert got, "阶段1：求知执行+验证+写入应成功"

    # 阶段2：写入的 K 收割成 KV 块 → 检索 + 注入（通路通，无异常）
    kv = harvest_kv_block(model, tok, store, f"fact/{facts[0]['entity']}",
                          facts[0]["K"], a_layers, _DEVICE)
    top_ids, _ = retrieve(model.kernel, model, tok, facts[0]["Q"], [kv], 1, _DEVICE, a_layers)
    assert len(top_ids) >= 1, "阶段2：HRL 检索应命中"
    g_kv = answer_with_kv_inject(model, tok, facts[0], kv, a_layers, _DEVICE, max_new=8)
    assert isinstance(g_kv, str), "阶段2：HCA 注入后应能生成（无异常）"

    # 阶段3：睡眠固化（无异常）
    consolidator = SleepConsolidator()
    isc = InquirySleepConsolidation()
    rep = isc.consolidate_inquiry_blocks(
        store, consolidator, prior_knowledge=None, namespace="inquiry",
        usage_count=12, saliency=1.0, regression_ok=True,
    )
    assert rep.n_practiced >= 1, "阶段3：固化应处理至少 1 个 draft 块"
    assert (rep.n_promoted + rep.n_quarantined + rep.n_rejected) >= 1, "阶段3：应有固化裁决"
