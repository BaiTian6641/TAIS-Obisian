"""求知执行器（Inquiry Executor）测试——主动求知闭环"执行+学习"pilot 验证。

对齐实现要求与 docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md §2/§3/§4/§7：
- Evidence 构造：字段齐全、credibility 默认（user>doc>web）。
- CrossVerifier：一致证据 verified；冲突证据标 conflict 不覆盖；多源一致提分。
- KnowledgeBlockWriter：写入 BlockStore 可 get；累积不覆盖（同 id 版本化/冲突保留双方）。
- InquiryExecutor：AskQuestion（mock ask_fn）→验证写入→True；CallTool 同理；
  Decline→False 不执行。
- 闭环：ActiveInquiryPipeline 低 certainty→求知→执行→重评估 certainty 升高→闭环。
- 红线：未验证证据不写入（裸自我修正防护 arXiv:2310.01798）；冲突保留双方标分歧。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tais_obsidian.model.inquiry_branch import (  # noqa: E402
    ActiveInquiryLoop,
    InquiryAction,
    InquiryBranch,
    InquiryRouter,
)
from tais_obsidian.model.inquiry_executor import (  # noqa: E402
    SOURCE_CREDIBILITY,
    ActiveInquiryPipeline,
    CrossVerifier,
    Evidence,
    InquiryExecutor,
    KnowledgeBlockWriter,
)
from tais_obsidian.model.manifold_bridge import ThoughtManifoldBridge  # noqa: E402
from tais_obsidian.model.reasoning_loop import ReasoningLoop  # noqa: E402
from tais_obsidian.model.thought_core import ThoughtCore  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Evidence 构造（字段齐全 + credibility 默认 user>doc>web）
# ---------------------------------------------------------------------------


def test_evidence_fields_and_default_credibility():
    """Evidence 字段齐全；credibility 缺省按 source 取默认（user>doc>web）。"""
    ev = Evidence(content="地球绕太阳转", source="user")
    assert ev.content == "地球绕太阳转"
    assert ev.source == "user"
    assert ev.credibility == SOURCE_CREDIBILITY["user"] == 0.9
    assert ev.timestamp > 0
    assert ev.verified is False
    # 三通道可信度排序：user(0.9) > doc(0.7) > web(0.5)（§4.1 信任度加权）
    assert (Evidence(content="x", source="user").credibility
            > Evidence(content="x", source="doc").credibility
            > Evidence(content="x", source="web").credibility)


def test_evidence_explicit_credibility_override_and_validation():
    """显式 credibility 覆盖默认；越界/未知 source 抛错（fail-closed）。"""
    ev = Evidence(content="x", source="web", credibility=0.3)
    assert ev.credibility == 0.3  # 显式覆盖默认 0.5
    with pytest.raises(ValueError):
        Evidence(content="x", source="unknown_src")
    with pytest.raises(ValueError):
        Evidence(content="x", source="user", credibility=1.5)


# ---------------------------------------------------------------------------
# CrossVerifier（一致 verified / 冲突标 conflict 不覆盖 / 多源提分）
# ---------------------------------------------------------------------------


def _fixed_embed_fn(dim: int = 16):
    """注入确定性 embed_fn：相同文本→相同向量，相似文本→高余弦（pilot mock 语义）。"""
    table = {}
    def embed(text: str) -> torch.Tensor:
        if text not in table:
            torch.manual_seed(abs(hash(text)) % (2**31))
            table[text] = torch.randn(dim)
        return table[text]
    return embed


def test_verifier_consistent_evidence_verified():
    """与既有知识一致的证据 verified=True（一致性>阈值 且 无冲突）。"""
    v = CrossVerifier(embed_fn=_fixed_embed_fn())
    old = Evidence(content="太阳从东方升起", source="doc", verified=True)
    # 新证据与既有一条**完全相同**内容 → 余弦≈1 → 高一致 verified
    new = Evidence(content="太阳从东方升起", source="user")
    verified, consistency, conflict = v.verify(new, [old])
    assert verified is True
    assert conflict is False
    assert consistency > v.consistency_threshold


def test_verifier_conflicting_evidence_flagged_not_verified():
    """与既有知识冲突（余弦<冲突阈值）→ conflict=True 且 verified=False（不静默覆盖）。"""
    dim = 16
    # 构造一对**相反**向量（余弦≈-1 → 必冲突）：embed_fn 按内容返回固定正/反向量
    pos = torch.ones(dim)
    neg = -torch.ones(dim)
    mapping = {"A": pos, "B": neg}
    v = CrossVerifier(embed_fn=lambda t: mapping[t], conflict_threshold=0.3)
    old = Evidence(content="A", source="doc", verified=True, embedding=pos)
    new = Evidence(content="B", source="user", embedding=neg)
    verified, consistency, conflict = v.verify(new, [old])
    assert conflict is True, "相反向量应判冲突"
    assert verified is False, "冲突未决不 verified（保留双方标分歧，不静默覆盖）"


def test_verifier_multi_source_boost_raises_score():
    """多源一致（既有知识含不同 source 且一致）→ consistency 提分（§3.2 多源一致性）。"""
    embed = _fixed_embed_fn()
    v = CrossVerifier(embed_fn=embed)
    new = Evidence(content="光速恒定", source="user")
    # 无多源（既有为空）vs 有多源（既有一条不同 source 一致证据）
    _, score_single, _ = v.verify(new, [])
    old_other_src = Evidence(content="光速恒定", source="doc", verified=True)
    _, score_multi, _ = v.verify(new, [old_other_src])
    assert score_multi > score_single, "多源一致应提分（多条独立来源一致）"


# ---------------------------------------------------------------------------
# KnowledgeBlockWriter（写入可 get / 累积不覆盖 / 冲突保留双方）
# ---------------------------------------------------------------------------


def test_writer_writes_verified_evidence_and_gettable():
    """verified 证据写入 BlockStore 可 get；payload 含 source_credibility 元数据（§4.1）。"""
    bs = BlockStore()
    w = KnowledgeBlockWriter()
    ev = Evidence(content="水在100°C沸腾", source="doc", verified=True)
    bid = w.write(ev, bs, namespace="inquiry", consistency=0.9)
    assert bid is not None
    payload = bs.get(bid)
    assert payload is not None
    assert payload["content"] == "水在100°C沸腾"
    assert payload["source_credibility"] == SOURCE_CREDIBILITY["doc"]
    assert payload["verified"] is True and payload["draft"] is True


def test_writer_rejects_unverified_evidence():
    """未验证（verified=False）证据拒绝写入（裸自我修正防护红线 arXiv:2310.01798）。"""
    bs = BlockStore()
    w = KnowledgeBlockWriter()
    ev = Evidence(content="未验证的主张", source="web", verified=False)
    assert w.write(ev, bs) is None, "未验证证据绝不写入（裸自我修正防护）"
    assert bs.stats()["L1"] == 0, "BlockStore 应保持空（未验证不写入）"


def test_writer_accumulates_versions_no_overwrite():
    """累积不覆盖：同内容重复写入 → 版本号自增（:v1/:v2），旧版保留（抗坍缩红线）。"""
    bs = BlockStore()
    w = KnowledgeBlockWriter()
    ev = Evidence(content="同一条知识", source="doc", verified=True)
    bid1 = w.write(ev, bs)
    bid2 = w.write(ev, bs)  # 同内容再写 → 新版本，不覆盖旧版
    assert bid1 is not None and bid2 is not None and bid1 != bid2
    assert bid1.endswith(":v1") and bid2.endswith(":v2"), "版本化 :v{n} 自增"
    assert bs.get(bid1) is not None and bs.get(bid2) is not None, "累积不覆盖：旧版保留"


def test_writer_conflict_preserves_both_and_flags_dispute():
    """冲突未决：新版本写入但标 conflict + dispute_note，保留双方（不静默覆盖红线）。"""
    bs = BlockStore()
    w = KnowledgeBlockWriter()
    ev_old = Evidence(content="旧观点", source="doc", verified=True)
    ev_new = Evidence(content="冲突的新观点", source="user", verified=True)
    bid_old = w.write(ev_old, bs, conflict=False)
    bid_new = w.write(ev_new, bs, conflict=True)  # 冲突未决 → 标分歧
    p_old, p_new = bs.get(bid_old), bs.get(bid_new)
    assert p_old is not None and p_new is not None, "冲突保留双方（不覆盖旧版）"
    assert p_new["conflict"] is True and p_new["dispute_note"] is not None
    assert "分歧" in p_new["dispute_note"]


# ---------------------------------------------------------------------------
# InquiryExecutor（AskQuestion/CallTool 验证写入→True；Decline→False 不执行）
# ---------------------------------------------------------------------------


def _make_executor(ask_out=None, tool_out=None, embed_fn=None):
    """搭建 InquiryExecutor（注入 mock ask_fn/tool_fn 与确定性 embed_fn）。"""
    bs = BlockStore()
    verifier = CrossVerifier(embed_fn=embed_fn or _fixed_embed_fn())
    ex = InquiryExecutor(
        blockstore=bs, verifier=verifier,
        ask_fn=(lambda q: ask_out) if ask_out is not None else None,
        tool_fn=(lambda q: tool_out) if tool_out is not None else None,
    )
    return ex, bs


def test_executor_ask_question_verified_writes_returns_true():
    """AskQuestion（mock ask_fn）→ 验证通过 → 写入 BlockStore → 返回 True（获新证据闭环）。"""
    ex, bs = _make_executor(ask_out="用户解释：答案是42")
    from tais_obsidian.model.inquiry_branch import InquiryDecision
    d = InquiryDecision(action=InquiryAction.ASK_QUESTION, certainty=0.5,
                        reason="什么是答案", ask_token="<|ask|>")
    ok = ex(d)
    assert ok is True, "验证通过应返回 True（求知成功获新证据，闭环重评估）"
    assert bs.stats()["L1"] >= 1, "验证通过应写入 BlockStore"


def test_executor_call_tool_verified_writes_returns_true():
    """CallTool（mock tool_fn）→ 验证 → 写入 → 返回 True。"""
    ex, bs = _make_executor(tool_out="文档：π≈3.14159")
    from tais_obsidian.model.inquiry_branch import InquiryDecision
    d = InquiryDecision(action=InquiryAction.CALL_TOOL, certainty=0.5,
                        reason="查π值", ask_token="<|ask|>")
    assert ex(d) is True
    assert bs.stats()["L1"] >= 1


def test_executor_decline_and_direct_answer_not_executed():
    """Decline/DirectAnswer 不执行求知动作（诚实降级/已掌握区），返回 False 不写入。"""
    ex, bs = _make_executor(ask_out="不应被问到")
    from tais_obsidian.model.inquiry_branch import InquiryDecision
    d_decl = InquiryDecision(action=InquiryAction.DECLINE, certainty=0.2, reason="空白")
    d_dir = InquiryDecision(action=InquiryAction.DIRECT_ANSWER, certainty=0.9, reason="已知")
    assert ex(d_decl) is False and ex(d_dir) is False
    assert bs.stats()["L1"] == 0, "Decline/DirectAnswer 不写入"


def test_executor_unverified_evidence_not_written_returns_false():
    """冲突未决证据：verified=False → 不写入、返回 False（裸自我修正防护）。"""
    dim = 16
    mapping = {"先验": torch.ones(dim), "冲突证据": -torch.ones(dim)}
    embed = lambda t: mapping.get(t, torch.randn(dim))
    bs = BlockStore()
    verifier = CrossVerifier(embed_fn=embed, conflict_threshold=0.3)
    ex = InquiryExecutor(blockstore=bs, verifier=verifier,
                         ask_fn=lambda q: "冲突证据")
    # 先注入一条先验知识（与即将到来的冲突证据相反）
    ex._knowledge.append(Evidence(content="先验", source="doc", verified=True,
                                  embedding=torch.ones(dim)))
    from tais_obsidian.model.inquiry_branch import InquiryDecision
    d = InquiryDecision(action=InquiryAction.ASK_QUESTION, certainty=0.5, reason="问")
    ok = ex(d)
    assert ok is False, "冲突未验证证据：不写入，返回 False（不闭环）"
    assert bs.stats()["L1"] == 0, "未验证证据绝不写入（裸自我修正防护）"


# ---------------------------------------------------------------------------
# 闭环：ActiveInquiryPipeline 低 certainty→求知→执行→重评估→闭环
# ---------------------------------------------------------------------------


def _make_pipeline(core_dim=256, manifold_dim=32, max_ticks=4, ask_out="用户解释"):
    """搭建 ActiveInquiryPipeline（复用 test_inquiry_branch 的 _make_loop 结构）。"""
    tc = ThoughtCore(core_dim=core_dim, n_groups=8, history=4,
                     max_ticks=max_ticks, manifold_dim=manifold_dim)
    bridge = ThoughtManifoldBridge(d_model=core_dim, manifold_dim=manifold_dim)
    rl = ReasoningLoop(thought_core=tc, bridge=bridge, kernel=None)  # mock certainty
    branch = InquiryBranch(router=InquiryRouter(), kernel=None)
    loop = ActiveInquiryLoop(reasoning_loop=rl, inquiry_branch=branch)
    bs = BlockStore()
    ex = InquiryExecutor(blockstore=bs, verifier=CrossVerifier(embed_fn=_fixed_embed_fn()),
                         ask_fn=(lambda q: ask_out) if ask_out is not None else None)
    pipe = ActiveInquiryPipeline(inquiry_loop=loop, executor=ex)
    # ActiveInquiryPipeline 非 nn.Module（组合封装）；.to() 挂到内部 loop 上
    pipe.inquiry_loop.to(DEVICE)
    return pipe, tc, bs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA（对齐测试风格 DEVICE cuda）")
def test_pipeline_low_certainty_inquiry_closes_loop():
    """低 certainty → 求知分支 → 执行器执行 → 验证写入 → 重评估 certainty（闭环）。"""
    pipe, tc, bs = _make_pipeline()
    B, T = 1, 2
    # 小范数输入 → mock certainty=sigmoid(norm)≈0.5（可学习区）→ 触发求知分支
    state0 = torch.randn(B, T, tc.core_dim, device=DEVICE) * 0.01
    target = torch.randn(B, tc.manifold_dim, device=DEVICE)
    final_state, traj, stop_tick, closed = pipe.run(
        state0, target_coord=target, hrl_hit_fn=lambda k, s: False, max_ticks=4,
    )
    assert final_state.shape == (B, T, tc.core_dim)
    # 求知 tick 触发（decision 非 None）
    inquire_decisions = [d for _, d in traj if d is not None]
    assert any(d.action in (InquiryAction.ASK_QUESTION, InquiryAction.CALL_TOOL)
               for d in inquire_decisions), "低 certainty 未命中应触发求知分支"
    # 闭环：求知成功（mock ask_fn 返回+验证通过）→ reason 含"闭环"重评估标注
    assert closed is True, "求知成功获得且验证通过新证据应重评估 certainty（闭环）"
    assert bs.stats()["L1"] >= 1, "验证通过的新证据应写入 BlockStore"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")
def test_pipeline_no_executor_fn_no_closure():
    """执行器无 ask_fn（求知通道缺失）→ 未获新证据 → 不闭环（诚实降级）。"""
    pipe, tc, bs = _make_pipeline(ask_out=None)  # ask_fn=None
    pipe.executor.ask_fn = None
    state0 = torch.randn(1, 2, tc.core_dim, device=DEVICE) * 0.01
    target = torch.randn(1, tc.manifold_dim, device=DEVICE)
    _, traj, _, closed = pipe.run(
        state0, target_coord=target, hrl_hit_fn=lambda k, s: False, max_ticks=3,
    )
    assert closed is False, "求知通道缺失未获新证据：不闭环（诚实降级，不伪造）"
    assert bs.stats()["L1"] == 0, "未获新证据不写入"
