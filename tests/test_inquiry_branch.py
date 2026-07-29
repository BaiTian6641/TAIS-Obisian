"""推理循环求知分支（Inquiry Branch）测试——主动求知闭环 pilot 验证。

对齐实现要求与 docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md §1/§6：
- 路由决策：高 certainty→DirectAnswer；低+命中→DirectAnswer；低+未命中+可学习区
  →AskQuestion/CallTool；完全空白→Decline。
- 审计 token：AskQuestion→<|ask|>、Decline→声明文本（含"暂不可用"）、DirectAnswer→None。
- 可学习区判定（RPL/LP）：mid<certainty<high 触发求知，certainty≤mid 触发 Decline。
- 与推理循环集成：maybe_inquire 在 reasoning_tick 后正确触发；ActiveInquiryLoop.run
  轨迹含 InquiryDecision。
- 真实 certainty 接入（可选，_kaltruth checkpoint）：known→DirectAnswer，
  fake→求知/Decline（certainty 低）。
- 诚实降级：Decline 声明文本含 certainty 与未命中信息。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tais_obsidian.model.inquiry_branch import (  # noqa: E402
    ASK_TOKEN,
    ActiveInquiryLoop,
    InquiryAction,
    InquiryBranch,
    InquiryDecision,
    InquiryRouter,
)
from tais_obsidian.model.manifold_bridge import ThoughtManifoldBridge  # noqa: E402
from tais_obsidian.model.reasoning_loop import (  # noqa: E402
    ReasoningLoop,
    ReasoningTickState,
)
from tais_obsidian.model.thought_core import ThoughtCore  # noqa: E402

# ---------------------------------------------------------------------------
# 路由决策（InquiryRouter.decide 纯规则，CPU 可跑）
# ---------------------------------------------------------------------------


def test_route_high_certainty_direct_answer():
    """高 certainty（≥high_threshold=0.7）→ DirectAnswer（P(IK) 高，已掌握区）。"""
    r = InquiryRouter()
    d = r.decide(certainty=0.85, hrl_hit=False)
    assert d.action == InquiryAction.DIRECT_ANSWER
    assert d.ask_token is None
    assert "P(IK) 高" in d.reason


def test_route_low_certainty_hrl_hit_direct_answer():
    """低 certainty + HRL 命中相关知识块 → DirectAnswer（有知识可答）。"""
    r = InquiryRouter()
    d = r.decide(certainty=0.5, hrl_hit=True)
    assert d.action == InquiryAction.DIRECT_ANSWER
    assert d.ask_token is None
    assert "HRL 命中" in d.reason


def test_route_learnable_zone_ask_question():
    """低+未命中+可学习区（mid<cert<high）且 priority 低/None → AskQuestion。"""
    r = InquiryRouter()
    d = r.decide(certainty=0.55, hrl_hit=False, priority=None)
    assert d.action == InquiryAction.ASK_QUESTION
    assert d.ask_token == ASK_TOKEN
    assert "可学习区" in d.reason and "RPL" in d.reason
    # priority 低于 call_tool_priority 也选 AskQuestion
    d2 = r.decide(certainty=0.55, hrl_hit=False, priority=0.2)
    assert d2.action == InquiryAction.ASK_QUESTION


def test_route_learnable_zone_call_tool():
    """低+未命中+可学习区且 priority 高 → CallTool（自我学习优先，§2.2）。"""
    r = InquiryRouter()
    d = r.decide(certainty=0.55, hrl_hit=False, priority=0.8)
    assert d.action == InquiryAction.CALL_TOOL
    assert d.ask_token == ASK_TOKEN
    assert "CallTool" in d.reason


def test_route_blank_zone_decline():
    """完全空白区（certainty ≤ mid=0.4）→ Decline（诚实降级，RPL 学习成本过高）。"""
    r = InquiryRouter()
    d = r.decide(certainty=0.2, hrl_hit=False)
    assert d.action == InquiryAction.DECLINE
    assert "暂不可用" in d.ask_token
    assert "诚实降级" in d.reason


# ---------------------------------------------------------------------------
# 审计 token（显式显形化红线）
# ---------------------------------------------------------------------------


def test_audit_token_ask_question():
    """AskQuestion → <|ask|> 显式 token（对齐"必须显式出现在 CoT"红线）。"""
    branch = InquiryBranch()
    d = branch.router.decide(certainty=0.55, hrl_hit=False)
    assert branch.inquiry_token(d) == ASK_TOKEN


def test_audit_token_decline_message():
    """Decline → 声明文本含"暂不可用"（诚实降级红线）。"""
    branch = InquiryBranch()
    d = branch.router.decide(certainty=0.2, hrl_hit=False)
    tok = branch.inquiry_token(d)
    assert tok is not None and "暂不可用" in tok


def test_audit_token_direct_answer_none():
    """DirectAnswer / None decision → None（无求知动作不显形）。"""
    branch = InquiryBranch()
    d = branch.router.decide(certainty=0.9, hrl_hit=False)
    assert branch.inquiry_token(d) is None
    assert branch.inquiry_token(None) is None


# ---------------------------------------------------------------------------
# 可学习区判定（RPL/LP 对齐：mid<cert<high 求知，cert≤mid Decline）
# ---------------------------------------------------------------------------


def test_learnable_zone_boundaries():
    """RPL/LP 边界：mid<certainty<high 触发求知；certainty≤mid 触发 Decline。"""
    r = InquiryRouter(high_threshold=0.7, mid_threshold=0.4)
    # 可学习区内（0.4 < c < 0.7）→ 求知（Ask/CallTool）
    assert r.decide(0.41, hrl_hit=False).action in (
        InquiryAction.ASK_QUESTION,
        InquiryAction.CALL_TOOL,
    )
    assert r.decide(0.69, hrl_hit=False).action in (
        InquiryAction.ASK_QUESTION,
        InquiryAction.CALL_TOOL,
    )
    # 边界 certainty == mid（0.4）→ 完全空白区 Decline（≤mid 不可学）
    assert r.decide(0.40, hrl_hit=False).action == InquiryAction.DECLINE
    # certainty 低于 mid → Decline
    assert r.decide(0.1, hrl_hit=False).action == InquiryAction.DECLINE
    # 边界 certainty == high（0.7）→ DirectAnswer（≥high 已掌握）
    assert r.decide(0.70, hrl_hit=False).action == InquiryAction.DIRECT_ANSWER


# ---------------------------------------------------------------------------
# 与推理循环集成（maybe_inquire + ActiveInquiryLoop.run）
# ---------------------------------------------------------------------------


def _make_loop(core_dim=256, manifold_dim=32, max_ticks=4, device="cpu"):
    """搭建 ReasoningLoop + InquiryBranch + ActiveInquiryLoop（pilot，无 kernel mock）。"""
    tc = ThoughtCore(
        core_dim=core_dim, n_groups=8, history=4,
        max_ticks=max_ticks, manifold_dim=manifold_dim,
    )
    bridge = ThoughtManifoldBridge(d_model=core_dim, manifold_dim=manifold_dim)
    rl = ReasoningLoop(thought_core=tc, bridge=bridge, kernel=None)  # mock certainty
    branch = InquiryBranch(router=InquiryRouter(), kernel=None)
    loop = ActiveInquiryLoop(reasoning_loop=rl, inquiry_branch=branch)
    return loop.to(device), tc


def _mock_tick_state(certainty: float, tick_index: int = 0, core_dim: int = 256, manifold_dim: int = 32):
    """构造一个 mock ReasoningTickState（maybe_inquire 只读 certainty）。"""
    coord = torch.zeros(1, 2, manifold_dim)
    disp = torch.zeros(1, 2, manifold_dim)
    return ReasoningTickState(
        tick_index=tick_index, current_coord=coord, disp=disp, certainty=certainty
    )


def test_maybe_inquire_triggers_on_low_certainty_miss():
    """maybe_inquire：certainty 低且未命中 → 返回求知决策；高/命中 → None。"""
    branch = InquiryBranch()
    # 低 certainty + 未命中 → 求知（可学习区 Ask / 空白区 Decline）
    d_low = branch.maybe_inquire(_mock_tick_state(0.5), hrl_hit=False)
    assert d_low is not None and d_low.action in (
        InquiryAction.ASK_QUESTION, InquiryAction.CALL_TOOL, InquiryAction.DECLINE,
    )
    # 高 certainty → None（DirectAnswer，不进求知分支）
    assert branch.maybe_inquire(_mock_tick_state(0.9), hrl_hit=False) is None
    # 低 certainty + 命中 → None（DirectAnswer，有知识可答）
    assert branch.maybe_inquire(_mock_tick_state(0.5), hrl_hit=True) is None


def test_active_inquiry_loop_trajectory_contains_decisions():
    """ActiveInquiryLoop.run：轨迹元素为 (tick_state, InquiryDecision|None)。"""
    loop, tc = _make_loop()
    B, T = 1, 2
    # 小范数输入使 mock certainty=sigmoid(norm) 落入可学习区（mid0.4<cert<high0.7）
    # norm≈0 → sigmoid≈0.5（可学习区），确保触发求知分支（非 DirectAnswer 早停）。
    state0 = torch.randn(B, T, tc.core_dim) * 0.01
    target = torch.randn(B, tc.manifold_dim)
    # hrl_hit_fn 恒 False（未命中）→ 低 certainty tick 触发求知分支
    final_state, traj, stop_tick = loop.run(
        state0, target_coord=target, hrl_hit_fn=lambda k, s: False, max_ticks=4,
    )
    assert final_state.shape == (B, T, tc.core_dim)
    assert 1 <= stop_tick <= 4
    assert len(traj) == stop_tick
    for tick_state, decision in traj:
        assert isinstance(tick_state, ReasoningTickState)
        assert decision is None or isinstance(decision, InquiryDecision)
    # mock certainty≈sigmoid(小norm)≈0.5 落在可学习区 → 至少一个 tick 触发求知（decision 非 None）
    assert any(d is not None for _, d in traj), "低 certainty 未命中应触发求知分支"


def test_active_inquiry_loop_inquiry_executor_closes_loop():
    """求知闭环：inquiry_executor 返回 True（获新证据）→ 重评估 certainty（reason 标注闭环）。"""
    loop, tc = _make_loop()
    B, T = 1, 2
    state0 = torch.randn(B, T, tc.core_dim)
    target = torch.randn(B, tc.manifold_dim)
    calls = []

    def executor(decision):
        calls.append(decision.action)
        return True  # 模拟求知成功获得新证据

    _, traj, _ = loop.run(
        state0, target_coord=target, hrl_hit_fn=lambda k, s: False,
        inquiry_executor=executor, max_ticks=2,
    )
    # Ask/CallTool 决策经执行器；成功后 reason 含"闭环"重评估标注
    inquire_decisions = [d for _, d in traj if d is not None]
    if any(d.action in (InquiryAction.ASK_QUESTION, InquiryAction.CALL_TOOL) for d in inquire_decisions):
        assert len(calls) >= 1, "Ask/CallTool 决策应调用 inquiry_executor"
        assert any("闭环" in d.reason for d in inquire_decisions), "求知成功应重评估 certainty"


def test_active_inquiry_loop_decline_no_executor():
    """Decline（完全空白）不调用执行器（诚实降级无求知动作）。"""
    loop, tc = _make_loop()
    B, T = 1, 2
    state0 = torch.randn(B, T, tc.core_dim) * 0.0  # norm≈0 → mock certainty≈0.5（可学习区）
    target = torch.randn(B, tc.manifold_dim)
    calls = []
    # 强制路由器 mid 高使 mock certainty 落入完全空白区 → Decline
    loop.inquiry_branch.router = InquiryRouter(high_threshold=0.9, mid_threshold=0.8)
    _, traj, _ = loop.run(
        state0, target_coord=target, hrl_hit_fn=lambda k, s: False,
        inquiry_executor=lambda d: calls.append(d.action) or True, max_ticks=2,
    )
    for _, d in traj:
        if d is not None and d.action == InquiryAction.DECLINE:
            pass
    # Decline 不进执行器（calls 只含 Ask/CallTool）
    assert all(a in (InquiryAction.ASK_QUESTION, InquiryAction.CALL_TOOL) for a in calls)


# ---------------------------------------------------------------------------
# 诚实降级（Decline 声明文本含 certainty 与未命中信息）
# ---------------------------------------------------------------------------


def test_decline_message_contains_certainty_and_miss():
    """Decline 声明文本含 certainty 数值与"HRL 未命中"信息（诚实降级红线）。"""
    branch = InquiryBranch()
    d = branch.router.decide(certainty=0.25, hrl_hit=False)
    assert d.action == InquiryAction.DECLINE
    assert "暂不可用" in d.ask_token
    assert "0.25" in d.ask_token, "声明文本应含 certainty 数值"
    assert "未命中" in d.ask_token, "声明文本应含未命中信息"
    # maybe_inquire 路径（用 branch.blank_message 模板）
    d2 = branch.maybe_inquire(_mock_tick_state(0.25), hrl_hit=False)
    assert d2 is not None and "暂不可用" in d2.ask_token and "0.25" in d2.ask_token


# ---------------------------------------------------------------------------
# 真实 certainty 接入（可选，_kaltruth checkpoint，CUDA）
# ---------------------------------------------------------------------------

TUNED_DIR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_kaltruth"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
READ_LAYER = 10  # kaltruth 选定末 GDN 层（kal-gdn2-truth-finetune.md）


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")
def test_real_certainty_kaltruth_checkpoint():
    """真实 KAL certainty（_kaltruth）：known 文本→DirectAnswer，fake 文本→求知/Decline。"""
    if not TUNED_DIR.exists():
        pytest.skip("kaltruth checkpoint 未产出（先跑 scripts/kal_truth_finetune_gdn2.py）")
    import numpy as np
    from safetensors.torch import load_file

    import kal_probe as kp
    from tais_obsidian.config import ModelConfig
    from tais_obsidian.model.model import TaisObsidianForCausalLM
    from tais_obsidian.tokenizer_io import TokenizerIO

    # 加载坑（kal-gdn2-truth-finetune.md）：先 attach_kernel 再 load_state_dict(strict=True)
    cfg = ModelConfig.from_json(TUNED_DIR / "config.json")
    model = TaisObsidianForCausalLM(cfg)
    model.attach_kernel()
    sd = load_file(str(TUNED_DIR / "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model = model.to("cuda").eval()
    tok = TokenizerIO(str(TOK))

    import diverse_truth_data as dt
    rng = np.random.default_rng(777)
    known_texts = dt.build_real_statements(rng, 8)
    fake_texts = kp.build_fake_fact_texts(rng, 8)

    @torch.no_grad()
    def p_known(texts):
        ids = kp.encode_fixed(tok, texts, 48)
        feats, _ = kp.forward_collect(model, ids, [READ_LAYER], "cuda", 16, "last")
        h = torch.from_numpy(feats[READ_LAYER]).to("cuda")
        probs = torch.softmax(model.kernel.kal_l1(h).float(), dim=-1)
        return probs[:, 0].cpu().numpy()

    router = InquiryRouter()
    pk_known = p_known(known_texts)
    pk_fake = p_known(fake_texts)
    print(f"\n[test] known P(known) 均值 {pk_known.mean():.3f} | fake P(known) 均值 {pk_fake.mean():.3f}")
    # known 文本 certainty 高 → DirectAnswer（多数样本）
    known_actions = [router.decide(float(c), hrl_hit=False).action for c in pk_known]
    assert known_actions.count(InquiryAction.DIRECT_ANSWER) >= len(known_actions) // 2, (
        f"known 文本应多数 DirectAnswer，实得 {known_actions}"
    )
    # fake 文本 certainty 低 → 求知（Ask/CallTool）或 Decline（不应硬答）
    fake_actions = [router.decide(float(c), hrl_hit=False).action for c in pk_fake]
    non_direct = sum(a != InquiryAction.DIRECT_ANSWER for a in fake_actions)
    assert non_direct >= len(fake_actions) // 2, (
        f"fake 文本应多数求知/Decline（非硬答），实得 {fake_actions}"
    )
