"""交互式全链验证系统回归测试（小尺度：2 facts、1 推理 prompt）。

覆盖（复用 scripts/interactive_validation_demo.py 的相位函数，验证链路不回归）：
  - report 结构完整（四相位 + criteria + known_baselines + honest_notes）；
  - Phase B 教学写入成功（2/2，CrossVerifier 验证 + 累积不覆盖）；
  - baseline ≤ KV 注入召回（方向性判据；小样本不追求 0.625 强度）；
  - Phase A 虚构事实 certainty < 0.5 方向阈（已知口径：虚构≈0）；
  - Phase D 睡眠固化有裁决产出（PROMOTE/QUARANTINE/REJECT ≥ 1）；
  - derive_qa 纯函数口径（自由文本 → Q/A 锚点推导）。

不追求强度指标（小样本），只验证链路不回归。
双卡分工：用计算卡 PRO 4000（CUDA_VISIBLE_DEVICES=1）。
运行：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/test_interactive_validation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")

CKPT = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_unified"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
_DEVICE = "cuda"

import interactive_validation_demo as ivd  # noqa: E402
from tais_obsidian.model.inquiry_branch import InquiryRouter  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402


@pytest.fixture(scope="module")
def loaded():
    """统一 checkpoint 加载（load_unified：gate_mlp 复挂坑处理）；缺失则 skip。"""
    if not CKPT.exists():
        pytest.skip("统一 checkpoint 未产出（先跑 scripts/build_unified_checkpoint.py）")
    model, tok, a_layers = ivd.load_model_and_tokenizer(str(CKPT), str(TOK), _DEVICE)
    return model, tok, a_layers


@pytest.fixture(scope="module")
def mini_run(loaded):
    """小尺度四相位主流程（2 facts、1 推理 prompt），一次跑完供各断言复用。"""
    model, tok, a_layers = loaded
    torch.manual_seed(0)
    store = BlockStore()
    router = InquiryRouter()
    executor, model_embed = ivd.make_executor(model, tok, a_layers, _DEVICE, store)
    manifold = ivd.make_manifold(model.config.d_model)
    facts = ivd._make_facts(2, seed=0)

    pa = ivd.phase_a_blank_decline(model, tok, facts, _DEVICE, router)
    taught = ivd.teach_facts(model, tok, facts, store, executor, router, _DEVICE, a_layers)
    pb = ivd.eval_recall(model, tok, taught, _DEVICE, a_layers, max_new=8)
    pc = ivd.phase_c_probes(model, tok, ivd.REASONING_PROMPTS[:1], _DEVICE, manifold,
                            trace_new=8, seed=0)
    pd = ivd.phase_d_sleep(model, tok, store, executor, model_embed, facts,
                           _DEVICE, a_layers)
    meta = {"ckpt": str(CKPT), "n_facts": 2, "seed": 0, "read_layer": ivd.READ_LAYER,
            "device": _DEVICE, "timestamp": "test"}
    report = ivd.build_report(pa, pb, pc, pd, meta)
    return report


def test_derive_qa_pure():
    """derive_qa 纯函数：自由文本事实 → K/Q/A 锚点（末句判对口径）。"""
    f = ivd.derive_qa("蓝鲸是一种虚构的哺乳动物，背上有银色的条纹。")
    assert f["K"] and f["Q"] and f["A"] and f["entity"]
    assert f["A"] == "背上有银色的条纹", "末句应为判对锚"
    f3 = {"K": "The Skadre engine runs on refined xenon.",
          "Q": "What does the Skadre engine run on?", "A": "xenon"}
    assert all(f3.values()), "显式三段式锚点直接用（REPL 解析侧语义）"


def test_report_structure_complete(mini_run):
    """report 结构完整：四相位 + criteria + known_baselines + honest_notes。"""
    for key in ("meta", "known_baselines", "phase_a_blank_decline",
                "phase_b_teach_recall", "phase_c_probes", "phase_d_sleep",
                "criteria", "all_pass", "honest_notes"):
        assert key in mini_run, f"report 缺 {key}"
    for phase in ("phase_a_certainty_low", "phase_a_decline_rate", "phase_b_write_rate",
                  "phase_b_baseline_zero", "phase_b_kv_recall", "phase_b_retrieval",
                  "phase_c_grid_no_code", "phase_d_verdicts_present"):
        c = mini_run["criteria"][phase]
        assert {"value", "expected", "pass", "note"} <= set(c), f"criteria.{phase} 字段不全"


def test_phase_b_write_success(mini_run):
    """Phase B：2 条事实全部写入成功（CrossVerifier 验证 + 累积不覆盖版本化）。"""
    pb = mini_run["phase_b_teach_recall"]
    assert pb["n_written"] == 2, f"写入 {pb['n_written']}/2，写入链路回归"
    assert pb["write_rate"] == 1.0


def test_phase_b_inject_ge_baseline(mini_run):
    """KV 注入召回 ≥ baseline（方向性；小样本不追求 0.625 强度）。"""
    pb = mini_run["phase_b_teach_recall"]
    assert pb["acc_kv_inject"] >= pb["acc_baseline"], (
        f"注入 {pb['acc_kv_inject']} < 基线 {pb['acc_baseline']}——召回链路回归")


def test_phase_a_fake_certainty_low(mini_run):
    """虚构事实 certainty 均值 < 0.5 方向阈（已知口径：虚构≈0）。"""
    pa = mini_run["phase_a_blank_decline"]
    assert pa["certainty_mean"] < 0.5, (
        f"虚构事实 certainty {pa['certainty_mean']:.3f} 应 <0.5（KAL 校准方向回归）")
    assert pa["n_decline"] >= 1, "完全空白区应至少部分 Decline 诚实降级"


def test_phase_d_verdicts_present(mini_run):
    """Phase D：睡眠固化裁决 + v1.1 自适应边缘带（doc 源经 RE_VERIFY 补验证固化）。

    【行为变更 v1.1】旧口径：doc 源（cred 0.7）consensus=0.8×0.85=0.68<0.7 被一刀切
    REJECT——本 fixture 旧期望为 PROMOTE 1（user）/QUARANTINE 1（冲突）/REJECT 1（doc）。
    新口径：doc 源证据加权共识 0.688 落边缘带 [0.62,0.7) → CrossVerifier 二次复核
    通过+有界加成 → PROMOTE——新期望 PROMOTE 2/QUARANTINE 1/REJECT 0。
    变更理由：修复信源可信度边缘效应（工具来源知识系统性进不了长期记忆）；
    冲突块 QUARANTINE 红线不变（自适应不触碰漂移拦截）。
    """
    pd = mini_run["phase_d_sleep"]
    n_verdicts = pd["n_promoted"] + pd["n_quarantined"] + pd["n_rejected"]
    assert n_verdicts >= 1, "固化应有裁决产出"
    assert len(pd["per_block"]) == n_verdicts or pd["n_practiced"] >= 1
    # 冲突块应被 QUARANTINE（保留双方标分歧红线；自适应不触碰漂移拦截）
    conflict_verdicts = [b["verdict"] for b in pd["per_block"] if b["conflict"]]
    assert conflict_verdicts and all(v == "QUARANTINE" for v in conflict_verdicts), (
        f"冲突块应 QUARANTINE：{conflict_verdicts}")
    # v1.1 自适应：已教事实（非冲突）全部固化——fact0 走 CallTool→doc 源（边缘效应
    # 现场信源）经 RE_VERIFY 补验证通过，fact1 走 AskQuestion→user 源直接 PROMOTE
    taught = [b for b in pd["per_block"] if not b["conflict"]]
    assert taught and all(b["verdict"] == "PROMOTE" for b in taught), (
        f"已教事实应全部固化（v1.1 修复信源边缘效应）："
        f"{[(b['source'], b['verdict']) for b in taught]}")
    doc_blocks = [b for b in taught if b["source"] == "doc"]
    assert doc_blocks, "n_facts=2 时 fact0（priority 0.6）应走 CallTool→doc 源"
    for b in doc_blocks:
        assert b["reverify"] and b["reverify"]["passed"] is True, (
            f"doc 源应经 RE_VERIFY 补验证通过（非直接放行）：{b['block_id']}")
