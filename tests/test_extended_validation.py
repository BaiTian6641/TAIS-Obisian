"""五场景扩展交互测试（extended_validation）回归测试：小尺度 S2/S3/S5 主链路。

覆盖（复用 scripts/extended_validation.py 的场景函数，断言结构 + 方向，不追求强度）：
  a) 版本化累积不覆盖：同内容教两次 → draft_versions == [1, 2]（:v{n} 自增）；
  b) S2 小尺度（3 facts × 2 rounds）：召回/检索曲线结构与值域、版本证据、
     失败重教只针对召回失败条目；
  c) S3 桥接方向：基线推不出 D、B'/C' 写入、检索命中、注入召回 > 不注入、
     轨迹-块邻近性数值结构（before/after min/mean）；
  d) S5 睡眠固化：有裁决产出、矛盾块被拦（QUARANTINE/REJECT，防放水红线）、
     tracker 前后快照、doc 源 RE_VERIFY→PROMOTE 路径存在、固化后召回不破坏；
  e) nearest_block_series 纯函数（合成数据，无模型）。

双卡分工：用计算卡 PRO 4000（CUDA_VISIBLE_DEVICES=1）。
运行：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe -m pytest tests/test_extended_validation.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")

CKPT = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_unified"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
PROJECTOR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_unified_manifold" / "projector.pt"

import extended_validation as ev  # noqa: E402

# 小尺度 S2 事实集（3 条引擎事实，teaching 分布对齐——OOV/自定义句式实测注入召回
# 不工作，见 extended_validation report honest_notes 的载体能力边界负结果）
MINI_FACTS = None  # fixture 里 ev.default_s2_facts(3, seed=1) 生成


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    """统一 checkpoint + 已训投影器装配（Ctx 内含 BlockStore/executor/router）。"""
    if not CKPT.exists():
        pytest.skip("统一 checkpoint 未产出（先跑 scripts/build_unified_checkpoint.py）")
    if not PROJECTOR.exists():
        pytest.skip("已训流形投影器未产出（先跑 scripts/train_manifold_projector.py）")
    out_dir = tmp_path_factory.mktemp("extended_validation")
    logger = ev.SessionLog(out_dir / "session_log.jsonl")
    c = ev.Ctx(str(CKPT), str(TOK), str(PROJECTOR), "cuda", Path(out_dir), logger)
    yield c
    c.log.close()


@pytest.fixture(scope="module")
def mini_chain(ctx):
    """小尺度 S2→S3→S5 顺序主链（共享 BlockStore，一次跑完供各断言复用）。"""
    torch.manual_seed(0)
    facts = ev.default_s2_facts(3, seed=0)
    s2 = ev.scenario_s2(ctx, facts=facts, n_rounds=2)
    s3 = ev.scenario_s3(ctx)
    kv_map = {}
    for f in facts:
        kv = ctx.store.get(f"s2/fact/{f['entity']}:kv")
        if kv is not None:
            kv_map[f["entity"]] = kv
    s5 = ev.scenario_s5(ctx, s2_facts=facts, s2_kv_map=kv_map)
    return {"s2": s2, "s3": s3, "s5": s5, "facts": facts}


def test_a_version_accumulate(ctx) -> None:
    """a) 同内容教两次 → :v1/:v2 共存（累积不覆盖，版本自增）。"""
    K = "The Quilmeb flower opens only under moonlight."
    r1 = ev.teach_one(ctx, K, "test/version/quilmeb")
    r2 = ev.teach_one(ctx, K, "test/version/quilmeb")
    assert r1["written"] and r2["written"], "两次教学均应写入（CrossVerifier 验证通过）"
    assert r2["versions"] == [1, 2], f"版本应自增共存 [1,2]，实得 {r2['versions']}"
    assert ev.draft_versions(ctx.store, K) == [1, 2]
    print(f"[a] 版本化累积不覆盖 OK：versions={r2['versions']}")


def test_b_s2_structure_direction(mini_chain) -> None:
    """b) S2 小尺度：曲线结构/值域 + 版本证据 + 重教只针对失败条目。"""
    s2 = mini_chain["s2"]
    facts = mini_chain["facts"]
    assert len(s2["recall_curve"]) == 2 and len(s2["retrieval_curve"]) == 2
    for v in s2["recall_curve"] + s2["retrieval_curve"] + s2["baseline_curve"]:
        assert 0.0 <= v <= 1.0
    assert set(s2["version_evidence"].keys()) == {f["entity"] for f in facts}
    assert all(len(v) >= 1 for v in s2["version_evidence"].values()), "每条至少 v1"
    # 方向：末轮检索命中率（同协议判据 0.938@n=16；n=3 小样本容忍 1 条失手）
    assert s2["retrieval_curve"][-1] >= 0.66, f"检索命中率 {s2['retrieval_curve']}"
    # 召回曲线单调不降（同内容重教确定性复现，平坦即满足）
    rc = s2["recall_curve"]
    assert rc[1] >= rc[0] - 1e-9
    # 第 2 轮重教集合 = 第 1 轮召回失败集合
    r1_fail = {e for e, ok in s2["rounds"][0]["per_fact"].items() if not ok["kv_ok"]}
    assert s2["rounds"][1]["n_taught"] == len(r1_fail)
    print(f"[b] S2 结构/方向 OK：recall={s2['recall_curve']} "
          f"retrieval={s2['retrieval_curve']} versions={s2['version_evidence']}")


def test_c_s3_bridge_direction(mini_chain) -> None:
    """c) S3 桥接：基线失败 → 补教 B'/C' → 检索命中 + 注入召回 > 不注入 + 邻近性结构。"""
    s3 = mini_chain["s3"]
    assert s3["baseline_ok"] is False, "教学前基线应推不出 D（宽松判对失败）"
    assert s3["taught"] == {"B'": True, "C'": True}
    assert len(s3["retrieval_top3"]) == 3
    assert any(t in ("s3/bridge/b1", "s3/bridge/c1") for t in s3["retrieval_top3"]), (
        f"top3 应含桥接块（{s3['retrieval_top3']}；D 问句式非训练分布，top-1 失手如实记录）")
    assert s3["inject_ok"] is True and s3["no_inject_after_teach_ok"] is False, (
        f"注入召回应 > 不注入：inject={s3['inject_ok']} no_inject={s3['no_inject_after_teach_ok']}")
    for lab, px in s3["proximity"].items():
        assert {"before_min", "after_min", "before_mean", "after_mean"} <= set(px)
        assert all(np.isfinite(v) for v in px.values())
    assert len(s3["series_before"]) > 0 and len(s3["series_after"]) > 0
    print(f"[c] S3 桥接方向 OK：inject={s3['inject_ok']} proximity={s3['proximity']}")


def test_d_s5_sleep(mini_chain) -> None:
    """d) S5：裁决产出 + 矛盾块被拦 + tracker 快照 + RE_VERIFY→PROMOTE + 固化不破坏召回。"""
    s5 = mini_chain["s5"]
    n_verdicts = s5["n_promoted"] + s5["n_quarantined"] + s5["n_rejected"]
    assert n_verdicts >= 1, "固化应有裁决产出"
    assert s5["conflict_block"]["blocked"] is True, (
        f"矛盾块必须仍 QUARANTINE/REJECT，实得 {s5['conflict_block']['verdict']}")
    assert set(s5["credibility_before"]) and set(s5["credibility_after"])
    assert s5["n_reverified"] >= 1 and any(
        p == "RE_VERIFY→PROMOTE" for p in s5["verdict_paths"]), (
        f"doc 源应经 RE_VERIFY 固化：{list(s5['verdict_paths'])}")
    assert s5["post_sleep_recall"] is not None
    assert s5["post_sleep_recall"] >= mini_chain["s2"]["recall_curve"][-1] - 1e-9
    print(f"[d] S5 OK：paths={ {k: len(v) for k, v in s5['verdict_paths'].items()} } "
          f"conflict={s5['conflict_block']['verdict']}")


def test_e_nearest_block_series() -> None:
    """e) 纯函数：轨迹-块最近序列结构（合成数据，无模型）。"""
    g = np.random.default_rng(0)
    coords = g.normal(size=(7, 16))
    blocks = g.normal(size=(3, 16))
    labels = ["b0", "b1", "b2"]
    series, dmat = ev.nearest_block_series(coords, blocks, labels)
    assert dmat.shape == (7, 3) and len(series) == 7
    for i, s in enumerate(series):
        assert s["step"] == i and s["block"] in labels
        assert abs(s["dist"] - dmat[i].min()) < 1e-9
        assert s["block"] == labels[int(dmat[i].argmin())]
    print(f"[e] nearest_block_series OK：{[s['block'] for s in series]}")
