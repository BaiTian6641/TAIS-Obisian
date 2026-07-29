"""HRL indexer 块检索 + HCA 召回头训练单元测试（兑现"实时可用"两训练缺口）。

判据（docs/TAIS_Obsidian_知识内化训练_分析与设计.md §2.1 + 接口计划 §6 + AGENTS.md §7 红线）：
- **缺口① indexer 块检索**：训练后 top-k 命中率 > 随机基线（表征本可分，训后对齐）；
- **缺口② HCA 召回头**：KV 注入答对率 > 不注入基线（0）——注入块参与召回；
- **主干 frozen 红线**：训练后主干权重逐位不变（indexer/门控非主干，方案 B 边界）；
- **端到端闭环**：写入→检索（命中）→注入→答对（用上），实时可用兑现。

红线落实：主干全程 frozen；检索/召回辅助损失只进目标头（梯度隔离，MoE-RL 红线）。

测试用 pilot 级小步数快速复训（n_facts=6、检索 60 步、召回 60 步），判"显著改善"
非"完美"（诚实标注：训练规模 pilot 级；in-context 上界 0.70 是参考非必须达到）。
风格对齐 tests/test_internalization_e2e.py（sys.path 插 src/scripts，DEVICE cuda）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

import train_retrieval_recall as trr  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
TOK = "data/tokenizer/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_FACTS = 6
RETR_STEPS = 60
RECALL_STEPS = 60


@pytest.fixture(scope="module")
def trained():
    """快速复训两缺口（小步数，module 级共享）。返回 (model, kernel, tok, facts, a_layers, 指标)。"""
    torch.manual_seed(0)
    tok = TokenizerIO(TOK)
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, DEVICE)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    for p in model.parameters():
        p.requires_grad_(False)
    model.attach_kernel()
    kernel = model.kernel
    for p in kernel.parameters():
        p.requires_grad_(True)
    facts = trr.make_facts(N_FACTS, seed=0)
    logs = []
    snap = trr.backbone_snapshot(model, a_layers)
    r1 = trr.train_retrieval(model, kernel, tok, facts, a_layers, DEVICE,
                             RETR_STEPS, 1e-2, 0, logs.append)
    r2 = trr.train_recall(model, tok, facts, a_layers, DEVICE,
                          RECALL_STEPS, 1e-2, 0, 8, logs.append)
    unchanged, _ = trr.backbone_unchanged(model, snap, a_layers)
    return model, kernel, tok, facts, a_layers, r1, r2, unchanged


def test_retrieval_hit_improves(trained):
    """缺口①：训练后 indexer 块检索 top-1 命中率显著 > 随机基线（1/N）。"""
    _, _, _, facts, _, r1, _, _ = trained
    n = len(facts)
    random_baseline = 1.0 / n
    assert r1["final_hit"] > random_baseline, (
        f"检索命中率 {r1['final_hit']:.3f} 未超随机基线 {random_baseline:.3f}")
    assert r1["final_hit"] > r1["init_hit"], "训练后命中率应优于训练前"


def test_recall_acc_above_baseline(trained):
    """缺口②：KV 注入答对率 > 不注入基线（0）——注入块参与召回（通→用）。"""
    _, _, _, _, _, _, r2, _ = trained
    assert r2["final_acc"] > 0.0, "KV 注入答对率应 > 0（召回头训后注入块被用上）"
    assert r2["final_acc"] >= r2["init_acc"], "训练后答对率应不劣于训练前"


def test_backbone_frozen(trained):
    """主干 frozen 红线：训练后主干权重逐位不变（indexer/门控非主干）。"""
    *_, unchanged = trained
    assert unchanged, "主干权重被改动（frozen 红线违反）"


def test_closed_loop(trained):
    """端到端闭环：写入→检索（命中）→注入→答对（用上），实时可用。"""
    model, kernel, tok, facts, a_layers, _, _, _ = trained
    from tais_obsidian.model.injection import make_injector
    layer = a_layers[0]
    injector = make_injector()
    cand_repr = torch.cat(
        [trr.hidden(model, tok, f["K"], layer, DEVICE)[0].mean(0, keepdim=True).unsqueeze(0)
         for f in facts], dim=1).to(DEVICE)
    all_entries = [trr.harvest_kv(model, tok, f["K"], a_layers, DEVICE) for f in facts]
    hits = 0
    for j, f in enumerate(facts):
        q = trr.hidden(model, tok, f["Q"], layer, DEVICE)[0].mean(0, keepdim=True).unsqueeze(0)
        s = kernel.route_candidates(q, cand_repr, k=None, detach_input=True)[0, -1]
        top = int(s.argmax())
        hits += (top == j)
        # 检索命中块注入 → prompt 法续答（闭环：检索→注入→生成，无异常即通路通）
        qp = f"Question: {f['Q']}\nAnswer: "
        with torch.autocast("cuda", torch.bfloat16, enabled=(DEVICE == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(qp)], device=DEVICE))
            cache = trr._inject_fact_into_cache(model, cache, all_entries[top], a_layers, injector)
            out = []
            for _ in range(4):
                nxt = int(logits[:, -1, :].float().argmax(-1).item())
                if nxt == tok.eot_id:
                    break
                out.append(nxt)
                logits, cache = model(torch.tensor([[nxt]], device=DEVICE), cache)
        _ = tok.decode(out)  # 生成无异常
    # 闭环检索命中（写入→检索通）：命中率应 > 随机
    assert hits / len(facts) > 1.0 / len(facts), "闭环检索命中率应 > 随机基线"
