"""数据管线 + SFT 评估测试（知识内化训练 · 阶段 1）。

覆盖（对齐任务要求）：
1. 样本结构：{K,Q,A,q_type} 齐全，三类比例合理；
2. K-依赖验证：虚构实体词不在 val shard 语料（先验不存在）；
3. 一致/矛盾标签正确；
4. SFT 数据格式：label mask 正确（K/Q mask=-100，只 A 计损失）；
5. （可选，若有 SFT checkpoint）评估：有 K 答对率 > 无 K 答对率。

纯 CPU 快速项（1-4）默认跑；GPU 评估项（5）无 checkpoint 时 skip。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import build_teaching_data as btd  # noqa: E402
import teaching_sft as ts  # noqa: E402
from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

TOK_PATH = ROOT / "data/tokenizer/tokenizer.json"
SFT_CKPT = ROOT / "checkpoints/pilot_0p1b_gdn2_10k_teaching"


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


@pytest.fixture(scope="module")
def tok():
    return TokenizerIO(TOK_PATH)


@pytest.fixture(scope="module")
def samples(rng):
    """生成一小批三类样本供结构/标签测试。"""
    return ([btd.build_fact(rng) for _ in range(20)]
            + [btd.build_chain(rng) for _ in range(20)]
            + [btd.build_consist(rng) for _ in range(20)])


# ---------------------------------------------------------------------------
# 1. 样本结构：{K,Q,A,q_type} 齐全，三类比例合理
# ---------------------------------------------------------------------------
def test_sample_structure(samples):
    for s in samples:
        for key in ("K", "Q", "A", "q_type", "label"):
            assert key in s, f"样本缺字段 {key}: {s}"
        assert s["q_type"] in ("fact", "chain", "consist")
        assert s["label"] in ("accept", "reject")
        assert isinstance(s["K"], str) and len(s["K"]) > 0
        assert isinstance(s["Q"], str) and len(s["Q"]) > 0
        assert isinstance(s["A"], str) and len(s["A"]) > 0


def test_three_type_ratio(rng):
    """三类比例合理：build 比例 [0.4,0.3,0.3] 下三类均非空且 fact 最多。"""
    n = 300
    nf, nc = int(n * 0.4), int(n * 0.3)
    gen = ([btd.build_fact(rng) for _ in range(nf)]
           + [btd.build_chain(rng) for _ in range(nc)]
           + [btd.build_consist(rng) for _ in range(n - nf - nc)])
    from collections import Counter
    dist = Counter(s["q_type"] for s in gen)
    assert dist["fact"] > 0 and dist["chain"] > 0 and dist["consist"] > 0
    assert dist["fact"] >= dist["chain"] and dist["fact"] >= dist["consist"]


def test_fact_answer_in_k(samples):
    """(a)/(b) K-依赖类：答案 A 必须是 K 中可出现的信息（事实/链条可从 K 读出）。"""
    for s in samples:
        if s["q_type"] in ("fact", "chain") and s["q_type"] != "consist":
            # 事实类答案应能在 K 中找到字面（如数字/颜色/食物）；链条 Yes/No 除外
            if s["A"] not in ("Yes", "No"):
                assert s["A"].lower() in s["K"].lower(), \
                    f"答案 '{s['A']}' 不在 K 中（K-依赖性破坏）: {s['K']}"


# ---------------------------------------------------------------------------
# 2. K-依赖验证：虚构实体词不在 val shard 语料（先验不存在）
# ---------------------------------------------------------------------------
def test_k_dependence_entity_not_in_val(rng):
    """虚构专名不出现在 val 语料——去掉 K 模型只能猜（先验不存在）。"""
    gen = [btd.build_fact(rng) for _ in range(40)] + [btd.build_chain(rng) for _ in range(40)]
    ver = btd.verify_k_dependence(gen, seed=0)
    assert ver["n_entities"] > 0
    assert ver["k_dep_ok"], f"虚构实体泄漏进 val 语料: {ver['leaked']}"
    assert ver["n_in_val"] == 0


def test_make_fake_word_is_novel(rng):
    """虚构专名生成器本身不产生常见英文词（音节拼接，首字母大写）。"""
    common = {"the", "planet", "water", "earth", "sun", "moon", "star", "france", "paris"}
    for _ in range(200):
        w = btd.make_fake_word(rng)
        assert w[0].isupper()
        assert w.lower() not in common
        assert len(w) >= 4


# ---------------------------------------------------------------------------
# 3. 一致/矛盾标签正确
# ---------------------------------------------------------------------------
def test_consist_labels(rng):
    """(c) 类：一致 K→accept+consistent，矛盾 K→reject+contradictory；K 含先验锚。"""
    seen = {"accept": 0, "reject": 0}
    for _ in range(60):
        s = btd.build_consist(rng)
        assert s["q_type"] == "consist"
        seen[s["label"]] += 1
        if s["label"] == "accept":
            assert s["A"] == "consistent"
        else:
            assert s["A"] == "contradictory"
        # K 由先验 P + 变体组成（两句以上）
        assert s["K"].count(".") >= 1
    assert seen["accept"] > 0 and seen["reject"] > 0, "两类变体都应出现"


def test_consist_contradictory_is_false_claim(rng):
    """矛盾变体必须是与先验冲突的错误陈述（如 'always boils at exactly 100°C everywhere'）。"""
    bad_variants = [v[2] for v in btd._CONSIST_POOL]
    ok_variants = [v[1] for v in btd._CONSIST_POOL]
    # 矛盾变体含绝对化/错误词；一致变体含 consistent
    for bad in bad_variants:
        assert "consistent" not in bad.lower()
    for ok in ok_variants:
        assert "consistent" in ok.lower()


# ---------------------------------------------------------------------------
# 4. SFT 数据格式：label mask 正确（K/Q mask=-100，只 A 计损失）
# ---------------------------------------------------------------------------
def test_label_mask_only_answer(tok, samples):
    """labels 中 K/Q 段全为 -100，仅 Answer 段（含 EOT）为真实 token id。"""
    s = next(x for x in samples if x["q_type"] == "fact")
    ids, labels = ts.encode_sample(tok, s, seq_len=192)
    p_ids = tok.encode(f"{s['K']}\nQuestion: {s['Q']}\nAnswer: ")
    # prompt 段全 mask
    assert all(l == ts.IGNORE for l in labels[:len(p_ids)]), "K/Q prompt 段应全 mask"
    # answer 段非 mask 且与 ids 一致
    a_ids = tok.encode(s["A"]) + [tok.eot_id]
    ans_labels = labels[len(p_ids):len(p_ids) + len(a_ids)]
    assert all(l != ts.IGNORE for l in ans_labels), "Answer 段不应 mask"
    assert ans_labels == ids[len(p_ids):len(p_ids) + len(a_ids)]


def test_build_batch_shift_and_pad(tok, samples):
    """build_batch：y 为 x 右移一位；padding 位置 mask=-100；只对 answer 计损失。"""
    batch = samples[:8]
    x, y = ts.build_batch(tok, batch, seq_len=192, device="cpu")
    assert x.shape == y.shape
    # 每行至少有一个非 -100（answer 段）
    for b in range(y.shape[0]):
        n_ans = int((y[b] != ts.IGNORE).sum())
        assert n_ans >= 1, f"第 {b} 行无 answer label"
    # loss 可计算且有限
    import torch
    logits = torch.randn(x.shape[0], x.shape[1], tok.vocab_size)
    loss = ts.masked_ce(logits, y)
    assert torch.isfinite(loss)


def test_masked_ce_ignores_mask(tok):
    """masked_ce 只统计非 -100 位置（全 mask 行不计入）。"""
    import torch
    V = tok.vocab_size
    logits = torch.randn(2, 5, V)
    y = torch.full((2, 5), ts.IGNORE)
    y[0, 3] = 10  # 只有一个有效位置
    loss = ts.masked_ce(logits, y)
    import torch.nn.functional as F
    ref = F.cross_entropy(logits[0, 3].float(), torch.tensor(10))
    assert abs(loss.item() - ref.item()) < 1e-4


# ---------------------------------------------------------------------------
# 5. （可选）SFT 评估：有 K 答对率 > 无 K 答对率
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not SFT_CKPT.exists(), reason="SFT checkpoint 未生成（先跑 teaching_sft.py）")
def test_sft_internalization_gap(tok, rng):
    """有 checkpoint 时：加载评估，有 K 答对率应 > 无 K 答对率（内化判据）。"""
    import torch
    from tais_obsidian.model.model import TaisObsidianForCausalLM
    if not torch.cuda.is_available():
        pytest.skip("需 GPU")
    device = "cuda"
    model = TaisObsidianForCausalLM.from_pretrained(str(SFT_CKPT), device).eval()
    dep = [btd.build_fact(rng) for _ in range(30)] + [btd.build_chain(rng) for _ in range(30)]
    with_k = sum(ts.answer_correct(
        ts.gen_answer(model, tok, f"{s['K']}\nQuestion: {s['Q']}\nAnswer: ", device), s["A"]) for s in dep) / len(dep)
    without_k = sum(ts.answer_correct(
        ts.gen_answer(model, tok, f"Question: {s['Q']}\nAnswer: ", device), s["A"]) for s in dep) / len(dep)
    assert with_k > without_k, f"内化判据不满足：有K {with_k:.3f} 应 > 无K {without_k:.3f}"
