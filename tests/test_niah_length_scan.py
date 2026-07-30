"""NIAH 长度扫描的结构与判据测试（scripts/eval_niah_length_scan.py）。

覆盖（fb1 长度扫描三要素 + 判据）：
1. 长度扫描结构：不同 target_length 生成样本 token 数 ≈ 目标长度（±开销），埋点分散；
2. 多 key 数结构：n_keys ∈ {8,32,128} 埋点数正确、key 互不相同（含超池扩展）；
3. 放宽判据：完整 VALUE 匹配（多位数字全对）实现正确——构造 mock 模型直接验证两判据；
4. 小扫描跑通：[256,512]×[8,16] 结构级冒烟（合成填充，不加载 checkpoint）。

不加载真实 checkpoint（结构/判据正确性测试）；显存/外推实测见脚本主流程。
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

import eval_niah_length_scan as scan  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

TOK_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"


@pytest.fixture(scope="module")
def tok():
    if not TOK_PATH.exists():
        pytest.skip("tokenizer 不存在（需先 prepare_data.py）")
    return TokenizerIO(TOK_PATH)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(0)


# ── 1. 长度扫描结构 ─────────────────────────────────────────────────────────


def test_sample_length_hits_target(tok, rng):
    """不同 target_length 下样本 token 数 ≈ 目标（埋点+查询开销内）。"""
    for L in (256, 512, 768):
        s = scan.build_niah_length_sample(rng, tok, None, L, n_keys=8)
        # 总长在 [L-8, L+2]：填充按 budget 精确，埋点/查询分词与估值差 ≤ 数 token
        assert L - 8 <= s["n_tokens"] <= L + 2, f"L={L} 实际 {s['n_tokens']}"


def test_facts_scattered(tok, rng):
    """埋点均匀分散在长上下文中（非集中在开头/末尾），查询在末尾。"""
    s = scan.build_niah_length_sample(rng, tok, None, 1024, n_keys=8)
    ends = s["facts_end"]
    assert len(ends) == 8
    # 相邻埋点间距 ≈ fill_budget/(n_keys+1)（均匀分散，最小间距 > 20 token）
    gaps = [ends[0]] + [ends[i + 1] - ends[i] for i in range(len(ends) - 1)]
    assert min(gaps) > 20, f"埋点过密 gaps={gaps}"
    # 查询前缀在最后一个埋点之后（查询在末尾）
    assert s["query_prefix_len"] > ends[-1]
    assert s["n_tokens"] > s["query_prefix_len"]


def test_facts_positions_correct(tok, rng):
    """facts_end 位置 = 各埋点结束 prefix 长度（解码验证该位前缀以埋点句结尾）。"""
    s = scan.build_niah_length_sample(rng, tok, None, 512, n_keys=8)
    for i, fe in enumerate(s["facts_end"]):
        prefix_text = tok.decode(s["ids"][:fe])
        # 该位前缀应以 "is {value}." 结尾（埋点句尾）
        assert prefix_text.rstrip().endswith(f"is {s['fact_values'][i]}."), (
            f"埋点 {i} 位置错位: ...{prefix_text[-60:]!r}"
        )


# ── 2. 多 key 数结构 ────────────────────────────────────────────────────────


@pytest.mark.parametrize("n_keys", [8, 32, 128])
def test_n_keys_structure(tok, rng, n_keys):
    """n_keys 扫描：埋点数正确、key 互不相同（含超 12 池的变体扩展）。"""
    L = max(1024, n_keys * 24 + 300)  # 长度随 key 数放大（每埋点 ~11 token + 间距）
    s = scan.build_niah_length_sample(rng, tok, None, L, n_keys=n_keys)
    assert len(s["facts_end"]) == n_keys
    assert len(s["fact_values"]) == n_keys
    keys = scan._key_variants(np.random.default_rng(1), n_keys)
    assert len(set(keys)) == n_keys, "key 存在重复"


def test_query_value_consistency(tok, rng):
    """查询指向的埋点 value 与 query_value 一致（判据真值正确）。"""
    s = scan.build_niah_length_sample(rng, tok, None, 512, n_keys=8)
    qi = s["query_fact_idx"]
    assert s["query_value"] == s["fact_values"][qi]
    # 查询 VALUE token 首元素存在（判据目标合法）
    assert len(s["query_value_ids"]) >= 1


# ── 3. 放宽判据（完整 VALUE 匹配）─────────────────────────────────────────────


class _MockModel:
    """可控 next-token 模型：按预设 token 表返回 argmax（验证判据逻辑）。"""

    def __init__(self, forced_ids: list[int], vocab: int = 33000):
        self.forced = list(forced_ids)
        self.vocab = vocab
        self.calls = 0

    def __call__(self, x, cache=None):
        self.calls += 1
        B, T = x.shape
        logits = torch.full((B, T, self.vocab), -10.0)
        if cache is None:
            # prefill 调用：逐位置填 forced（前 len(forced) 位），其余给 0 号
            for t in range(T):
                tok_id = self.forced[t] if t < len(self.forced) else 0
                logits[0, t, tok_id] = 10.0
        else:
            # 增量调用：按 prefill 长度+cache pos 推进 forced 指针
            pos = int(cache["pos"]) if isinstance(cache, dict) else 0
            idx = pos  # 预测的 next-token 位置
            tok_id = self.forced[idx] if idx < len(self.forced) else 0
            logits[0, -1, tok_id] = 10.0
        new_cache = {"pos": (0 if cache is None else cache["pos"]) + T, "layers": []}
        return logits, new_cache


def test_full_value_criterion_hit_and_miss(tok, rng):
    """完整 VALUE 判据：全对才命中（首对+尾错=不命中；全对=命中）。"""
    scan._TOK = tok  # 注入模块级 tokenizer
    s = scan.build_niah_length_sample(rng, tok, None, 256, n_keys=4)
    qv_ids = s["query_value_ids"]
    qpl = s["query_prefix_len"]

    # Case A：模型预测全对 → first/full 均命中
    # _MockModel prefill 分支语义：位置 t 的输出 logits[0,t] 预测 forced[t]（即 next@t=forced[t]）。
    # 查询判定位 = prefill 到 qpl 后位置 qpl-1 的 next-token——故 forced[qpl-1] 须=qv_ids[0]，
    # forced[qpl:]=qv_ids[1:]（增量步续答）。off-by-one：forced=[0]*(qpl-1)+qv_ids。
    forced = [0] * (qpl - 1) + qv_ids
    m = _MockModel(forced)
    # 模拟 eval_one_sample 的判据段：prefill 至 qpl，逐位判定
    cache = None
    q_logits = None
    for st in range(0, qpl, 512):
        # seg 截到 qpl（对齐 eval_one_sample“prefill 到查询前缀”语义：末段不得越入
        # 查询 ids，否则 q_logits 取到查询内而非 qpl 位）
        seg = s["ids"][st : min(st + 512, qpl)]
        logits, cache = m(torch.tensor([seg]), cache)
        if st + len(seg) >= qpl:
            q_logits = logits[0, -1].float()
    pred_first = int(q_logits.argmax().item())
    assert pred_first == qv_ids[0]  # 首 token 命中
    gen = scan._gen_argmax(m, cache, pred_first, len(qv_ids) - 1, "cpu")
    assert [pred_first] + gen == qv_ids, "完整 VALUE 应全对命中"

    # Case B：首 token 对、第二 token 错 → first 命中、full 不命中（判据区分度）
    wrong = (qv_ids[-1] + 1) % 33000
    # off-by-one 同 Case A：forced_b[qpl-1]=qv_ids[0]（位置 qpl-1 预测首 token），
    # forced_b[qpl]=wrong（增量步第二 token 错）
    forced_b = [0] * (qpl - 1) + [qv_ids[0], wrong]
    m_b = _MockModel(forced_b)
    cache = None
    q_logits = None
    for st in range(0, qpl, 512):
        seg = s["ids"][st : min(st + 512, qpl)]  # 截到 qpl（同 Case A）
        logits, cache = m_b(torch.tensor([seg]), cache)
        if st + len(seg) >= qpl:
            q_logits = logits[0, -1].float()
    pred_first_b = int(q_logits.argmax().item())
    assert pred_first_b == qv_ids[0]  # 首 token 仍命中
    gen_b = scan._gen_argmax(m_b, cache, pred_first_b, len(qv_ids) - 1, "cpu")
    assert [pred_first_b] + gen_b != qv_ids, "尾 token 错 → 完整 VALUE 不命中"


def test_first_token_criterion_matches_legacy(tok):
    """首 token 判据与原脚本口径一致（VALUE 首 token id，带空格前缀）。"""
    scan._TOK = tok
    assert scan._value_first_tok("4821") == tok.encode(" 4821")[0]
    assert scan._value_first_tok("9999") == tok.encode(" 9999")[0]


# ── 4. 小扫描跑通（结构级，mock 模型）────────────────────────────────────────


def test_small_scan_smoke(tok, rng):
    """小扫描 [256,512]×[4,8] 结构跑通：样本构造+判据流程无异常。"""
    scan._TOK = tok
    for L in (256, 512):
        for nk in (4, 8):
            s = scan.build_niah_length_sample(rng, tok, None, L, n_keys=nk)
            assert s["n_tokens"] >= L - 8
            assert len(s["facts_end"]) == nk
            # mock 模型（全 0 预测）走一遍判据流程（不崩即可）
            m = _MockModel([0] * (s["n_tokens"] + 8))
            cache = None
            qpl = s["query_prefix_len"]
            for st in range(0, qpl, 512):
                # seg 截到 qpl（对齐 eval_one_sample“prefill 到查询前缀”语义：
                # 末段 chunk 不得越入查询 ids，否则 cache pos > qpl）
                seg = s["ids"][st : min(st + 512, qpl)]
                _, cache = m(torch.tensor([seg]), cache)
            assert cache["pos"] == qpl


def test_chunk_prefill_positions(tok):
    """分块 prefill 边界：埋点结束位跨 chunk 边界时 logits 收集正确（位置不丢）。"""
    scan._TOK = tok
    rng = np.random.default_rng(7)
    s = scan.build_niah_length_sample(rng, tok, None, 640, n_keys=8)
    # chunk=100（非对齐），逐块收集应有全部 8 个埋点 logits
    qpl = s["query_prefix_len"]
    forced_len = qpl + 8
    m = _MockModel([0] * forced_len)
    cache = None
    facts_logits: dict[int, torch.Tensor] = {}
    for st in range(0, qpl, 100):
        # seg 截到 qpl（末段 chunk 不得越入查询 ids，否则 cache pos > qpl）
        seg = s["ids"][st : min(st + 100, qpl)]
        logits, cache = m(torch.tensor([seg]), cache)
        for i, fe in enumerate(s["facts_end"]):
            if st < fe <= st + len(seg):
                facts_logits[i] = logits[0, fe - 1 - st].float().cpu()
    assert len(facts_logits) == 8, f"跨 chunk 埋点收集丢失: {sorted(facts_logits)}"
    assert cache["pos"] == qpl
