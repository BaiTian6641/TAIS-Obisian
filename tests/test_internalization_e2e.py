"""内化-检索-注入端到端单元测试（知识内化"实时可用"承诺）。

判据（docs/TAIS_Obsidian_知识内化训练_分析与设计.md §2.1 + 接口计划 §6 载体能力边界红线）：
- 知识块写入 BlockStore 可 get（内化/存储通路）；
- 检索表征可分：embedding 余弦相似度命中刚写入的 K 块（top-k 含目标）——
  HRL indexer 未训时随机（不强制），但表征本身可分是检索可用的前提；
- 载体能力边界：token 寻址载体（kv）能事实召回、向量（steering）只能 steer——
  0.1B 上两者生成答对率均 ≈ 基线（KV 召回头未训通而未用、向量本就不能事实召回）；
- 端到端（写入→检索→注入→答对）跑通无异常；
- 运行时注入不动权重（零梯度，区别于 teaching_sft 的离线 SFT）。

诚实边界（0.1B 原型实测）：teaching checkpoint 未训"经 HCA 注入块做事实召回"
（门控对 HCA 分支权重≈0.016），故 KV 注入后生成不改变——本测试验证**通路通**，
不断言 0.1B 已能注入召回（那是 E+ 待训目标）。

测试风格对齐 tests/test_injection.py（sys.path 插 src，DEVICE cuda）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
from tais_obsidian.model.injection import make_injector  # noqa: E402
from tais_obsidian.model.memlayer import make_memory_layer  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

import internalization_e2e as e2e  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
TOK = "data/tokenizer/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW = 8


@pytest.fixture(scope="module")
def model_and_tok():
    tok = TokenizerIO(TOK)
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, DEVICE)
    model.eval()
    model.attach_kernel()
    return model, tok


@pytest.fixture(scope="module")
def a_layers(model_and_tok):
    model, _ = model_and_tok
    return [i for i, t in enumerate(model.config.layer_types) if t == "A"]


@pytest.fixture(scope="module")
def facts():
    return e2e._make_facts(5, seed=0)


# ---------------- 内化（写入 BlockStore）----------------

def test_blockstore_put_get(model_and_tok, a_layers, facts):
    """知识块写入 BlockStore 可 get（内化/存储通路通）。"""
    model, tok = model_and_tok
    store = BlockStore()
    block = e2e.internalize(model, tok, store, facts[0], a_layers, DEVICE, model.config.d_model)
    got = store.get(block["block_id"])
    assert got is not None
    assert got["block_id"] == block["block_id"]
    assert got["kind"] == "kv"  # token 寻址载体（事实召回，红线）
    assert set(got["entries"].keys()) == set(a_layers)  # 各 CSA 层均有 (k,v)
    # 载体形状：[B, n_kv, N, head_dim]（inject_hca_entries 所需）
    k, v = got["entries"][a_layers[0]]
    assert k.shape[1] == model.config.n_kv_heads and k.shape[-1] == model.config.head_dim


def test_internalize_no_weight_change(model_and_tok, a_layers, facts):
    """运行时内化为零梯度写入——模型权重不变（实时可用 vs 离线 SFT 内化的红线）。"""
    model, tok = model_and_tok
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    store = BlockStore()
    for f in facts[:3]:
        e2e.internalize(model, tok, store, f, a_layers, DEVICE, model.config.d_model)
    for n, p in model.named_parameters():
        assert torch.equal(before[n], p), f"运行时写入改动了权重 {n}（红线：运行时不动权重）"


# ---------------- 检索（表征可分）----------------

def test_retrieval_repr_separable(model_and_tok, a_layers, facts):
    """检索表征可分：embedding 余弦相似度 top-1 命中刚写入的 K 块。

    HRL indexer 未训时打分随机（不强制命中）；但 CSA 层隐藏态表征本身须可分
    （不同实体表征不同），这是检索可用的前提（demo 实测余弦基线 100%）。
    """
    model, tok = model_and_tok
    store = BlockStore()
    blocks = [e2e.internalize(model, tok, store, f, a_layers, DEVICE, model.config.d_model) for f in facts]
    hits = 0
    for f, b in zip(facts, blocks):
        q_r = e2e.hidden(model, tok, f["Q"], a_layers[0], DEVICE)[0].mean(0)  # [d]
        cand_r = torch.stack([c["repr"][0, 0] for c in blocks])  # [N,d]
        cos = torch.nn.functional.cosine_similarity(q_r.unsqueeze(0), cand_r, dim=-1)
        if blocks[int(cos.argmax())]["block_id"] == b["block_id"]:
            hits += 1
    assert hits / len(facts) >= 0.6, f"检索表征不可分（余弦命中 {hits}/{len(facts)}）"


def test_route_candidates_runs(model_and_tok, a_layers, facts):
    """HRL route_candidates 打分通路跑通（输出形状/数值合法）。"""
    model, tok = model_and_tok
    store = BlockStore()
    blocks = [e2e.internalize(model, tok, store, f, a_layers, DEVICE, model.config.d_model) for f in facts[:3]]
    top_ids, cand_score = e2e.retrieve(model.kernel, model, tok, facts[0]["Q"], blocks, 2, DEVICE, a_layers)
    assert len(top_ids) == 2
    assert cand_score.shape[0] == len(blocks)
    assert torch.isfinite(cand_score).all()


# ---------------- 注入（载体能力边界）----------------

def test_kv_inject_into_hca(model_and_tok, a_layers, facts):
    """KV（token 寻址）注入：namespace 校验通过、条目进入 HCA 区（n_hca_inj>0）。"""
    model, tok = model_and_tok
    store = BlockStore()
    block = e2e.internalize(model, tok, store, facts[0], a_layers, DEVICE, model.config.d_model)
    injector = make_injector()
    qprompt = f"Question: {facts[0]['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(DEVICE == "cuda")):
        _, cache = model(torch.tensor([tok.encode(qprompt)], device=DEVICE))
    for i in a_layers:
        mixer = model.layers[i].mixer
        ns = make_namespace(model.config, i, cache["layers"][i]["k"].dtype)
        k, v = block["entries"][i]
        payload = BlockPayload(block_id=block["block_id"], compiled_kind="kv",
                               entries=(k, v), layer_ns=tuple(ns.values()))
        k_inj, v_inj = injector.inject(payload, namespace=ns)  # namespace 校验 fail-closed
        new_state = mixer.inject_hca_entries(cache["layers"][i], (k_inj, v_inj), ns)
        assert "hca_inj_k" in new_state and new_state["hca_inj_k"].shape[2] == k.shape[2]
        cache["layers"][i] = new_state


def test_kv_namespace_fail_closed(model_and_tok, a_layers, facts):
    """KV 注入 namespace 不匹配 → inject_hca_entries fail-closed 拒注（红线：注入即攻击面）。

    inject_hca_entries 内部按"模型实际层 namespace"逐字段校验传入 namespace，
    任一不匹配即抛 NamespaceMismatchError（调用方走重算/文本 RAG 回退）。
    """
    from tais_obsidian.model.blockpath import NamespaceMismatchError
    model, tok = model_and_tok
    store = BlockStore()
    block = e2e.internalize(model, tok, store, facts[0], a_layers, DEVICE, model.config.d_model)
    qprompt = f"Question: {facts[0]['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(DEVICE == "cuda")):
        _, cache = model(torch.tensor([tok.encode(qprompt)], device=DEVICE))
    i = a_layers[0]
    mixer = model.layers[i].mixer
    good_ns = make_namespace(model.config, i, cache["layers"][i]["k"].dtype)
    bad_ns = dict(good_ns, layer_idx=999)  # 层号不匹配
    k, v = block["entries"][i]
    with pytest.raises(NamespaceMismatchError):
        mixer.inject_hca_entries(cache["layers"][i], (k, v), bad_ns)


def test_factual_recall_flag(model_and_tok):
    """载体能力边界标注：kv(token寻址)=True，steering(向量)=False（__post_init__ 强校验）。"""
    p_kv = BlockPayload(block_id="a", compiled_kind="kv", entries=(torch.randn(1, 4, 2, 64),) * 2)
    p_vec = BlockPayload(block_id="b", compiled_kind="steering", vector=torch.randn(768))
    assert p_kv.factual_recall is True   # token 寻址能事实召回
    assert p_vec.factual_recall is False  # 向量只能 steer 不能事实召回


def test_vector_carrier_cannot_recall(model_and_tok, a_layers, facts):
    """向量载体（steering）注入不能事实召回（红线：向量当事实用必失败）。

    0.1B 实测：steering 改变生成（steer 行为有效）但答对率 ≈ 基线（≈0）——
    只能 steer 不能事实召回。此处断言向量注入答对率不显著高于基线。
    """
    model, tok = model_and_tok
    store = BlockStore()
    blocks = [e2e.internalize(model, tok, store, f, a_layers, DEVICE, model.config.d_model) for f in facts]
    n = len(facts)
    base = sum(e2e.answer_correct(e2e.answer_baseline(model, tok, f, DEVICE, MAX_NEW), f["A"]) for f in facts)
    vec = sum(e2e.answer_correct(
        e2e.answer_with_vector_inject(model, tok, model.kernel, f, b, a_layers, DEVICE, max_new=MAX_NEW),
        f["A"]) for f, b in zip(facts, blocks))
    # 向量载体不能事实召回：答对数应 ≈ 基线（都≈0），不显著更高
    assert vec <= base + 1, f"向量载体疑似事实召回（vec={vec} base={base}，违反载体边界）"


# ---------------- 端到端跑通 ----------------

def test_e2e_pipeline_runs(model_and_tok, a_layers, facts):
    """端到端（写入→检索→注入→答对）跑通无异常；三条件对照数值合法。"""
    model, tok = model_and_tok
    store = BlockStore()
    # ① 内化写入
    blocks = [e2e.internalize(model, tok, store, f, a_layers, DEVICE, model.config.d_model) for f in facts]
    assert store.stats()["L1"] == len(facts)
    # ②③④ 检索 + 注入 + 三条件答对
    for f, b in zip(facts, blocks):
        top_ids, _ = e2e.retrieve(model.kernel, model, tok, f["Q"], blocks, 1, DEVICE, a_layers)
        g_base = e2e.answer_baseline(model, tok, f, DEVICE, MAX_NEW)
        g_kv = e2e.answer_with_kv_inject(model, tok, f, b, a_layers, DEVICE, MAX_NEW)
        g_ic = e2e.answer_incontext(model, tok, f, DEVICE, MAX_NEW)
        assert isinstance(g_base, str) and isinstance(g_kv, str) and isinstance(g_ic, str)


def test_incontext_upper_bound(model_and_tok, facts):
    """in-context 上界：K 作 token 上下文时模型答对率显著 > 0（证明知识本会答）。

    缺口定位：in-context（滑窗读 K token）能答对，而 KV 注入 HCA 不能——
    证明"知识可用、运行时注入召回头未训"（0.1B 诚实边界，E+ 待训）。
    """
    model, tok = model_and_tok
    n = len(facts)
    ic = sum(e2e.answer_correct(e2e.answer_incontext(model, tok, f, DEVICE, MAX_NEW), f["A"]) for f in facts)
    base = sum(e2e.answer_correct(e2e.answer_baseline(model, tok, f, DEVICE, MAX_NEW), f["A"]) for f in facts)
    assert ic / n > base / n, f"in-context 上界 {ic}/{n} 未超基线 {base}/{n}（知识不可用？异常）"
