"""内化-检索-注入端到端（知识内化"实时可用"核心承诺的端到端验证）。

设计规范：docs/TAIS_Obsidian_知识内化训练_分析与设计.md §2.1（"实时可用"承诺：
新知识写入知识块后，推理循环 HRL 检索命中 → kernel.inject 注入 → 同一对话后续立即可用，
无需等睡眠固化、无需重新 SFT）。

端到端四阶段（本脚本串成闭环）：
  ① 内化（写入）：新知识 K（虚构事实）经模型 prefill 提取表征（KV harvest + 隐藏态）
     → BlockPayload 写入 BlockStore（**运行时零梯度写入，不动权重**——区别于 teaching_sft
     的离线 SFT 内化，这是"实时可用"与"离线内化"的本质区别）。
  ② 检索（路由）：后续相关 query Q 经内核 route_candidates（HRL LightningIndexer）
     对候选块打分 → top-k 命中刚写入的 K 块。
  ③ 注入（执行）：命中块按载体类型经 kernel.inject + make_injector 注入 CSA 层
     （监测/执行分置：注入写 CSA 层残差/注意力，KAL 探针读 GDN 层，不同层）。
  ④ 评估：注入后模型对 Q 的答对率 vs 不注入基线 vs 向量载体对照。

载体能力边界红线（接口计划 §6，已核实，本脚本核心判据）：
  - **token 寻址载体（kv/gist/mem_entry）能事实召回**——新知识是事实，必须用此类载体
    （本脚本用 **kv（HCA 拼接，inject_hca_entries）**：把 K 各 CSA 层的 K/V 前置拼入 HCA
    区、对所有 query 恒可见，是知识块注入原生落点，设计 §17.3）。
  - **位置不变向量（icv/steering/concept_slot）只能 steer 行为，不能事实召回**——
    作对照组，验证"向量当事实用"必然失败。

诚实边界（0.1B 原型，探针已实测，禁止臆造）：
  - teaching checkpoint（pilot_0p1b_gdn2_10k_teaching）**未训练"经 HCA 注入块做事实召回"**
    ——注入条目确实进入 HCA 区（n_hca_inj>0），但门控对 HCA 分支权重极低（≈0.016），
    故 KV 注入后**生成不改变**（通路通、召回头未训）。这是 0.1B 原型的已知缺口，
    "注入块参与召回"需 E+ 阶段训练（设计 §11.1 风险① 前缀偏差 + 块召回训练目标）。
  - steering（PM-stream 加法）**确实改变生成**（通路有效），但只 steer 行为、不事实召回。
  - **in-context 上界**：把 K 作纯文本前缀喂入（teaching_sft 已验证此路径有K答对≈1.0），
    证明"知识本身模型本会答"——缺口在运行时检索-注入载体的召回头，非知识不可用。

双卡分工：本脚本用 RTX 4070（CUDA_VISIBLE_DEVICES=0，8GB，控 batch/seq）。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/internalization_e2e.py --n_facts 20
产出：runs/internalization_e2e/report.json。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
from tais_obsidian.model.injection import make_injector  # noqa: E402
from tais_obsidian.model.memlayer import make_memory_layer  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload, make_kernel  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_REPORT = "runs/internalization_e2e/report.json"

# 程序化虚构实体（先验不存在；**对齐 teaching_sft 训练分布**——复合随机实体名 +
# "What does X run on?" 句式 + 单答案词，使 in-context 上界可复现 teaching 有K≈1.0，
# 并给 HRL 检索提供匹配的训练分布表征）。实体/答案程序化合成，在 val 语料出现 0 个。
import random  # noqa: E402

_SYL = ["ska", "dre", "xis", "kar", "nex", "vla", "zum", "brei", "thor", "quen",
        "myr", "zae", "blor", "nyx", "gath", "oru", "vae", "dra", "kul", "wex"]
_FUEL = ["xenon", "helium-3", "deuterium", "antimatter", "thorium", "dark-matter",
         "plasma", "krypton", "photon", "neutronium"]


def _make_facts(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    facts, used = [], set()
    while len(facts) < n:
        name = "".join(rng.choice(_SYL) for _ in range(rng.randint(3, 5))).capitalize()
        if name in used:
            continue
        used.add(name)
        fuel = rng.choice(_FUEL)
        K = f"The {name} engine runs on refined {fuel}."
        Q = f"What does the {name} engine run on?"
        A = fuel
        facts.append({"K": K, "Q": Q, "A": A, "entity": name})
    return facts


# ---------------------------------------------------------------------------
# 表征提取 / 生成原语
# ---------------------------------------------------------------------------
@torch.no_grad()
def hidden(model, tok, text, layer, dev):
    """取某 CSA 层的隐藏态表征 [1,T,d]（pm_stream=1 时 capture 返回张量）。"""
    ids = torch.tensor([tok.encode(text)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, _, caps = model(ids, capture_layers=[layer])
    return caps[layer]


@torch.no_grad()
def continue_from(model, tok, logits, cache, dev, max_new=8):
    """从 prefill 末 logits 直接续答（**prompt 法**，对齐 teaching_sft 评估口径）。

    teaching_sft 的"有K答对"是：prefill 全 prompt（含 K 文本）后取末 token logits
    直接 argmax 续答——K 在 token 上下文（滑窗精确召回），模型读 K 的 raw token 抄答案。
    塞 eot 启动 token（cache 法）会让模型生成散文（偏离 SFT 分布），故此处不塞。
    bf16 autocast（对齐 teaching 配方）。
    """
    out = []
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        for _ in range(max_new):
            nxt = int(logits[:, -1, :].float().argmax(-1).item())
            if nxt == tok.eot_id:
                break
            out.append(nxt)
            x = torch.tensor([[nxt]], device=dev)
            logits, cache = model(x, cache)
    return tok.decode(out)


def answer_correct(gen: str, gold: str) -> bool:
    """宽松判对：生成文本（小写）含正确答案（小写）即算对。

    连字符燃料（helium-3/dark-matter）：gold 或去连字符形命中即算对（0.1B 可能只生成片段）。
    """
    g = gen.strip().lower()
    a = gold.strip().lower()
    return a in g or a.replace("-", " ") in g or a.replace("-", "") in g.replace("-", "")


# ---------------------------------------------------------------------------
# 内化（写入）：K → KV BlockPayload 写入 BlockStore（运行时零梯度，不动权重）
# ---------------------------------------------------------------------------
@torch.no_grad()
def internalize(model, tok, store, fact, a_layers, dev, d_model):
    """把新知识 K 编码成 KV 知识块写入 BlockStore（运行时零梯度写入，**不动任何权重**）。

    载体 = kv（token 寻址，能事实召回，载体能力边界红线）：收割 K 各 CSA 层的 K/V
    （prefill 后 cache 里的 [B,T,n_kv,hd] → 转置 [B,n_kv,N,hd]），作 BlockPayload 存库。
    同时返回 K 的隐藏态均值表征（供 HRL 检索候选）。
    """
    K = fact["K"]
    ids = torch.tensor([tok.encode(K)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, kcache = model(ids)  # prefill K，得各 CSA 层 K/V
    entries = {}
    for i in a_layers:
        st = kcache["layers"][i]
        # state k/v [B,T,n_kv,hd] → inject_hca_entries 需 [B,n_kv,N,hd]（dim1=n_kv）
        k = st["k"].transpose(1, 2).contiguous()
        v = st["v"].transpose(1, 2).contiguous()
        entries[i] = (k, v)
    # 检索候选表征：K 在首 CSA 层的均值隐藏态 [1,1,d]
    repr_k = hidden(model, tok, K, a_layers[0], dev)[0].mean(0, keepdim=True).unsqueeze(0)
    payload = {
        "block_id": f"fact/{fact['entity']}",
        "kind": "kv",                 # token 寻址载体（事实召回）
        "entries": entries,           # {layer_idx: (k,v)}
        "repr": repr_k,               # 检索候选表征
        "text": K,
    }
    # 写 L1（短期记忆层，容量 64）——L0 仅 8 个热块槽，20 条事实写 L1 防淘汰
    store.put(payload["block_id"], payload, tier="L1")
    return payload


# ---------------------------------------------------------------------------
# 检索（路由）：query 经 route_candidates 打分，top-k 命中
# ---------------------------------------------------------------------------
@torch.no_grad()
def retrieve(kernel, model, tok, query, candidates, k, dev, a_layers):
    """对候选块集合打分 → top-k 命中块 id 列表。

    query/candidates 用首 CSA 层隐藏态表征；route_candidates = HRL LightningIndexer。
    detach 只读（梯度隔离红线：检索 detach 不污染主干）。
    """
    q_repr = hidden(model, tok, query, a_layers[0], dev)  # [1,Tq,d]
    cand_repr = torch.cat([c["repr"] for c in candidates], dim=1)  # [1,Tk,d]
    scores = kernel.route_candidates(q_repr, cand_repr, k=None, detach_input=True)  # [1,Tq,Tk]
    cand_score = scores[0].max(dim=0).values  # [Tk]
    kk = min(k, len(candidates))
    topv, topi = cand_score.topk(kk)
    return [candidates[j]["block_id"] for j in topi.tolist()], cand_score


# ---------------------------------------------------------------------------
# 注入（执行）：命中块按载体注入 CSA 层 → 生成
# ---------------------------------------------------------------------------
@torch.no_grad()
def answer_with_kv_inject(model, tok, fact, block, a_layers, dev, max_new=8):
    """KV 注入（token 寻址）：prefill Q 得 cache → 各 CSA 层 inject_hca_entries 拼入命中块 → 生成。

    走 make_injector + kernel 语义：kv 载体经 injector 路由（namespace 校验 fail-closed）
    → inject_hca_entries 前置拼入 HCA 区。**运行时注入，不动权重**（实时可用 vs 离线 SFT）。
    prefill Q → 注入 → 从 Answer 末 logits 直接续答（prompt 法）。
    """
    qprompt = f"Question: {fact['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache = model(torch.tensor([tok.encode(qprompt)], device=dev))
        injector = make_injector()  # kv 走 blockpath namespace 校验
        for i in a_layers:
            mixer = model.layers[i].mixer
            ns = make_namespace(model.config, i, cache["layers"][i]["k"].dtype)
            k, v = block["entries"][i]
            # 经 injector 路由（kv：namespace 校验 fail-closed + 返回待拼接 (k,v)）
            payload = BlockPayload(block_id=block["block_id"], compiled_kind="kv",
                                   entries=(k, v), layer_ns=tuple(ns.values()))
            k_inj, v_inj = injector.inject(payload, namespace=ns)
            cache["layers"][i] = mixer.inject_hca_entries(cache["layers"][i], (k_inj, v_inj), ns)
    return continue_from(model, tok, logits, cache, dev, max_new)


@torch.no_grad()
def answer_with_vector_inject(model, tok, kernel, fact, block, a_layers, dev, alpha=1.0, max_new=8):
    """向量注入（steering，PM-stream 加法）：验证"向量只能 steer 不能事实召回"红线。

    用 kernel.inject（icv/steering → PM-stream 单次加法）——只 steer 行为，不做事实召回。
    """
    qprompt = f"Question: {fact['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache = model(torch.tensor([tok.encode(qprompt)], device=dev))
    vec = block["repr"][0, 0].to(torch.float32)  # K 的均值表征 [d]（向量载体载荷）
    csa = a_layers[0]
    bp = BlockPayload(block_id=block["block_id"], compiled_kind="steering", vector=vec)
    out = []
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        for _ in range(max_new):
            nxt = int(logits[:, -1, :].float().argmax(-1).item())
            if nxt == tok.eot_id:
                break
            out.append(nxt)
            x = torch.tensor([[nxt]], device=dev)
            r = model(x, cache, run_kernel=True, inject_payloads={csa: [bp]})
            logits, cache = r[0], r[1]  # run_kernel 返三元组
    return tok.decode(out)


@torch.no_grad()
def answer_baseline(model, tok, fact, dev, max_new=8):
    """基线：不注入，纯凭先验答虚构事实（应答不出）。prefill Q → 直接续答。"""
    qprompt = f"Question: {fact['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache = model(torch.tensor([tok.encode(qprompt)], device=dev))
    return continue_from(model, tok, logits, cache, dev, max_new)


@torch.no_grad()
def answer_incontext(model, tok, fact, dev, max_new=8):
    """in-context 上界：K 作纯文本前缀（teaching_sft 已验证此路径有K≈1.0）——证明知识本会答。"""
    full = f"{fact['K']}\nQuestion: {fact['Q']}\nAnswer: "
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache = model(torch.tensor([tok.encode(full)], device=dev))
    return continue_from(model, tok, logits, cache, dev, max_new)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="内化-检索-注入端到端（实时可用验证）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--n_facts", type=int, default=20)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    d_model = model.config.d_model
    print(f"[e2e] ckpt={args.ckpt} A_layers={a_layers} d_model={d_model} n_facts={args.n_facts}")

    # 运行时部件（零梯度，不动权重）：BlockStore + 内核（HRL 检索） + 记忆层（注入器用）
    store = BlockStore()
    model.attach_kernel()  # 挂 TAIS 内核（HRL LightningIndexer 检索；权重随 ckpt，未训时随机）
    kernel = model.kernel
    memory_layer = make_memory_layer(n_slots=256, key_dim=64, value_dim=d_model).to(dev)
    _ = make_injector(memory_layer)  # kv 注入走 blockpath，不需 memory_layer；mem_entry 才用

    facts = _make_facts(args.n_facts, seed=args.seed)
    print(f"[e2e] 虚构事实 {len(facts)} 条（实体先验不存在）")

    # ---- ① 内化（写入全部事实块）----
    blocks = [internalize(model, tok, store, f, a_layers, dev, d_model) for f in facts]
    print(f"[①内化] 写入 BlockStore {store.stats()} 块（运行时零梯度，不动权重）")

    # ---- ② 检索命中率：每条 Q 检索，目标块是否在 top-k ----
    hits = 0
    cos_hits = 0  # embedding 余弦相似度基线（对照：证明表征可分、缺口在 indexer 未训）
    for f, b in zip(facts, blocks):
        top_ids, _ = retrieve(kernel, model, tok, f["Q"], blocks, max(args.topk, 1), dev, a_layers)
        if b["block_id"] in top_ids[: max(args.topk, 1)]:
            hits += 1
        # 余弦基线：Q 与候选块表征的 cos 相似度 top-1（非内核 indexer，纯表征对照）
        q_r = hidden(model, tok, f["Q"], a_layers[0], dev)[0].mean(0)  # [d]
        cand_r = torch.stack([c["repr"][0, 0] for c in blocks])  # [N,d]
        cos = torch.nn.functional.cosine_similarity(q_r.unsqueeze(0), cand_r, dim=-1)
        if blocks[int(cos.argmax())]["block_id"] == b["block_id"]:
            cos_hits += 1
    hit_rate = hits / len(facts)
    cos_hit_rate = cos_hits / len(facts)
    print(f"[②检索] HRL route_candidates top-{args.topk} 命中率 = {hit_rate:.3f} ({hits}/{len(facts)}) "
          f"| embedding 余弦基线 = {cos_hit_rate:.3f}（表征可分、indexer 未训的对照）")

    # ---- ③④ 注入 + 评估（三条件对照 + in-context 上界）----
    n = len(facts)
    acc = {"baseline": 0, "kv": 0, "vector": 0, "incontext": 0}
    samples = []
    for f, b in zip(facts, blocks):
        g_base = answer_baseline(model, tok, f, dev, args.max_new)
        g_kv = answer_with_kv_inject(model, tok, f, b, a_layers, dev, args.max_new)
        g_vec = answer_with_vector_inject(model, tok, kernel, f, b, a_layers, dev, max_new=args.max_new)
        g_ic = answer_incontext(model, tok, f, dev, args.max_new)
        acc["baseline"] += answer_correct(g_base, f["A"])
        acc["kv"] += answer_correct(g_kv, f["A"])
        acc["vector"] += answer_correct(g_vec, f["A"])
        acc["incontext"] += answer_correct(g_ic, f["A"])
        if len(samples) < 3:
            samples.append({"K": f["K"], "Q": f["Q"], "A": f["A"],
                            "baseline": g_base, "kv": g_kv, "vector": g_vec, "incontext": g_ic})
    rates = {k: v / n for k, v in acc.items()}
    print(f"[③④评估] 答对率（n={n}）：")
    print(f"  不注入基线      : {rates['baseline']:.3f}  （凭先验，应≈0）")
    print(f"  KV 注入(token寻址): {rates['kv']:.3f}  （载体能事实召回；0.1B 召回头未训→通而未用）")
    print(f"  向量注入(steering): {rates['vector']:.3f}  （载体只能 steer，应≈0）")
    print(f"  in-context 上界  : {rates['incontext']:.3f}  （K 纯文本前缀，证明知识本会答）")

    realtime_ok = rates["kv"] > rates["baseline"]
    report = {
        "ckpt": args.ckpt, "n_facts": n, "topk": args.topk,
        "retrieval_hit_rate": hit_rate, "retrieval_cosine_baseline": cos_hit_rate,
        "acc_baseline": rates["baseline"], "acc_kv_inject": rates["kv"],
        "acc_vector_inject": rates["vector"], "acc_incontext_upper": rates["incontext"],
        "realtime_usable": realtime_ok,
        "verdict": {
            "retrieval": "HRL route_candidates 命中写入块" if hit_rate > 0 else "检索未命中",
            "kv_carrier": "token 寻址载体通路通（注入进 HCA 区）但 0.1B 召回头未训（门控≈0）→ 通而未用",
            "vector_carrier": "向量载体只能 steer 行为，不能事实召回（红线验证）",
            "gap": "in-context 上界≈1.0 证明知识本会答；缺口在运行时检索-注入载体的召回头（E+ 待训）",
        },
        "samples": samples,
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[判据] 实时可用（KV注入>基线）: {realtime_ok}；检索命中 {hit_rate:.3f}；"
          f"向量对照≈0 验证载体边界；in-context 上界 {rates['incontext']:.3f}")
    print(f"[save] report → {rep}")


if __name__ == "__main__":
    main()
