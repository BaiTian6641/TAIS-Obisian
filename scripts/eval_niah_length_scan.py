"""NIAH 长度扫描评估（GDN 状态饱和行为实测：GDN-1 vs GDN-2 10k 有界）。

fb1 评审最有价值批评（runs/feedback/fb1.md + repo 记忆 fb1-feedback-verification.md）：
现有 scripts/eval_retrieval_niah.py 是 8-key **短上下文**（数十 token）合成检索 + 单首 token
判据，绝对值 0.217 不能带"已验证"错觉进 1.5B/1M（设计 §9 #4）。本脚本做**长度 × key 数 ×
判据**三维扫描，实测 GDN 递归状态在长上下文/多干扰下的检索衰减行为。

设计（纯新增，不改原 eval_retrieval_niah.py）：
- **长度扫描**：总上下文 token 数 target_length（默认 [512,1024]；>1024 为 0.1B 训练 max_seq
  外推，RoPE 缓存上限=模型 config.max_seq=1024，超出自动裁剪并诚实标注）。
- **多 key 数扫描**：n_keys ∈ [8,32,128]（干扰强度递增；KEY_POOL 与原脚本共用 12 虚构
  专名，超出池自动扩展词形变体 key_i，avoid 训练语料捷径）。
- **放宽判据**：两个判据并列报告——first-token（原脚本同口径，VALUE 首 token argmax 命中）
  与 full-VALUE（增量逐 token argmax 生成完整 VALUE 全对，更严格对照）——定位 0.217 低值
  是 GDN 状态饱和还是判据过严。
- **填充文本**：val shard 真实 token 随机切片 + 末段合成无关句（英文维基式模板），埋点
  均匀分散到填充中（埋点间 gap≈均等，测长上下文检索而非邻近检索）。
- **增量生成注意**：prefill 按 prefill_chunk（默认 512 token）分块走 cache 增量（GDN chunked
  递归 + 注意力 KV cache 递增拼接），不重算全前缀；VALUE 逐 token argmax 走 T=1 naive 递归。
- **0.1B 外推诚实标注**：训练 seq_len=1024；>1024 靠 GDN 递归状态 + 三级注意力（滑窗 L0 512
  + CSA stride-4 + HCA 128:1）外推，RoPE 滑窗分支受 config.max_seq=1024 硬限（超出位置无
  cos/sin 缓存，本脚本裁剪 target_length 并记录 boundary 字段）。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/eval_niah_length_scan.py \
      [--lengths 512 1024] [--n_keys 8 32] [--n_queries 50] [--prefill_chunk 512] [--scan]
输出：控制台表 + runs/niah_length_scan/report.json（GDN-1 vs GDN-2 各 cell 两判据检索率）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 与原 eval_retrieval_niah.py 同一虚构专名池（避免训练语料记忆捷径）；12 个词
KEY_POOL = ["zephyr", "quillon", "marvex", "tandril", "obscura", "vellith",
            "noryx", "kalmesh", "dravok", "silune", "prexia", "jundar"]

# 填充合成无关句模板（val 真实文本之外的干扰段；无事实性数字，避免与埋点数字混淆）
FILLER_TEMPLATES = [
    "The committee reviewed the proposal and deferred the vote to the next session.",
    "Researchers noted that the methodology required further validation before publication.",
    "The committee meeting adjourned after a lengthy discussion of the budget.",
    "A spokesperson said the findings would be released in due course.",
    "The report highlighted several areas where additional study was recommended.",
    "Local officials expressed cautious optimism about the new transit plan.",
    "The museum's new exhibit draws on artifacts from the early medieval period.",
    "Analysts expect the market to remain volatile through the end of the quarter.",
    "The team published their preliminary observations in an open-access journal.",
    "Organizers said the festival would return next year with an expanded program.",
]


def _key_variants(rng: np.random.Generator, n_keys: int) -> list[str]:
    """取 n_keys 个互不相同的 key：池内（≤12）全用原词；超池在原词后扩展变体后缀。"""
    if n_keys <= len(KEY_POOL):
        return list(rng.choice(KEY_POOL, size=n_keys, replace=False))
    # 超池：key_i 变体（虚构词+编号仍是专名式，token 成本≈原词+1）
    keys = [f"{k}_{i}" for i, k in enumerate(KEY_POOL * ((n_keys // len(KEY_POOL)) + 1))]
    return keys[:n_keys]


def build_niah_length_sample(
    rng: np.random.Generator,
    tok: TokenizerIO,
    val_ids: np.ndarray | None,
    target_length: int,
    n_keys: int,
    query_key: str | None = None,
) -> dict:
    """构造一个长度扫描 NIAH 样本（token 空间操作，返回 token 序列与判据元数据）。

    结构：n_keys 个埋点（"The passcode for {KEY} is {VALUE}."）均匀分散进填充文本
    （val shard 真实 token + 末段合成无关句），末尾接查询
    （"What is the passcode for {KEY}? The passcode for {KEY} is"）。

    埋点 token 定位（与分词无关、无需字符串对齐）：锚 = encode("The passcode for {key} is")
    恰好是埋点段去尾 3 token 的前缀（value 2 token + "."），逐埋点验证后取下一段起点。

    返回 dict：
      ids (list[int]) 完整序列（含查询）；facts_end (list[int]) 各埋点结束位置（prefix 长度，
        模型在该位置的 next-token 预测目标 = 该埋点 VALUE 首 token）；fact_values (list[str])；
      query_fact_idx 被查询埋点下标；query_prefix_len 查询判定点位置（len(ids）截断
        query_value 前的长度）；query_value 正确 VALUE 字符串；query_value_ids 其 token 序列。
    """
    keys = _key_variants(rng, n_keys)
    values = [str(rng.integers(1000, 9999)) for _ in range(n_keys)]

    fact_ids: list[list[int]] = []
    for k, v in zip(keys, values):
        full = tok.encode(f"The passcode for {k} is {v}.")
        anchor = tok.encode(f"The passcode for {k} is")
        # 校验锚为前缀（value token 数不固定：1000-9999 数字 BPE 2-4 token + 句号）
        assert full[: len(anchor)] == anchor and len(full) - len(anchor) >= 2, (
            f"埋点分词结构变化：{k}/{v} full={full} anchor={anchor}"
        )
        fact_ids.append(full)

    # 查询埋点（随机或指定）
    qi = int(rng.integers(n_keys)) if query_key is None else keys.index(query_key)
    qk, qv = keys[qi], values[qi]
    # 查询 token：与前文拼接时 BPE 会在段首 ' What' 合并为单 token（1786）——
    # 与独立编码（'What'=1622）差 1 token；为精确总长，这里直接取拼接语境下的编码
    query_text = f" What is the passcode for {qk}? The passcode for {qk} is"
    query_ids = tok.encode(query_text)
    qv_ids = tok.encode(" " + qv)  # 查询后 VALUE 前带空格（与埋点句式一致）

    overhead = sum(len(f) for f in fact_ids) + len(query_ids)
    fill_budget = max(target_length - overhead, 0)

    # 填充：val 真实 token 占 ~80% + 末段合成无关句 ~20%（真实+受控混合干扰）
    filler: list[int] = []
    synth_budget = fill_budget // 5
    real_budget = fill_budget - synth_budget
    if val_ids is not None and real_budget > 0:
        off = int(rng.integers(0, max(val_ids.size - real_budget, 1)))
        filler.extend(int(x) for x in val_ids[off : off + real_budget])
    if synth_budget > 0:
        n_sents = max(synth_budget // 15, 1)  # 每句约 15 token
        sents = [FILLER_TEMPLATES[int(rng.integers(len(FILLER_TEMPLATES)))] for _ in range(n_sents)]
        synth_ids = tok.encode("\n\n" + " ".join(sents))
        filler.extend(synth_ids[:synth_budget])
    # 合成句截断后可能短于 fill_budget（句子边界截断），用 val token 补足到精确预算
    if len(filler) < fill_budget and val_ids is not None:
        need = fill_budget - len(filler)
        off = int(rng.integers(0, max(val_ids.size - need, 1)))
        filler.extend(int(x) for x in val_ids[off : off + need])
    # val_ids=None（结构测试/无 checkpoint）时用合成句填充到精确预算（否则 n_tokens 远小于
    # target——val 缺失时 synth_budget 仅 1/5 且无补足路径）。n_sents 按保守 ~10 token/句
    # 估算保证充足（每句实际 ~15-20 token），编码后截到精确 need。
    if len(filler) < fill_budget:
        need = fill_budget - len(filler)
        n_sents = max(need // 10 + 1, 1)
        sents = [FILLER_TEMPLATES[int(rng.integers(len(FILLER_TEMPLATES)))] for _ in range(n_sents)]
        filler.extend(tok.encode(" " + " ".join(sents))[:need])
    filler = filler[:fill_budget]  # 截断到精确预算（filler 盈余时）

    # 埋点均匀分散：切成 n_keys+1 段，埋点置于段边界（gap≈fill_budget/(n_keys+1)）
    gaps = np.full(n_keys + 1, fill_budget // (n_keys + 1), dtype=np.int64)
    gaps[: fill_budget % (n_keys + 1)] += 1  # 余数摊到前几段
    ids: list[int] = []
    facts_end: list[int] = []
    pos = 0
    for i in range(n_keys):
        seg = filler[pos : pos + int(gaps[i])]
        ids.extend(seg)
        pos += int(gaps[i])
        ids.extend(fact_ids[i])
        facts_end.append(len(ids))  # 该埋点结束后的 prefix 长度（预测位 = VALUE 首 token）
    ids.extend(filler[pos:])  # 末段填充（埋点与查询之间的检索距离尾段）
    query_prefix_len = len(ids)
    ids.extend(query_ids)

    return {
        "ids": ids,
        "facts_end": facts_end,
        "fact_values": values,
        "query_fact_idx": qi,
        "query_prefix_len": query_prefix_len,
        "query_value": qv,
        "query_value_ids": qv_ids,
        "n_tokens": len(ids),
    }


@torch.no_grad()
def _prefill_chunked(model, ids: list[int], device: str, chunk: int):
    """分块 prefill（cache 增量，不重算全前缀）：GDN 走 chunked 递归、注意力 KV 递增拼接。

    返回每步的 logits（仅保留最后一块的完整 logits 足够判据使用；为埋点判据需逐块收集）。
    """
    cache = None
    logits = None
    for s in range(0, len(ids), chunk):
        seg = ids[s : s + chunk]
        x = torch.tensor([seg], dtype=torch.long, device=device)
        logits, cache = model(x, cache)
    return logits, cache


@torch.no_grad()
def _gen_argmax(model, cache, next_id: int, n_tokens: int, device: str) -> list[int] | None:
    """VALUE 完整匹配：逐 token argmax 增量生成 n_tokens 个（T=1 naive 递归路径）。

    外推硬限：生成使 cache 长度 > max_seq 时，A 层 `k_rope=_rope(k,0)` 从 0 全量重算 RoPE
    越界（cos/sin 仅 max_seq 行）——这是模型架构的真实外推边界（训练 seq=max_seq 下从未
    生成超 max_seq）。捕获该 RuntimeError 返回 None（full 判据不可达，诚实标注，非 crash）。
    """
    out: list[int] = []
    cur = next_id
    for _ in range(n_tokens):
        x = torch.tensor([[cur]], dtype=torch.long, device=device)
        try:
            logits, cache = model(x, cache)
        except RuntimeError as e:
            if "size of tensor" in str(e) and "must match" in str(e):
                return None  # RoPE 外推硬限：生成超 max_seq，full 判据不可达
            raise
        nxt = int(logits[0, -1].float().argmax().item())
        out.append(nxt)
        cur = nxt
    return out


@torch.no_grad()
def eval_one_sample(model, sample: dict, device: str, chunk: int) -> dict:
    """评估一个样本：首 token 判据 + 完整 VALUE 判据 + 埋点级诊断。"""
    ids = sample["ids"]
    qpl = sample["query_prefix_len"]
    # 一次性 prefill 到查询前缀（分块增量），收集查询判定点的 next-token
    cache = None
    q_logits = None
    facts_logits: dict[int, torch.Tensor] = {}
    for s in range(0, qpl, chunk):
        seg = ids[s : s + chunk]
        x = torch.tensor([seg], dtype=torch.long, device=device)
        logits, cache = model(x, cache)
        # 收集落在本块内的埋点结束位与查询前缀末位
        for i, fe in enumerate(sample["facts_end"]):
            if s < fe <= s + len(seg):
                facts_logits[i] = logits[0, fe - 1 - s].float().cpu()
        if s + len(seg) >= qpl:
            q_logits = logits[0, -1].float()
    assert q_logits is not None and len(facts_logits) == len(sample["facts_end"])

    # 判据 1（首 token）：查询位 argmax == VALUE 首 token
    pred_first = int(q_logits.argmax().item())
    qv_first = sample["query_value_ids"][0]
    hit_first = pred_first == qv_first

    # 判据 2（完整 VALUE）：逐 token argmax 生成 len(qv_ids) 个，全对才算命中。
    # gen=None = 生成超 max_seq 触发 RoPE 外推硬限 → full 判据不可达（诚实标注，非 hit）。
    gen = _gen_argmax(model, cache, pred_first, len(sample["query_value_ids"]) - 1, device)
    full_unreachable = gen is None
    pred_full = [pred_first] + gen if gen is not None else [pred_first]
    hit_full = (pred_full == sample["query_value_ids"]) if gen is not None else False

    # 埋点级诊断（各埋点首 token 命中率——检索近/远埋点衰减用）
    fact_hits = []
    for i, fe in enumerate(sample["facts_end"]):
        fv_first = None
        full = sample["fact_values"][i]
        ids_v = None
        fact_hits.append(int(int(facts_logits[i].argmax().item()) == _value_first_tok(full)))
    return {"hit_first": int(hit_first), "hit_full": int(hit_full), "fact_hits": fact_hits,
            "full_unreachable": full_unreachable}


# 模块级 tokenizer 引用（_value_first_tok 用；eval 前由 main 注入）
_TOK: TokenizerIO | None = None
_VALUE_FIRST_CACHE: dict[str, int] = {}


def _value_first_tok(value: str) -> int:
    """VALUE（4 位数字串）首 token id（带空格前缀，与埋点/查询句式一致）。"""
    if value not in _VALUE_FIRST_CACHE:
        _VALUE_FIRST_CACHE[value] = _TOK.encode(" " + value)[0]
    return _VALUE_FIRST_CACHE[value]


def run_scan(
    model,
    tok: TokenizerIO,
    val_ids: np.ndarray | None,
    lengths: list[int],
    n_keys_list: list[int],
    n_queries: int,
    seed: int,
    device: str,
    chunk: int,
    max_len_cap: int,
) -> dict:
    """跑完整扫描：lengths × n_keys 网格，每 cell n_queries 个样本（同 seed 公平对比）。"""
    global _TOK
    _TOK = tok
    results: dict[str, dict] = {}
    for L in lengths:
        for nk in n_keys_list:
            t0 = time.time()
            rng = np.random.default_rng(seed)  # 每 cell 同 seed → 同埋点集（公平）
            hit_first = hit_full = 0
            fact_hit_sum = fact_hit_n = 0
            n_tok_obs = []
            n_trunc = 0
            n_full_unreach = 0
            for _ in range(n_queries):
                sample = build_niah_length_sample(rng, tok, val_ids, L, nk)
                # RoPE 硬限：cache 语义下 key 需从 0 全量 RoPE（tri_attention.forward
                # `k_rope = self._rope(k, 0)`），Tk>max_seq 越界（cos/sin 仅 max_seq 行）
                # → 样本截断到 ≤max_len_cap（含查询）并诚实标注 truncated。
                if sample["n_tokens"] > max_len_cap:
                    ov = sample["n_tokens"] - max_len_cap
                    sample["ids"] = sample["ids"][:max_len_cap]
                    sample["query_prefix_len"] -= ov
                    sample["n_tokens"] = max_len_cap
                    # 截断后 facts_end 重定位：fe_new = fe_old − ov，仅保留落在
                    # [1, query_prefix_len] 内的（被截掉的埋点从判据剔除，诚实不计入）。
                    sample["facts_end"] = [fe - ov for fe in sample["facts_end"]
                                           if 1 <= fe - ov <= sample["query_prefix_len"]]
                    n_trunc += 1
                n_tok_obs.append(sample["n_tokens"])
                r = eval_one_sample(model, sample, device, chunk)
                hit_first += r["hit_first"]
                hit_full += r["hit_full"]
                fact_hit_sum += sum(r["fact_hits"])
                fact_hit_n += len(r["fact_hits"])
                n_full_unreach += int(r["full_unreachable"])
            key = f"L{L}_k{nk}"
            results[key] = {
                "target_length": L,
                "n_keys": nk,
                "n_queries": n_queries,
                "acc_first": round(hit_first / n_queries, 4),
                "acc_full": round(hit_full / n_queries, 4),
                "fact_acc_first": round(fact_hit_sum / max(fact_hit_n, 1), 4),
                "avg_tokens": round(float(np.mean(n_tok_obs)), 1),
                "sec": round(time.time() - t0, 1),
                # L>max_len_cap = 外推截断（样本截到 max_len_cap 后评测，非全长真实检索）
                "clipped": L > max_len_cap,
                "extrapolated_clipped": L > max_len_cap,
                "truncated": n_trunc,
                # 生成超 max_seq 触发 RoPE 外推硬限的样本数（full 判据不可达，诚实标注）
                "full_unreachable": n_full_unreach,
                "gen_boundary_hit": n_full_unreach > 0,
            }
            tr = f" trunc={n_trunc}" if n_trunc else ""
            fu = f" full_unreach={n_full_unreach}" if n_full_unreach else ""
            print(f"  [scan] {key}: first={results[key]['acc_first']:.3f} "
                  f"full={results[key]['acc_full']:.3f} fact={results[key]['fact_acc_first']:.3f} "
                  f"({results[key]['sec']}s{tr}{fu})", flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="NIAH 长度扫描（GDN-1 vs GDN-2，长度×key×判据）")
    ap.add_argument("--ckpt_gdn1", default="checkpoints/pilot_0p1b_gdn1/final")
    ap.add_argument("--ckpt_gdn2", default="checkpoints/pilot_0p1b_gdn2_bounded_10k/final",
                    help="GDN-2 对比端（默认 10k 有界 decay checkpoint）")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--data_dir", default="data/shards", help="val shard 目录（填充真实文本）")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024])
    ap.add_argument("--n_keys", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--n_queries", type=int, default=50)
    ap.add_argument("--prefill_chunk", type=int, default=512,
                    help="prefill 分块大小（token）；≤训练 seq_len 保持数值语义一致")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/niah_length_scan/report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--scan", action="store_true", help="完整扫描模式（默认即扫描；保留兼容）")
    args = ap.parse_args()

    tok = TokenizerIO(args.tokenizer)
    # val shard 真实文本作填充（失败则退化为纯合成句填充，诚实标注）
    val_ids: np.ndarray | None = None
    try:
        vs = Shards(args.data_dir, "val")
        val_ids = vs._maps[0]
        print(f"[niah-scan] 填充文本：val shard（{vs.total} tokens）+ 合成无关句")
    except Exception as e:  # noqa: BLE001
        print(f"[niah-scan] ⚠️ val shard 不可用（{e}），退化为纯合成无关句填充")

    print(f"[niah-scan] lengths={args.lengths} n_keys={args.n_keys} "
          f"n_queries={args.n_queries} prefill_chunk={args.prefill_chunk} seed={args.seed}")

    all_results: dict[str, dict] = {}
    boundary_notes: list[str] = []
    for tag, ckpt in [("gdn1", args.ckpt_gdn1), ("gdn2", args.ckpt_gdn2)]:
        if not Path(ckpt).exists():
            print(f"[niah-scan] {tag} checkpoint 不存在: {ckpt}，跳过")
            all_results[tag] = None
            continue
        model = TaisObsidianForCausalLM.from_pretrained(ckpt, args.device, strict=False)
        model.eval()
        max_seq = int(model.config.max_seq)
        # RoPE 硬限：Tk>max_seq 越界（cache 语义 key 从 0 全量 RoPE，cos/sin 仅 max_seq 行）。
        # >max_seq 目标 = 0.1B 训练 max_seq 外推：保目标长度 L 标签、样本截断到 ≤max_seq
        # （run_scan 内做），boundary 诚实标注"外推截断"——非全长真实检索。
        for L in args.lengths:
            if L > max_seq:
                note = (f"target_length={L} 超模型 max_seq={max_seq}（RoPE 缓存硬限）→ "
                        f"样本截断到 ≤{max_seq}（外推截断，非全长真实检索）；" 
                        f">1024 全长外推需 max_seq 扩展后实测，如实标注边界")
                if note not in boundary_notes:
                    boundary_notes.append(note)
                    print(f"[niah-scan] ⚠️ {note}")
        lengths = sorted(set(args.lengths))
        print(f"[niah-scan] ── {tag} ({ckpt}) max_seq={max_seq} lengths={lengths} ──")
        t0 = time.time()
        all_results[tag] = run_scan(
            model, tok, val_ids, lengths, args.n_keys, args.n_queries,
            args.seed, args.device, args.prefill_chunk, max_seq,
        )
        print(f"[niah-scan] {tag} 完成，{time.time() - t0:.0f}s")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # 汇总表（GDN-1 vs GDN-2 各 cell 两判据）
    if all_results.get("gdn1") and all_results.get("gdn2"):
        print("\n════ NIAH 长度扫描对比（first-token / full-VALUE）════")
        hdr = f"{'cell':<14}{'gdn1_first':>11}{'gdn2_first':>11}{'Δfirst':>9}{'gdn1_full':>10}{'gdn2_full':>10}{'Δfull':>8}"
        print(hdr)
        for key in all_results["gdn1"]:
            r1, r2 = all_results["gdn1"][key], all_results["gdn2"][key]
            print(f"{key:<14}{r1['acc_first']:>11.3f}{r2['acc_first']:>11.3f}"
                  f"{r2['acc_first']-r1['acc_first']:>+9.3f}{r1['acc_full']:>10.3f}"
                  f"{r2['acc_full']:>10.3f}{r2['acc_full']-r1['acc_full']:>+8.3f}")

    report = {
        "config": {
            "lengths": args.lengths, "n_keys": args.n_keys, "n_queries": args.n_queries,
            "prefill_chunk": args.prefill_chunk, "seed": args.seed,
            "ckpt_gdn1": args.ckpt_gdn1, "ckpt_gdn2": args.ckpt_gdn2,
            "filler": "val_shard+synthetic" if val_ids is not None else "synthetic_only",
        },
        "boundary_notes": boundary_notes,
        "results": all_results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[niah-scan] report → {out}")


if __name__ == "__main__":
    main()
