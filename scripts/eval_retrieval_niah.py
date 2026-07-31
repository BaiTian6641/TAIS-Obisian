"""合成检索任务评估（NIAH 式 key-value 埋点查询，GDN-1 vs GDN-2 对比）。

设计（对齐 RULER/NIAH 检索范式 + GDN-2 erase gate 主场）：
- 在序列中埋入若干 key-value 对（"The passcode for {KEY} is {VALUE}."），后接查询
  （"What is the passcode for {KEY}?"），测模型对正确 VALUE 的 next-token 预测准确率。
- 多 key 干扰（K 个 key 埋点，查其一）——GDN-2 的 erase gate 应在写新 key 时**保护
  已存 key 关联**（选择性擦除），GDN-1 标量 β 无差别覆盖 → GDN-2 检索准确率应更高。
  这正是 §25.2 已知 GDN 固定状态检索短板 + GDN-2 RULER 增益（S-NIAH-3 63.2→89.8）的
  最小复现。
- 对 GDN-1 / GDN-2 两 checkpoint 同协议评估（同埋点集、同 seed）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/eval_retrieval_niah.py \
      --ckpt_gdn1 checkpoints/pilot_0p1b_gdn1/final \
      --ckpt_gdn2 checkpoints/pilot_0p1b_gdn2/final [--n_keys 8] [--n_queries 100]
输出：控制台 + runs/retrieval_niah/report.json（两模型检索准确率对比）。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 埋点 key/value 词表（虚构专名 + 数字码，避免与训练语料重叠的记忆捷径）
KEY_POOL = ["zephyr", "quillon", "marvex", "tandril", "obscura", "vellith",
            "noryx", "kalmesh", "dravok", "silune", "prexia", "jundar"]


def build_niah_sample(rng: np.random.Generator, n_keys: int, query_key: str | None = None):
    """构造一个 NIAH 样本：n_keys 个 key-value 埋点 + 一个查询。

    返回 (context_ids_text, query_text, correct_value)。
    """
    keys = list(rng.choice(KEY_POOL, size=n_keys, replace=False))
    values = [str(rng.integers(1000, 9999)) for _ in range(n_keys)]
    # 埋点（顺序随机）
    facts = [f"The passcode for {k} is {v}." for k, v in zip(keys, values)]
    qk = query_key or str(keys[int(rng.integers(n_keys))])
    qv = values[keys.index(qk)]
    context = " ".join(facts)
    query = f"What is the passcode for {qk}? The passcode for {qk} is"
    return context, query, qv


@torch.no_grad()
def eval_retrieval(model, tok, rng, n_keys, n_queries, device):
    """测 n_queries 个埋点查询的 top-1 检索准确率（正确 VALUE 首位数字 token）。"""
    correct = 0
    for _ in range(n_queries):
        context, query, qv = build_niah_sample(rng, n_keys)
        text = context + " " + query
        ids = torch.tensor(tok.encode(text)).unsqueeze(0).to(device)
        logits, _ = model(ids)
        # 下一个 token 应是 qv 的首 token（数字码首位）
        next_logits = logits[0, -1]
        pred_id = int(next_logits.argmax().item())
        pred_tok = tok.decode([pred_id]).strip()
        # 命中：预测 token 与正确值的首 token 匹配（数字码逐 token）
        correct_first = tok.encode(qv)[0] if tok.encode(qv) else None
        if correct_first is not None and pred_id == correct_first:
            correct += 1
        elif pred_tok and qv.startswith(pred_tok):
            correct += 1
    return correct / n_queries


def main() -> None:
    ap = argparse.ArgumentParser(description="NIAH 检索评估（GDN-1 vs GDN-2）")
    ap.add_argument("--ckpt_gdn1", default="checkpoints/pilot_0p1b_gdn1/final")
    ap.add_argument("--ckpt_gdn2", default="checkpoints/pilot_0p1b_gdn2/final")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--n_keys", type=int, default=8)
    ap.add_argument("--n_queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/retrieval_niah/report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = TokenizerIO(args.tokenizer)
    print(f"[niah] n_keys={args.n_keys} n_queries={args.n_queries} seed={args.seed}")
    results = {}
    for tag, ckpt in [("gdn1", args.ckpt_gdn1), ("gdn2", args.ckpt_gdn2)]:
        if not Path(ckpt).exists():
            print(f"[niah] {tag} checkpoint 不存在: {ckpt}，跳过")
            results[tag] = None
            continue
        model = TaisObsidianForCausalLM.from_pretrained(ckpt, args.device, strict=False)
        model.eval()
        rng = np.random.default_rng(args.seed)  # 同 seed → 同埋点集（公平对比）
        acc = eval_retrieval(model, tok, rng, args.n_keys, args.n_queries, args.device)
        results[tag] = acc
        print(f"[niah] {tag} ({ckpt.split('/')[-2]}): 检索准确率 = {acc:.3f}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    if results.get("gdn1") is not None and results.get("gdn2") is not None:
        delta = results["gdn2"] - results["gdn1"]
        verdict = ("✅ GDN-2 检索优于 GDN-1" if delta > 0.02
                   else ("🟡 相当（|Δ|<0.02）" if abs(delta) <= 0.02 else "⚠️ GDN-2 检索反而劣"))
        print(f"\n对比：GDN-1 {results['gdn1']:.3f} vs GDN-2 {results['gdn2']:.3f}（Δ={delta:+.3f}）")
        print(f"判定：{verdict}")
        results["delta"] = delta
        results["verdict"] = verdict

    results["n_keys"] = args.n_keys
    results["n_queries"] = args.n_queries
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[niah] report → {out}")


if __name__ == "__main__":
    main()
