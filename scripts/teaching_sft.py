"""教学式 SFT pilot（知识内化训练 · 阶段 1：教模型"用上 K"的内化行为）。

设计规范：docs/TAIS_Obsidian_知识内化训练_分析与设计.md §2.1/§3。

**红线**：这是 **SFT 教学**（离线训练，教模型内化行为），**不是知识块写入**——
与运行时 steering/KV/记忆层零梯度写入（W1–W2）不同。此处是离线全参数小 lr SFT，
属规范 §3 阶段 1（SFT 教行为，离线），不触碰"运行时不动权重"红线。

基座：checkpoints/pilot_0p1b_gdn2_10k/final（GDN-2，d_model=768，GDN-2 门已收敛）。

SFT 格式：把 {K, Q, A} 拼成教学序列，next-token 损失**只算 Answer 部分**
（mask K/Q 的 label=-100，对齐 SFT 惯例——只学答不学背）：

    "{K}\nQuestion: {Q}\nAnswer: {A}<|endoftext|>"

训练循环对齐 src/tais_obsidian/train.py 配方（bf16 autocast + fp32 参数、
grad clip 1.0、AdamW β=(0.9,0.95) eps=1e-8），lr 小（默认 1e-4，几百步）。

微调策略：**全参数小 lr SFT**（0.1B 小模型，~几百步）。备选：只微调注意力/MLP
上层（注释说明）——但全参数更简单且 0.1B 显存够（~0.5GB 模型 + 小 batch）。

评估（关键验证判据，规范 §2.1）：
- **有 K 答对率**：给 K + K-依赖 Q，模型答对率。
- **无 K 答对率**：去掉 K 只给 Q，模型答对率（应低，证明 K-依赖）。
- **内化判据**：有 K ≫ 无 K（差值越大=内化越强）；SFT 后差值应更大（教学有效）。
- **退联检验判据**：一致 K(accept) 内化率 > 矛盾 K(reject) 内化率（模型学会区分）。

双卡分工：本脚本用 RTX 4070（CUDA_VISIBLE_DEVICES=0，8GB，控制 batch/seq）。

用法：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/teaching_sft.py --steps 300
产出：checkpoints/pilot_0p1b_gdn2_10k_teaching/ + runs/teaching_sft/report.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k/final"
DEFAULT_DATA = "runs/teaching_data/teaching_samples.jsonl"
DEFAULT_OUT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
DEFAULT_REPORT = "runs/teaching_sft/report.json"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
IGNORE = -100  # label mask：K/Q 不计损失，只学 Answer（SFT 惯例）


# ---------------------------------------------------------------------------
# 数据编码：拼教学序列 + label mask（K/Q mask，只 A 计损失）
# ---------------------------------------------------------------------------
def encode_sample(tok: TokenizerIO, s: dict, seq_len: int) -> tuple[list[int], list[int]]:
    """把 {K,Q,A} 编码为 (input_ids, labels)；labels 中 K/Q 部分 = -100，仅 A+EOT 计损失。

    序列： "{K}\nQuestion: {Q}\nAnswer: {A}<EOT>"
    next-token 预测：input_ids[:-1] → labels[1:]（对齐因果 LM）。
    我们只对 Answer 段（含其后的 EOT）计损失——模型学"看到 Question 后产出 A"。
    """
    prompt = f"{s['K']}\nQuestion: {s['Q']}\nAnswer: "
    answer = f"{s['A']}"
    p_ids = tok.encode(prompt)
    a_ids = tok.encode(answer) + [tok.eot_id]
    ids = (p_ids + a_ids)[:seq_len]
    # labels：先全 -100，再把 Answer 段（含 EOT）置为真实 id
    labels = [IGNORE] * len(ids)
    a_start = min(len(p_ids), len(ids))
    for i in range(a_start, len(ids)):
        labels[i] = ids[i]
    return ids, labels


def build_batch(tok: TokenizerIO, samples: list[dict], seq_len: int,
                device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """编码一批样本 → (x [B,T], y [B,T])；y 已右移且含 -100 mask，padding 亦 mask。"""
    enc = [encode_sample(tok, s, seq_len) for s in samples]
    T = max(len(ids) for ids, _ in enc)
    T = min(T, seq_len)
    pad = tok.eot_id
    bx = np.full((len(enc), T), pad, dtype=np.int64)
    by = np.full((len(enc), T), IGNORE, dtype=np.int64)
    for b, (ids, labels) in enumerate(enc):
        n = min(len(ids), T)
        # x = ids[:-1]，y = labels[1:]（因果 LM 右移）
        bx[b, : n - 1] = ids[: n - 1]
        by[b, : n - 1] = labels[1:n]
    return torch.from_numpy(bx).to(device), torch.from_numpy(by).to(device)


def masked_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """只对非 -100 位置计 cross entropy（分块 fp32，对齐 train.py.chunked_ce）。"""
    flat = logits.reshape(-1, logits.size(-1)).float()
    tgt = targets.reshape(-1)
    return F.cross_entropy(flat, tgt, ignore_index=IGNORE)


# ---------------------------------------------------------------------------
# 生成式答对评估：给 prompt（有/无 K）生成短答案，判是否含正确 A
# ---------------------------------------------------------------------------
@torch.no_grad()
def gen_answer(model, tok, prompt: str, device: str, max_new: int = 12) -> str:
    """贪心生成短答案（temperature=0），截到换行/EOT。"""
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out: list[int] = []
    with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
        logits, cache = model(x)
        for _ in range(max_new):
            nxt = int(logits[:, -1, :].float().argmax(-1).item())
            if nxt == tok.eot_id:
                break
            out.append(nxt)
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, cache = model(x, cache)
    return tok.decode(out)


def answer_correct(gen: str, gold: str) -> bool:
    """宽松判对：生成文本（小写）含正确答案（小写）即算对（短答案精确匹配对 0.1B 过苛）。"""
    return gold.strip().lower() in gen.strip().lower()


@torch.no_grad()
def evaluate(model, tok, samples: list[dict], device: str) -> dict:
    """SFT 前/后评估：有 K vs 无 K 答对率 + 一致/矛盾内化率（退联检验判据）。

    - fact/chain（k_dep=True）：有 K prompt = K+Question，无 K prompt = 仅 Question。
    - consist：判 "accept/reject" 行为——一致 K 应判 consistent（内化/接受），
      矛盾 K 应判 contradictory（拒/标分歧）。内化率=判 "consistent" 的比例。
    """
    dep = [s for s in samples if s["q_type"] in ("fact", "chain")]
    cons = [s for s in samples if s["q_type"] == "consist"]

    # 有 K / 无 K 答对率（事实+链条）
    with_k = sum(answer_correct(gen_answer(model, tok, f"{s['K']}\nQuestion: {s['Q']}\nAnswer: ", device), s["A"]) for s in dep)
    without_k = sum(answer_correct(gen_answer(model, tok, f"Question: {s['Q']}\nAnswer: ", device), s["A"]) for s in dep)
    n_dep = max(len(dep), 1)

    # 退联检验：一致 vs 矛盾 K 的"内化率"（判 consistent 的比例）
    ok = [s for s in cons if s["label"] == "accept"]
    bad = [s for s in cons if s["label"] == "reject"]

    def consist_accept_rate(group):
        # 判 "consistent"（且非 contradictory）为接受/内化；只生成一次
        if not group:
            return 0.0
        c = 0
        for s in group:
            g = gen_answer(model, tok, f"{s['K']}\nQuestion: {s['Q']}\nAnswer: ", device).lower()
            if "consistent" in g and "contradictory" not in g:
                c += 1
        return c / len(group)

    acc_ok = consist_accept_rate(ok)
    acc_bad = consist_accept_rate(bad)
    return {
        "n_dep": n_dep,
        "with_k_acc": with_k / n_dep,       # 有 K 答对率
        "without_k_acc": without_k / n_dep,  # 无 K 答对率（应低，证明 K-依赖）
        "internalization_gap": (with_k - without_k) / n_dep,  # 内化差值（越大越强）
        "consist_accept_rate": acc_ok,       # 一致 K 内化率（应高）
        "contradict_accept_rate": acc_bad,   # 矛盾 K 内化率（应低）
        "consist_gap": acc_ok - acc_bad,     # 退联检验差值（>0 = 学会区分）
    }


# ---------------------------------------------------------------------------
# 主流程：加载基座 → SFT 前评估 → SFT 训练 → SFT 后评估 → 保存 + report
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="教学式 SFT pilot（知识内化 · 阶段 1）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--micro_batch", type=int, default=16, help="8GB 卡：OOM 则减小")
    ap.add_argument("--seq_len", type=int, default=192, help="教学序列短（K+Q+A）")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_n", type=int, default=120, help="评估抽样数（各 split）")
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)

    # 读教学样本，切 train/eval
    samples = [json.loads(l) for l in Path(args.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    rng.shuffle(samples)
    n_eval = min(args.eval_n, len(samples) // 5)
    eval_samples, train_samples = samples[:n_eval], samples[n_eval:]
    print(f"[sft] 教学样本 {len(samples)}：train {len(train_samples)} / eval {len(eval_samples)}")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, device)
    model.config.grad_checkpoint = False  # SFT 序列短，省重算开销换速度
    model.train()

    # ---- SFT 前评估（基线）----
    print("[eval] SFT 前评估…")
    model.eval()
    pre = evaluate(model, tok, eval_samples, device)
    model.train()
    print(f"  前: 有K {pre['with_k_acc']:.3f} 无K {pre['without_k_acc']:.3f} "
          f"内化差 {pre['internalization_gap']:.3f} | 一致 {pre['consist_accept_rate']:.3f} "
          f"矛盾 {pre['contradict_accept_rate']:.3f} 退联差 {pre['consist_gap']:.3f}")

    # ---- 优化器（对齐 train.py：AdamW β=(0.9,0.95)，decay 分组）----
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (decay if p.ndim >= 2 and "embed" not in name else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95), eps=1e-8, fused=torch.cuda.is_available())

    def lr_at(step):
        return args.lr * (step + 1) / args.warmup if step < args.warmup else args.lr

    # ---- SFT 训练循环（bf16 autocast + grad clip，对齐 train.py）----
    print(f"[sft] 开始训练 steps={args.steps} micro={args.micro_batch} seq={args.seq_len} lr={args.lr}")
    loss_hist, t0 = [], time.time()
    for step in range(args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = rng.integers(0, len(train_samples), size=args.micro_batch)
        batch = [train_samples[i] for i in idx]
        x, y = build_batch(tok, batch, args.seq_len, device)
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(x)
            loss = masked_ce(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        loss_hist.append(float(loss.item()))
        if step % args.log_every == 0 or step == args.steps - 1:
            mem = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0
            print(f"  step {step:4d} | loss {np.mean(loss_hist[-args.log_every:]):.4f} | "
                  f"lr {lr_at(step):.2e} | {mem:.2f}GB | {(time.time()-t0):.0f}s")

    # ---- SFT 后评估 ----
    print("[eval] SFT 后评估…")
    model.eval()
    post = evaluate(model, tok, eval_samples, device)
    print(f"  后: 有K {post['with_k_acc']:.3f} 无K {post['without_k_acc']:.3f} "
          f"内化差 {post['internalization_gap']:.3f} | 一致 {post['consist_accept_rate']:.3f} "
          f"矛盾 {post['contradict_accept_rate']:.3f} 退联差 {post['consist_gap']:.3f}")

    # ---- 保存 + report ----
    out_dir = Path(args.out_dir)
    model.save_pretrained(out_dir)
    report = {
        "ckpt_base": args.ckpt, "steps": args.steps, "lr": args.lr,
        "micro_batch": args.micro_batch, "seq_len": args.seq_len,
        "n_train": len(train_samples), "n_eval": len(eval_samples),
        "loss_first50": float(np.mean(loss_hist[:50])),
        "loss_last50": float(np.mean(loss_hist[-50:])),
        "loss_hist": loss_hist,
        "pre": pre, "post": post,
        "delta_internalization_gap": post["internalization_gap"] - pre["internalization_gap"],
        "delta_consist_gap": post["consist_gap"] - pre["consist_gap"],
        "train_seconds": time.time() - t0,
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] checkpoint → {out_dir}；report → {rep}")
    print(f"[判据] 内化差 {pre['internalization_gap']:.3f}→{post['internalization_gap']:.3f} "
          f"(Δ{report['delta_internalization_gap']:+.3f})；退联差 "
          f"{pre['consist_gap']:.3f}→{post['consist_gap']:.3f} (Δ{report['delta_consist_gap']:+.3f})")


if __name__ == "__main__":
    main()
