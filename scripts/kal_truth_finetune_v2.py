"""KAL 真值锚微调 v2（校准 P1：0.769→≥0.8，锚集扩充 + 预测反馈循环，2026-07-31）。

对齐设计 §27.2 与 fb1 评审（runs/feedback/fb1.md，校准是全链最弱环）：
- **锚集扩充**（diverse_truth_data_v2）：known 侧加扩展真实事实库 + 多领域真实文本
  （math/code 英文片段，data/raw）+ val 解码段；unknown 侧加 near-miss 细粒度错误
  （真实事实改关键数字/日期/属性）+ 跨领域混搭 + 领域伪事实（假定理/假库）——
  强迫头学语义级真假而非"含虚构词=假"表面启发（评估是 kal_probe 模板 OOD，泛化靠
  多样性撑，记忆坑 4）。
- **预测反馈循环**（§27.2"元认知训练需预测+反馈循环"）：可验证 cloze 题池（程序
  算术/常识事实/较难事实 + 干扰项）→ **候选打分**（模型对 正确+干扰候选 logprob
  argmax = 模型自己的预测，批量短前向，比贪心生成快约一个数量级）→ 按预测对错打标
  （答对=known/答错=unknown）→ 与锚集混训一轮。样本 = **prompt 本身**（P(IK) 决策点
  是"看到问题时的表征"，非模型自己产出的对错答案文本——v2 首轮教训：prompt+补全
  会把"错答案文本=unknown"的表面特征混进来，伤 OOD）。
- **消融**：脚本内报告 仅锚集扩充（Phase A 后）vs +反馈循环（Phase B 后）双口径 AUROC，
  **择优保存**（B 臂 OOD 均值 < A 臂则回滚到 A 臂头权重——反馈循环是消融项非必赢项）。

红线（继承 kal_truth_finetune_gdn2 + 记忆坑 5 条）：
- detach 主干（no_grad 提取 hidden，梯度只进 kal_l1）；微调前后 val next-token loss
  逐位一致（脚本自验，--val_check）。
- kernel=None 加载坑：from_pretrained 前先 attach_kernel()（记忆坑 1）。
- 评估 @torch.no_grad()；Shards.get_batch 已右移对齐勿再 [:,:-1]。
- GPU 纪律：0.5B 双卡训练后台跑，本脚本 CUDA_VISIBLE_DEVICES=1，micro ≤8，
  OOM 自动折半重试（绝不能 OOM 影响训练进程）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_truth_finetune_v2.py \
      [--steps_a 600] [--feedback_rounds 1] [--steps_b 150] [--lr_b 3e-3] [--fb_frac 0.5]
产出：checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2/ + runs/kal_truth_v2/report.json。
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
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import kal_probe as kp  # noqa: E402
import diverse_truth_data as dt1  # noqa: E402
import diverse_truth_data_v2 as dt2  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k/final"
DEFAULT_OUT = "checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2"
DEFAULT_REPORT = "runs/kal_truth_v2/report.json"
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10]  # G2G2G2A×3，注意力层 3,7,11 除外
EVAL_SEEDS = [999, 1999, 2999]  # 双口径稳定性：3 个评估 seed（数据抽取不同）


class OomGuard:
    """hidden 提取 micro batch 的 OOM 保护：OOM 时清缓存折半（下限 2）。"""

    def __init__(self, bs: int):
        self.bs = bs

    def collect(self, model, id_list, layer, device, pooling):
        while True:
            try:
                feats, _ = kp.forward_collect(model, id_list, [layer], device, self.bs, pooling)
                return feats[layer]
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if self.bs <= 2:
                    raise
                self.bs = max(2, self.bs // 2)
                print(f"  ⚠️ OOM → 提取 micro batch 折半至 {self.bs} 重试")


@torch.no_grad()
def val_next_token_loss(model, shards_dir, device, n=16, T=48):
    """val next-token 平均 CE（与 tests/test_kal_gdn2_truth 同口径，rng=123 固定）。"""
    val = Shards(shards_dir, "val")
    rng = np.random.default_rng(123)
    x, y = val.get_batch(n, T, "cpu", rng)  # get_batch 已右移对齐，勿再 [:,:-1]
    logits, _ = model(x.to(device))
    lp = torch.log_softmax(logits.float(), dim=-1)
    ce = -lp.gather(-1, y.to(device).unsqueeze(-1)).squeeze(-1)
    return float(ce.mean().item())


@torch.no_grad()
def score_mc_predictions(model, tok, items, device, bs=8):
    """候选打分预测：每题 (prompt, answer, distractors)，候选 logprob argmax = 模型预测。

    全部 (prompt+候选) 序列批量短前向（bs≤8 红线），候选跨度取 mean logprob（长度归一），
    返回每题预测是否正确的 bool 列表。比贪心逐 token 生成快约一个数量级。
    """
    seqs, spans, owners = [], [], []
    for qi, (prompt, ans, ds) in enumerate(items):
        p_ids = tok.encode(prompt)
        for cand in [ans] + list(ds):  # 候选 0 = 正确答案
            c_ids = tok.encode(" " + cand)
            seqs.append(p_ids + c_ids)
            spans.append(len(p_ids))  # 候选起点（含前导空格 token）
            owners.append(qi)
    scores = np.empty(len(seqs), dtype=np.float64)
    for s in range(0, len(seqs), bs):
        chunk = seqs[s : s + bs]
        T = max(len(x) for x in chunk)
        ids = np.full((len(chunk), T), tok.eot_id, dtype=np.int64)
        for b, x in enumerate(chunk):
            ids[b, : len(x)] = x
        ids_t = torch.from_numpy(ids).to(device)
        logits, _ = model(ids_t)
        lp = torch.log_softmax(logits.float(), dim=-1)
        for b, x in enumerate(chunk):
            st = spans[s + b]
            tgt = ids_t[b, st:len(x)]
            scores[s + b] = lp[b, st - 1: len(x) - 1].gather(
                -1, tgt.unsqueeze(-1)).squeeze(-1).mean().item()
        del ids_t, logits, lp
    preds_ok: list[bool] = []
    for qi in range(len(items)):
        idx = [j for j, o in enumerate(owners) if o == qi]
        preds_ok.append(int(np.argmax(scores[idx])) == 0)
    return preds_ok


def collect_feedback(model, tok, rng, device, n_pool):
    """预测反馈循环采样：cloze 题 → 模型候选打分预测 → 按对错打标（对=known 0 / 错=unknown 2）。

    样本 = **prompt 本身**（P(IK) 决策点是问题表征；不把模型产出的答案文本混入训练样本）。
    两类均衡下采样。返回 (texts, labels, model_acc)。
    """
    items = dt2.build_cloze_pool_mc(rng, n_pool)
    preds_ok = score_mc_predictions(model, tok, items, device, bs=8)
    correct = [p for (p, _, _), ok in zip(items, preds_ok) if ok]
    wrong = [p for (p, _, _), ok in zip(items, preds_ok) if not ok]
    n_take = min(len(correct), len(wrong))
    acc = len(correct) / max(1, len(items))
    if n_take == 0:
        print(f"  ⚠️ 反馈池单边为空（correct={len(correct)} wrong={len(wrong)}），本轮反馈跳过")
        return [], np.zeros(0, dtype=np.int64), acc
    ci = rng.choice(len(correct), size=n_take, replace=False)
    wi = rng.choice(len(wrong), size=n_take, replace=False)
    texts = [correct[int(i)] for i in ci] + [wrong[int(i)] for i in wi]
    labels = np.array([0] * n_take + [2] * n_take, dtype=np.int64)
    print(f"  [feedback] cloze {len(items)} 题：模型答对 {len(correct)}（acc {acc:.2f}）"
          f"答错 {len(wrong)} → 均衡取 {n_take}+{n_take}")
    return texts, labels, acc


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL 真值锚微调 v2（锚集扩充+预测反馈循环）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--shards", default=kp.DEFAULT_SHARDS)
    ap.add_argument("--raw_dir", default="data/raw")
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--layer", type=int, default=10, help="读点层（kaltruth v1 已扫 ℓ10 最优）")
    ap.add_argument("--scan", action="store_true", help="可选：扫描 ℓ8/ℓ9/ℓ10 init AUROC 再选层")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_each", type=int, default=64, help="每步 known/unknown 各 n_each 条")
    ap.add_argument("--steps_a", type=int, default=600, help="Phase A：仅锚集扩充微调步数")
    ap.add_argument("--feedback_rounds", type=int, default=1)
    ap.add_argument("--feedback_n", type=int, default=240, help="每轮 cloze 题池大小")
    ap.add_argument("--steps_b", type=int, default=150, help="Phase B：反馈混训步数/轮")
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--lr_b", type=float, default=3e-3, help="Phase B 反馈混训 lr（保守，防冲垮 A 臂）")
    ap.add_argument("--fb_frac", type=float, default=0.5, help="反馈样本混入比例（相对 n_each）")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--extract_bs", type=int, default=8, help="hidden 提取 micro batch（红线 ≤8）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    print(f"[kaltruth-v2] ckpt={args.ckpt} layer=ℓ{args.layer} steps_a={args.steps_a} "
          f"feedback_rounds={args.feedback_rounds} steps_b={args.steps_b}")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    if model.kernel is None:
        # 记忆坑 1：10k kernel_enabled=False 训练 → kernel=None，先 attach 再微调
        print("[kaltruth-v2] model.kernel=None → attach_kernel()（内核头随机初始化，需真值锚微调）")
        model.attach_kernel()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.kernel.kal_l1.parameters():
        p.requires_grad_(True)
    head = model.kernel.kal_l1
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    guard = OomGuard(args.extract_bs)
    layer = args.layer
    if layer not in GDN_LAYERS:
        print(f"❌ ℓ{layer} 非 GDN 层（{GDN_LAYERS}）")
        sys.exit(1)

    # 主干未污染自验：微调前 val loss
    loss_before = val_next_token_loss(model, args.shards, args.device)
    print(f"[kaltruth-v2] 微调前 val next-token loss {loss_before:.5f}")

    # extra_known：val shard 解码段（分布内文本作 known 多样性）
    val_shards = Shards(args.shards, "val")
    x_extra, _ = val_shards.get_batch(256, args.seq_len, "cpu", np.random.default_rng(args.seed + 5))
    extra_known = [tok.decode(row) for row in x_extra.numpy().tolist()]

    @torch.no_grad()
    def eval_auroc(n_eval, eval_seed, read_layer=None):
        """真值 AUROC：known(val) vs fake（kal_probe 模板，OOD），score=logit[0]−logit[2]。"""
        rl = layer if read_layer is None else read_layer
        ids, labels_np, subset = kp.build_l1_dataset(
            tok, args.shards, np.random.default_rng(eval_seed), n_eval, n_eval // 2, 0, args.seq_len)
        hidden = guard.collect(model, ids, rl, args.device, args.pooling)
        h = torch.from_numpy(hidden).to(args.device)
        logits = head(h).float()
        scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
        known_binary = (labels_np == 1).astype(np.int64)
        fake_mask = (subset == "known") | (subset == "fake")
        return (kp.auroc(scores, known_binary),
                kp.auroc(scores[fake_mask], known_binary[fake_mask]))

    def eval_multi(n_eval):
        """双口径稳定性：3 评估 seed 的 overall/fake AUROC 列表 + 均值±std。"""
        ovs, fks = [], []
        for es in EVAL_SEEDS:
            ov, fk = eval_auroc(n_eval, es)
            ovs.append(ov)
            fks.append(fk)
        return {"seeds": EVAL_SEEDS, "overall": [round(v, 4) for v in ovs],
                "fake": [round(v, 4) for v in fks],
                "overall_mean": float(np.mean(ovs)), "overall_std": float(np.std(ovs)),
                "fake_mean": float(np.mean(fks)), "fake_std": float(np.std(fks))}

    # 可选读点扫描（先确认 1+2 收益前的读点微调能力保留）
    if args.scan:
        for cand in (8, 9, 10):
            ov, fk = eval_auroc(200, EVAL_SEEDS[0], read_layer=cand)
            print(f"  [scan] ℓ{cand}: init overall AUROC {ov:.3f} | fake {fk:.3f}")

    # ------------------------------------------------------------------
    # 训练循环（Phase A 锚集 / Phase B 反馈混训共用）
    # ------------------------------------------------------------------
    def lr_at(step, total, base_lr):
        if step < args.warmup:
            return base_lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, total - args.warmup)
        return base_lr * (1.0 - 0.9 * prog)

    def anchor_batch():
        texts, labels_np = dt2.build_diverse_truth_dataset_v2(
            rng, args.n_each, args.n_each, extra_known=extra_known, raw_dir=args.raw_dir)
        ids = kp.encode_fixed(tok, texts, args.seq_len)
        hidden = guard.collect(model, ids, layer, args.device, args.pooling)
        return (torch.from_numpy(hidden).to(args.device),
                torch.from_numpy(labels_np).to(args.device))

    def train_steps(total, tag, fb_ids=None, fb_labels=None, base_lr=None):
        base_lr = args.lr if base_lr is None else base_lr
        head.train()
        for step in range(total):
            lr = lr_at(step, total, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            hidden, labels = anchor_batch()
            if fb_ids is not None and len(fb_ids) > 0:
                # 反馈样本按 fb_frac 混入（每步随机抽取；样本 = prompt 本身）
                k = min(max(1, int(args.n_each * args.fb_frac)), len(fb_ids))
                sel = rng.choice(len(fb_ids), size=k, replace=False)
                fb_chunk = [fb_ids[int(i)] for i in sel]
                fb_hidden = guard.collect(model, fb_chunk, layer, args.device, args.pooling)
                hidden = torch.cat([hidden, torch.from_numpy(fb_hidden).to(args.device)], dim=0)
                labels = torch.cat([labels, torch.from_numpy(fb_labels[sel]).to(args.device)], dim=0)
            logits = head(hidden)
            ce = F.cross_entropy(logits.float(), labels)
            ce.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            if (step + 1) % 100 == 0 or step == 0:
                print(f"  [{tag}] step {step+1:4d}/{total} ce={ce.item():.4f} lr={lr:.2e}")

    t0 = time.time()
    # ---- Phase A：仅锚集扩充（消融臂 A）----
    print(f"[kaltruth-v2] Phase A：锚集扩充微调 {args.steps_a} 步（v2 锚集）")
    train_steps(args.steps_a, "A")
    eval_a = {"script_n200": eval_multi(200), "test_n400": eval_multi(400)}
    print(f"[kaltruth-v2] Phase A 后（仅锚集扩充）："
          f"脚本口径(n200) overall {eval_a['script_n200']['overall_mean']:.3f}±{eval_a['script_n200']['overall_std']:.3f} | "
          f"测试口径(n400) overall {eval_a['test_n400']['overall_mean']:.3f}±{eval_a['test_n400']['overall_std']:.3f}")
    # A 臂头权重快照（择优保存：B 臂伤 OOD 则回滚）
    head_a = {k: v.detach().clone() for k, v in head.state_dict().items()}

    # ---- Phase B：预测反馈循环（消融臂 B = A + 反馈，保守 lr）----
    fb_stats = []
    for rd in range(args.feedback_rounds):
        print(f"[kaltruth-v2] Phase B 轮 {rd+1}/{args.feedback_rounds}：候选打分预测→对错打标→混训"
              f"（lr {args.lr_b}，混入 {args.fb_frac:.0%}）")
        fb_texts, fb_labels_np, fb_acc = collect_feedback(
            model, tok, np.random.default_rng(args.seed + 100 + rd), args.device, args.feedback_n)
        fb_stats.append({"round": rd + 1, "pool": args.feedback_n, "model_acc": fb_acc,
                         "n_balanced_each": int((fb_labels_np == 0).sum())})
        if len(fb_texts) == 0:
            continue
        fb_ids = kp.encode_fixed(tok, fb_texts, args.seq_len)
        train_steps(args.steps_b, f"B{rd+1}", fb_ids=fb_ids, fb_labels=fb_labels_np,
                    base_lr=args.lr_b)
    dur = time.time() - t0

    # ---- 最终双口径评估（3 seed 均值±std）----
    eval_b = {"script_n200": eval_multi(200), "test_n400": eval_multi(400)}
    for name, r in eval_b.items():
        print(f"[kaltruth-v2] Phase B 后 {name}: overall {r['overall_mean']:.3f}±{r['overall_std']:.3f} "
              f"{r['overall']} | fake {r['fake_mean']:.3f}±{r['fake_std']:.3f}")

    # ---- 择优保存：双口径均值之和 A ≥ B 则回滚到 A 臂（反馈循环是消融项非必赢项）----
    score_a = eval_a["script_n200"]["overall_mean"] + eval_a["test_n400"]["overall_mean"]
    score_b = eval_b["script_n200"]["overall_mean"] + eval_b["test_n400"]["overall_mean"]
    selected_arm = "A+B" if score_b > score_a else "A"
    if selected_arm == "A":
        head.load_state_dict(head_a)
        print(f"[kaltruth-v2] 择优：A 臂（{score_a:.3f}）≥ B 臂（{score_b:.3f}）→ 回滚保存 A 臂头权重")
    else:
        print(f"[kaltruth-v2] 择优：B 臂（{score_b:.3f}）> A 臂（{score_a:.3f}）→ 保存 A+反馈 头权重")
    final = eval_a if selected_arm == "A" else eval_b  # 判定/报告以实际保存臂为准

    # ---- certainty 方向语义复验（known 高 / fake 低）----
    @torch.no_grad()
    def prob_known(texts):
        ids = kp.encode_fixed(tok, texts, args.seq_len)
        hidden = guard.collect(model, ids, layer, args.device, args.pooling)
        h = torch.from_numpy(hidden).to(args.device)
        return torch.softmax(head(h).float(), dim=-1)[:, 0].cpu().numpy()

    probe_rng = np.random.default_rng(args.seed + 777)
    pk_known = prob_known(dt1.build_real_statements(probe_rng, 8))
    pk_fake = prob_known(kp.build_fake_fact_texts(probe_rng, 8))
    certainty_ok = bool(pk_known.mean() > 0.5 and pk_fake.mean() < 0.5)
    print(f"[kaltruth-v2] certainty 抽查：known P(known) {pk_known.mean():.3f}（应高）| "
          f"fake P(known) {pk_fake.mean():.3f}（应低）→ {'✅' if certainty_ok else '⚠️'}")

    # ---- 主干未污染自验：微调后 val loss 逐位一致 ----
    loss_after = val_next_token_loss(model, args.shards, args.device)
    loss_diff = abs(loss_after - loss_before)
    print(f"[kaltruth-v2] val loss 微调前 {loss_before:.5f} | 微调后 {loss_after:.5f} | "
          f"漂移 {loss_diff:.2e} → {'✅ 逐位一致' if loss_diff < 1e-4 else '⚠️ 有漂移'}")

    # ---- 判定：双口径 3 seed 均值都 ≥0.8 且最低 seed ≥0.78（稳定非卡边）----
    ok = (final["script_n200"]["overall_mean"] >= 0.8 and final["test_n400"]["overall_mean"] >= 0.8
          and min(final["script_n200"]["overall"]) >= 0.78 and min(final["test_n400"]["overall"]) >= 0.78)
    verdict = ("✅ 双口径达标（3 seed 均值≥0.8，最低 seed≥0.78）" if ok
               else "⚠️ 未稳定达标（需更多步/数据或读点调整）")
    print(f"判定：{verdict}")

    out_dir = Path(args.out_dir)
    model.save_pretrained(out_dir)
    report = {
        "ckpt": args.ckpt, "layer": layer, "seed": args.seed,
        "steps_a": args.steps_a, "feedback_rounds": args.feedback_rounds, "steps_b": args.steps_b,
        "n_each": args.n_each, "seq_len": args.seq_len, "lr": args.lr, "lr_b": args.lr_b,
        "fb_frac": args.fb_frac, "selected_arm": selected_arm,
        "method": "锚集扩充（diverse_truth_data_v2：near-miss/跨域混搭/领域伪事实/多领域真实文本）"
                  "+预测反馈循环（cloze 多候选：模型 logprob argmax 预测→对错打标→prompt 样本混入，"
                  "保守 lr_b）；双臂择优保存；detach 主干只训 kal_l1",
        "anchor_composition": {
            "known": "v1 真实句/contrast 真实 + 扩展真实事实库 60 + 否定真实 10 + "
                     "math/code 领域真实文本片段 + val 解码段 256",
            "unknown": "v1 短句/疑问/contrast 虚构 + near-miss 细粒度错误 + 跨领域混搭 + "
                       "数学假定理/代码假库（程序化虚构名）",
        },
        "feedback": fb_stats,
        "eval_phase_a_anchor_only": eval_a,
        "eval_phase_b_with_feedback": eval_b,
        "eval_final": final,
        "certainty_probe": {"known_p_known_mean": float(pk_known.mean()),
                            "fake_p_known_mean": float(pk_fake.mean()),
                            "direction_ok": certainty_ok},
        "val_loss": {"before": loss_before, "after": loss_after, "diff": loss_diff},
        "verdict": verdict, "duration_s": round(dur, 1), "out_dir": str(out_dir),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kaltruth-v2] checkpoint → {out_dir}；report → {rp}；用时 {dur:.0f}s")


if __name__ == "__main__":
    main()
