"""内生 KAL 探针评估（KAL 完整训练后的 T1 验证）。

与 scripts/kal_probe.py 的关键区别：kal_probe 是**事后线性探针**（在 frozen hidden 上
手训外部逻辑回归），回答"hidden state 里有没有可线性读出的'知/不知'信号"（M2，AUROC 0.945）。
本脚本评估**内生 KAL 头**（kal_aux_weight>0 训练时随优化的 kernel.kal_l1 权重），
回答"内生头本身是否已学会 P(IK)、直接产出 AUROC≥0.8"——这才是 KAL 内生训练的目标
（设计 §8.3 / 部件详细计划 Part B：内生头随 checkpoint 存取，运行时零额外训练）。

复用 kal_probe 的 L1 数据集构建（known/fake/shuffled）与 forward_collect 特征提取，
但**不训练任何探针**：直接用内生 kal_l1 头对 pooled hidden 打分，算 AUROC。

内生头打分口径（对齐 train.py kal_pik_aux_loss 的二分类退化）：
  三态头 [d,3] = 知道/不确定/空白，训练时只用 [0,2] 两类（"不确定"无标签），
  故评估用 score = logit[0] - logit[2]（知道 vs 空白的对数几率，越大越"知道"）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/eval_intrinsic_kal.py \
      [--ckpt checkpoints/pilot_0p1b_kal/final] [--layers 8]
输出：控制台 + runs/kal_intrinsic/report.json。
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
sys.path.insert(0, str(ROOT / "scripts"))  # 复用 kal_probe 的数据集/特征提取
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import kal_probe as kp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="内生 KAL 探针评估（T1，KAL 完整训练后）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_kal/final")
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--shards", default=kp.DEFAULT_SHARDS)
    ap.add_argument("--out", default="runs/kal_intrinsic/report.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_known", type=int, default=400)
    ap.add_argument("--n_fake", type=int, default=200)
    ap.add_argument("--n_shuffled", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    print(f"[intrinsic] checkpoint={args.ckpt} layers={args.layers} pooling={args.pooling}")

    # 复用 kal_probe 的 L1 数据集（known=1 / fake+shuffled=0）
    l1_ids, l1_labels, l1_subset = kp.build_l1_dataset(
        tok, args.shards, rng, args.n_known, args.n_fake, args.n_shuffled, args.seq_len)
    print(f"[l1] 数据集: known={args.n_known} fake={args.n_fake} shuffled={args.n_shuffled}")

    # 加载 KAL 训练 checkpoint（须含 kernel.kal_l1 已训练权重）
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    if model.kernel is None:
        print("❌ model.kernel=None——该 checkpoint 无内生 KAL 头（非 KAL 训练 run）。"
              "请用 --ckpt checkpoints/pilot_0p1b_kal/final")
        sys.exit(1)
    print("[intrinsic] 内核已挂载，kal_l1 头已加载（内生权重，非随机初始化）")

    # 复用特征提取（pooled hidden per layer）
    # 复用特征提取（pooled hidden per layer），模型保留供内生头打分
    feats, mean_logprob = kp.forward_collect(model, l1_ids, args.layers, args.device,
                                             args.batch_size, args.pooling)

    # 内生头打分 + AUROC（不训练任何探针）
    # FLARE 基线：mean logprob 越高=越"知道"
    flare_overall = kp.auroc(mean_logprob, l1_labels)
    fake_mask = (l1_subset == "known") | (l1_subset == "fake")
    flare_fake = kp.auroc(mean_logprob[fake_mask], l1_labels[fake_mask])

    results = {}
    for layer in args.layers:
        h = torch.from_numpy(feats[layer]).to(args.device)
        with torch.no_grad():
            logits = model.kernel.kal_l1(h).float()  # [N,3]
        # 知道(0) vs 空白(2) 的对数几率（与 train.py 二分类退化口径一致）
        scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
        results[layer] = {
            "auroc_overall": kp.auroc(scores, l1_labels),
            "auroc_fake": kp.auroc(scores[fake_mask], l1_labels[fake_mask]),
        }
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ---- 判定（如实）----
    print("\n===== 内生 KAL 探针（不训练，直接评估内核权重）=====")
    print(f"FLARE 基线：overall {flare_overall:.3f} | fake {flare_fake:.3f}")
    verdicts = {}
    for layer, r in results.items():
        ok_overall = r["auroc_overall"] >= 0.8
        ok_fake = r["auroc_fake"] >= 0.8
        beat_flare = r["auroc_fake"] > flare_fake
        verdicts[layer] = (
            f"ℓ{layer}: 内生 overall AUROC {r['auroc_overall']:.3f}（{'✅≥0.8' if ok_overall else '⚠️<0.8'}）"
            f" | fake {r['auroc_fake']:.3f}（{'✅≥0.8' if ok_fake else '⚠️<0.8'}，"
            f"{'优于' if beat_flare else '未优于'} FLARE {flare_fake:.3f}）"
        )
        print(verdicts[layer])

    report = {
        "ckpt": args.ckpt, "layers": args.layers, "pooling": args.pooling,
        "seq_len": args.seq_len, "seed": args.seed, "eval": "intrinsic_kal_l1_no_training",
        "score_spec": "logit[know=0] - logit[blank=2]（对齐 train.py kal_pik_aux_loss 二分类退化）",
        "flare_baseline": {"overall": flare_overall, "fake": flare_fake},
        "intrinsic": results,
        "verdicts": verdicts,
        "duration_s": round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[intrinsic] report → {out}")


if __name__ == "__main__":
    main()
