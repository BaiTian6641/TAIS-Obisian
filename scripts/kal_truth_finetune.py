"""KAL 真值锚微调（T1 迭代，修正"在线自标注 P(IK) 目标错位"）。

背景（诚实负结果，runs/kal_intrinsic/report.json）：kal_aux_weight>0 的在线自标注
（伪标签=主干 next-token 预测正确性）训练的内生 KAL 头 ℓ8 AUROC 仅 0.433——
预测正确性测的是**文本局部流畅度**而非**事实真假**：known(FineWeb 真实文本)语言建模
高熵常预测错→错标"未知"；fake/shuffled 局部 n-gram 仍可预测→错标"知道"。
正合 2606.02628：幻觉检测信号在线性 hidden state，但**训练目标须锚真值(fake vs real)**。

本脚本（方案 1，真值锚）：
- 加载已训 checkpoint（pilot_0p1b_kal/final，主干+内核已就位），**冻结主干与内核其余部件**；
- 仅用真值标签微调 kernel.kal_l1 头：fake（kal_probe 伪事实模板）=unknown(空白类2)、
  分布内文本（val shard）=known(知道类0)；三态头沿用 [0,2] 二分类退化（"不确定"无标签）；
- 红线：detach 主干 hidden（监测/执行分置 + 探针冻结语义），梯度只进 kal_l1；
- 轻量短校准（默认 500 步），可选 --init_from_probe 用 M2 事后探针方向热启动。

与 kal_probe.py 的关键区别：kal_probe 是**事后诊断**（frozen hidden 手训外部探针，权重不随
checkpoint）；本脚本把**真值语义写回内生头权重**（随 checkpoint 存取，运行时零额外训练）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_truth_finetune.py \
      [--ckpt checkpoints/pilot_0p1b_kal/final] [--steps 500] [--layers 8]
产出：{ckpt 目录}/final_kal_truth/（含微调后的内核权重）+ runs/kal_truth/report.json。
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


@torch.no_grad()
def collect_hidden(model, id_list, layer, device, batch_size, pooling):
    """复用 kal_probe.forward_collect 取 pooled hidden [N,d]（单流=内容流）。"""
    feats, _ = kp.forward_collect(model, id_list, [layer], device, batch_size, pooling)
    return feats[layer]


def make_batch(tok, val_shards, rng, n_each, T, layer, model, device, batch_size, pooling,
               data_source: str = "template"):
    """构造一批真值样本：n_each 条 known + n_each 条 unknown，编码→hidden→标签。

    data_source：
    - "template"：known=val 分布内文本，unknown=kal_probe 8 类长模板伪事实（v1，模板同分布）；
    - "diverse"：known/unknown 经 diverse_truth_data 多样化生成（多句式短句/疑问/否定/
      contrast-pair + 程序化虚构词 + 真实事实句），解决 v1 的 OOD 泛化短板（规范 §7.2）。

    返回 (hidden [2*n_each, d]（no_grad 提取）, labels [2*n_each]（known=0/unknown=2）)。
    """
    if data_source == "diverse":
        import diverse_truth_data as dt
        texts, labels_np = dt.build_diverse_truth_dataset(rng, n_each, n_each)
        ids = kp.encode_fixed(tok, texts, T)
        labels = labels_np
    else:  # template（v1）
        x, _ = val_shards.get_batch(n_each, T, "cpu", rng)
        known_ids = x.numpy().tolist()
        fake_ids = kp.encode_fixed(tok, kp.build_fake_fact_texts(rng, n_each), T)
        ids = known_ids + fake_ids
        labels = np.array([0] * n_each + [2] * n_each, dtype=np.int64)  # 0=知道, 2=空白
    hidden = collect_hidden(model, ids, layer, device, batch_size, pooling)
    return torch.from_numpy(hidden).to(device), torch.from_numpy(labels).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL 真值锚微调（T1 迭代）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_kal/final")
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--shards", default=kp.DEFAULT_SHARDS)
    ap.add_argument("--out_dir", default=None, help="默认 {ckpt}_kaltruth")
    ap.add_argument("--report", default="runs/kal_truth/report.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_each", type=int, default=64, help="每批 known/fake 各 n_each 条")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_source", choices=["template", "diverse"], default="template",
                    help="template=v1 模板伪事实（同分布）；diverse=v2 多样化真值（OOD 鲁棒，规范 §7.2）")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    layer = args.layers[0]  # 单读点（M2 发现 ℓ8 最强；多读点留待多头融合）
    print(f"[kaltruth] ckpt={args.ckpt} layer=ℓ{layer} steps={args.steps} lr={args.lr}")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    if model.kernel is None:
        print("❌ model.kernel=None——需 KAL 训练 checkpoint（含内核）")
        sys.exit(1)
    # 冻结全部，只解冻 kal_l1（红线：梯度只进 KAL 头，不污染主干/内核其余部件）
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.kernel.kal_l1.parameters():
        p.requires_grad_(True)
    head = model.kernel.kal_l1
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    val_shards = Shards(args.shards, "val")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_auroc(n_eval=200):
        """真值 AUROC：known(val) vs fake，score = logit[0]-logit[2]。"""
        ids, labels_np, subset = kp.build_l1_dataset(
            tok, args.shards, np.random.default_rng(args.seed + 999), n_eval, n_eval // 2, 0, args.seq_len)
        hidden = collect_hidden(model, ids, layer, args.device, args.batch_size, args.pooling)
        h = torch.from_numpy(hidden).to(args.device)
        logits = head(h).float()
        scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
        known_binary = (labels_np == 1).astype(np.int64)  # build_l1_dataset: known=1
        fake_mask = (subset == "known") | (subset == "fake")
        return (kp.auroc(scores, known_binary),
                kp.auroc(scores[fake_mask], known_binary[fake_mask]))

    init_overall, init_fake = eval_auroc()
    print(f"[kaltruth] 微调前：overall AUROC {init_overall:.3f} | fake {init_fake:.3f}")

    t0 = time.time()
    head.train()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        hidden, labels = make_batch(tok, val_shards, rng, args.n_each, args.seq_len,
                                    layer, model, args.device, args.batch_size, args.pooling,
                                    data_source=args.data_source)
        logits = head(hidden)  # [B,3]（hidden 已 detach 提取，梯度只进 head）
        ce = F.cross_entropy(logits.float(), labels)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps} ce={ce.item():.4f} lr={lr:.2e}")

    final_overall, final_fake = eval_auroc()
    dur = time.time() - t0
    print(f"[kaltruth] 微调后：overall AUROC {init_overall:.3f}→{final_overall:.3f} | "
          f"fake {init_fake:.3f}→{final_fake:.3f} 用时 {dur:.0f}s")
    verdict = ("✅ 真值锚达标（overall≥0.8）" if final_overall >= 0.8
               else "⚠️ 未达 0.8（真值锚有效但需更多步/数据，或 ℓ8 读点/单头限制）")
    print(f"判定：{verdict}")

    # 保存微调后的 checkpoint（含内核权重）
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{args.ckpt}_kaltruth")
    model.save_pretrained(out_dir)
    report = {
        "ckpt": args.ckpt, "layer": layer, "steps": args.steps, "lr": args.lr,
        "n_each": args.n_each, "seq_len": args.seq_len, "seed": args.seed,
        "method": "真值锚（fake=unknown/real=known，detach 主干，梯度只进 kal_l1）",
        "init": {"overall": init_overall, "fake": init_fake},
        "final": {"overall": final_overall, "fake": final_fake},
        "verdict": verdict, "duration_s": round(dur, 1), "out_dir": str(out_dir),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kaltruth] checkpoint → {out_dir}；report → {rp}")


if __name__ == "__main__":
    main()
