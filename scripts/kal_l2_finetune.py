"""KAL L2 情感头真值微调（VA 正交回归，T1 观测 1.5B 可复现性）。

设计依据（article_ref/07 §3，逐条已核实）：
- **VA 线性子空间**（Anthropic 2604.07729 / Sun 2604.03147）：valence/arousal 在残差流
  线性可解码、circumplex 组织、v/a 近似正交（r≈-0.02）。形式 = 残差流 → 学两**正交**轴
  → (v,a) 回归；`W[d,2]` 两列加**正交化约束**（非两维独立 BCE——那会让 v/a 轴纠缠）。
- **功能不对称（McGaugh 04 / Mather ABC 2011）**：arousal 进写门（巩固增益主驱动）、
  valence 进极性。本脚本产出 arousal 显著性，供 CA1 巩固门加权（写显著性门控落地）。

与 kal_probe.train_l2_probe 的关键区别：
- train_l2_probe 是**事后诊断**（两维独立 BCE，frozen hidden，权重不随 checkpoint，无正交约束）；
- 本脚本微调**内生 kernel.kal_l2**（随 checkpoint 存取），损失 = VA 两维 MSE 回归 +
  λ·正交惩罚（W 两列余弦相似度平方），红线 detach 主干（梯度只进 kal_l2，监测/执行分置）。

数据：kal_probe.build_l2_dataset（dair-ai/emotion 6 类 → VA 粗标签，或词表 fallback）。
VA 标签映射为连续坐标：valence {0,1}→{-1,+1}、arousal {0,1}→{-1,+1}（circumplex 四象限）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_l2_finetune.py \
      [--ckpt checkpoints/pilot_0p1b_kal/final] [--steps 400] [--layers 8]
产出：{ckpt}_kall2/（含微调 kal_l2）+ runs/kal_l2/report.json。
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

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import kal_probe as kp  # noqa: E402


@torch.no_grad()
def collect_hidden(model, id_list, layer, device, batch_size, pooling):
    feats, _ = kp.forward_collect(model, id_list, [layer], device, batch_size, pooling)
    return feats[layer]


def ortho_penalty(head: torch.nn.Module) -> torch.Tensor:
    """W[d,2] 两列正交惩罚：余弦相似度平方（v 轴 ⊥ a 轴，circumplex 组织）。"""
    W = head.proj.weight  # [2, d]（Linear(d,2)）
    v_axis = F.normalize(W[0], dim=0)
    a_axis = F.normalize(W[1], dim=0)
    cos = torch.dot(v_axis, a_axis)
    return cos * cos


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL L2 情感头 VA 正交回归微调（T1 观测）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_kal/final")
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--shards", default=kp.DEFAULT_SHARDS)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--report", default="runs/kal_l2/report.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--l2_per_class", type=int, default=150)
    ap.add_argument("--l2_max_len", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ortho_lambda", type=float, default=0.1, help="正交惩罚权重")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    layer = args.layers[0]
    print(f"[kall2] ckpt={args.ckpt} layer=ℓ{layer} steps={args.steps} ortho_λ={args.ortho_lambda}")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    if model.kernel is None:
        print("❌ model.kernel=None——需 KAL 训练 checkpoint（含内核）")
        sys.exit(1)
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.kernel.kal_l2.parameters():
        p.requires_grad_(True)
    head = model.kernel.kal_l2
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    # 一次性构建 L2 数据集（VA 粗标签 → 连续坐标 ±1）
    ids, yv, ya, source = kp.build_l2_dataset(tok, rng, args.l2_per_class, args.l2_max_len)
    print(f"[kall2] 数据集 source={source} n={len(ids)} "
          f"v+={int(yv.sum())}/v-={int((1-yv).sum())} a+={int(ya.sum())}/a-={int((1-ya).sum())}")
    # 提取 hidden（一次前向，detach）
    hidden_np = collect_hidden(model, ids, layer, args.device, args.batch_size, args.pooling)
    H = torch.from_numpy(hidden_np).to(args.device)  # [N,d]，已 detach
    # VA 连续坐标：{0,1}→{-1,+1}
    V = torch.from_numpy(yv.astype(np.float32) * 2 - 1).to(args.device)
    A = torch.from_numpy(ya.astype(np.float32) * 2 - 1).to(args.device)
    N = H.shape[0]

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_va():
        """VA 分类 AUROC（sign(v) 作 score）+ 正交度（两列余弦）。"""
        head.eval()
        out = head(H).float()  # [N,2]
        v_score = out[:, 0].cpu().numpy()
        a_score = out[:, 1].cpu().numpy()
        v_auroc = kp.auroc(v_score, yv)
        a_auroc = kp.auroc(a_score, ya)
        W = head.proj.weight
        cos = float(F.cosine_similarity(W[0], W[1], dim=0))
        head.train()
        return v_auroc, a_auroc, cos

    init_v, init_a, init_cos = eval_va()
    print(f"[kall2] 微调前：valence AUROC {init_v:.3f} arousal AUROC {init_a:.3f} cos={init_cos:.3f}")

    t0 = time.time()
    head.train()
    idx_all = np.arange(N)
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        rng.shuffle(idx_all)
        bidx = torch.from_numpy(idx_all[: args.batch_size]).to(args.device)
        out = head(H[bidx])  # [B,2]
        mse = F.mse_loss(out[:, 0], V[bidx]) + F.mse_loss(out[:, 1], A[bidx])
        pen = ortho_penalty(head)
        loss = mse + args.ortho_lambda * pen
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps} mse={mse.item():.4f} ortho={pen.item():.4f} lr={lr:.2e}")

    final_v, final_a, final_cos = eval_va()
    dur = time.time() - t0
    print(f"[kall2] 微调后：valence AUROC {init_v:.3f}→{final_v:.3f} | "
          f"arousal {init_a:.3f}→{final_a:.3f} | cos {init_cos:.3f}→{final_cos:.3f} 用时 {dur:.0f}s")
    # T1 观测判据：VA 两维 AUROC 均 >0.6（超越 chance 0.5，0.1B 弱信号如实；1.5B 待验）
    verdict = ("✅ VA 两维均可解码（>0.6，T1 观测达标）" if (final_v > 0.6 and final_a > 0.6)
               else "⚠️ VA 信号弱（0.1B 情感子空间未强线性化，1.5B 待验——诚实负结果如实记）")
    print(f"判定：{verdict}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{args.ckpt}_kall2")
    model.save_pretrained(out_dir)
    report = {
        "ckpt": args.ckpt, "layer": layer, "steps": args.steps, "lr": args.lr,
        "ortho_lambda": args.ortho_lambda, "data_source": source, "n": len(ids),
        "method": "VA 两维 MSE 回归 + λ·正交惩罚（W 两列余弦²），detach 主干红线",
        "init": {"valence_auroc": init_v, "arousal_auroc": init_a, "cos": init_cos},
        "final": {"valence_auroc": final_v, "arousal_auroc": final_a, "cos": final_cos},
        "verdict": verdict, "duration_s": round(dur, 1), "out_dir": str(out_dir),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kall2] checkpoint → {out_dir}；report → {rp}")


if __name__ == "__main__":
    main()
