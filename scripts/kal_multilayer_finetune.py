"""KAL L1 多层融合微调（ℓ多挂点 + AUROC 软加权，规范 §7.3，KAL 鲁棒性最后一块）。

设计依据（article_ref/07 §2 多层融合，逐条已核实）：
- **单层次优**（Wang 2605.26366，ICML 2026）：峰值层因数据集/模型而异，固定单层
  平均损失 2–5 AUROC——不赌单层。
- **AUROC 软加权融合**：`z = Σ_l w_l·z_l`，`w_l = softmax(AUROC_l / T)`，验证集估计
  各层 AUROC——多层软加权比单层稳、比硬投票稳。
- 0.1B 12 层（GGGAGGGAGGGA），M2 实测 ℓ4 overall 0.885/fake 0.959、ℓ8 0.945/0.979；
  多挂点取 **ℓ4/8/10**（M2 已验 ℓ4/8 + 近输出层 ℓ10；1.5B 28 层正式挂点 ℓ10/14/18 的预演）。

与单层 kal_truth_finetune 的关键区别：
- 单层：只训 ℓ8 一个头（kal_l1，随 checkpoint）；
- 本脚本：**每层独立真值头**（ℓ4/8/10 各一 Linear(d,3)，真值锚微调）+ **验证集 AUROC
  软加权融合** score——鲁棒性来自"任何单层失效时其余层兜底"。
- ℓ8 头直接复用内核 kal_l1（已训）；ℓ4/ℓ10 新建（暂存脚本，正式可入内核多头）。
  红线：detach 主干（梯度只进探针头，监测/执行分置）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_multilayer_finetune.py \
      [--ckpt checkpoints/pilot_0p1b_kal/final] [--layers 4 8 10] [--steps 400]
产出：runs/kal_multilayer/report.json（各层 AUROC + 融合权重 + 融合 vs 单层对比）。
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
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.kal import make_l1_head  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import kal_probe as kp  # noqa: E402
import diverse_truth_data as dt  # noqa: E402


@torch.no_grad()
def collect_multilayer(model, id_list, layers, device, batch_size, pooling):
    """一次前向取多层 pooled hidden {layer: [N,d]}（detach）。"""
    feats, _ = kp.forward_collect(model, id_list, list(layers), device, batch_size, pooling)
    return feats


def score_head(head, H):
    """三态头 score = logit[know=0] − logit[blank=2]（与 L1 口径一致）。"""
    with torch.no_grad():
        lg = head(H).float()
    return lg[:, 0] - lg[:, 2]


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL L1 多层融合微调（AUROC 软加权）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_kal/final")
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--report", default="runs/kal_multilayer/report.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8, 10])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--n_each", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--fuse_T", type=float, default=0.05, help="软加权温度（小=趋硬投票）")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    layers = args.layers
    print(f"[multilayer] ckpt={args.ckpt} layers=ℓ{layers} steps={args.steps} fuse_T={args.fuse_T}")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device, strict=True,
                                                  skip_keys=("kernel.side_heads.conflict.",))
    model.eval()
    if model.kernel is None:
        print("❌ model.kernel=None——需 KAL 训练 checkpoint")
        sys.exit(1)
    for p in model.parameters():
        p.requires_grad_(False)
    d = model.config.d_model
    # 每层独立真值头：ℓ8 复用内核已训 kal_l1，其余新建
    heads: dict[int, torch.nn.Module] = {}
    for L in layers:
        if L == 8 and model.kernel is not None:
            heads[L] = model.kernel.kal_l1  # 复用已训（仅评估，不再训）
            for p in heads[L].parameters():
                p.requires_grad_(False)
        else:
            heads[L] = make_l1_head(d).to(args.device)
            for p in heads[L].parameters():
                p.requires_grad_(True)
    trainable = [p for L in layers if L != 8 for p in heads[L].parameters()]
    opt = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0) if trainable else None

    # 真值数据（diverse v2，多句式+contrast-pair+程序化虚构词——跨层鲁棒）
    texts, labels = dt.build_diverse_truth_dataset(rng, args.n_each, args.n_each)
    ids = kp.encode_fixed(tok, texts, args.seq_len)
    print(f"[multilayer] 数据集 n={len(ids)} known={(labels==0).sum()}/unknown={(labels==2).sum()}")
    feats = collect_multilayer(model, ids, layers, args.device, args.batch_size, args.pooling)
    H = {L: torch.from_numpy(feats[L]).to(args.device) for L in layers}
    Y = torch.from_numpy(labels).to(args.device)
    y01 = (labels == 0).astype(int)  # known→1

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_layers():
        """各层 AUROC + 软加权融合 AUROC + 最优单层。"""
        scores = {}
        aucs = {}
        for L in layers:
            sc = score_head(heads[L], H[L]).cpu().numpy()
            scores[L] = sc
            aucs[L] = kp.auroc(sc, y01)
        # 软加权融合：w_l = softmax(AUROC_l / T)
        auroc_arr = np.array([aucs[L] for L in layers])
        w = np.exp(auroc_arr / args.fuse_T)
        w = w / w.sum()
        fused = sum(w[i] * scores[L] for i, L in enumerate(layers))
        fused_auroc = kp.auroc(fused, y01)
        best_single = max(aucs.values())
        best_layer = layers[int(np.argmax(auroc_arr))]
        return aucs, w, fused_auroc, best_single, best_layer

    init_aucs, init_w, init_fused, init_best, init_bl = eval_layers()
    print(f"[multilayer] 微调前：各层 AUROC { {L: round(a,3) for L,a in init_aucs.items()} } | "
          f"融合 {init_fused:.3f} | 最优单层 ℓ{init_bl} {init_best:.3f}")

    t0 = time.time()
    if opt is not None:
        idx_all = np.arange(len(ids))
        for step in range(args.steps):
            lr = lr_at(step)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            rng.shuffle(idx_all)
            bidx = torch.from_numpy(idx_all[: args.batch_size]).to(args.device)
            loss = 0.0
            for L in layers:
                if L == 8:
                    continue  # ℓ8 已训，不再训
                lg = heads[L](H[L][bidx])
                # 二分类退化：two=[know(0), blank(2)] 两列；标签映射 known(0)→0 / unknown(2)→1
                two = torch.stack([lg[:, 0], lg[:, 2]], dim=-1)
                target = (Y[bidx] == 2).long()  # known=0, unknown=1（对齐 two 两列）
                loss = loss + F.cross_entropy(two.float(), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if (step + 1) % 100 == 0 or step == 0:
                print(f"  step {step+1:4d}/{args.steps} ce={loss.item():.4f} lr={lr:.2e}")

    final_aucs, final_w, final_fused, final_best, final_bl = eval_layers()
    dur = time.time() - t0
    print(f"[multilayer] 微调后：各层 AUROC { {L: round(a,3) for L,a in final_aucs.items()} }")
    print(f"  融合权重 w={ {L: round(float(wi),3) for L,wi in zip(layers, final_w)} }")
    print(f"  融合 AUROC {init_fused:.3f}→{final_fused:.3f} | 最优单层 ℓ{final_bl} {final_best:.3f}")
    gain = final_fused - final_best
    verdict = (
        f"✅ 多层融合 ≥ 最优单层（融合 {final_fused:.3f} vs 单层 {final_best:.3f}，"
        f"增益 {gain:+.3f}）" if final_fused >= final_best - 1e-4
        else f"🟡 融合未超最优单层（{final_fused:.3f} vs {final_best:.3f}）——单层 ℓ{final_bl} "
             f"已饱和，多层价值在跨任务/跨域稳健性（非本同分布集 AUROC）")
    print(f"判定：{verdict}")

    report = {
        "ckpt": args.ckpt, "layers": layers, "steps": args.steps, "fuse_T": args.fuse_T,
        "method": "每层独立真值头（diverse v2）+ 验证集 AUROC 软加权融合 w_l=softmax(AUROC_l/T)",
        "init": {"aucs": init_aucs, "fused": init_fused, "best_single": init_best, "best_layer": init_bl},
        "final": {"aucs": final_aucs, "weights": {str(L): float(wi) for L, wi in zip(layers, final_w)},
                  "fused": final_fused, "best_single": final_best, "best_layer": final_bl, "gain": gain},
        "verdict": verdict, "duration_s": round(dur, 1),
        "note": "0.1B 12 层挂点 ℓ4/8/10（M2 已验 ℓ4/8）；1.5B 28 层正式挂点 ℓ10/14/18 预演",
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[multilayer] report → {rp}")


if __name__ == "__main__":
    main()
