"""KAL L3 冲突头真值微调（中层 logistic 三态 + conflict/surprise 消融，T1 观测）。

设计依据（article_ref/07 §4，逐条已核实）：
- **中层残差流冲突信号**（2410.16090，NeurIPS 2024）：LLM 在残差流内部寄存知识冲突
  信号，中层（Llama3-8B 第13层≈40%深度）升起，**简单 logistic 探针即 90% 准确率**；
  context-memory 冲突（parametric vs contextual），取**整体残差非少数头硬路由**
  （2503.10996：memory/context 头非互斥、superposition，硬路由不稳健）。
- **三态分类**（Xu 2403.08319 框架）：一致(consistent) / 参数优先(parametric) /
  上下文优先(contextual)——模型在冲突下偏向哪个知识源。
- **存争议（T1 消融）**：dACC 编码 unsigned surprise 非 signed RPE（Hayden 2011）——
  L3 该测"冲突方向"还是"无符号惊讶"？本脚本**同时训练两候选头做消融**。

与 L1/L2 同纪律：微调内生侧信道头（side_heads.conflict，随 checkpoint），detach 主干
（监测/执行分置红线：L3 只读残差不触碰主干权重），真值标签。

数据构建（自包含，合成 context-memory 冲突）：
- **一致(0)**：语境与常识一致（"The capital of France is Paris. Paris is the capital of France."）；
- **参数优先(1)**：语境给出与常识冲突的信息但模型应信记忆——用弱断言语气
  （"Some claim the capital of France is London, but actually the capital of France is Paris."）；
- **上下文优先(2)**：语境强陈述覆盖常识（虚构/反事实，模型须用上下文——
  "In this fictional world, the capital of France is London. The story begins in London."）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_l3_finetune.py \
      [--ckpt checkpoints/pilot_0p1b_kal/final] [--steps 400] [--layers 8]
产出：{ckpt}_kall3/ + runs/kal_l3/report.json（含 conflict vs surprise 消融对比）。
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

# ---------------------------------------------------------------------------
# L3 冲突数据集（context-memory 三态，自包含合成）
# ---------------------------------------------------------------------------
# 常识事实（实体 + 正确属性），用于构造一致/冲突语境
_FACTS = [
    ("capital of France", "Paris"), ("capital of Japan", "Tokyo"),
    ("largest planet", "Jupiter"), ("chemical symbol for water", "H2O"),
    ("boiling point of water", "100 degrees Celsius"),
    ("author of Hamlet", "Shakespeare"), ("currency of the USA", "the dollar"),
    ("largest ocean", "the Pacific Ocean"), ("speed of light", "300000 km/s"),
    ("organ that pumps blood", "the heart"),
]
_WRONG = ["London", "Mars", "Napoleon", "the euro", "50 degrees", "Mercury",
          "Tokyo", "the liver", "1000 km/s", "Oslo"]


def build_l3_dataset(rng: np.random.Generator, n_per_class: int) -> tuple[list[str], np.ndarray]:
    """构造三态冲突数据集。labels: 0=一致, 1=参数优先, 2=上下文优先。"""
    texts, labels = [], []
    for _ in range(n_per_class):
        fact, right = _FACTS[int(rng.integers(len(_FACTS)))]
        wrong = str(rng.choice(_WRONG))
        # 一致：语境与常识一致
        texts.append(f"The {fact} is {right}. Indeed, {right} is the {fact}.")
        labels.append(0)
        # 参数优先：语境有冲突主张但模型应信记忆（弱断言被纠正）
        texts.append(f"Some mistakenly claim the {fact} is {wrong}, but in fact the {fact} is {right}.")
        labels.append(1)
        # 上下文优先：语境强陈述反事实（虚构世界，须用上下文）
        texts.append(f"In this fictional world, the {fact} is {wrong}. The story begins in {wrong}.")
        labels.append(2)
    return texts, np.array(labels, dtype=np.int64)


@torch.no_grad()
def collect_hidden(model, id_list, layer, device, batch_size, pooling):
    feats, _ = kp.forward_collect(model, id_list, [layer], device, batch_size, pooling)
    return feats[layer]


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL L3 冲突头三态微调（T1 观测 + 消融）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_kal/final")
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--report", default="runs/kal_l3/report.json")
    ap.add_argument("--layers", type=int, nargs="+", default=[8])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--n_per_class", type=int, default=120)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    layer = args.layers[0]
    print(f"[kall3] ckpt={args.ckpt} layer=ℓ{layer} steps={args.steps}")

    # strict=True（用户纪律）+ skip_keys 剔除形状演进的 conflict 键：旧 checkpoint 的
    # side_heads.conflict 为 1 态占位（Linear(d,1)），新内核升级为三态 Linear(d,3)——
    # 剔除旧 1 态权重（三态头随机初始化待本脚本微调），其余权重严格载入。
    model = TaisObsidianForCausalLM.from_pretrained(
        args.ckpt, args.device, strict=True,
        skip_keys=("kernel.side_heads.conflict.",))
    model.eval()
    if model.kernel is None:
        print("❌ model.kernel=None——需 KAL 训练 checkpoint（含内核）")
        sys.exit(1)
    for p in model.parameters():
        p.requires_grad_(False)

    d = model.config.d_model
    # L3 三态头写入内核 side_heads.conflict（Linear(d,3)，随 checkpoint 存取——红线：
    # 内生头须持久化，不临时新建）；surprise 对照头临时（消融候选，d→1，不随 checkpoint）。
    conflict_head = model.kernel.side_heads.conflict
    assert conflict_head.out_features == 3, "side_heads.conflict 须为三态 Linear(d,3)"
    surprise_head = torch.nn.Linear(d, 1).to(args.device)  # unsigned surprise（消融，临时）
    for p in conflict_head.parameters():
        p.requires_grad_(True)
    for p in surprise_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(list(conflict_head.parameters()) + list(surprise_head.parameters()),
                            lr=args.lr, betas=(0.9, 0.95), weight_decay=1e-4)

    texts, labels = build_l3_dataset(rng, args.n_per_class)
    ids = kp.encode_fixed(tok, texts, args.seq_len)
    print(f"[kall3] 数据集 n={len(ids)} 一致/参数/上下文={np.bincount(labels).tolist()}")
    H = torch.from_numpy(collect_hidden(model, ids, layer, args.device,
                                        args.batch_size, args.pooling)).to(args.device)
    Y3 = torch.from_numpy(labels).to(args.device)              # 三态
    Ysur = torch.from_numpy((labels > 0).astype(np.int64)).to(args.device)  # 冲突(1,2) vs 一致(0)
    N = H.shape[0]

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_l3():
        conflict_head.eval(); surprise_head.eval()
        logits3 = conflict_head(H).float()
        pred3 = logits3.argmax(dim=-1)
        acc3 = (pred3 == Y3).float().mean().item()
        # 三态各类一-vs-余 AUROC
        aucs = {}
        for c, name in [(0, "consistent"), (1, "parametric"), (2, "contextual")]:
            score = logits3[:, c].cpu().numpy()
            binary = (labels == c).astype(int)
            aucs[name] = kp.auroc(score, binary)
        # surprise 头：冲突检测 AUROC（无符号）
        sur_score = surprise_head(H).float().squeeze(-1).cpu().numpy()
        sur_auroc = kp.auroc(sur_score, (labels > 0).astype(int))
        conflict_head.train(); surprise_head.train()
        return acc3, aucs, sur_auroc

    init_acc, init_aucs, init_sur = eval_l3()
    print(f"[kall3] 微调前：三态 acc {init_acc:.3f} | surprise AUROC {init_sur:.3f}")

    t0 = time.time()
    conflict_head.train(); surprise_head.train()
    idx_all = np.arange(N)
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        rng.shuffle(idx_all)
        bidx = torch.from_numpy(idx_all[: args.batch_size]).to(args.device)
        ce3 = F.cross_entropy(conflict_head(H[bidx]).float(), Y3[bidx])
        bce = F.binary_cross_entropy_with_logits(
            surprise_head(H[bidx]).float().squeeze(-1), Ysur[bidx].float())
        loss = ce3 + bce
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(conflict_head.parameters()) + list(surprise_head.parameters()), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps} ce3={ce3.item():.4f} bce={bce.item():.4f} lr={lr:.2e}")

    final_acc, final_aucs, final_sur = eval_l3()
    dur = time.time() - t0
    print(f"[kall3] 微调后：三态 acc {init_acc:.3f}→{final_acc:.3f} | "
          f"surprise AUROC {init_sur:.3f}→{final_sur:.3f} 用时 {dur:.0f}s")
    print(f"  三态 AUROC: consistent={final_aucs['consistent']:.3f} "
          f"parametric={final_aucs['parametric']:.3f} contextual={final_aucs['contextual']:.3f}")
    # 判据（2410.16090：冲突检测中层 logistic 高准确率；三态 acc>0.6 超 chance 0.33）
    conflict_ok = final_sur > 0.7  # 冲突检测（无符号）
    three_ok = final_acc > 0.6     # 三态方向
    verdict = (
        "✅ L3 冲突检测达标（surprise>0.7 且三态>0.6，T1 观测）" if (conflict_ok and three_ok)
        else ("🟡 冲突检测可（surprise>0.7）但三态方向弱——信号=无符号惊讶为主，"
              "方向（parametric/contextual）待 1.5B/更多数据（诚实，正合 Hayden 2011 争议）" if conflict_ok
              else "⚠️ L3 冲突信号弱（0.1B 中层未强编码，1.5B 待验——诚实负结果）"))
    print(f"判定：{verdict}")
    print(f"  消融结论：{'unsigned-surprise 优于方向判别' if final_sur > final_acc else '方向判别（三态）更优'}")

    out_dir = Path(args.out_dir) if args.out_dir else Path(f"{args.ckpt}_kall3")
    model.save_pretrained(out_dir)
    report = {
        "ckpt": args.ckpt, "layer": layer, "steps": args.steps, "lr": args.lr,
        "n_per_class": args.n_per_class, "method": "L3 三态 logistic（一致/参数/上下文）+ surprise 对照头，detach 主干",
        "init": {"acc3": init_acc, "aucs": init_aucs, "surprise_auroc": init_sur},
        "final": {"acc3": final_acc, "aucs": final_aucs, "surprise_auroc": final_sur},
        "ablation": "unsigned-surprise vs 三态方向（Hayden 2011 争议）",
        "verdict": verdict, "duration_s": round(dur, 1), "out_dir": str(out_dir),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kall3] checkpoint → {out_dir}；report → {rp}")


if __name__ == "__main__":
    main()
