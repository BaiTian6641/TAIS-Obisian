"""训练思考流形投影器（ThoughtManifoldProjector）——独立 sidecar 训练，不碰主干权重。

背景（训练需求判定，2026-07-31 核实）：
- 统一 checkpoint `checkpoints/pilot_0p1b_gdn2_10k_unified` 的 state_dict 共 266 键
  （embed 1 + layers 243 + norm_f 1 + kernel 21），**无任何 manifold/projector 键**；
  各 demo（thinking_real_adapter_demo 等）均现场 `ThoughtManifoldProjector(...)` 随机实例化，
  权重统计与 PyTorch 默认初始化完全一致（std 0.02086 ≈ 理论 0.02083）。
- 思考流形层文档（docs/memories/thinking-manifold-layer.md）把"独立训练该投影器
  （不触碰主干权重）"列为**待接**事项。→ 投影器从未训练，本脚本补齐。

训练目标（复用 manifold.py 既有损失，**不重写**）：
  ThoughtManifold.loss = w_conformal·conformal_isometry_loss + w_decor·decorrelation_loss
  - 共形等距（尺度不变）：相邻思考段的流形位移 ∝ 语义关系步长；
  - VICReg 去相关兜底（防坍缩红线，w_decor>0 不可省）。

监督信号（自监督，无需人工标注）：
  - 输入：主干（**冻结 + detach 红线**）在 data/shards 文本上的多层 hidden
    （capture_layers=[4,7,10]，早/中/晚三层，同一文本的三条视角轨迹）；
  - 思考段：每条 128-token 窗口切成 16 段 × 8 token，段内均值池化 → 段表征；
  - 语义步长：相邻段表征的余弦距离 1−cos（由冻结主干自身几何提供，detach 常数）。

红线纪律：
  - 主干全参数 requires_grad_(False) + hidden .detach()；优化器只含投影器参数；
  - **训练前后主干 state_dict 逐位一致校验**（torch.equal 全 266 键）；
  - 投影器存为**独立 sidecar** `checkpoints/pilot_0p1b_gdn2_10k_unified_manifold/projector.pt`，
    不改 unified 主 checkpoint。

训练后自验（脚本内建，数值写入报告）：
  ① 同类语义聚簇：4 主题 × 4 句，主题内 vs 跨主题余弦距离对比度（已训 vs 随机基线）；
  ② 等距性：val shard 窗口上位移-步长 Pearson（已训 vs 随机基线）+ 训练损失曲线；
  ③ 轨迹语义性：真实生成轨迹的"直线度" straightness = ||c_T−c_1||/Σ||disp||
     （随机游走≈0，有方向推进→更大；已训 vs 随机基线）。

用法：
  CUDA_VISIBLE_DEVICES=1 python -u scripts/train_manifold_projector.py [--steps 1500]
产出：
  checkpoints/pilot_0p1b_gdn2_10k_unified_manifold/projector.pt（投影器 sidecar）
  runs/manifold_training/report.json（损失曲线 + 三项验证数值）
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # build_unified_checkpoint.load_unified
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 兼容
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.manifold import ThoughtManifold, ThoughtManifoldProjector  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

from build_unified_checkpoint import load_unified  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_gdn2_10k_unified"
OUT_DIR = Path("checkpoints/pilot_0p1b_gdn2_10k_unified_manifold")
OUT_PT = OUT_DIR / "projector.pt"
REPORT = Path("runs/manifold_training/report.json")
TOK = "data/tokenizer/tokenizer.json"
SHARDS = "data/shards"

CAPTURE_LAYERS = [4, 7, 10]   # 早/中/晚三层（ℓ10 = KAL/演示标准读点）
READ_LAYER = 10               # 验证用读点（对齐 kaltruth/全链 demo 口径）
SEG_LEN = 8                   # 每思考段 token 数
N_SEG = 16                    # 每窗口段数 → 窗口 = 128 token
W_CONFORMAL = 1.0
W_DECOR = 0.1                 # 去相关兜底权重（坍缩红线：不可为 0，对齐 manifold.py 默认）

# 聚簇验证句集（4 主题 × 4 句；语料以 FineWeb-Edu 英文为主，用英文句）
TOPICS = {
    "math": ["The derivative of x squared is two x.",
             "To solve the equation, isolate the variable first.",
             "A prime number has exactly two divisors.",
             "The integral of cosine is sine."],
    "cooking": ["Preheat the oven before baking the bread.",
                "Chop the onions and fry them in olive oil.",
                "Simmer the soup for twenty minutes.",
                "Season the sauce with salt and pepper."],
    "history": ["The Roman Empire collapsed in the fifth century.",
                "World War Two ended in nineteen forty five.",
                "Ancient Egypt built the pyramids of Giza.",
                "The French Revolution began in seventeen eighty nine."],
    "biology": ["Mitochondria produce energy for the cell.",
                "DNA carries genetic information in chromosomes.",
                "Plants convert sunlight into chemical energy.",
                "The human heart pumps blood through arteries."],
}


# ---------------------------------------------------------------------------
# 表征提取原语（冻结主干，detach 红线）
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_hidden(model, ids, layers, dev):
    """ids [B,T] → {layer: hidden [B,T,d] fp32 detached}（pm_stream=1 时张量；dict 取 content）。"""
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, _, caps = model(ids, capture_layers=layers)
    out = {}
    for l in layers:
        h = caps[l]
        if isinstance(h, dict):
            h = h["content"]
        out[l] = h.float().detach()
    return out


@torch.no_grad()
def text_hidden(model, tok, text, dev, layer=READ_LAYER):
    """单文本 ℓlayer 均值池化表征 [1, d]（聚簇验证用）。"""
    ids = torch.tensor([tok.encode(text)], device=dev)
    h = batch_hidden(model, ids, [layer], dev)[layer]
    return h.mean(dim=1)  # [1, d]


def make_segments(h):
    """hidden [B,T,d] → 段表征 [B,N_SEG,d]（段内均值池化）。"""
    B, T, d = h.shape
    assert T == SEG_LEN * N_SEG, f"窗口长度 {T} ≠ {SEG_LEN}×{N_SEG}"
    return h.view(B, N_SEG, SEG_LEN, d).mean(dim=2)


def semantic_steps_from(segs):
    """段表征 [B,N,d] → 语义步长监督 [B,N-1]：相邻段余弦距离 1−cos（≥0，detach 常数）。"""
    a = torch.nn.functional.normalize(segs[:, 1:, :], dim=-1)
    b = torch.nn.functional.normalize(segs[:, :-1, :], dim=-1)
    return (1.0 - (a * b).sum(dim=-1)).clamp_min(0.0).detach()


# ---------------------------------------------------------------------------
# 验证三件套（已训 vs 随机基线共用同一套评估函数）
# ---------------------------------------------------------------------------
@torch.no_grad()
def eval_clustering(model, tok, projector, dev):
    """① 聚簇对比度：主题内余弦距离均值 vs 跨主题均值（流形坐标上）。"""
    crd = {}
    for t, sents in TOPICS.items():
        e = torch.cat([text_hidden(model, tok, s, dev) for s in sents])  # [4, d]
        c = projector.project(e)  # [4, 64]
        crd[t] = c / c.norm(dim=-1, keepdim=True)
    within, across = [], []
    for t, c in crd.items():
        D = 1 - c @ c.T
        within += [D[i, j].item() for i in range(4) for j in range(i + 1, 4)]
    import itertools
    for t1, t2 in itertools.combinations(crd, 2):
        D = 1 - crd[t1] @ crd[t2].T
        across += D.flatten().tolist()
    mw, ma = float(np.mean(within)), float(np.mean(across))
    return {"within": mw, "across": ma, "contrast": ma / max(mw, 1e-8)}


@torch.no_grad()
def eval_isometry(model, manifold, shards_val, projector, dev, rng, n_windows=48):
    """② 等距性：val 窗口上位移-步长 Pearson（conformal 诊断，越大越好）。"""
    ids, _ = shards_val.get_batch(n_windows, SEG_LEN * N_SEG, dev, rng)
    hmap = batch_hidden(model, ids, [READ_LAYER], dev)
    segs = make_segments(hmap[READ_LAYER])
    steps = semantic_steps_from(segs)
    coords = projector.project(segs)
    _, diag = manifold.loss(coords, steps, w_conformal=W_CONFORMAL, w_decor=W_DECOR)
    return {"pearson": diag["pearson"], "conformal": diag["conformal"],
            "decorrelation": diag["decorrelation"]}


@torch.no_grad()
def eval_generation_traj(model, tok, projector, dev, prompt, max_new=24):
    """③ 生成轨迹语义性：greedy 续答，逐步取 ℓ10 → 流形轨迹直线度 straightness。

    straightness = ||c_T − c_1|| / Σ||相邻位移||：随机游走 → ≈1/√T（小）；
    有方向推进的轨迹 → 更大。配合逐维 std（坐标使用范围）报告。
    """
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache, caps = model(ids, capture_layers=[READ_LAYER])
    h = caps[READ_LAYER]
    if isinstance(h, dict):
        h = h["content"]
    hs = [h[:, -1:, :].float()]  # 末 prompt token 作轨迹起点
    n_new = 0
    for _ in range(max_new):
        nxt = int(logits[:, -1, :].float().argmax(-1).item())
        if nxt == tok.eot_id:
            break
        x = torch.tensor([[nxt]], device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache, caps = model(x, cache, capture_layers=[READ_LAYER])
        h = caps[READ_LAYER]
        if isinstance(h, dict):
            h = h["content"]
        hs.append(h[:, -1:, :].float())
        n_new += 1
    coords = projector.project(torch.cat(hs, dim=1))[0]  # [T, 64]
    disp = (coords[1:] - coords[:-1]).norm(dim=-1)
    end_to_end = (coords[-1] - coords[0]).norm()
    straightness = float(end_to_end / disp.sum().clamp_min(1e-8))
    return {"n_steps": int(coords.shape[0]), "straightness": straightness,
            "mean_disp": float(disp.mean()), "coord_dim_std_min": float(coords.std(0).min()),
            "coord_dim_std_max": float(coords.std(0).max())}


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="训练思考流形投影器（冻结主干，sidecar 保存）")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8, help="每步窗口数（×3 层 = 24 条轨迹）")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT_PT))
    ap.add_argument("--report", default=str(REPORT))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(TOK)

    print("=" * 70)
    print("【思考流形投影器训练】冻结主干 + VICReg 去相关 + 共形等距（复用 manifold.py 损失）")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 加载统一 checkpoint（冻结红线）+ 逐位快照（训练后校验不污染）
    # ------------------------------------------------------------------
    model = load_unified(args.ckpt, dev)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    backbone_before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_bb_params = sum(v.numel() for v in backbone_before.values())
    print(f"[load] {args.ckpt}（{len(backbone_before)} 键 / {n_bb_params/1e6:.1f}M 参数，全冻结）")

    d_model = model.config.d_model
    manifold_dim = model.config.manifold_dim
    manifold = ThoughtManifold(d_model, manifold_dim).to(dev)
    # 随机基线投影器（与 demos 同一实例化方式：manual_seed(42) 后新建）——对照组
    torch.manual_seed(42)
    proj_rand = ThoughtManifoldProjector(d_model, manifold_dim).to(dev).eval()
    torch.manual_seed(args.seed)  # 恢复训练种子（避免基线实例化污染训练 RNG）
    trainable = [p for p in manifold.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable)
    print(f"[init] 投影器可训练参数 {n_tr}（proj.weight+bias；view3d 固定无梯度）")
    assert all(p.requires_grad for p in trainable) and not any(
        p.requires_grad for p in model.parameters()), "冻结红线：主干不得可训练"

    # ------------------------------------------------------------------
    # 训练前基线（随机投影器三项验证数值——对照组）
    # ------------------------------------------------------------------
    shards_val = Shards(SHARDS, "val")
    rng_eval = np.random.default_rng(999)
    print("\n[基线] 随机投影器（未训练）三项验证：")
    base_clu = eval_clustering(model, tok, proj_rand, dev)
    base_iso = eval_isometry(model, manifold, shards_val, proj_rand, dev, rng_eval)
    base_traj = eval_generation_traj(model, tok, proj_rand, dev,
                                     "The derivative of x squared is")
    print(f"  ① 聚簇对比度 {base_clu['contrast']:.3f}（within {base_clu['within']:.4f} / "
          f"across {base_clu['across']:.4f}）")
    print(f"  ② 等距 Pearson {base_iso['pearson']:.4f}（conformal {base_iso['conformal']:.4f}）")
    print(f"  ③ 轨迹直线度 {base_traj['straightness']:.3f}（{base_traj['n_steps']} 步）")

    # ------------------------------------------------------------------
    # 训练循环（AdamW + cosine；梯度只进投影器）
    # ------------------------------------------------------------------
    shards_train = Shards(SHARDS, "train")
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps, eta_min=args.lr * 0.1)
    seq = SEG_LEN * N_SEG
    curve = []
    t0 = time.time()
    print(f"\n[train] steps={args.steps} batch={args.batch}×{len(CAPTURE_LAYERS)} 层 "
          f"窗口={seq} token lr={args.lr} w_conformal={W_CONFORMAL} w_decor={W_DECOR}")
    for step in range(args.steps):
        ids, _ = shards_train.get_batch(args.batch, seq, dev, rng)
        hmap = batch_hidden(model, ids, CAPTURE_LAYERS, dev)  # 冻结主干前向（no_grad）
        # 多层 hidden → 多条轨迹样本（同一坐标空间；拼接 batch 维）
        segs = torch.cat([make_segments(hmap[l]) for l in CAPTURE_LAYERS], dim=0)  # [B*3,N,d]
        steps_sup = semantic_steps_from(segs)  # [B*3,N-1]
        coords = manifold.project(segs)        # 梯度路径：仅投影器
        loss, diag = manifold.loss(coords, steps_sup,
                                   w_conformal=W_CONFORMAL, w_decor=W_DECOR)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # 红线断言：主干参数不得有梯度（投影器训练不得污染主干）
        assert not any(p.grad is not None for p in model.parameters()), \
            "红线破坏：主干参数出现梯度"
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            curve.append({"step": step, "loss": float(loss.item()),
                          "conformal": diag["conformal"], "decorrelation": diag["decorrelation"],
                          "pearson": diag["pearson"], "lr": sched.get_last_lr()[0]})
            print(f"  step {step:5d} | loss {loss.item():.4f} "
                  f"(conf {diag['conformal']:.4f} decor {diag['decorrelation']:.4f}) "
                  f"pearson {diag['pearson']:+.4f}")
    train_sec = time.time() - t0
    print(f"[train] 完成 {args.steps} 步，耗时 {train_sec:.1f}s（{args.steps/train_sec:.1f} step/s）")

    # ------------------------------------------------------------------
    # 主干逐位一致校验（红线：训练不碰主干权重）
    # ------------------------------------------------------------------
    backbone_after = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    bitwise_same = all(torch.equal(backbone_before[k], backbone_after[k])
                       for k in backbone_before)
    print(f"\n[红线] 主干 state_dict 训练前后逐位一致：{bitwise_same}"
          f"（{len(backbone_before)} 键）")
    assert bitwise_same, "红线破坏：主干权重被改动"

    # ------------------------------------------------------------------
    # 训练后验证（已训投影器 vs 随机基线）
    # ------------------------------------------------------------------
    proj = manifold.projector.eval()
    rng_eval2 = np.random.default_rng(999)  # 与基线同一评估流（同窗同句，公平对照）
    print("\n[验证] 已训投影器三项验证（vs 随机基线）：")
    new_clu = eval_clustering(model, tok, proj, dev)
    new_iso = eval_isometry(model, manifold, shards_val, proj, dev, rng_eval2)
    new_traj = eval_generation_traj(model, tok, proj, dev,
                                    "The derivative of x squared is")
    print(f"  ① 聚簇对比度 {new_clu['contrast']:.3f}（基线 {base_clu['contrast']:.3f}；"
          f"within {new_clu['within']:.4f}→{base_clu['within']:.4f} 基线，"
          f"across {new_clu['across']:.4f}→{base_clu['across']:.4f} 基线）")
    print(f"  ② 等距 Pearson {new_iso['pearson']:.4f}（基线 {base_iso['pearson']:.4f}）")
    print(f"  ③ 轨迹直线度 {new_traj['straightness']:.3f}（基线 {base_traj['straightness']:.3f}）")

    # ------------------------------------------------------------------
    # 保存 sidecar + 报告
    # ------------------------------------------------------------------
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": {k: v.cpu() for k, v in proj.state_dict().items()},
        "d_model": d_model, "manifold_dim": manifold_dim,
        "meta": {"ckpt": args.ckpt, "steps": args.steps, "batch": args.batch,
                 "lr": args.lr, "seed": args.seed, "capture_layers": CAPTURE_LAYERS,
                 "seg_len": SEG_LEN, "n_seg": N_SEG,
                 "w_conformal": W_CONFORMAL, "w_decor": W_DECOR,
                 "supervision": "semantic_steps = 1-cos(相邻段冻结主干表征)，detach 常数"},
    }, out)
    print(f"\n[save] 投影器 sidecar → {out}")

    report = {
        "ckpt": args.ckpt, "out": str(out),
        "train": {"steps": args.steps, "batch": args.batch, "layers": CAPTURE_LAYERS,
                  "lr": args.lr, "seed": args.seed, "train_seconds": train_sec,
                  "curve": curve},
        "backbone_bitwise_identical": bitwise_same,
        "baseline_random": {"clustering": base_clu, "isometry": base_iso, "trajectory": base_traj},
        "trained": {"clustering": new_clu, "isometry": new_iso, "trajectory": new_traj},
        "honesty_note": "0.1B 主干语义弱时，训练后轨迹仍可能近似噪声——如实报告数值，不粉饰",
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] 训练报告 → {rep}")


if __name__ == "__main__":
    main()
