"""流形推理预览可视化（Manifold Preview）——向量空间中渲染模型推理过程。

功能：输入一段 prompt，模型续答（cache 增量生成），逐生成步取 ℓ10 hidden →
ThoughtManifold.project → project_3d → 渲染 3D 轨迹图（matplotlib，Carbon 色板
#0f62fe/#24a148/#d12771/#161616）。轨迹按生成步着色（渐变），起点/终点标注；
可选叠加知识块位置（同一投影器打上），显示推理轨迹是否"路过"知识块附近
（A→D 桥接验证的关键视图）。

坏路径检测：每生成步构造 ReasoningTickState（certainty 来自已校准 KAL L1 sense
只读读出），经 thought_visualizer.ThoughtVisualizer 四类检测（信心膨胀/漂移/
早停失败/recall 风暴），触发的类别写进图注。**3D 仅人类视图**（design §1.1 红线）；
本脚本纯只读推理（no_grad，监测/执行分置），不回流改变任何状态。

投影器来源（两模式）：
  - 默认加载已训 sidecar checkpoints/pilot_0p1b_gdn2_10k_unified_manifold/projector.pt
    （train_manifold_projector.py 产物）；
  - `--random` 用随机初始化投影器（manual_seed(42)，与 demos 同一实例化方式）——对照组；
  - `--compare` 并排渲染 随机 | 已训 双面板（同一次生成、同一批 hidden，唯一差异=投影器）。

用法：
  CUDA_VISIBLE_DEVICES=1 python -u scripts/manifold_preview.py \
      --prompt "The derivative of x squared is" --max_new 24
  CUDA_VISIBLE_DEVICES=1 python -u scripts/manifold_preview.py \
      --prompt "..." --compare --blocks_text runs/manifold_preview/blocks_sample.txt
产出：runs/manifold_preview/<slug>.png + <slug>.npz（轨迹 64 维坐标/3D/certainty/
  坏路径标记/知识块坐标，供下游分析）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无显示环境（服务器/工作站 headless）
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

# CJK 字体回退（坏路径类别中文标注；缺字体会渲染成方框）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                                   "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # build_unified_checkpoint.load_unified
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 兼容
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import ThoughtManifoldProjector  # noqa: E402
from tais_obsidian.model.reasoning_loop import ReasoningTickState  # noqa: E402
from tais_obsidian.model.thought_visualizer import ThoughtVisualizer  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

from build_unified_checkpoint import load_unified  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_gdn2_10k_unified"
PROJECTOR_PT = "checkpoints/pilot_0p1b_gdn2_10k_unified_manifold/projector.pt"
TOK = "data/tokenizer/tokenizer.json"
OUT_DIR = Path("runs/manifold_preview")
READ_LAYER = 10  # ℓ10（KAL/全链 demo 标准读点）
RECALL_THRESHOLD = 0.3  # 对齐 ReasoningLoop.should_recall 默认

# Carbon 色板（IBM Carbon Design）
C_BLUE = "#0f62fe"    # 轨迹
C_GREEN = "#24a148"   # 知识块
C_MAGENTA = "#d12771"  # 坏路径
C_GRAY100 = "#161616"  # 文字/起点终点


# ---------------------------------------------------------------------------
# 纯函数部件（可测，无模型依赖）
# ---------------------------------------------------------------------------
def slugify(prompt: str, max_len: int = 40) -> str:
    """prompt → 文件名安全 slug（小写字母数字连字符，截断）。"""
    s = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return (s[:max_len].strip("-") or "empty")


def load_projector(path: str | None, d_model: int, manifold_dim: int,
                   device: str) -> tuple[ThoughtManifoldProjector, str]:
    """加载投影器：path 给定时读 sidecar，否则随机初始化（seed 42 对照组）。

    返回 (projector, label)——label 用于图注（"trained" / "random"）。
    """
    proj = ThoughtManifoldProjector(d_model, manifold_dim).to(device).eval()
    if path is not None:
        blob = torch.load(path, map_location="cpu", weights_only=True)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        proj.load_state_dict(sd)
        return proj, "trained"
    # 随机对照：与 demos 同一实例化方式（manual_seed(42) 后新建 → 重新实例化对齐）
    torch.manual_seed(42)
    proj = ThoughtManifoldProjector(d_model, manifold_dim).to(device).eval()
    return proj, "random"


def build_tick_states(coords64: torch.Tensor, certainties: list[float],
                      early_stop_last: bool,
                      recall_threshold: float = RECALL_THRESHOLD
                      ) -> list[ReasoningTickState]:
    """逐生成步 64 维流形坐标 + KAL certainty → ReasoningTickState 列表。

    coords64: [T, manifold_dim]（每生成步一个流形坐标点）。
    early_stop_last: 生成因 eot 自然停止 → 末步记 early_stop=True（自适应算力语义：
    模型自己说完即"停"）；跑满 max_new → False（末步可能触发"早停失败"检测）。
    """
    T = coords64.shape[0]
    states = []
    for i in range(T):
        cert = float(certainties[i])
        cur = coords64[i].detach().float().view(1, 1, -1)
        # disp：生成步语义 = 到下一生成步的流形位移（末步无后继 → 零位移）
        if i < T - 1:
            disp = (coords64[i + 1] - coords64[i]).detach().float().view(1, 1, -1)
        else:
            disp = torch.zeros_like(cur)
        states.append(ReasoningTickState(
            tick_index=i,
            current_coord=cur,
            disp=disp,
            certainty=cert,
            early_stop=(i == T - 1) and early_stop_last,
            recall_triggered=cert < recall_threshold,
        ))
    return states


def detect_bad_path(tick_states: list[ReasoningTickState],
                    projector: ThoughtManifoldProjector):
    """接入 thought_visualizer 四类坏路径检测。

    返回 dict：{"n_bad": int, "classes": [去重的触发类别名], "bad_idx": [坏点 tick 序号],
                "trajectory": ThoughtTrajectory}。信心膨胀检测需忠实性诊断
    （speak_do_consistency），本脚本无 CoT 审计 → 该项不适用（诚实标注）。
    """
    viz = ThoughtVisualizer()
    traj = viz.build(tick_states, projector)
    bad_idx = [p.tick_index for p in traj.points if p.is_bad_path]
    classes: list[str] = []
    for p in traj.points:
        for reason in (p.bad_reason.split(";") if p.bad_reason else []):
            m = re.match(r"^(信心膨胀|漂移|早停失败|recall风暴)", reason)
            if m and m.group(1) not in classes:
                classes.append(m.group(1))
    return {"n_bad": traj.n_bad_points, "classes": classes,
            "bad_idx": bad_idx, "trajectory": traj}


def project_blocks(projector: ThoughtManifoldProjector,
                   block_reprs: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """知识块表征 [N, d_model] → (64 维坐标 [N,64], 3D 坐标 [N,3])（同一投影器）。"""
    with torch.no_grad():
        c64 = projector.project(block_reprs.float())
        c3 = projector.project_3d(c64)
    return c64.cpu().numpy(), c3.cpu().numpy()


def trajectory_block_distances(coords64: np.ndarray,
                               blocks64: np.ndarray) -> np.ndarray:
    """轨迹点 ↔ 知识块最近距离矩阵（64 维空间欧氏）：返回 [T] 每步到最近块距离。"""
    if blocks64.size == 0:
        return np.full(coords64.shape[0], np.nan)
    d = np.linalg.norm(coords64[:, None, :] - blocks64[None, :, :], axis=-1)  # [T,N]
    return d.min(axis=1)


def _step_cmap():
    """生成步渐变色图（Carbon：#161616 → #0f62fe）。"""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("carbon_step", [C_GRAY100, C_BLUE])


def render_trajectory(xyz: np.ndarray, out_png: str | Path | None, *,
                      blocks3d: np.ndarray | None = None,
                      block_labels: list[str] | None = None,
                      bad_idx: list[int] | None = None,
                      bad_classes: list[str] | None = None,
                      title: str = "", views2d: bool = False,
                      ax=None, show: bool = False) -> Path | None:
    """渲染 3D 轨迹图（或 2D 三视图）到 PNG。ax 给定时画进既有 Axes3D（compare 用，
    此时 out_png 传 None，由外层 fig 统一保存）。

    轨迹按生成步着色（#161616→#0f62fe 渐变）；起点 S / 终点 E 标注；
    知识块 #24a148 菱形；坏路径点 #d12771 ×。
    """
    out = Path(out_png) if out_png is not None else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    bad_set = set(bad_idx or [])
    T = xyz.shape[0]
    steps = np.arange(T)

    if ax is None:
        if views2d:
            fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
            fig.patch.set_facecolor("white")
            for k, (i, j, lab) in enumerate([(0, 1, "xy"), (0, 2, "xz"), (1, 2, "yz")]):
                a = axes[k]
                a.scatter(xyz[:, i], xyz[:, j], c=steps, cmap=_step_cmap(), s=36, zorder=3)
                a.plot(xyz[:, i], xyz[:, j], color=C_BLUE, alpha=0.35, lw=1.2, zorder=2)
                a.scatter(*xyz[0, [i, j]], marker="o", s=90, facecolor="white",
                          edgecolor=C_GRAY100, linewidth=1.8, zorder=4)
                a.annotate("S", xyz[0, [i, j]], fontsize=9, color=C_GRAY100, weight="bold")
                a.scatter(*xyz[-1, [i, j]], marker="s", s=90, facecolor=C_BLUE,
                          edgecolor=C_GRAY100, linewidth=1.2, zorder=4)
                a.annotate("E", xyz[-1, [i, j]], fontsize=9, color=C_GRAY100, weight="bold")
                if blocks3d is not None and len(blocks3d):
                    a.scatter(blocks3d[:, i], blocks3d[:, j], marker="D", s=70,
                              color=C_GREEN, edgecolor=C_GRAY100, linewidth=0.8, zorder=4)
                for bi in bad_set:
                    a.scatter(*xyz[bi, [i, j]], marker="x", s=110, color=C_MAGENTA,
                              linewidth=2.4, zorder=5)
                a.set_xlabel(lab[0]); a.set_ylabel(lab[1])
                a.set_title(f"{lab} 视图", fontsize=10, color=C_GRAY100)
            bad_txt = ("坏路径：" + "/".join(bad_classes)) if bad_classes else "坏路径：无"
            fig.suptitle(f"{title}　{bad_txt}", fontsize=11, color=C_GRAY100)
            fig.tight_layout()
            fig.savefig(out, dpi=160)
            plt.close(fig)
            return out
        fig = plt.figure(figsize=(8.4, 7))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, projection="3d")
        own_fig = fig
    else:
        own_fig = None

    sc = ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=steps, cmap=_step_cmap(),
                    s=42, zorder=3)
    ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=C_BLUE, alpha=0.35, lw=1.4, zorder=2)
    ax.scatter(*xyz[0], marker="o", s=110, facecolor="white",
               edgecolor=C_GRAY100, linewidth=2.0, zorder=4)
    ax.text(*xyz[0], "S", fontsize=10, color=C_GRAY100, weight="bold")
    ax.scatter(*xyz[-1], marker="s", s=110, facecolor=C_BLUE,
               edgecolor=C_GRAY100, linewidth=1.4, zorder=4)
    ax.text(*xyz[-1], "E", fontsize=10, color=C_GRAY100, weight="bold")
    if blocks3d is not None and len(blocks3d):
        ax.scatter(blocks3d[:, 0], blocks3d[:, 1], blocks3d[:, 2], marker="D", s=90,
                   color=C_GREEN, edgecolor=C_GRAY100, linewidth=0.9, zorder=4,
                   label="knowledge block")
        if block_labels:
            for pt, lab in zip(blocks3d, block_labels):
                ax.text(*pt, lab, fontsize=7.5, color=C_GREEN)
    for bi in bad_set:
        ax.scatter(*xyz[bi], marker="x", s=150, color=C_MAGENTA, linewidth=2.8, zorder=5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    bad_txt = ("坏路径：" + "/".join(bad_classes)) if bad_classes else "坏路径：无"
    ax.set_title(f"{title}　{bad_txt}", fontsize=10.5, color=C_GRAY100)

    if own_fig is not None:
        cbar = own_fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.08)
        cbar.set_label("generation step", color=C_GRAY100)
        own_fig.tight_layout()
        own_fig.savefig(out, dpi=160)
        plt.close(own_fig)
    return out


def save_npz(path: str | Path, **arrays) -> Path:
    """保存轨迹数据（np.savez：coords64/xyz/certainty/bad/blocks 等）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


# ---------------------------------------------------------------------------
# 模型侧原语（真实生成 + 逐步 hidden + KAL certainty，只读 detach）
# ---------------------------------------------------------------------------
@torch.no_grad()
def generate_with_hidden(model, tok, prompt: str, dev: str, max_new: int = 24,
                         layer: int = READ_LAYER, temperature: float = 0.0,
                         top_k: int = 0) -> dict:
    """prompt 续答（cache 增量），逐步取 ℓlayer hidden + KAL P(known)。

    返回 {"hiddens": [T, d_model] fp32（末 prompt token + 每生成步）,
          "tokens": [生成 token 文本], "certainties": [T], "stopped_by_eot": bool,
          "generated_text": str}。temperature=0 → greedy argmax（确定性，对照可复现）。
    """
    ids = torch.tensor([tok.encode(prompt)], device=dev)

    def _forward(x, cache):
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            return model(x, cache, capture_layers=[layer])

    def _h_of(caps):
        h = caps[layer]
        if isinstance(h, dict):
            h = h["content"]
        return h

    def _certainty(h_step):
        """KAL L1 sense 只读读出（known 类概率；kernel 已校准时为真值锚语义）。"""
        if model.kernel is None:
            return float("nan")
        sense = model.kernel.sense(h_step)
        probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)
        return float(probs[:, 0].mean().item())

    logits, cache, caps = _forward(ids, None)
    h = _h_of(caps)
    hiddens = [h[0, -1, :].float().cpu()]
    certs = [_certainty(h[:, -1:, :])]
    out_tokens: list[int] = []
    stopped = False
    for _ in range(max_new):
        step_logits = logits[:, -1, :].float()
        if temperature and temperature > 0:
            step_logits = step_logits / temperature
            if top_k > 0:
                v, _ = torch.topk(step_logits, top_k)
                step_logits = step_logits.masked_fill(
                    step_logits < v[:, -1:], float("-inf"))
            nxt = int(torch.multinomial(torch.softmax(step_logits, -1), 1).item())
        else:
            nxt = int(step_logits.argmax(-1).item())
        if nxt == tok.eot_id:
            stopped = True
            break
        out_tokens.append(nxt)
        x = torch.tensor([[nxt]], device=dev)
        logits, cache, caps = _forward(x, cache)
        h = _h_of(caps)
        hiddens.append(h[0, -1, :].float().cpu())
        certs.append(_certainty(h))
    return {"hiddens": torch.stack(hiddens),           # [T, d]
            "tokens": [tok.decode([t]) for t in out_tokens],
            "certainties": certs,
            "stopped_by_eot": stopped,
            "generated_text": tok.decode(out_tokens)}


@torch.no_grad()
def block_reprs_from_texts(model, tok, texts: list[str], dev: str,
                           layer: int = READ_LAYER) -> torch.Tensor:
    """知识文本列表 → ℓlayer 均值池化表征 [N, d]（与 harvest_kv_block 的 repr 同口径）。"""
    reps = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        ids = torch.tensor([tok.encode(t)], device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            _, _, caps = model(ids, capture_layers=[layer])
        h = caps[layer]
        if isinstance(h, dict):
            h = h["content"]
        reps.append(h.float().mean(dim=1).cpu())  # [1, d]
    return torch.cat(reps) if reps else torch.zeros(0, model.config.d_model)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="流形推理预览可视化（3D 轨迹 + 知识块叠加 + 坏路径）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new", type=int, default=24)
    ap.add_argument("--projector", default=PROJECTOR_PT,
                    help="已训投影器 sidecar 路径（缺省文件不存在时回退随机对照并标注）")
    ap.add_argument("--random", action="store_true", help="强制随机初始化投影器（对照组）")
    ap.add_argument("--compare", action="store_true",
                    help="并排渲染 随机|已训 双面板（同一次生成同一批 hidden）")
    ap.add_argument("--blocks_text", default=None,
                    help="知识块文本文件（每行一条知识；经同一投影器叠加到图上）")
    ap.add_argument("--views2d", action="store_true", help="2D 三视图（xy/xz/yz）替代 3D")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(TOK)
    model = load_unified(args.ckpt, dev)
    model.eval()
    d_model, m_dim = model.config.d_model, model.config.manifold_dim
    print(f"[load] {args.ckpt} d_model={d_model} manifold_dim={m_dim} dev={dev}")

    # ------------------------------------------------------------------
    # ① 生成（一次；compare 双面板共用同一批 hidden，唯一差异=投影器）
    # ------------------------------------------------------------------
    gen = generate_with_hidden(model, tok, args.prompt, dev, args.max_new,
                               temperature=args.temperature, top_k=args.top_k)
    hiddens = gen["hiddens"].to(dev)  # [T, d]
    print(f"[gen] {len(gen['tokens'])} tokens：{gen['generated_text']!r}"
          f"（{'eot 自然停止' if gen['stopped_by_eot'] else '跑满 max_new'}）")
    print(f"[kal] certainty 均值 {np.nanmean(gen['certainties']):.3f} "
          f"min {np.nanmin(gen['certainties']):.3f}（kernel 已校准读数，只读监测）")

    # ------------------------------------------------------------------
    # ② 知识块（可选）：文本 → ℓ10 repr → 同一投影器
    # ------------------------------------------------------------------
    block_reprs, block_labels = None, None
    if args.blocks_text:
        lines = Path(args.blocks_text).read_text(encoding="utf-8").splitlines()
        block_labels = [l.strip()[:24] for l in lines if l.strip()]
        block_reprs = block_reprs_from_texts(model, tok, lines, dev).to(dev)
        print(f"[blocks] {len(block_labels)} 条知识块（{args.blocks_text}）")

    # ------------------------------------------------------------------
    # ③ 投影 + 坏路径检测 + 渲染（单投影器 or compare 双面板）
    # ------------------------------------------------------------------
    slug = slugify(args.prompt)
    out_dir = Path(args.out_dir)

    def _project_and_detect(proj):
        with torch.no_grad():
            coords64 = proj.project(hiddens)            # [T, 64]
            xyz = proj.project_3d(coords64)             # [T, 3]
        ticks = build_tick_states(coords64, gen["certainties"], gen["stopped_by_eot"])
        det = detect_bad_path(ticks, proj)
        return coords64, xyz, det

    def _blocks_of(proj):
        if block_reprs is None:
            return None, None, None
        b64, b3 = project_blocks(proj, block_reprs)
        return b64, b3, block_labels

    if args.compare:
        proj_r, _ = load_projector(None, d_model, m_dim, dev)
        proj_t, label_t = load_projector(
            args.projector if Path(args.projector).exists() else None,
            d_model, m_dim, dev)
        c_r, x_r, d_r = _project_and_detect(proj_r)
        c_t, x_t, d_t = _project_and_detect(proj_t)
        b64_r, b3_r, labs = _blocks_of(proj_r)
        b64_t, b3_t, _ = _blocks_of(proj_t)

        fig = plt.figure(figsize=(16.5, 7.2))
        fig.patch.set_facecolor("white")
        ax1 = fig.add_subplot(121, projection="3d")
        ax2 = fig.add_subplot(122, projection="3d")
        render_trajectory(x_r.cpu().numpy(), None, blocks3d=b3_r, block_labels=labs,
                          bad_idx=d_r["bad_idx"], bad_classes=d_r["classes"],
                          title=f"random（未训练对照）", ax=ax1)
        render_trajectory(x_t.cpu().numpy(), None, blocks3d=b3_t, block_labels=labs,
                          bad_idx=d_t["bad_idx"], bad_classes=d_t["classes"],
                          title=f"{label_t}（已训 sidecar）", ax=ax2)
        fig.suptitle(f"流形推理轨迹对比：{args.prompt!r} → {gen['generated_text']!r}",
                     fontsize=11, color=C_GRAY100)
        out_png = out_dir / f"{slug}_compare.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        out_npz = save_npz(
            out_dir / f"{slug}_compare.npz",
            coords64_random=c_r.cpu().numpy(), xyz_random=x_r.cpu().numpy(),
            coords64_trained=c_t.cpu().numpy(), xyz_trained=x_t.cpu().numpy(),
            certainty=np.array(gen["certainties"]),
            bad_idx_random=np.array(d_r["bad_idx"]), bad_idx_trained=np.array(d_t["bad_idx"]),
            blocks64_trained=b64_t if b64_t is not None else np.zeros((0, m_dim)),
            prompt=np.array([args.prompt]),
            generated=np.array([gen["generated_text"]]),
            tokens=np.array(gen["tokens"]))
        results = {"random": {"bad": d_r["classes"], "n_bad": d_r["n_bad"]},
                   "trained": {"bad": d_t["classes"], "n_bad": d_t["n_bad"]}}
        print(f"[bad] random：{d_r['classes'] or '无'}；trained：{d_t['classes'] or '无'}")
        if b64_t is not None:
            dist = trajectory_block_distances(c_t.cpu().numpy(), b64_t)
            print(f"[blocks] 轨迹到最近知识块距离（64 维，trained）："
                  f"min {np.nanmin(dist):.3f} mean {np.nanmean(dist):.3f}")
            results["block_dist_trained"] = {"min": float(np.nanmin(dist)),
                                             "mean": float(np.nanmean(dist))}
    else:
        proj_path = None if args.random else (
            args.projector if Path(args.projector).exists() else None)
        if not args.random and proj_path is None:
            print(f"[warn] 已训投影器 {args.projector} 不存在 → 回退随机对照（先跑 "
                  f"train_manifold_projector.py）")
        proj, label = load_projector(proj_path, d_model, m_dim, dev)
        coords64, xyz, det = _project_and_detect(proj)
        b64, b3, labs = _blocks_of(proj)
        suffix = "_2d" if args.views2d else ""
        out_png = out_dir / f"{slug}_{label}{suffix}.png"
        render_trajectory(xyz.cpu().numpy(), out_png, blocks3d=b3, block_labels=labs,
                          bad_idx=det["bad_idx"], bad_classes=det["classes"],
                          title=f"{label}：{args.prompt!r}", views2d=args.views2d)
        out_npz = save_npz(
            out_dir / f"{slug}_{label}{suffix}.npz",
            coords64=coords64.cpu().numpy(), xyz=xyz.cpu().numpy(),
            certainty=np.array(gen["certainties"]),
            bad_idx=np.array(det["bad_idx"]),
            blocks64=b64 if b64 is not None else np.zeros((0, m_dim)),
            prompt=np.array([args.prompt]),
            generated=np.array([gen["generated_text"]]),
            tokens=np.array(gen["tokens"]))
        results = {"label": label, "bad": det["classes"], "n_bad": det["n_bad"]}
        print(f"[bad] {det['classes'] or '无'}（n={det['n_bad']}；信心膨胀项需忠实性诊断，"
              f"本脚本不适用）")
        if b64 is not None:
            dist = trajectory_block_distances(coords64.cpu().numpy(), b64)
            print(f"[blocks] 轨迹到最近知识块距离（64 维）："
                  f"min {np.nanmin(dist):.3f} mean {np.nanmean(dist):.3f}")
            results["block_dist"] = {"min": float(np.nanmin(dist)),
                                     "mean": float(np.nanmean(dist))}

    print(f"[save] {out_png}")
    print(f"[save] {out_npz}")
    meta_path = Path(str(out_npz).replace(".npz", ".json"))
    meta_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {meta_path}")


if __name__ == "__main__":
    main()
