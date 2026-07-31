"""生成英文版报告图表（reports/assets/en/*.png）——全部数据取自 runs/*/report.json 与训练日志（不重跑模型）。

与 _make_report_assets.py（中文版）同数据、同 Carbon 风格；供英文版学术文档引用。
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "assets" / "en"
OUT.mkdir(parents=True, exist_ok=True)

INK, SUB, FAINT, LINE = "#161616", "#525252", "#8d8d8d", "#e0e0e0"
BLUE, PURPLE, TEAL, GREEN, MAGENTA, ORANGE, RED = "#0f62fe", "#8a3ffc", "#009d9a", "#24a148", "#d12771", "#ff832b", "#da1e28"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=SUB, labelsize=9)
    ax.grid(axis="y", color=LINE, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor="white", dpi=150)
    plt.close(fig)
    print("written", name)


# 1) Training curve (real log)
log = (ROOT / "runs/_gdn2_10k_train_out.txt").read_text(encoding="utf-8", errors="replace")
steps, losses, emas = [], [], []
for m in re.finditer(r"step\s+(\d+) \| loss ([\d.]+) \(ema ([\d.]+)\)", log):
    steps.append(int(m.group(1))); losses.append(float(m.group(2))); emas.append(float(m.group(3)))
fig, ax = plt.subplots(figsize=(8.6, 4.2))
style_ax(ax)
ax.plot(steps, losses, color=LINE, lw=1.0, label="train loss (every 50 steps)")
ax.plot(steps, emas, color=BLUE, lw=2.2, label="EMA smoothed")
ax.set_xlabel("training step", color=SUB); ax.set_ylabel("next-token loss", color=SUB)
ax.set_title("0.1B GDN-2 pre-training, 10k steps (runs/_gdn2_10k_train_out.txt)", loc="left", color=INK)
ax.legend(frameon=False)
save(fig, "chart_training_curve.png")

# 2) Ablation
fig, ax = plt.subplots(figsize=(7.6, 4.2))
style_ax(ax)
names = ["hybrid\nbaseline", "+tri-level\nstack", "+PM-\nstream", "combined"]
vals = [3.768, 3.762, 3.744, 3.743]
bars = ax.bar(names, vals, color=[FAINT, ORANGE, TEAL, BLUE], width=0.58, zorder=3)
ax.set_ylim(3.70, 3.80)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.axhline(vals[0], color=FAINT, ls="--", lw=1)
ax.set_ylabel("val loss (lower is better)", color=SUB)
ax.set_title("Native-component ablation (0.1B, 2000 steps): PM-stream −0.024 / combined −0.025 nats", loc="left", color=INK)
save(fig, "chart_ablation.png")

# 3) KAL calibration
fig, ax = plt.subplots(figsize=(7.6, 4.2))
style_ax(ax)
groups = ["v1 truth anchor\n(confidence)", "v2 anchor expansion\n(script n=200)", "v2 anchor expansion\n(test n=400)"]
means = [0.769, 0.84475, 0.829225]
stds = [0.0, 0.024172, 0.007420]
bars = ax.bar(groups, means, yerr=stds, color=[FAINT, BLUE, BLUE], width=0.5, zorder=3,
              error_kw=dict(ecolor=SUB, capsize=4, lw=1.2))
ax.axhline(0.8, color=GREEN, ls="--", lw=1.4)
ax.text(2.42, 0.804, "target ≥ 0.8", fontsize=9, color=GREEN, ha="right")
for b, v in zip(bars, means):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.set_ylim(0.70, 0.90)
ax.set_ylabel("truth-calibration AUROC", color=SUB)
ax.set_title("KAL metacognition calibration: anchor expansion reaches ≥0.8 (3 seeds, mean±std)", loc="left", color=INK)
save(fig, "chart_kal.png")

# 4) Full-chain strengths
fig, ax = plt.subplots(figsize=(8.6, 4.2))
style_ax(ax)
items = ["HRL block\nretrieval top-1", "HCA injection\nrecall", "in-context\n(after mem-layer fix)", "honest decline\n(fabricated)"]
vals = [0.938, 0.625, 0.688, 1.000]
refs = [(0.062, "baseline 0.062"), (0.188, "linear gate 0.188"), (0.250, "gating side-effect 0.250"), (0.0, "")]
bars = ax.bar(items, vals, color=[PURPLE, GREEN, TEAL, MAGENTA], width=0.52, zorder=3)
for b, v, (rv, rl) in zip(bars, vals, refs):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold", color=INK)
    if rv > 0:
        ax.plot([b.get_x(), b.get_x() + b.get_width()], [rv, rv], color=RED, lw=1.4, ls="--")
        ax.text(b.get_x() + b.get_width() / 2, rv + 0.015, rl, ha="center", fontsize=7.5, color=RED)
ax.set_ylim(0, 1.12)
ax.set_ylabel("metric", color=SUB)
ax.set_title("Unified-checkpoint full-chain trained strengths (0.1B, n=16)", loc="left", color=INK)
save(fig, "chart_fullchain.png")

# 5) NIAH length scan
fig, ax = plt.subplots(figsize=(8.6, 4.2))
style_ax(ax)
labels = ["512\nk8", "512\nk32", "1024\nk8", "1024\nk32", "2048\nk8", "2048\nk32", "4096\nk8", "4096\nk32"]
g1 = [0.100, 0.060, 0.080, 0.040, 0, 0, 0, 0]
g2 = [0.120, 0.040, 0.100, 0.060, 0, 0, 0, 0]
x = np.arange(len(labels)); w = 0.36
ax.bar(x - w / 2, g1, w, color=FAINT, label="GDN-1", zorder=3)
ax.bar(x + w / 2, g2, w, color=TEAL, label="GDN-2 bounded 10k", zorder=3)
ax.axvspan(3.5, 7.5, color="#fde7e7", zorder=1)
ax.text(5.5, 0.105, "max_seq=1024 hard limit (truncated to 0)\n→ RoPE expansion + YaRN project", ha="center", fontsize=8.5, color=RED)
ax.set_xticks(x, labels)
ax.set_xlabel("context length × needle count", color=SUB)
ax.set_ylabel("NIAH first-token hit", color=SUB)
ax.set_ylim(0, 0.14)
ax.legend(frameon=False)
ax.set_title("NIAH length scan (50 queries/cell): 1024 hard limit motivates the 256K project", loc="left", color=INK)
save(fig, "chart_niah.png")

# 6) CA1 adaptive: before/after verdicts
fig, ax = plt.subplots(figsize=(7.6, 4.2))
style_ax(ax)
cats = ["PROMOTE", "QUARANTINE", "REJECT"]
before = [3, 1, 3]
after = [6, 1, 0]
x = np.arange(3); w = 0.36
b1 = ax.bar(x - w / 2, before, w, color=FAINT, label="CA1 v1.0 (static)", zorder=3)
b2 = ax.bar(x + w / 2, after, w, color=GREEN, label="CA1 v1.1 (adaptive RE_VERIFY)", zorder=3)
for bs in (b1, b2):
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, str(int(b.get_height())), ha="center", fontsize=11, fontweight="bold", color=INK)
ax.set_xticks(x, cats)
ax.set_ylabel("blocks (n=6 taught + 1 conflict)", color=SUB)
ax.set_ylim(0, 7.2)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Sleep-consolidation verdicts: source-credibility edge effect fixed;\nconflict block still QUARANTINE (anti-poison red line held)", loc="left", color=INK, fontsize=10.5)
save(fig, "chart_ca1_adaptive.png")

# 7) Manifold projector: random vs trained
fig, ax = plt.subplots(figsize=(7.6, 4.2))
style_ax(ax)
metrics = ["clustering contrast\n(higher=better)", "isometry Pearson\n(higher=better)"]
rand = [1.558, 0.882]
trained = [1.989, 0.977]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, rand, w, color=FAINT, label="random init (untrained)", zorder=3)
b2 = ax.bar(x + w / 2, trained, w, color=PURPLE, label="trained (1500 steps, frozen backbone)", zorder=3)
for bs in (b1, b2):
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x, metrics)
ax.set_ylim(0, 2.5)
ax.legend(frameon=False, loc="upper right")
ax.set_title("Thought-manifold projector: was never trained; training makes geometry semantic\n(math-prompt trajectory's 4 nearest blocks = all 4 math blocks)", loc="left", color=INK, fontsize=10.5)
save(fig, "chart_manifold_training.png")

# 8) S3 bridge: proximity before/after + answer
fig, ax = plt.subplots(figsize=(7.6, 4.2))
style_ax(ax)
names = ["B' (Zorblax→xenon)", "C' (xenon→krypton)"]
bfore = [8.628, 6.121]
after = [8.628, 5.535]
x = np.arange(2); w = 0.36
b1 = ax.bar(x - w / 2, bfore, w, color=FAINT, label="before teaching", zorder=3)
b2 = ax.bar(x + w / 2, after, w, color=BLUE, label="after teaching B'/C'", zorder=3)
for bs in (b1, b2):
    for b in bs:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}", ha="center", fontsize=10, fontweight="bold", color=INK)
ax.set_xticks(x, names)
ax.set_ylabel("min trajectory-block distance (lower=closer)", color=SUB)
ax.legend(frameon=False)
ax.set_title("S3 bridge: teaching only B'/C' lets the model answer D ('krypton') via injection;\nreasoning trajectory moves closer to the taught blocks", loc="left", color=INK, fontsize=10.5)
save(fig, "chart_s3_bridge.png")

# 9) S4 concept neighbors
sims = [("silver", 0.184, 1), ("metal", 0.309, 1), ("iron", 0.334, 1), ("copper", 0.196, 1),
        ("democracy", 0.202, 0), ("banana", 0.261, 0), ("algebra", 0.111, 0), ("window", 0.098, 0)]
fig, ax = plt.subplots(figsize=(8.6, 4.2))
style_ax(ax)
names = [s[0] for s in sims]
vals = [s[1] for s in sims]
colors = [GREEN if s[2] else FAINT for s in sims]
bars = ax.bar(names, vals, color=colors, width=0.56, zorder=3)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, color=INK)
ax.set_ylabel("cosine similarity to 'Xylon' concept slot", color=SUB)
ax.set_title("S4 dynamic vocabulary: Kaplan-extracted concept lands near metals (related 0.256 > unrelated 0.168)", loc="left", color=INK, fontsize=10.5)
save(fig, "chart_s4_neighbors.png")

print("ALL EN CHARTS DONE")
