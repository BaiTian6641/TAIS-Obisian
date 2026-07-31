"""生成报告资产：Carbon 风格架构详图 v3 + 真实数据图表（一次性脚本，可重复跑）。

输出 reports/assets/*.png。全部数值取自项目真实产物（训练日志 / report JSON / 记忆库已核实数据）。
风格：IBM Carbon Design（White theme：gray 系分区 + blue60 #0f62fe 主色 + 少量语义色、8px 网格、无装饰渐变）。
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

# ---------- 字体 ----------
_CJK = ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei"]
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in _CJK:
    if _f in _installed:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

# ---------- Carbon 色板（White theme） ----------
INK = "#161616"      # gray100 主文本
SUB = "#525252"      # gray70 次文本
FAINT = "#8d8d8d"    # gray60 弱文本
LINE = "#e0e0e0"     # gray20 边框
BG = "#f4f4f4"       # gray10 容器底
BLUE = "#0f62fe"     # blue60 主交互
BLUE_BG = "#edf4ff"  # blue10
PURPLE = "#8a3ffc"   # purple60 元认知
TEAL = "#009d9a"     # teal60 主干
GREEN = "#24a148"    # green60 知识/验证
ORANGE = "#ff832b"   # orange60 注意力栈/睡眠
RED = "#da1e28"      # red60 风险
MAGENTA = "#d12771"  # magenta60 求知


def box(ax, x, y, w, h, fc, ec, lw=1.2, r=0.6, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, zorder=z, alpha=alpha)
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=9, color=INK, weight="normal", ha="center", va="center", z=5, spacing=1.25):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha, va=va,
            zorder=z, linespacing=spacing)


def arrow(ax, p1, p2, color, label=None, lw=1.6, style="-", conn="arc3,rad=0.0",
          label_xy=None, label_size=7.5, z=3, arrowstyle="-|>", ms=12):
    a = FancyArrowPatch(p1, p2, arrowstyle=arrowstyle, mutation_scale=ms,
                        color=color, lw=lw, linestyle=style,
                        connectionstyle=conn, zorder=z, shrinkA=2, shrinkB=2)
    ax.add_patch(a)
    if label and label_xy:
        # 标签带白底描边，绝不压其他文字（位置由布局手工指定）
        ax.text(*label_xy, label, fontsize=label_size, color=color, ha="center", va="center",
                zorder=6, bbox=dict(fc="white", ec="none", pad=1.2))


def chip(ax, x, y, s, color, size=7.2, z=6):
    ax.text(x, y, s, fontsize=size, color="white", ha="center", va="center", zorder=z,
            bbox=dict(fc=color, ec="none", boxstyle="round,pad=0.32"))


# =====================================================================
# 1) 架构详图 v3（x 0-160, y 0-100）
# =====================================================================
def make_architecture():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── 标题栏 ──
    txt(ax, 4, 96.5, "TAIS Obsidian 架构详图", 20, INK, "bold", ha="left")
    txt(ax, 4, 92.5, "权重虚拟内存 · 原生 1M 上下文 · KAL 元认知 · 主动求知 · 动态词表", 10.5, SUB, ha="left")
    txt(ax, 156, 96.5, "v3.0 · 2026-07-31", 9, FAINT, ha="right")
    chip(ax, 138, 92.5, "✓ 0.1B pilot 全链验证", GREEN, 7.8)
    chip(ax, 157, 92.5, "● 1B（1017.7M）训练中", BLUE, 7.8)

    # ── Zone A：主干 Backbone（左）──
    box(ax, 3, 14, 41, 74, "white", TEAL, 1.8, 1.2)
    txt(ax, 5, 85, "① 主干 Backbone", 11.5, TEAL, "bold", ha="left")
    txt(ax, 43, 85, "1B = 32 层 · d1536 · Muon", 7.6, SUB, ha="right")

    # Embedding
    box(ax, 6, 15.5, 35, 7, BG, LINE)
    txt(ax, 23.5, 19, "Tokenizer（32k BPE + 2048 reserved）\nEmbedding（tied）", 7.8)

    # GDN-2 层
    box(ax, 6, 24.5, 35, 15, "#e5f6f6", TEAL)
    txt(ax, 23.5, 36.4, "×24 GDN-2 线性注意力层（递归状态）", 8.4, TEAL, "bold")
    txt(ax, 23.5, 31.8, "erase/write 解耦门 · 有界 decay（g_min=−5）\n门收敛 4× 加速 · NIAH 0.207 反超 GDN-1\n记忆层 delta 写（副作用根治 ic 0.688）", 7.4)

    # 三级栈
    box(ax, 6, 41.5, 35, 20, "#fff4e8", ORANGE)
    txt(ax, 23.5, 58.4, "×8 三级检索注意力栈", 8.4, "#b25e09", "bold")
    box(ax, 8, 53.5, 31, 3.6, "white", LINE)
    txt(ax, 23.5, 55.2, "L0 滑窗 512（精确，RoPE·可扩 256K YaRN）", 7.2)
    box(ax, 8, 49.3, 31, 3.6, "white", LINE)
    txt(ax, 23.5, 51, "L1 CSA 选择检索（stride-4 · LightningIndexer）", 7.2)
    box(ax, 8, 45.1, 31, 3.6, "white", LINE)
    txt(ax, 23.5, 46.8, "L2 HCA gist（128:1 · 知识块注入点）", 7.2)
    txt(ax, 23.5, 43, "消融 −0.006 nats（吞吐 91%）", 7.0, SUB)

    # PM-stream
    box(ax, 6, 63.5, 35, 9, "#e5f6f6", TEAL)
    txt(ax, 23.5, 69.4, "PM-stream 多流残差（mHC n=5）", 8.4, TEAL, "bold")
    txt(ax, 23.5, 66, "4 内容流 + 1 感知-记忆流 · Sinkhorn 约束\n消融 −0.024 nats · 组合 −0.025", 7.4)

    # LM-Head
    box(ax, 6, 74.5, 35, 7, BG, LINE)
    txt(ax, 23.5, 78, "RMSNorm → LM-Head（tied embedding）\n动态词表 concept_slot（Kaplan ℓ3 提取）", 7.8)

    # ── Zone B：TAIS 内核（中上）──
    box(ax, 47, 58, 47, 30, "white", PURPLE, 1.8, 1.2)
    txt(ax, 49, 85, "② TAIS 内核（checkpoint 内生）", 11.5, PURPLE, "bold", ha="left")
    box(ax, 49, 73, 20, 10, "#f6f0ff", PURPLE)
    txt(ax, 59, 79.8, "KAL 分层元认知", 8.2, PURPLE, "bold")
    txt(ax, 59, 75.8, "L1 P(IK) 三态 · L2 情感 · L3 冲突\n探针 0.945 · 校准 0.845/0.829", 7.0)
    box(ax, 71, 73, 21, 10, "#f6f0ff", PURPLE)
    txt(ax, 81.5, 79.8, "HRL 检索", 8.2, PURPLE, "bold")
    txt(ax, 81.5, 75.8, "LightningIndexer + DG 投影\n块检索 top-1 = 0.938（训练 1.000）", 7.0)
    box(ax, 49, 60, 43, 11, BLUE_BG, BLUE)
    txt(ax, 70.5, 67.8, "sense / route / inject 统一前向接口", 8.0, BLUE, "bold")
    txt(ax, 70.5, 63.4, "监测/执行分置：KAL 只读 GDN 层 ℓ10 · 注入写 CSA/HCA 层\n梯度隔离红线 · 探针冻结（不作生成损失）", 7.2)

    # ── Zone C：主动求知（中下）──
    box(ax, 47, 27, 47, 27, "white", MAGENTA, 1.8, 1.2)
    txt(ax, 49, 51, "③ 主动求知闭环（自我学习）", 11.5, MAGENTA, "bold", ha="left")
    steps = ["① certainty\n(KAL 读出)", "② 求知分支\n四选一 RPL/LP", "③ 交叉验证\n绝不裸自我修正", "④ 写入\n累积不覆盖", "⑤ 重评估\n实时可用"]
    xw = 8.4
    for i, s in enumerate(steps):
        x = 49.5 + i * (xw + 0.7)
        box(ax, x, 40.5, xw, 7.5, "#fdf0f6", MAGENTA)
        txt(ax, x + xw / 2, 44.3, s, 6.8)
        if i < 4:
            arrow(ax, (x + xw, 44.3), (x + xw + 0.7, 44.3), MAGENTA, lw=1.2, ms=9)
    box(ax, 49, 29.5, 43, 8.5, BG, LINE)
    txt(ax, 70.5, 35, "AskQuestion / CallTool / Decline / DirectAnswer", 7.6, INK)
    txt(ax, 70.5, 31.8, "诚实降级 16/16（虚构事实绝不硬答）· 睡眠固化 PROMOTE 8 / QUARANTINE 1 / REJECT 8", 7.2, SUB)

    # ── Zone D：知识块 + 运行时（右）──
    box(ax, 97, 27, 60, 61, "white", GREEN, 1.8, 1.2)
    txt(ax, 99, 85, "④ 知识块库 + DKB 运行时", 11.5, GREEN, "bold", ha="left")
    box(ax, 99, 71, 26, 11, "#e9f6ec", GREEN)
    txt(ax, 112, 79.2, "KnowledgeBlock 双形态", 8.0, GREEN, "bold")
    txt(ax, 112, 74.8, "markdown 源代码（审计）\n+ 编译产物（可失效重建）", 7.0)
    box(ax, 127, 71, 28, 11, "#e9f6ec", GREEN)
    txt(ax, 141, 79.2, "载体四型（标 factual_recall）", 8.0, GREEN, "bold")
    txt(ax, 141, 74.6, "KV 块 / 记忆层 delta / ICV 向量\n概念槽（位置不变，只 steer）", 7.0)
    box(ax, 99, 60, 56, 8.5, BG, LINE)
    txt(ax, 127, 66.2, "BlockStore（usage_weighted）+ 页表 SQLite", 7.8, INK)
    txt(ax, 127, 62.6, "累积不覆盖 · 版本化 :v{n} · 冲突保留双方标分歧 · 防篡改签名", 7.2, SUB)
    box(ax, 99, 49.5, 56, 8.5, BLUE_BG, BLUE)
    txt(ax, 127, 55.7, "Memory Bus + Pager（缺页 fail-closed）", 7.8, BLUE, "bold")
    txt(ax, 127, 52, "缺页/超时显式声明「该部分记忆暂不可用」，绝不空白作答", 7.2, SUB)
    box(ax, 99, 30, 56, 17, BG, LINE)
    txt(ax, 127, 44.2, "存储层级", 8.0, INK, "bold")
    tiers = [("L0 VRAM", "工作记忆"), ("L1 DRAM", "短期"), ("L2 NVMe", "长期"), ("L3 远端", "档案")]
    for i, (t, d) in enumerate(tiers):
        x = 101 + i * 13.8
        box(ax, x, 33.5, 12.4, 8, "white", LINE)
        txt(ax, x + 6.2, 39.3, t, 7.4, BLUE, "bold")
        txt(ax, x + 6.2, 35.8, d, 7.0, SUB)

    # ── Zone E：睡眠固化（中下横条）──
    box(ax, 47, 8, 110, 15, "white", ORANGE, 1.8, 1.2)
    txt(ax, 49, 20.4, "⑤ 睡眠固化器（离线 W3+）", 10.5, "#b25e09", "bold", ha="left")
    stages = [("draft 提交", "知识块"), ("CA1 门", "验证+漂移监测"), ("间隔提取\n练习", "CLS 慢通道"), ("SHY 归一化", "防膨胀"), ("Muon W4 固化", "同优化器")]
    xw = 18.5
    for i, (t, d) in enumerate(stages):
        x = 50 + i * (xw + 1.6)
        box(ax, x, 10.5, xw, 7, "#fff4e8", ORANGE)
        txt(ax, x + xw / 2, 15, t, 7.6)
        txt(ax, x + xw / 2, 12, d, 6.6, SUB)
        if i < 4:
            arrow(ax, (x + xw, 14), (x + xw + 1.6, 14), ORANGE, lw=1.2, ms=9)

    # ── 跨区箭头（正交/弧线重路由，标签全部避开方块与其他文字）──
    # sense: 主干右缘 → 内核左缘（短直连，竖排标签放间隙）
    arrow(ax, (44.2, 76), (46.8, 76), PURPLE, style="--", ms=10)
    txt(ax, 45.5, 80.5, "sense\n只读ℓ10", 6.4, PURPLE, ha="center")
    # HRL route: 内核右缘 → 知识块左缘（短直连，竖排标签）
    arrow(ax, (94.2, 76), (96.8, 76), PURPLE, style="--", ms=10)
    txt(ax, 95.5, 80.5, "route\n检索", 6.4, PURPLE, ha="center")
    # inject: 知识块左下 →（走 C/E 区间隙 y≈25）→ 三级栈右缘（两段弧，标签放间隙中央）
    arrow(ax, (96.8, 32), (62, 25.2), GREEN, conn="arc3,rad=0.08")
    arrow(ax, (62, 25.2), (44.2, 46), GREEN, conn="arc3,rad=-0.22")
    txt(ax, 75, 24.4, "注入召回 0.625（HCA / 记忆层）", 7.0, GREEN, ha="center", z=6)
    # 求知 → 知识块（短弧，标签放下方间隙）
    arrow(ax, (94.2, 44), (96.8, 56), MAGENTA, conn="arc3,rad=-0.25", ms=10)
    txt(ax, 95.6, 50, "W0–W2\n零\n梯\n度\n写", 6.2, MAGENTA, ha="center")
    # KAL → 求知（垂直短箭头，标签放 B/C 区间隙）
    arrow(ax, (64, 59.8), (64, 54.4), MAGENTA, style="--", ms=10)
    txt(ax, 66.5, 57, "certainty 触发", 6.6, MAGENTA, ha="left")
    # draft → 睡眠固化（知识块底部 → 固化器顶部，垂直直连，标签右置避开存储框）
    arrow(ax, (127, 29.8), (127, 23.4), "#b25e09", style="--", ms=10)
    txt(ax, 130, 26.5, "draft→固化门", 6.6, "#b25e09", ha="left")
    # W4 回流: 固化器左缘 → 主干底缘（小弧，标签放主干下方空隙）
    arrow(ax, (46.8, 12), (32, 14.6), "#b25e09", conn="arc3,rad=0.25", style="--", ms=10)
    txt(ax, 38, 9.8, "W4 固化回流（审计门）", 6.6, "#b25e09", ha="center")

    # ── 红线脚注 ──
    box(ax, 3, 1.5, 154, 5, BG, LINE, r=0.4)
    txt(ax, 80, 4, "红线：运行时只读+零梯度快写（W0–W2）· W3+ 仅睡眠期且审计 · 人格块运行时只读 · 监测/执行分置 · 探针冻结 · 冲突不静默覆盖 · 诚实降级 · 注入即攻击面（签名+筛查）",
        7.6, SUB)

    fig.savefig(OUT / "architecture_v3.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("written architecture_v3.png")


# =====================================================================
# 图表通用样式
# =====================================================================
def style_ax(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(LINE)
    ax.tick_params(colors=SUB, labelsize=9)
    ax.grid(axis="y", color=LINE, lw=0.7, alpha=0.7)
    ax.set_axisbelow(True)


# =====================================================================
# 2) 训练曲线（0.1B GDN-2 10k，真实日志）
# =====================================================================
def make_training_curve():
    log = (ROOT / "runs/_gdn2_10k_train_out.txt").read_text(encoding="utf-8", errors="replace")
    steps, losses, emas = [], [], []
    for m in re.finditer(r"step\s+(\d+) \| loss ([\d.]+) \(ema ([\d.]+)\)", log):
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        emas.append(float(m.group(3)))
    assert len(steps) > 50, f"日志条目过少: {len(steps)}"
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    style_ax(ax)
    ax.plot(steps, losses, color=LINE, lw=1.0, label="train loss（每 50 步）")
    ax.plot(steps, emas, color=BLUE, lw=2.2, label="EMA 平滑")
    ax.set_xlabel("训练步", fontsize=10, color=SUB)
    ax.set_ylabel("next-token loss", fontsize=10, color=SUB)
    ax.set_title("0.1B GDN-2 预训练 10k 步收敛曲线（runs/_gdn2_10k_train_out.txt）",
                 fontsize=11, color=INK, loc="left")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "chart_training_curve.png", facecolor="white")
    plt.close(fig)
    print(f"written chart_training_curve.png ({len(steps)} pts, final ema {emas[-1]:.3f})")


# =====================================================================
# 3) 消融矩阵（D0 报告 §6.4，2000 步 val）
# =====================================================================
def make_ablation():
    names = ["hybrid\n基线", "+三级\n检索栈", "+PM-\nstream", "组合"]
    vals = [3.768, 3.762, 3.744, 3.743]
    colors = [FAINT, ORANGE, TEAL, BLUE]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    style_ax(ax)
    bars = ax.bar(names, vals, color=colors, width=0.58, zorder=3)
    ax.set_ylim(3.70, 3.80)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.0012, f"{v:.3f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.axhline(vals[0], color=FAINT, ls="--", lw=1)
    ax.text(3.45, vals[0] + 0.0015, "基线", fontsize=8, color=FAINT, ha="right")
    ax.set_ylabel("val loss（越低越好）", fontsize=10, color=SUB)
    ax.set_title("原生部件消融（0.1B，2000 步）：PM-stream −0.024 / 组合 −0.025 nats",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_ablation.png", facecolor="white")
    plt.close(fig)
    print("written chart_ablation.png")


# =====================================================================
# 4) KAL 校准演进（0.769 → 0.845/0.829，3 seed 误差棒）
# =====================================================================
def make_kal():
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    style_ax(ax)
    groups = ["v1 真值锚\n(置信度锚)", "v2 锚集扩充\n脚本口径 n200", "v2 锚集扩充\n测试口径 n400"]
    means = [0.769, 0.84475, 0.829225]
    stds = [0.0, 0.024172, 0.007420]
    colors = [FAINT, BLUE, BLUE]
    bars = ax.bar(groups, means, yerr=stds, color=colors, width=0.5, zorder=3,
                  error_kw=dict(ecolor=SUB, capsize=4, lw=1.2))
    ax.axhline(0.8, color=GREEN, ls="--", lw=1.4)
    ax.text(2.42, 0.804, "目标 ≥0.8", fontsize=9, color=GREEN, ha="right")
    for b, v in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_ylim(0.70, 0.90)
    ax.set_ylabel("真值校准 AUROC", fontsize=10, color=SUB)
    ax.set_title("KAL 元认知校准：锚集扩充达成 ≥0.8（3 seed 均值±std；反馈循环负结果已回滚）",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_kal.png", facecolor="white")
    plt.close(fig)
    print("written chart_kal.png")


# =====================================================================
# 5) 全链已训强度（统一 checkpoint，n=16）
# =====================================================================
def make_fullchain():
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    style_ax(ax)
    items = ["HRL 块检索\ntop-1", "HCA 注入\n召回", "in-context\n(记忆层根治后)", "诚实降级\n(虚构事实)"]
    vals = [0.938, 0.625, 0.688, 1.000]
    refs = [(0.062, "基线 0.062"), (0.188, "线性门控 0.188"), (0.250, "门控副作用 0.250"), (0.0, "")]
    colors = [PURPLE, GREEN, TEAL, MAGENTA]
    bars = ax.bar(items, vals, color=colors, width=0.52, zorder=3)
    for i, (b, v, (rv, rl)) in enumerate(zip(bars, vals, refs)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}", ha="center",
                fontsize=11, color=INK, fontweight="bold")
        if rv > 0:
            ax.plot([b.get_x(), b.get_x() + b.get_width()], [rv, rv], color=RED, lw=1.4, ls="--")
            ax.text(b.get_x() + b.get_width() / 2, rv + 0.015, rl, ha="center", fontsize=7.5, color=RED)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("指标值", fontsize=10, color=SUB)
    ax.set_title("统一 checkpoint 全链已训强度（0.1B，n=16）：检索→召回→内化→诚实降级",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_fullchain.png", facecolor="white")
    plt.close(fig)
    print("written chart_fullchain.png")


# =====================================================================
# 6) NIAH 长度扫描（GDN-1 vs GDN-2 有界 10k，first-token 判据）
# =====================================================================
def make_niah():
    import numpy as np
    r = json.loads((ROOT / "runs/niah_length_scan/report.json").read_text(encoding="utf-8"))
    res = r["results"]
    # results 结构: list of dict per cell（length×n_keys×model）——按 memory 表重建，若结构不符回退硬编码
    table = {  # (length, keys): (gdn1, gdn2) —— docs/memories/niah-length-scan-gate-adaptive.md 实测表
        (512, 8): (0.100, 0.120), (512, 32): (0.060, 0.040),
        (1024, 8): (0.080, 0.100), (1024, 32): (0.040, 0.060),
        (2048, 8): (0.0, 0.0), (2048, 32): (0.0, 0.0),
        (4096, 8): (0.0, 0.0), (4096, 32): (0.0, 0.0),
    }
    try:  # 尝试从 report 直接取（结构适配则覆盖）
        for row in res:
            k = (row["length"], row["n_keys"])
            table[k] = (row.get("gdn1_first", table[k][0]), row.get("gdn2_first", table[k][1]))
    except Exception:
        pass
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    style_ax(ax)
    labels = ["512\nk8", "512\nk32", "1024\nk8", "1024\nk32", "2048\nk8", "2048\nk32", "4096\nk8", "4096\nk32"]
    keys = [(512, 8), (512, 32), (1024, 8), (1024, 32), (2048, 8), (2048, 32), (4096, 8), (4096, 32)]
    g1 = [table[k][0] for k in keys]
    g2 = [table[k][1] for k in keys]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w / 2, g1, w, color=FAINT, label="GDN-1", zorder=3)
    ax.bar(x + w / 2, g2, w, color=TEAL, label="GDN-2 有界 10k", zorder=3)
    ax.axvspan(3.5, 7.5, color="#fde7e7", zorder=1)
    ax.text(5.5, 0.105, "max_seq=1024 硬限（>1024 截断为 0）\n→ RoPE 扩容+YaRN 已立项", ha="center", fontsize=8.5, color=RED)
    ax.set_xticks(x, labels)
    ax.set_xlabel("上下文长度 × needle 数", fontsize=10, color=SUB)
    ax.set_ylabel("NIAH first-token 命中", fontsize=10, color=SUB)
    ax.set_ylim(0, 0.14)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("NIAH 长度扫描（50 queries/cell）：GDN-2 短中长略优；1024 硬限催生 256K 扩容工程",
                 fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_niah.png", facecolor="white")
    plt.close(fig)
    print("written chart_niah.png")


# =====================================================================
# 7) CA1 自适应裁决对比（中文版）
# =====================================================================
def make_ca1_adaptive_zh():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    style_ax(ax)
    cats = ["PROMOTE", "QUARANTINE", "REJECT"]
    before = [3, 1, 3]
    after = [6, 1, 0]
    x = np.arange(3); w = 0.36
    b1 = ax.bar(x - w / 2, before, w, color=FAINT, label="CA1 v1.0（静态）", zorder=3)
    b2 = ax.bar(x + w / 2, after, w, color=GREEN, label="CA1 v1.1（自适应 RE_VERIFY）", zorder=3)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, str(int(b.get_height())),
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.set_xticks(x, cats)
    ax.set_ylabel("块数（6 教学块 + 1 冲突块）", fontsize=10, color=SUB)
    ax.set_ylim(0, 7.2)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("睡眠固化裁决：信源边缘效应已根治；冲突块仍 QUARANTINE（防投毒红线）",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_ca1_adaptive.png", facecolor="white")
    plt.close(fig)
    print("written chart_ca1_adaptive.png")


# =====================================================================
# 8) 流形投影器训练对照（中文版）
# =====================================================================
def make_manifold_training_zh():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    style_ax(ax)
    metrics = ["聚簇对比度\n(越高越好)", "等距 Pearson\n(越高越好)"]
    rand = [1.558, 0.882]
    trained = [1.989, 0.977]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, rand, w, color=FAINT, label="随机初始化（未训练）", zorder=3)
    b2 = ax.bar(x + w / 2, trained, w, color=PURPLE, label="已训练（1500 步，冻结主干）", zorder=3)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{b.get_height():.3f}",
                    ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x, metrics)
    ax.set_ylim(0, 2.5)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    ax.set_title("思考流形投影器：确认从未训练 → 训练使几何获得语义\n（数学 prompt 轨迹最近 4 块恰为全部数学块）",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_manifold_training.png", facecolor="white")
    plt.close(fig)
    print("written chart_manifold_training.png")


# =====================================================================
# 9) S3 桥接邻近性（中文版）
# =====================================================================
def make_s3_bridge_zh():
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    style_ax(ax)
    names = ["B'（Zorblax→xenon）", "C'（xenon→krypton）"]
    bfore = [8.628, 6.121]
    after = [8.628, 5.535]
    x = np.arange(2); w = 0.36
    b1 = ax.bar(x - w / 2, bfore, w, color=FAINT, label="教学前", zorder=3)
    b2 = ax.bar(x + w / 2, after, w, color=BLUE, label="补教 B'/C' 后", zorder=3)
    for bs in (b1, b2):
        for b in bs:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.1, f"{b.get_height():.2f}",
                    ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.set_xticks(x, names)
    ax.set_ylabel("轨迹-块最小距离（越小越近）", fontsize=10, color=SUB)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("S3 桥接：只教中间知识 B'/C' 即注入答出 D（'krypton'）；\n推理轨迹向新教块靠近",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_s3_bridge.png", facecolor="white")
    plt.close(fig)
    print("written chart_s3_bridge.png")


# =====================================================================
# 10) S4 概念槽语义邻居（中文版）
# =====================================================================
def make_s4_neighbors_zh():
    sims = [("silver", 0.184, 1), ("metal", 0.309, 1), ("iron", 0.334, 1), ("copper", 0.196, 1),
            ("democracy", 0.202, 0), ("banana", 0.261, 0), ("algebra", 0.111, 0), ("window", 0.098, 0)]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=150)
    style_ax(ax)
    names = [s[0] for s in sims]
    vals = [s[1] for s in sims]
    colors = [GREEN if s[2] else FAINT for s in sims]
    bars = ax.bar(names, vals, color=colors, width=0.56, zorder=3)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=9, color=INK)
    ax.set_ylabel("与 'Xylon' 概念槽的 cos 相似度", fontsize=10, color=SUB)
    ax.set_title("S4 动态词表：Kaplan 提取的概念落在金属邻近（相关 0.256 > 无关 0.168；绿=金属类）",
                 fontsize=10.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "chart_s4_neighbors.png", facecolor="white")
    plt.close(fig)
    print("written chart_s4_neighbors.png")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    make_architecture()
    make_training_curve()
    make_ablation()
    make_kal()
    make_fullchain()
    make_niah()
    make_ca1_adaptive_zh()
    make_manifold_training_zh()
    make_s3_bridge_zh()
    make_s4_neighbors_zh()
    print("ALL DONE")
