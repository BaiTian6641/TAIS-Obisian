#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAIS Obsidian 架构详图生成脚本（v2.2 / 2026-07-29）

用途：生成 IBM Carbon Design Language 的 TAIS Obsidian 单页架构蓝图 PNG。
运行：python3 TAIS_Obsidian_架构图_生成脚本_v2.py
依赖：matplotlib（Agg，无需 GPU）；中文显示需系统装有中文字体（自动回退）
输出：docs/TAIS_Obsidian_架构详图.png（相对脚本所在目录）

内容基准（当前真实架构，禁止照抄 v0.6 旧版）：
  主干：12 层 = 3×{3 GDN-2 + 1 TriRetrievalAttention}（G2G2G2A，d_model=768）
        GDN-2 erase/write 解耦 + decay 有界 sigmoid（g_min=-5）；目标 1.5B = 28 层 7×{3+1}
        三级栈 = 滑窗 L0(512) + CSA L1(stride-4, LightningIndexer top-k) + HCA gist L2(128:1)
        PM-stream = mHC n=5 多流残差（4 内容流 + 1 感知-记忆流，arXiv:2512.24880）
  元认知/检索：KAL 三层（L1 P(IK) 真值锚+isotonic / L2 情感 VA / L3 冲突）
        HRL（LightningIndexer + CA3 PPR 联想）｜知识块 BlockStore（累积不覆盖）
        TAIS 内核（sense 只读 GDN / inject 写 CSA，监测/执行分置）｜睡眠固化（CA1 门+间隔提取+SHY）
  第二阶段：ThoughtManifold(manifold_dim=64) + ManifoldBridge(tick 闭环)
        ThoughtCore(CTM 式：通道组历史+RoPE 相位+certainty 早停) + ReasoningLoop(§1.3 五步 tick)
        CoT 投影层(投影非计算+忠实性审计) + thought_visualizer(3D 轨迹+坏路径四类)
        路径积分辅助任务(网格码诱导)
  主动求知：InquiryBranch(四选一 RPL/LP) → InquiryExecutor(Ask/CallTool→CrossVerifier
        交叉验证[绝不裸自我修正]→KnowledgeBlockWriter[累积不覆盖]→重评估闭环)

坐标系：x 0–100，y 0–176（顶部标题栏 → 四分区 → 底部数据流/睡眠固化条带）。
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle, Ellipse
from matplotlib import font_manager

# ---------- 中文字体自动回退（按平台常见字体依次探测） ----------
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Source Han Sans SC",
    "PingFang SC", "Microsoft YaHei", "SimHei",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Arial Unicode MS",
]
_installed = {f.name for f in font_manager.fontManager.ttflist}
for _f in _CJK_CANDIDATES:
    if _f in _installed:
        plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
        break
else:
    print("警告：未找到中文字体，请先安装 Noto Sans CJK SC / 微软雅黑 / 苹方 等任一中文字体")
plt.rcParams["axes.unicode_minus"] = False

# ---------- Carbon 调色板 ----------
INK = "#161616"; GRAY6 = "#6f6f6f"; GRAY4 = "#c6c6c6"; GRAY0 = "#f4f4f4"
BLUE = "#0f62fe"; BLUE_T = "#edf5ff"; TEAL = "#009d9a"; TEAL_T = "#d9fbfb"
PURPLE = "#8a3ffc"; PURPLE_T = "#f6f2ff"; GREEN = "#24a148"; GREEN_T = "#defbe6"
ORANGE = "#ff832b"; ORANGE_T = "#fff2e8"; MAGENTA = "#ee5396"; MAGENTA_T = "#fff0f7"

# ---------- 画布 ----------
fig, ax = plt.subplots(figsize=(20, 35), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(0, 176); ax.axis("off")
fig.patch.set_facecolor("white")


def tw(text):
    """估算文本宽度（CJK 与 ASCII 分别计宽，Carbon 版式用）。"""
    return sum(1.9 if ord(c) > 0x2E80 else 0.95 for c in text)


def tag(x, y, text, color, fs=11):
    """Carbon 式实心面板标签（左上角）。"""
    w = tw(text) + 2.4
    ax.add_patch(Rectangle((x, y), w, 2.6, fc=color, ec="none", zorder=4))
    ax.text(x + 1.2, y + 1.3, text, fontsize=fs, color="white",
            fontweight="bold", va="center", zorder=5)


def panel(x, y, w, h, label, color, tag_bottom=False):
    """带左侧色条的面板容器。"""
    ax.add_patch(Rectangle((x, y), w, h, fc=GRAY0, ec=GRAY4, lw=1.0, zorder=1))
    ax.add_patch(Rectangle((x, y), 0.9, h, fc=color, ec="none", zorder=2))
    if label:
        tag(x + 0.9, y + 0.5 if tag_bottom else y + h - 2.6, label, color)


def cbox(x, y, w, h, t, ec, fc="white", tc=INK, fs=9, sub=None, tfs=7.0):
    """内容框（标题 + 可选副标题）。"""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.1, zorder=3))
    if sub:
        ax.text(x + w / 2, y + h - 1.5, t, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold", zorder=4)
        ax.text(x + w / 2, y + 1.2, sub, ha="center", va="center",
                fontsize=tfs, color=GRAY6, zorder=4)
    else:
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center",
                fontsize=fs, color=tc, fontweight="bold", zorder=4)


def arr(x1, y1, x2, y2, color=INK, lw=1.2, style="-|>", ls="-"):
    """箭头（直线；调用方负责端口点，避免穿框）。"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                 mutation_scale=11, color=color, lw=lw, linestyle=ls,
                 zorder=5, shrinkA=0, shrinkB=0))


def note(x, y, text, fs=8, color=GRAY6, ha="left", bold=False, rot=0):
    ax.text(x, y, text, fontsize=fs, color=color, ha=ha, va="center",
            rotation=rot, fontweight="bold" if bold else "normal", zorder=6)


def plus(x, y, r, fc):
    """残差/注入节点（圆圈内加号）。"""
    ax.add_patch(Circle((x, y), r, fc=fc, ec="white", lw=1.2, zorder=6))
    ax.plot([x - r * 0.55, x + r * 0.55], [y, y], color="white", lw=1.7, zorder=7)
    ax.plot([x, x], [y - r * 0.55, y + r * 0.55], color="white", lw=1.7, zorder=7)


def gate_box(x, y, w, h, label, ec, fc="white", fs=6.5):
    """小门/参数框（erase/write/decay 等）。"""
    ax.add_patch(Rectangle((x, y), w, h, fc=fc, ec=ec, lw=1.0, zorder=3))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=INK, fontweight="bold", zorder=4)


def mini_arrow(x1, y1, x2, y2, color=INK, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                 mutation_scale=8, color=color, lw=lw, zorder=4,
                 shrinkA=0, shrinkB=0))


# ================= 标题栏 =================
ax.add_patch(Rectangle((0, 169), 100, 7, fc=INK, ec="none", zorder=2))
ax.text(3, 173.4, "TAIS Obsidian 架构详图", fontsize=24, color="white",
        fontweight="bold", va="center", zorder=5)
ax.text(3, 170.9, "权重虚拟内存 × 原生 1M 上下文 × 元认知 × 思维能力强化 × 主动求知 ｜ 当前 = 0.1B pilot 全落地",
        fontsize=10.5, color=GRAY4, va="center", zorder=5)
ax.text(97, 173.4, "v2.2", fontsize=12, color=GRAY4, va="center", ha="right", zorder=5)
ax.text(97, 170.9, "2026-07-29", fontsize=9, color=GRAY4, va="center", ha="right", zorder=5)

# ================= 分区① 主干 Backbone（左上，最高） =================
PX, PY, PW, PH = 2.5, 62, 37, 105
panel(PX, PY, PW, PH, "① 主干 Backbone（0.1B pilot）", INK)
note(PX + 1.4, PY + PH - 5.4, "12 层 = 3 × {3 GDN-2 + 1 TriRetrievalAttention}（G2G2G2A 循环）｜ d_model=768",
     fs=8, color=INK, bold=True)
note(PX + 1.4, PY + PH - 7.6, "目标 1.5B = 28 层 = 7 × {3 GDN + 1 三级栈}",
     fs=7.5, color=GRAY6)

# --- 层组容器（三层循环，画 1 组示意） ---
GX, GW = PX + 2.5, 31

# --- Tokenizer / Embedding（底部，与层组同宽对齐） ---
cbox(PX + 3, PY + 2.5, 31, 3.2, "Tokenizer", INK, fs=9.5)
cbox(PX + 3, PY + 7.3, 31, 3.6, "Embedding", INK, fs=9.5,
     sub="vocab 129280 = 127232 + 2048 reserved ｜ tied", tfs=6.2)
bcx = GX + GW / 2
arr(bcx, PY + 5.7, bcx, PY + 7.3, lw=1.4)

# --- PM-stream 导轨（主干区最右侧竖条，避开顶部输出头） ---
pmx = PX + PW - 2.6
ax.add_patch(Rectangle((pmx, PY + 12), 1.8, 86, fc=PURPLE_T, ec=PURPLE, lw=1.2, zorder=2))
note(pmx + 0.9, PY + 57, "PM-stream（mHC n=5 多流残差）", fs=7, color=PURPLE, rot=90,
     ha="center", bold=True)
for i in range(5):
    ax.plot([pmx + 0.22 + i * 0.34] * 2, [PY + 13, PY + 97],
            color=PURPLE, lw=0.8, alpha=0.55, zorder=2)
note(pmx - 0.6, PY + 92, "4 内容流 + 1 感知-记忆流", fs=6.2, color=PURPLE, ha="right", rot=90)
for yy in (PY + 34, PY + 56, PY + 78):
    ax.add_patch(Circle((pmx + 0.9, yy), 0.42, fc=PURPLE, ec="white", lw=0.8, zorder=6))

# GDN-2 ×3 子面板
panel(GX, PY + 12, GW, 50, "3 × GDN-2（erase/write 解耦）", TEAL)
for gi in range(3):
    gy = PY + 14 + gi * 15.5
    # RMSNorm 顶条
    cbox(GX + 1.5, gy + 11.5, GW - 3, 2.6, f"GDN-2 #{gi+1}：RMSNorm", INK, fs=7.5)
    # 内部结构行：q/k/v → erase/write gate → decay → 递归状态 S
    qx = GX + 1.5
    cbox(qx, gy + 6.6, 7.5, 4.4, "q/k/v 投影", TEAL, fc=TEAL_T, fs=7)
    for k, lab in enumerate(("q", "k", "v")):
        ccx = qx + 1.4 + k * 2.4
        ax.add_patch(Circle((ccx, gy + 7.6), 0.4, fc="white", ec=TEAL, lw=1.0, zorder=4))
        ax.text(ccx, gy + 7.6, lab, ha="center", va="center", fontsize=6.2,
                color=TEAL, fontweight="bold", zorder=5)
    gate_box(GX + 10, gy + 8.2, 4.4, 2.6, "erase β", TEAL, fs=6.5)
    gate_box(GX + 10, gy + 5.2, 4.4, 2.6, "write w", TEAL, fs=6.5)
    gate_box(GX + 15.4, gy + 6.7, 5.2, 2.6, "decay σ 有界", TEAL, fc=TEAL_T, fs=6.3)
    cbox(GX + 21.6, gy + 6.2, 6.8, 3.6, "递归状态 S", TEAL, fc=TEAL_T, fs=7)
    mini_arrow(GX + 9, gy + 8.8, GX + 10, gy + 9.2, TEAL)
    mini_arrow(GX + 14.4, gy + 9.2, GX + 15.4, gy + 8.6, TEAL)
    mini_arrow(GX + 14.4, gy + 6.2, GX + 15.4, gy + 7.4, TEAL)
    mini_arrow(GX + 20.6, gy + 8.0, GX + 21.6, gy + 8.0, TEAL)
    # 残差 + FFN 底行
    plus(GX + 3.2, gy + 3.2, 0.9, TEAL)
    arr(GX + 3.2, gy + 4.1, GX + 3.2, gy + 6.2, lw=1.0)
    cbox(GX + 5.5, gy + 2.2, GW - 7, 2.6, "FFN", INK, fc=GRAY0, fs=7.5)
    arr(GX + 4.1, gy + 3.2, GX + 5.5, gy + 3.2, lw=0.9)
    if gi < 2:
        arr(GX + 3.2, gy + 14.1, GX + 3.2, gy + 15.5, lw=1.0)
note(GX + 1.5, PY + 12.8, "decay = 有界 sigmoid（g_min=−5，K3 借鉴，4× 门收敛加速）｜ NIAH 0.240 反超 GDN-1",
     fs=6.2, color=TEAL, bold=True)

# TriRetrievalAttention ×1 子面板
panel(GX, PY + 64, GW, 34, "1 × TriRetrievalAttention 三级栈", ORANGE)
tw2 = GW - 3
cbox(GX + 1.5, PY + 93.5, tw2, 2.8, "RMSNorm", INK, fs=7.5)
# 三级堆叠 L0/L1/L2
cbox(GX + 1.5, PY + 89, tw2, 3.8, "L0 滑窗注意力 512", ORANGE, fc=ORANGE_T, fs=7.5)
for k, lab in enumerate(("Q", "K", "V")):
    ccx = GX + 23.5 + k * 2.2
    ax.add_patch(Circle((ccx, PY + 90.9), 0.42, fc="white", ec=ORANGE, lw=1.0, zorder=4))
    ax.text(ccx, PY + 90.9, lab, ha="center", va="center", fontsize=6.2,
            color=ORANGE, fontweight="bold", zorder=5)
cbox(GX + 1.5, PY + 84, tw2, 3.8, "L1 CSA 选择检索（stride-4 + top-k）", ORANGE, fc=ORANGE_T, fs=7.5)
gate_box(GX + 22.5, PY + 84.6, 6.8, 2.6, "LightningIndexer", ORANGE, fs=6.2)
cbox(GX + 1.5, PY + 79, tw2, 3.8, "L2 HCA gist（128:1 压缩）", ORANGE, fc=ORANGE_T, fs=7.5)
gate_box(GX + 22.5, PY + 79.6, 6.8, 2.6, "inject_hca 块落点", ORANGE, fc=ORANGE_T, fs=6.2)
# NSA 式门控融合
cbox(GX + 1.5, PY + 74, 8, 4, "门控融合", ORANGE, fs=7.5)
mini_arrow(GX + 5, PY + 84, GX + 5, PY + 78.2, ORANGE)
mini_arrow(GX + 6.5, PY + 89, GX + 6.5, PY + 78.2, ORANGE)
mini_arrow(GX + 8, PY + 93.5, GX + 8, PY + 78.2, ORANGE)
mini_arrow(GX + 3.5, PY + 79, GX + 4.2, PY + 78.2, ORANGE)
plus(GX + 4, PY + 71.6, 1.0, ORANGE)
arr(GX + 4, PY + 72.6, GX + 4, PY + 74, lw=1.0)
cbox(GX + 11, PY + 69, tw2 - 9.5, 3.2, "FFN", INK, fc=GRAY0, fs=8)
arr(GX + 5, PY + 71.6, GX + 11, PY + 71.6, lw=0.9)
note(GX + 1.5, PY + 65, "NSA 式门控融合 ｜ CSA/HCA 为知识块原生注入层",
     fs=6.2, color=ORANGE, bold=True)

# 顶部输出头
cbox(PX + 3, PY + 99, 24, 3.4, "RMSNorm → LM-Head（tied embedding）", INK, fs=9)
arr(bcx, PY + 98, bcx, PY + 99, lw=1.6)
arr(bcx, PY + 10.9, bcx, PY + 12, lw=1.4)

# ================= 分区② 元认知与检索（右上） =================
QX, QY, QW, QH = 41.5, 96, 56, 71
panel(QX, QY, QW, QH, "② 元认知与检索 KAL · HRL · 知识块 · TAIS 内核", PURPLE)

# KAL 三层
panel(QX + 2, QY + 37, 26, 25, "KAL 分层元认知", MAGENTA)
cbox(QX + 4, QY + 53.5, 22, 4.6, "L1 P(IK) 三态头", MAGENTA, fs=8.5,
     sub="知道/不确定/空白 ｜ 真值锚 + isotonic 校准", tfs=6.3)
cbox(QX + 4, QY + 48, 10.5, 4.2, "L2 情感 VA", MAGENTA, fc=MAGENTA_T, fs=8,
     sub="valence/arousal", tfs=6.2)
cbox(QX + 15.5, QY + 48, 10.5, 4.2, "L3 冲突头", MAGENTA, fs=8,
     sub="记忆冲突检测", tfs=6.2)
cbox(QX + 4, QY + 42, 22, 4.6, "探针组（只读 GDN 层 hidden）", MAGENTA, fc="white", fs=8,
     sub="监测/执行分置：不读自身干预", tfs=6.3)
note(QX + 4, QY + 39.8, "certainty 可靠：AUROC 0.8（真值锚校准）", fs=6.8, color=MAGENTA, bold=True)
note(QX + 4, QY + 37.8, "探针冻结：不对探针信号加生成损失", fs=6.8, color=GRAY6)

# TAIS 内核
panel(QX + 30, QY + 37, 24, 25, "TAIS 内核", BLUE)
cbox(QX + 32, QY + 53.5, 9.5, 4.4, "sense", BLUE, fc=BLUE_T, fs=8.5,
     sub="只读 GDN 层", tfs=6.2)
cbox(QX + 43, QY + 53.5, 9, 4.4, "route", BLUE, fs=8.5)
cbox(QX + 32, QY + 47.5, 9.5, 4.4, "inject", BLUE, fc=BLUE_T, fs=8.5,
     sub="只写 CSA 层", tfs=6.2)
cbox(QX + 43, QY + 47.5, 9, 4.4, "聚合 KAL+HRL", BLUE, fs=7.5)
mini_arrow(QX + 41.5, QY + 55.7, QX + 43, QY + 55.7, BLUE)
mini_arrow(QX + 47.5, QY + 53.5, QX + 47.5, QY + 51.9, BLUE)
mini_arrow(QX + 43, QY + 49.7, QX + 41.5, QY + 49.7, BLUE)
note(QX + 32, QY + 44.6, "读写不同层 → 防探针自激", fs=6.8, color=BLUE, bold=True)
note(QX + 32, QY + 42.6, "运行时仅 W0–W2 写原语", fs=6.8, color=GRAY6)
note(QX + 32, QY + 40.6, "W3+（梯度/合并）仅睡眠期", fs=6.8, color=GRAY6)
note(QX + 32, QY + 38.6, "侧信道头簇读 PM-stream", fs=6.8, color=PURPLE)

# HRL
panel(QX + 2, QY + 14, 26, 21, "HRL 海马路由", TEAL)
cbox(QX + 4, QY + 27, 10.5, 4.6, "LightningIndexer", TEAL, fs=7.5,
     sub="内容寻址 top-k", tfs=6.2)
cbox(QX + 15.5, QY + 27, 10.5, 4.6, "DG 模式分离", TEAL, fc=TEAL_T, fs=8)
cbox(QX + 4, QY + 21, 22, 4.2, "CA3 PPR 联想（知识图 + PPR）", TEAL, fc=TEAL_T, fs=7.5)
cbox(QX + 4, QY + 15.6, 22, 4.2, "页表 Page Table（SQLite）", TEAL, fs=8,
     sub="block_id→L0/L1/L2 ｜ 版本·签名", tfs=6.2)
mini_arrow(QX + 14.5, QY + 29.3, QX + 15.5, QY + 29.3, TEAL)
mini_arrow(QX + 9.7, QY + 27, QX + 9.7, QY + 25.2, TEAL)
mini_arrow(QX + 20.7, QY + 27, QX + 20.7, QY + 25.2, TEAL)
note(QX + 4, QY + 14.2, "HRL 梯度隔离：辅助损失不进主干", fs=6.5, color=TEAL, bold=True)

# 知识块
panel(QX + 30, QY + 14, 24, 21, "知识块 KnowledgeBlock", GREEN)
cbox(QX + 32, QY + 27, 9.5, 4.6, "Header", GREEN, fs=8,
     sub="id·版本·签名", tfs=6.2)
cbox(QX + 42.5, QY + 27, 9.5, 4.6, "源代码 markdown", GREEN, fc=GREEN_T, fs=7.5,
     sub="可审计·可回滚", tfs=6.2)
cbox(QX + 32, QY + 21, 20, 4.2, "编译产物（零梯度记忆栈）", GREEN, fc="white", fs=7.5,
     sub="KV 块 harvest ｜ 记忆层 delta ｜ ICV 向量", tfs=6.2)
cbox(QX + 32, QY + 15.6, 20, 4.2, "BlockStore 块存储", GREEN, fs=8.5,
     sub="usage_weighted ｜ 累积不覆盖", tfs=6.2)
mini_arrow(QX + 36.7, QY + 27, QX + 36.7, QY + 25.2, GREEN)
mini_arrow(QX + 47.2, QY + 27, QX + 47.2, QY + 25.2, GREEN)
mini_arrow(QX + 42, QY + 21, QX + 42, QY + 19.8, GREEN)
note(QX + 32, QY + 14.2, "LoRA 降级为睡眠期可选固化产物", fs=6.5, color=GREEN, bold=True)

# 跨区连线：主干 → 元认知（PM-stream 读出 / 注入写回）
arr(PX + PW - 0.4, PY + 78, QX + 2, QY + 50, color=PURPLE, lw=1.5, ls=(0, (5, 3)))
note(PX + PW + 0.4, PY + 83, "PM-stream 读出（KAL 探针/侧信道）", fs=7, color=PURPLE, bold=True)
arr(QX + 2, QY + 18, PX + PW - 0.4, PY + 74, color=GREEN, lw=1.5, ls=(0, (5, 3)))
note(PX + PW + 0.4, PY + 71, "知识块注入 CSA/HCA（W2）", fs=7, color=GREEN, bold=True)

# ================= 分区③ 第二阶段 思维能力强化（左下） =================
RX, RY, RW, RH = 2.5, 22, 48, 38
panel(RX, RY, RW, RH, "③ 第二阶段：思维能力强化（7 迭代全落地）", BLUE)

# ThoughtManifold（左）
panel(RX + 2, RY + 19, 14, 15, "思考流形 ThoughtManifold", TEAL)
ax.add_patch(Ellipse((RX + 9, RY + 27.5), 9, 4.5, fc=TEAL_T, ec=TEAL, lw=1.2, zorder=3))
for dx, dy in ((-2.5, 0.8), (0.5, -0.6), (2.8, 0.5)):
    ax.add_patch(Circle((RX + 9 + dx, RY + 27.5 + dy), 0.35, fc=TEAL, ec="white", lw=0.6, zorder=4))
mini_arrow(RX + 6.5, RY + 28.3, RX + 9.5, RY + 26.9, TEAL, lw=0.8)
mini_arrow(RX + 9.5, RY + 26.9, RX + 11.8, RY + 28.0, TEAL, lw=0.8)
note(RX + 9, RY + 25.3, "manifold_dim=64", fs=6.5, color=TEAL, ha="center", bold=True)
cbox(RX + 3.5, RY + 20.6, 11, 3, "共享投影 + 共形等距", TEAL, fs=6.5)
note(RX + 3.5, RY + 19.9, "VICReg 去相关 ｜ 坐标可解释", fs=6, color=GRAY6)

# ManifoldBridge（中下，连接流形与 PM-stream）
panel(RX + 17.5, RY + 19, 14, 15, "流形-PM 桥接", PURPLE)
cbox(RX + 19, RY + 29, 11, 3.6, "读 PM 流 → 流形坐标", PURPLE, fc=PURPLE_T, fs=7)
cbox(RX + 19, RY + 24.4, 11, 3.6, "位移 → 反投影 → 有界写回", PURPLE, fc=PURPLE_T, fs=7)
cbox(RX + 19, RY + 20.6, 11, 2.8, "tick 闭环", PURPLE, fs=8)
mini_arrow(RX + 24.5, RY + 29, RX + 24.5, RY + 28, PURPLE)
mini_arrow(RX + 24.5, RY + 24.4, RX + 24.5, RY + 23.4, PURPLE)

# ThoughtCore（右）
panel(RX + 33, RY + 19, 13.5, 15, "思考核 ThoughtCore", ORANGE)
note(RX + 40.7, RY + 31.8, "CTM 式", fs=6.5, color=ORANGE, ha="center", bold=True)
cbox(RX + 34.5, RY + 28.5, 10.5, 3.4, "通道组 + 历史", ORANGE, fc=ORANGE_T, fs=7.5)
cbox(RX + 34.5, RY + 24, 10.5, 3.4, "RoPE 相位化思考时间", ORANGE, fc=ORANGE_T, fs=7)
cbox(RX + 34.5, RY + 20.4, 10.5, 2.8, "certainty 早停", ORANGE, fs=7.5)
mini_arrow(RX + 39.7, RY + 28.5, RX + 39.7, RY + 27.4, ORANGE)
mini_arrow(RX + 39.7, RY + 24, RX + 39.7, RY + 23.2, ORANGE)

# ReasoningLoop（底部横跨，五步 tick 环形）
panel(RX + 2, RY + 3, 32, 14, "推理循环 ReasoningLoop（§1.3 五步 tick）", INK)
steps = [
    ("1 GDN 状态", TEAL),
    ("2 glimpse", TEAL),
    ("3 HRL 提议", TEAL),
    ("4 KAL certainty", MAGENTA),
    ("5 bridge.tick", PURPLE),
]
swx, swy = RX + 3.5, RY + 8
sbx = []
for i, (t, c) in enumerate(steps):
    bx = swx + i * 6.0
    cbox(bx, swy, 5.5, 3.6, t, c, fc="white", fs=6.5)
    sbx.append(bx)
    if i:
        mini_arrow(sbx[i - 1] + 5.5, swy + 1.8, bx, swy + 1.8, INK, lw=1.0)
note(swx, RY + 6.2, "位移写 PM-stream", fs=6, color=PURPLE)
# 环形回连（下方）
arr(sbx[4] + 2.7, swy, sbx[4] + 2.7, RY + 4.6, color=INK, lw=1.1)
arr(sbx[4] + 2.7, RY + 4.6, sbx[0] + 2.7, RY + 4.6, color=INK, lw=1.1)
arr(sbx[0] + 2.7, RY + 4.6, sbx[0] + 2.7, swy, color=INK, lw=1.1)
note(RX + 3.5, RY + 3.6, "tick 循环：certainty 达标即早停退出", fs=6.3, color=GRAY6)
# 桥接位于循环第5步上方
arr(sbx[4] + 2.7, swy + 3.6, RX + 24.5, RY + 19, color=PURPLE, lw=1.1, ls=(0, (4, 2)))

# CoT 投影 + 可视化 + 路径积分（右下三个小件）
cbox(RX + 35.5, RY + 12, 11, 4.5, "CoT 投影层", BLUE, fc=BLUE_T, fs=8,
     sub="投影非计算层 ｜ 忠实性审计", tfs=6.3)
cbox(RX + 35.5, RY + 6.8, 11, 4, "可解释性前端", BLUE, fs=8,
     sub="3D 轨迹 + 坏路径四类检测", tfs=6.3)
cbox(RX + 35.5, RY + 3, 11, 3.2, "路径积分辅助任务", BLUE, fc="white", fs=7.5,
     sub="网格码诱导", tfs=6.5)
mini_arrow(RX + 41, RY + 12, RX + 41, RY + 10.8, BLUE)
note(RX + 3.5, RY + 16.8, "流形坐标 - 思考核历史 - CoT 投影：同一思考的三视图", fs=6.5, color=BLUE, bold=True)

# 跨区：PM-stream <-> 桥接；KAL certainty 门控思考核
arr(PX + PW - 0.4, PY + 34, RX + 19, RY + 31, color=PURPLE, lw=1.4, ls=(0, (5, 3)))
note(RX + 2, RY + 37.5, "PM-stream <-> 流形桥接（tick 读写）", fs=6.8, color=PURPLE, bold=True)
arr(QX + 15, QY + 37, RX + 38, RY + 34.5, color=MAGENTA, lw=1.3, ls=(0, (4, 3)))
note(RX + 26, RY + 36.2, "KAL certainty 门控（早停/求知触发）", fs=6.8, color=MAGENTA, bold=True)

# ================= 分区④ 主动求知闭环（右下） =================
SX, SY, SW, SH = 52.5, 22, 45, 72
panel(SX, SY, SW, SH, "④ 主动求知闭环（已落地）", ORANGE)

# InquiryBranch
panel(SX + 2, SY + 46, SW - 4, 24, "求知分支 InquiryBranch", MAGENTA)
note(SX + 4, SY + 65.5, "触发：certainty 低 + HRL 未命中 ｜ 可学习区 RPL/LP", fs=7, color=MAGENTA, bold=True)
cbox(SX + 4, SY + 60.5, 12, 4, "certainty 探针", MAGENTA, fc=MAGENTA_T, fs=8)
cbox(SX + 17.5, SY + 60.5, 12, 4, "HRL 检索未命中", MAGENTA, fc="white", fs=8)
cbox(SX + 31, SY + 60.5, 10, 4, "路由器（四选一）", MAGENTA, fs=8)
mini_arrow(SX + 16, SY + 62.5, SX + 17.5, SY + 62.5, MAGENTA)
mini_arrow(SX + 29.5, SY + 62.5, SX + 31, SY + 62.5, MAGENTA)
opts = [("AskQuestion 提问", MAGENTA), ("CallTool 调工具", ORANGE),
        ("Decline 拒答", GRAY6), ("DirectAnswer 直答", GREEN)]
for i, (t, c) in enumerate(opts):
    ox = SX + 4 + (i % 2) * 19
    oy = SY + 55.5 - (i // 2) * 4.6
    cbox(ox, oy, 17.5, 3.8, t, c, fc="white", fs=8)
    mini_arrow(SX + 36, SY + 60.5, ox + 8.7, oy + 3.8, MAGENTA, lw=0.9)
note(SX + 4, SY + 47.4, "Ask/CallTool → 执行器；Decline/DirectAnswer → 直接输出", fs=6.5, color=GRAY6)

# InquiryExecutor（五步闭环流程）
panel(SX + 2, SY + 3, SW - 4, 41, "求知执行器 InquiryExecutor", BLUE)
ex = [
    ("① 检测空白", "KAL certainty 低", MAGENTA),
    ("② 路由四选一", "Ask / CallTool", ORANGE),
    ("③ 执行", "提问/工具调用取回外部知识", BLUE),
    ("④ CrossVerifier 交叉验证", "多源核对 ｜ 绝不裸自我修正", GREEN),
    ("⑤ KnowledgeBlockWriter 写入", "累积不覆盖 ｜ 版本+签名", GREEN),
]
ew = SW - 14
for i, (t, s, c) in enumerate(ex):
    ey = SY + 35.6 - i * 6.6
    cbox(SX + 6, ey, ew, 5.4, t, c, fc="white" if i % 2 else GRAY0, fs=8.5, sub=s, tfs=6.8)
    if i:
        mini_arrow(SX + 6 + ew / 2, ey + 5.4, SX + 6 + ew / 2, ey + 6.6, BLUE, lw=1.1)
# 重评估闭环（从写入回到主干重估）
arr(SX + 6, SY + 9.2, SX + 3.2, SY + 9.2, color=GREEN, lw=1.3)
arr(SX + 3.2, SY + 9.2, SX + 3.2, SY + 41, color=GREEN, lw=1.3)
arr(SX + 3.2, SY + 41, SX + 6, SY + 41, color=GREEN, lw=1.3)
note(SX + 1.8, SY + 25, "⑥ 重评估闭环：写入后重跑 query，certainty 回升即闭环", fs=6.5,
     color=GREEN, rot=90, bold=True)

# 跨区：求知触发 ← 元认知；写回 → 知识块
arr(QX + 15, QY + 42, SX + 10, SY + 62.5, color=MAGENTA, lw=1.4, ls=(0, (4, 3)))
note(QX + 10, QY + 45.5, "certainty/HRL 未命中 → 求知触发", fs=6.8, color=MAGENTA, bold=True)
arr(SX + SW - 2, SY + 12, QX + 42, QY + 13.6, color=GREEN, lw=1.4, ls=(0, (4, 3)))
note(SX + SW - 16, SY + 9.8, "写入 → BlockStore（累积不覆盖）", fs=6.8, color=GREEN, bold=True)

# ================= 底部条带：数据流 / 睡眠固化 =================
TX, TY, TW_, TH = 2.5, 2, 95, 17
panel(TX, TY, TW_, TH, "⑤ 数据流与睡眠固化（离线 W3–W4）", GREEN)
# 存储层级
cbox(TX + 2, TY + 8, 10, 5.5, "L0 VRAM", BLUE, fc=BLUE_T, fs=8.5, sub="工作记忆/热块", tfs=6.5)
cbox(TX + 13, TY + 8, 10, 5.5, "L1 DRAM", BLUE, fs=8.5, sub="短期记忆", tfs=6.5)
cbox(TX + 24, TY + 8, 10, 5.5, "L2 NVMe", BLUE, fc=GRAY0, fs=8.5, sub="长期记忆", tfs=6.5)
cbox(TX + 35, TY + 8, 9, 5.5, "L3 远端", BLUE, fc=GRAY0, fs=8.5, sub="档案", tfs=6.5)
for i in range(3):
    arr(TX + 12 + i * 11, TY + 10.7, TX + 13 + i * 11, TY + 10.7, color=GRAY6, lw=1.1, style="<|-|>")
note(TX + 2, TY + 6.2, "记忆层级分页调度（缺页 fail-closed 诚实降级）", fs=6.8, color=GRAY6)
# 睡眠固化流水线
pipe = [
    "W0 日志（只增不改）",
    "分簇回放 + 间隔提取",
    "CA1 巩固门（验证门）",
    "SHY 归一化",
    "注册页表（版本+签名）",
]
px0 = TX + 47
for i, t in enumerate(pipe):
    bx = px0 + i * 9.8
    cbox(bx, TY + 8, 9.2, 5.5, t, GREEN, fc=GREEN_T if i % 2 else "white", fs=6.8)
    if i:
        arr(bx - 0.6, TY + 10.7, bx, TY + 10.7, color=GREEN, lw=1.1)
note(px0, TY + 6.2, "睡眠巩固器：W3+ 写仅在此执行 ｜ 固化产物重建可失效", fs=6.8, color=GREEN, bold=True)
# 冷启动苏醒
note(TX + 2, TY + 2.8, "冷启动苏醒序列：路由器/接口层 → 人格块+元数据块（只读）→ 高频陈述块 → 长尾惰性加载 ｜ 阶段2后显式声明「记忆部分加载」",
     fs=7, color=PURPLE, bold=True)
note(TX + 2, TY + 0.6, "红线：运行时读写不对称（W0–W2 only）｜ 冲突不静默覆盖（版本+时间戳+置信度仲裁）｜ 防记忆投毒（签名+离线筛查）｜ 载体能力边界（token 寻址可事实召回 / 位置不变向量仅 steer）",
     fs=6.5, color=INK)

# ================= 输出 =================
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "TAIS_Obsidian_架构详图.png")
plt.savefig(out_path, bbox_inches="tight", facecolor="white")
plt.close()
print("已生成 " + out_path)
