#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TAIS Obsidian 1.5B 架构详图生成脚本（v0.6 / 对应设计文档 v1.1）

用途：生成 Carbon 设计语言的 TAIS Obsidian 单页架构蓝图 PNG。
运行：python3 TAIS_Obsidian_架构图_生成脚本.py
依赖：matplotlib（无需其他第三方库）；中文显示需系统装有中文字体
输出：TAIS_Obsidian_架构详图.png（当前工作目录）

坐标系：x 0–100，y -70–150（底部为流水线/苏醒/闭环/谱系条带）。
修改提示：所有部件由 panel()/cbox()/note()/arr() 四个图元构成，
        按坐标网格摆放；改布局时优先调整各面板的 (x, y, w, h)。
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
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
ORANGE = "#ff832b"; ORANGE_T = "#fff2e8"; MAGENTA = "#ee5396"

fig, ax = plt.subplots(figsize=(17, 32), dpi=200)
ax.set_xlim(0, 100); ax.set_ylim(-70, 150); ax.axis("off")
fig.patch.set_facecolor("white")


def tw(text):
    """估算文本宽度（CJK 与 ASCII 分别计宽）。"""
    return sum(1.9 if ord(c) > 0x2E80 else 0.95 for c in text)


def tag(x, y, text, color, fs=11):
    """Carbon 式实心面板标签。"""
    w = tw(text) + 2.2
    ax.add_patch(Rectangle((x, y), w, 2.6, fc=color, ec="none", zorder=4))
    ax.text(x + 1.1, y + 1.3, text, fontsize=fs, color="white",
            fontweight="bold", va="center", zorder=5)


def panel(x, y, w, h, label, color, tag_bottom=False):
    """带左侧色条的面板容器。"""
    ax.add_patch(Rectangle((x, y), w, h, fc=GRAY0, ec=GRAY4, lw=1.0, zorder=1))
    ax.add_patch(Rectangle((x, y), 0.9, h, fc=color, ec="none", zorder=2))
    if label:
        tag(x + 0.9, y + 0.5 if tag_bottom else y + h - 2.6, label, color)


def cbox(x, y, w, h, t, ec, fc="white", tc=INK, fs=9, sub=None, tfs=7.2):
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


# ================= 标题栏 =================
ax.add_patch(Rectangle((0, 143.5), 100, 6.5, fc=INK, ec="none"))
ax.text(3, 147.6, "TAIS Obsidian 1.5B", fontsize=22, color="white",
        fontweight="bold", va="center")
ax.text(3, 145.2, "tais-obsidian ｜ 知识感知 × 海马记忆 × 视觉空间 × 原生 1M ｜ Reasoning-native ｜ 增强A–F（v1.3）",
        fontsize=10, color=GRAY4, va="center")
ax.text(97, 147.6, "v0.8", fontsize=12, color=GRAY4, va="center", ha="right")
ax.text(97, 145.2, "2026-07-25", fontsize=9, color=GRAY4, va="center", ha="right")

# ================= 左侧主栈 =================
MX, MW = 4, 23; cx = MX + MW / 2
panel(2.5, 3, 27, 138, "基础主干 Backbone", INK)
note(1.2, 72, "7 × {3 GDN-MemBlock + 1 CSA-AttnBlock} = 28 层",
     fs=9, color=INK, bold=True, rot=90)
cbox(MX, 6, MW, 3.4, "Tokenizer", INK, fs=10)
cbox(MX, 11, MW, 3.4, "Embedding", INK, fs=10,
     sub="[seq_len, 2048] ｜ vocab 129280 ｜ tied")
arr(cx, 9.6, cx, 11)
ax.add_patch(Circle((cx, 15.6), 0.5, fc=INK, zorder=6))
note(16.4, 15.6, "vision tokens", fs=7)
arr(MX + 1.5, 15.6, cx - 0.55, 15.6, lw=1.0)
arr(cx, 14.6, cx, 15.3)
panel(3.8, 18, 24.4, 90, "", GRAY6)

# PM-stream 导轨
ax.add_patch(Rectangle((27.7, 18), 0.6, 110, fc=PURPLE_T, ec=PURPLE, lw=1.0, zorder=2))
note(28.0, 78, "PM-stream（mHC 5 流残差）", fs=6.5, color=PURPLE, rot=90,
     ha="center", bold=True)
for yy in (35.6, 50.6, 89.6, 104.6):
    ax.add_patch(Circle((28.0, yy), 0.32, fc=PURPLE, ec="white", lw=0.8, zorder=6))

# 3× GDN-MemBlock
panel(5.2, 20, 21, 52, "3 × GDN-MemBlock", TEAL)
cbox(MX, 22, MW - 2.5, 3.2, "RMSNorm", INK, fs=9)
arr(cx, 25.2, cx, 26.4)
cbox(MX + 1.2, 26.4, MW - 5, 7, "Gated DeltaNet（KDA 式）", TEAL, fc=TEAL_T, fs=9,
     sub="递归状态 = 工作记忆寄存器（原生无界）")
arr(cx, 33.4, cx, 34.6)
plus(cx, 35.6, 1.2, TEAL)
arr(cx, 36.8, cx, 38)
cbox(MX, 38, MW - 2.5, 3.2, "RMSNorm", INK, fs=9)
arr(cx, 41.2, cx, 42.4)
cbox(MX + 1.2, 42.4, MW - 5, 6, "FFN（dense 1.5B 版）", INK, fc=GRAY0, fs=9,
     sub="MoE 变体：64 选 4 + 1 共享")
arr(cx, 48.4, cx, 49.6)
plus(cx, 50.6, 1.2, INK)
plus(MX + MW - 4, 50.6, 1.0, PURPLE)
note(MX + MW - 2.8, 52.3, "注入点", fs=7, color=PURPLE)
note(6.5, 55, "（第 2、3 层同构）", fs=8, color=GRAY6)
ax.add_patch(Circle((MX + 0.8, 60), 0.55, fc=PURPLE, ec="white", lw=1.2, zorder=6))
note(MX + 2.0, 61.2, "KAL 探针挂载 ℓ10/14/18", fs=7, color=PURPLE)
note(6.5, 65.5, "侧信道头簇（读 PM-stream）", fs=7, color=PURPLE, bold=True)
note(6.5, 63.7, "ℓ8 预取｜ℓ14 写显著(+arousal)·冲突｜ℓ18 归因·联想",
     fs=6.3, color=PURPLE)
note(6.5, 68.5, "增强A：旁挂稀疏KV可写记忆层（delta rule 写入）",
     fs=6.8, color=TEAL, bold=True)
note(6, 73, "CSA 原生块通路（双向）：导出=harvest 压缩KV｜注入=拼接（增强B）",
     fs=6.8, color=TEAL, bold=True)

# 1× CSA-AttnBlock（V4 混合压缩注意力三级 = 三级记忆，增强F §17）
panel(5.2, 74, 21, 29, "1 × CSA-AttnBlock（V4 混合压缩）", ORANGE, tag_bottom=True)
cbox(MX, 76, MW - 2.5, 3.2, "RMSNorm", INK, fs=9)
arr(cx, 79.2, cx, 80.4)
cbox(MX + 1.2, 80.4, MW - 5, 7, "混合压缩注意力（V4 三级）", ORANGE, fc=ORANGE_T, fs=9,
     sub="滑窗·CSA·HCA = L0/L1/L2 记忆")
arr(cx, 87.4, cx, 88.6)
plus(cx, 89.6, 1.2, ORANGE)
arr(cx, 90.8, cx, 92)
cbox(MX, 92, MW - 2.5, 3.2, "RMSNorm", INK, fs=9)
arr(cx, 95.2, cx, 96.4)
cbox(MX + 1.2, 96.4, MW - 5, 5, "FFN / MoE（三维统一路由）", INK, fc=GRAY0, fs=8.5,
     sub="token域/块域/专家域 同构打分头")
arr(cx, 101.4, cx, 102.6)
plus(cx, 103.6, 1.2, INK)
note(6, 105.5, "HCA 区 = 块注入原生落点（消前缀偏差）｜ HCA=<|gist|>架构版 ｜ sinks=诚实降级",
     fs=6.3, color=ORANGE, bold=True)
note(6, 108, "默认 Reasoning（CoT）｜ 推理中发 <|recall|>/<|gist|>",
     fs=7, color=INK, bold=True)

# 顶部
cbox(MX, 110, MW, 3.4, "RMSNorm", INK, fs=10)
arr(cx, 104.8, cx, 110)
cbox(MX, 114.8, MW, 3.4, "LM-Head（DoLa 开关）", INK, fs=9.5)
arr(cx, 113.4, cx, 114.8)
cbox(MX + 2.5, 120, MW - 5, 4, "MTP 头", GRAY6, fc=GRAY0, fs=9)
arr(cx, 118.2, cx, 120)
arr(cx, 124, cx, 127.5, lw=1.8)

# ================= TAIS Memory Bus =================
ax.add_patch(Rectangle((31.2, 20), 0.7, 120, fc=BLUE, ec="none", zorder=2))
note(31.55, 100, "TAIS Memory Bus", fs=7.5, color="white", rot=90,
     ha="center", bold=True)
note(32.6, 68, "读通道（双通道载荷）", fs=7.5, color=TEAL, rot=90, bold=True)
arr(MX + 1.4, 60, 30.3, 60, color=PURPLE, lw=1.3)
arr(30.3, 60, 30.3, 128, color=PURPLE, lw=1.3)
arr(30.3, 128, 33, 128, color=PURPLE, lw=1.3)
arr(33, 88, 31.9, 88, color=TEAL, lw=1.6)
arr(31.55, 88, 31.55, 50.6, color=TEAL, lw=1.6)
arr(31.55, 50.6, MX + MW - 2.9, 50.6, color=TEAL, lw=1.6)
arr(29.5, 20, 31.55, 20, color=INK, lw=1.3)
arr(31.55, 20, 31.55, 10.5, color=INK, lw=1.3)
arr(31.55, 10.5, 36, 10.5, color=INK, lw=1.3, ls=(0, (4, 2)))
note(20, 17.3, "写通道：W0 日志 →", fs=7.5, color=INK, bold=True)

# ================= KAL（含增强C 情感头） =================
panel(33, 116, 31, 26, "知识感知层 KAL（分层元认知）", PURPLE)
cbox(35, 133, 12, 4.5, "探针组 ℓ10/14/18", PURPLE, fs=8)
cbox(48.5, 133, 7, 4.5, "三态头 L1", PURPLE, fs=8)
cbox(57, 133, 4.5, 4.5, "情感 L2", PURPLE, fc=MAGENTA, fs=7.5)
note(35, 131, "读 PM-stream ｜ L1/L2 同期训练", fs=7)
cbox(35, 125.5, 26.5, 4, "知道 ｜ 不确定 ｜ 空白", PURPLE, fc=PURPLE_T, fs=9)
cbox(35, 120, 13, 4, "ITI 干预向量", PURPLE, fs=8)
cbox(48.5, 120, 13, 4, "DoLa 开关", PURPLE, fs=8)
cbox(35, 116.8, 26.5, 2.8, "回想 query 生成", PURPLE, fs=8.5)
arr(41, 133, 41, 129.7, color=PURPLE, lw=1)
arr(52, 133, 52, 129.7, color=PURPLE, lw=1)
arr(59.2, 133, 59.2, 129.7, color=MAGENTA, lw=1)
arr(41.5, 125.5, 41.5, 124.2, color=PURPLE, lw=1)
arr(55, 125.5, 55, 124.2, color=PURPLE, lw=1)
arr(48, 125.5, 48, 119.8, color=PURPLE, lw=1)
arr(61.5, 118, 61.5, 113.2, color=PURPLE, lw=1.4)
# affect 调制总线（品红虚线）
arr(59.2, 132.7, 59.2, 126, color=MAGENTA, lw=1.2, ls=(0, (3, 2)))
arr(40, 116.5, 40, 113.2, color=MAGENTA, lw=1.4, ls=(0, (3, 2)))
note(41, 114.6, "affect", fs=6.5, color=MAGENTA, bold=True)

# ================= 行为塑形 =================
panel(66, 116, 32, 26, "行为塑形（离线训练）", GRAY6)
cbox(68, 133, 28, 4.5, "TruthRL 三元奖励 · GRPO", GRAY6, fs=9)
cbox(68, 127.5, 28, 4, "答对 +1 ｜ 拒答 +0.3 ｜ 幻觉 −1", GRAY6, fc=GRAY0, fs=8.5)
cbox(68, 122, 28, 4, "KAL 监督 + Persona 预防性 steering", GRAY6, fc=GRAY0, fs=8)
note(68, 119, "让「承认空白→回想」成为内生行为", fs=7.5)
arr(66, 130, 62, 130, color=GRAY6, lw=1.2, ls=(0, (4, 2)))

# ================= HRL =================
panel(33, 84, 65, 29, "海马路由层 HRL（类 MoE · 双向 · 可写）", TEAL)
cbox(35, 104, 18, 5, "DG · 模式分离", TEAL, fs=9,
     sub="Linear→TopK 稀疏｜persona 投影")
cbox(55, 104, 18, 5, "Indexer · 内容寻址", TEAL, fs=9,
     sub="FP8 分块归并 top-k")
cbox(75, 100.5, 21, 8.5, "页表 Page Table", TEAL, fs=9.5,
     sub="block_id → L0/L1/L2 ｜ 版本·签名")
cbox(35, 93.5, 29, 5.5, "CA3 · 模式补全", TEAL, fc=TEAL_T, fs=9,
     sub="知识图 + PPR，ε≈0.1 远邻联想")
cbox(66, 93.5, 30, 5.5, "预测预取器（分支预测式）", TEAL, fs=9,
     sub="按思考段提前 1–2 段")
cbox(35, 86.5, 29, 5, "CA1 · 巩固门", TEAL, fs=9,
     sub="写许可：验证门+版本+置信度")
cbox(66, 86.5, 30, 5, "读通道载荷组装", TEAL, fc=TEAL_T, fs=9,
     sub="文本+LoRA/KV/steering+TTL")
arr(53, 106.5, 55, 106.5, color=TEAL, lw=1.1)
arr(73, 106.5, 75, 105.5, color=TEAL, lw=1.1)
arr(44, 104, 44, 99.2, color=TEAL, lw=1.1)
arr(64, 96.2, 75, 100.5, color=TEAL, lw=1.1)
arr(85.5, 100.5, 85.5, 99.2, color=TEAL, lw=1.1)
arr(64, 89, 66, 89, color=TEAL, lw=1.1)
note(35, 84.8, "HRL = TEM 认知地图导航器：内在潜空间几何 + 外在物理坐标 ｜ landmark 块为 PPR 锚点", fs=6.6, color=TEAL, bold=True)
note(35, 82.3, "增强E：affect 调制总线 → CA1 固化优先级 ｜ DG 检索维度 ｜ 情感匹配召回 ｜ 块 affect 字段", fs=6.6, color=MAGENTA, bold=True)

# ================= 知识块 =================
panel(33, 52, 31, 29, "知识块 KnowledgeBlock", GREEN)
cbox(35, 72, 12, 4.5, "Header", GREEN, fs=8.5, sub="id·type·版本·签名")
cbox(48.5, 72, 13, 4.5, "源代码 markdown", GREEN, fs=8, sub="可读·可审计·可回滚")
cbox(35, 64.5, 12, 5, "产物 A：LoRA", GREEN, fc=GREEN_T, fs=8, sub="A/B[2048,16]")
cbox(48.5, 64.5, 13, 5, "产物 B：KV prefix", GREEN, fc=GREEN_T, fs=8, sub="[64,32,128]")
cbox(35, 57.5, 12, 5, "产物 C：steering", GREEN, fc=GREEN_T, fs=8, sub="[2048]")
cbox(48.5, 57.5, 13, 5, "route_key", GREEN, fs=8.5, sub="[1024] 稀疏+图边")
note(35, 54.5, "metadata：confidence·ttl·usage·载体适用性 ｜ +affect(V/A)", fs=6.8)
note(35, 52.8, "类型：陈述/程序/元数据/人格（RO）｜+记忆层条目（增强A）", fs=6.8)
arr(61.5, 72, 75, 99, color=TEAL, lw=1.2, ls=(0, (4, 2)))

# ================= DKB-Runtime =================
panel(66, 52, 32, 29, "DKB-Runtime 软件对接层", BLUE)
cbox(68, 72, 28, 4.5, "API 网关（回想/注入/记录）", BLUE, fs=8.5)
cbox(68, 66.5, 28, 4.5, "Pager 守护（换入/换出/预取）", BLUE, fs=8.5)
cbox(68, 61, 28, 4.5, "BlockStore（页表+向量库+对象）", BLUE, fs=8.5)
cbox(68, 55.5, 28, 4.5, "注入中间件（vLLM / HF hooks）", BLUE, fs=8.5)
note(68, 53.5, "与模型进程解耦 · 语言无关 ABI", fs=7)
arr(68, 63.7, 64, 63.7, color=BLUE, lw=1.2, style="<|-|>")
arr(82, 66.5, 82, 49.5, color=BLUE, lw=1.2)

# ================= 视觉空间区 =================
panel(33, 22, 31, 27, "视觉空间区 Visual Primitives", ORANGE)
cbox(35, 41, 12, 4.5, "ViT（冻结）", ORANGE, fs=8.5, sub="14×14 patch")
cbox(48.5, 41, 13, 4.5, "3×3 空间压缩", ORANGE, fs=8.5, sub="→ 324 tokens")
cbox(35, 35.5, 26.5, 4, "CSA 再压 4× → ~81 KV（≈7000×）", ORANGE, fc=ORANGE_T, fs=8)
cbox(35, 29.5, 26.5, 4.5, "<|ref|>/<|box|> 推理内交织", ORANGE, fs=8,
     sub="点与包围盒 = 最小思维单元")
note(35, 25.5, "视觉经验 → 空间记忆块（坐标邻近度边）", fs=7)
note(35, 23.5, "训练：V1 对齐 → V2 原语 SFT → V3 空间 RL", fs=7)
arr(35, 31.5, 16.2, 15.8, color=ORANGE, lw=1.3)

# ================= 记忆层级 =================
panel(66, 22, 32, 27, "记忆层级（分页调度）", GRAY6)
cbox(68, 41, 28, 4.5, "L0 VRAM · 工作记忆", BLUE, fc=BLUE_T, fs=9,
     sub="热块 + 人格块（Persona 冻结·RO）")
cbox(68, 34.5, 28, 4, "L1 DRAM · 短期记忆", BLUE, fc="white", fs=9)
cbox(68, 28, 28, 4, "L2 NVMe · 长期记忆", BLUE, fc=GRAY0, fs=9)
arr(82, 41, 82, 38.7, color=GRAY6, lw=1.1, style="<|-|>")
arr(82, 34.5, 82, 32.2, color=GRAY6, lw=1.1, style="<|-|>")
note(68, 24, "淘汰：LRU × 重要性 × 置信度", fs=7)
arr(66, 31, 31.9, 31, color=GRAY6, lw=1.1, style="<|-|>")

# ================= 睡眠巩固器 =================
panel(33, 5, 65, 14, "睡眠巩固器（离线，W3–W4 级写仅在此执行）", GREEN)
cbox(36, 8, 14, 5, "W0 日志", GREEN, fs=9, sub="只增不改")
cbox(52, 8, 14, 5, "重放 + 验证门", GREEN, fc=GREEN_T, fs=9, sub="近期优先（CLS）")
cbox(68, 8, 14, 5, "编译固化", GREEN, fs=9, sub="text→LoRA/KV")
cbox(84, 8, 12, 5, "注册页表", GREEN, fs=9, sub="版本+签名")
arr(50, 10.5, 52, 10.5, color=GREEN, lw=1.2)
arr(66, 10.5, 68, 10.5, color=GREEN, lw=1.2)
arr(82, 10.5, 84, 10.5, color=GREEN, lw=1.2)
arr(90, 13, 61, 55, color=GREEN, lw=1.2, ls=(0, (4, 2)))
note(84, 30, "固化为新块", fs=7, color=GREEN)

# ================= 训练与数据流水线 =================
panel(2.5, -20, 95.5, 15, "训练与数据流水线（T0–T5）", BLUE)
stages = [
    ("T0 外挂验证", "Qwen3.5-4B + Runtime", BLUE),
    ("T1 从零预训练", "Dolma3 取样 20–30B｜本机", BLUE),
    ("T2 信号对齐", "KAL 监督 + indexer KL", TEAL),
    ("T3 行为塑形", "三元奖励 GRPO｜Dolci", TEAL),
    ("T4 长上下文", "Longmino 128K→1M｜云端", ORANGE),
    ("T5 视觉+双轨", "原语 SFT→空间 RL｜4B", PURPLE),
]
sw = 14.8; sx = 4.5
for i, (t, s, c) in enumerate(stages):
    cbox(sx + i * (sw + 0.9), -16, sw, 7, t, c, fs=8.5, sub=s, tfs=6.8)
    if i:
        arr(sx + i * (sw + 0.9) - 0.9, -12.5, sx + i * (sw + 0.9), -12.5,
            color=GRAY6, lw=1.1)
note(4.5, -18.3, "数据：Dolma 3 Mix / Dolmino / Longmino / Stack v2 / Dolci ｜ 精度：bf16-mixed（8-bit AdamW 备选）｜ T3.5 技能习得（Verilator verifier）", fs=7.2)

# ================= 苏醒序列 =================
panel(2.5, -34, 95.5, 11, "苏醒序列（冷启动加载顺序）", PURPLE)
wake = [("0 页表自检", "签名校验"), ("1 舞台就位", "路由器·KAL·接口"),
        ("2 执行功能", "人格块+核心程序块"), ("3 高频陈述块", "按优先级换入"),
        ("4 长尾惰性", "缺页时再取")]
ww = 17.6; wx = 4.5
for i, (t, s) in enumerate(wake):
    cbox(wx + i * (ww + 0.9), -31, ww, 5.5, t, PURPLE, fc=PURPLE_T,
         fs=8.5, sub=s, tfs=6.8)
    if i:
        arr(wx + i * (ww + 0.9) - 0.9, -28.2, wx + i * (ww + 0.9), -28.2,
            color=PURPLE, lw=1.1)

# ================= Reasoning-native 闭环 =================
panel(2.5, -52, 95.5, 15, "Reasoning-native：推理中的记忆调用闭环", INK)
flow = [("CoT 思考段", "PM-stream 信号"), ("KAL 监测", "三态判定"),
        ("空白 → <|recall|>", "推理流内动作"), ("HRL 路由", "top-k + PPR 联想"),
        ("块注入", "记忆层 / LoRA ≈0 token"), ("带记忆继续推理", "全程落 W0 日志")]
fw = 14.6; fx = 4.5
for i, (t, s) in enumerate(flow):
    cbox(fx + i * (fw + 0.9), -48, fw, 7, t, INK, fc=GRAY0, fs=8, sub=s, tfs=6.6)
    if i:
        arr(fx + i * (fw + 0.9) - 0.9, -44.5, fx + i * (fw + 0.9), -44.5,
            color=INK, lw=1.1)
note(4.5, -50.3, "训练：三元奖励 GRPO + SKILL0 课程撤出 ｜ 理论桥：ICL=低秩权重更新（arXiv:2507.16003）｜ Verilator = verifier（Verilog RLVR 已验证）", fs=7.2)

# ================= 谱系与扩展路径 =================
panel(2.5, -68, 95.5, 13, "谱系与扩展路径（原生部件 ABI 跨规模不变）", TEAL)
fam = [("Obsidian 1.5B", "28 层 dense｜想法验证｜本机"),
       ("Obsidian 9B", "40 层｜千块库干扰测试"),
       ("Obsidian 27B", "基础意识主力｜mHC 原生规模"),
       ("Obsidian 30B-A6B", "MoE｜块即专家｜生产")]
fw2 = 22.2; fx2 = 4.5
for i, (t, s) in enumerate(fam):
    cbox(fx2 + i * (fw2 + 1.1), -64.5, fw2, 7, t, TEAL, fc=TEAL_T, fs=9,
         sub=s, tfs=7)
    if i:
        arr(fx2 + i * (fw2 + 1.1) - 1.1, -61, fx2 + i * (fw2 + 1.1), -61,
            color=TEAL, lw=1.2)
note(4.5, -66.5, "只扩主干：KAL / HRL / PM-stream / 注入点 / Block Spec 全部不变 ｜ 动态参数账本：27B 冻结基座 + 千块级技能库（块 ≈ 0.1–0.2% 基座参数/块）", fs=7.2)

note(2, -69.8, "硬件：RTX PRO 4000 SFF 24GB ｜ 原生 1M：渐进课程 32K→128K→1M + 训练内 YaRN（仅 CSA 层）｜ mHC n=5：4 内容流 + 1 PM-stream（双随机约束 ≤1.6×）",
     fs=7.8, color=INK, bold=True)

# ================= 输出 =================
plt.savefig("TAIS_Obsidian_架构详图.png", bbox_inches="tight", facecolor="white")
plt.close()
print("已生成 TAIS_Obsidian_架构详图.png")
