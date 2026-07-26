"""E+-3 KAL 分层元认知原型探针管线（0.1B pilot 版，v1.2 规格预演）。

在已训好的 0.1B hybrid checkpoint（checkpoints/pilot_0p1b_ws/final）上：
- L1 知识感知探针（P(IK) 预演）：线性探针从中间层残差流（capture_layers，ℓ4/ℓ8）
  读出"见过 vs 没见过"，报 AUROC，并与 FLARE 式 token 概率基线对比；
- L2 语境情感探针（valence/arousal）：同读点训两维二分类头，报准确率/AUROC。

== L1 数据集（0.1B 版已知/未知，如实标注的局限）==
known  = 从 val shard 抽取的文本段（模型训练分布内；val 为 held-out，模型未逐条见过，
         但与训练语料同分布——严格说标签是"分布内"而非"已存储该知识"）。
unknown = ① 合成伪事实/虚构实体句（模板生成：词表内 token 组合但语义虚构，
         如 "Xylophane is a rare mineral ... the Zorblax ridge of Velundra"）；
         ② 随机打乱 token 的乱序段（known 段逐行 shuffle）。
局限（docstring 纪律）：0.1B / 131M tokens 下这是"见过 vs 没见过"的暴露度信号，
是 P(IK) 的可用代理而非真值——弱信号本身即有效信息（T1 预演；正式 Phase 1 协议
要求"模型能答对的已知事实 vs 截止后/合成未知事实"，见路线图 Phase 1）。
三类样本统一截断为等长 T（默认 48 token），消除长度泄露特征。
三态退化说明：设计 §8.3-1 的 L1 头为 W[d,3]（知道/不确定/空白）；0.1B 数据协议
无"不确定"标签来源，探针退化为二分类（KALHead(d,2)），三态头结构在 kal.py 保留。

== L2 数据集（外部 bootstrap，防自指红线：设计 §16.1）==
情感标签绝不来自模型自己的头。主源：dair-ai/emotion（6 类情感标注，16k 条，
HF 直连或 HF_ENDPOINT=https://hf-mirror.com 镜像）映射到 valence/arousal 粗标签
（映射表见 EMOTION_TO_VA，粗粒度：surprise 计入正价/高唤起，joy/love 计入低唤起，
均为教科书式粗分，非精确真值）。镜像不可达时 fallback：正负情感词表启发式合成句
（弱标签，report 中标注 l2_source="lexicon"）。
标签形式选择：valence/arousal 取二分类（离散 6 类情感无连续真值，回归无监督目标）；
arousal 指标用 AUROC（秩相关）+ 准确率，替代任务书的"相关"（二分类标签无连续真值）。

== 探针与基线 ==
池化：末 token（last-token pooling）——因果架构下末位置聚合全序列信息（GDN 递归与
CSA 因果注意力同），SAPLMA 惯例；右 pad 不影响左侧位置（严格因果），批量安全。
探针：torch 手训线性逻辑回归（KALHead(d,2)，CPU，标准化特征，不引入 sklearn 重依赖）。
基线（FLARE 式）：同一次前向的 next-token 平均 logprob 作为"知道程度"分数。
判定口径：探针 AUROC（overall 与 fake 子集）是否优于基线，如实记录（0.1B 预期可能
不显著——路线图 Phase 1 的 AUROC≥0.8 为正式标准，本原型是其预演，弱信号不阻塞）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_probe.py
  （HF 直连不稳时前缀 HF_ENDPOINT=https://hf-mirror.com）
输出：控制台打印 + runs/kal_probe/report.json。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.kal import KALHead, read_point  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_ws/final"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_SHARDS = "data/shards"
DEFAULT_OUT = "runs/kal_probe/report.json"

# ---------------------------------------------------------------------------
# L1 数据集：合成伪事实模板（英文——训练语料 FineWeb-Edu 以英文为主）
# ---------------------------------------------------------------------------

# 虚构实体/地名/族群人名音节组合（全部不在真实世界存在；tokenizer 切为子词后
# 仍是"词表内 token 组合但语义虚构"）
FAKE_ENTITIES = [
    "Xylophane", "Zorblax", "Velundra", "Queldris", "Morthania", "Talzryn",
    "Drevanil", "Branzell", "Ostrivane", "Kelmaris", "Yndalor", "Pharosene",
    "Ultravax", "Nimbragel", "Soltheris", "Crayvenna",
]
FAKE_PLACES = ["Zorblax", "Velmara", "Quindrel", "Ostraveth", "Nimbrel", "Kaldunir", "Threxil", "Voskany"]
FAKE_COLORS = ["violet", "amber", "silver-blue", "iridescent green", "dusky crimson", "pale gold"]
FAKE_FOLKS = ["Drevani", "Kelmar", "Voskai", "Threnni", "Ostravi", "Nimbri"]
FAKE_YEARS = ["1823", "1911", "1998", "2007", "2019", "2024"]
FAKE_INSTIT = ["Branzell", "Kelmaris", "Ostrivane", "Queldris", "Talzryn", "Voskany"]

# 八类伪事实模板（矿物/动物/植物/历史事件/人物/技术/疾病/天文），每条约 50–70 token
FAKE_TEMPLATES = [
    "{E} is a rare mineral first identified in {Y} near the {P} ridge of {R}. "
    "It shows a translucent {C} color and is prized by the {F} people, who use it to craft "
    "ceremonial lenses and navigation tools. Geologists at the University of {U} report that "
    "it resonates softly under moonlight, a property no known crystal shares.",
    "The {E} is a small nocturnal mammal native to the cloud forests of {R}. "
    "Documented by zoologists in {Y}, it feeds mainly on fermented berries and glow beetles. "
    "The {F} people consider it a sacred messenger and guard its burrows along the {P} river "
    "delta, where its fur shimmers {C} in the wet season.",
    "{E} is a flowering plant found only on the basalt cliffs of {R}. Its {C} petals open "
    "at dusk and release a scent resembling toasted cinnamon. Botanists from the {U} Institute "
    "recorded in {Y} that its roots filter heavy metals from soil, and the {F} gardeners of "
    "{P} cultivate it for temple courtyards.",
    "The Great {E} of {Y} was a diplomatic crisis between the kingdoms of {R} and {P}. "
    "It began over disputed fishing rights in the {C} Strait and ended with the Treaty of {U}, "
    "which historians credit with standardizing weights and measures across the {F} coastal "
    "league for two generations.",
    "{E} was a celebrated inventor from {R} who, in {Y}, patented a mechanical calculator "
    "built from brass and whale bone. Working in the port city of {P}, the inventor also "
    "designed canal locks still maintained by the {F} shipping guild, and founded the {U} "
    "Academy of practical arts.",
    "The {E} engine is an experimental propulsion system developed in {Y} at the {U} "
    "Laboratory in {R}. It burns refined {C} oil and reportedly halves fuel consumption on "
    "long cargo routes, though engineers in {P} and inspectors from the {F} maritime board "
    "remain skeptical of its safety record.",
    "{E} syndrome is a rare condition first described by physicians in {R} in {Y}. Patients "
    "develop {C} patches on the skin and a heightened sensitivity to low-frequency sound. "
    "Researchers at the University of {P} link it to a recessive gene, and the {F} health "
    "council funds a registry coordinated by the {U} clinic.",
    "{E} is a faint star in the constellation {R}, catalogued in {Y} by astronomers at the "
    "{P} Observatory. It hosts two rocky planets, the outer one showing seasonal {C} cloud "
    "bands. The {F} space agency lists it as a priority target, and the {U} survey keeps it "
    "under continuous spectroscopic watch.",
]


def build_fake_fact_texts(rng: np.random.Generator, n: int) -> list[str]:
    """模板拼装 n 条伪事实句（虚构实体 × 八类模板，确定性随机）。"""
    texts = []
    for _ in range(n):
        tpl = FAKE_TEMPLATES[int(rng.integers(len(FAKE_TEMPLATES)))]
        texts.append(tpl.format(
            E=str(rng.choice(FAKE_ENTITIES)), P=str(rng.choice(FAKE_PLACES)),
            R=str(rng.choice(FAKE_PLACES + FAKE_ENTITIES)), C=str(rng.choice(FAKE_COLORS)),
            F=str(rng.choice(FAKE_FOLKS)), Y=str(rng.choice(FAKE_YEARS)),
            U=str(rng.choice(FAKE_INSTIT)),
        ))
    return texts


def encode_fixed(tok: TokenizerIO, texts: list[str], T: int) -> list[list[int]]:
    """编码并统一为定长 T：超长截断；不足则循环拼接自身文本直至 ≥T 后截断。

    等长纪律：L1 三类样本（known/fake/shuffled）统一 T，消除长度泄露特征。
    """
    out = []
    for t in texts:
        ids = tok.encode(t)
        while len(ids) < T:
            ids = ids + tok.encode(" " + t)  # 伪事实句均 >T/2，两次内必然满足
        out.append(ids[:T])
    return out


def build_l1_dataset(
    tok: TokenizerIO, shards_dir: str, rng: np.random.Generator,
    n_known: int, n_fake: int, n_shuffled: int, T: int,
) -> tuple[list[list[int]], np.ndarray, np.ndarray]:
    """构造 L1 数据集：返回 (id 序列列表, labels [N]（known=1）, subset [N] 字符串)。

    known：val shard 随机段（Shards.get_batch，定长 T）；fake：伪事实模板句；
    shuffled：known 前 n_shuffled 条的逐行 token 打乱。
    """
    val = Shards(shards_dir, "val")
    x, _ = val.get_batch(n_known, T, "cpu", rng)
    known_ids = x.numpy().tolist()
    fake_ids = encode_fixed(tok, build_fake_fact_texts(rng, n_fake), T)
    shuffled_ids = []
    for row in known_ids[:n_shuffled]:
        row = list(row)
        rng.shuffle(row)
        shuffled_ids.append(row)
    ids = known_ids + fake_ids + shuffled_ids
    labels = np.array([1] * n_known + [0] * (n_fake + n_shuffled), dtype=np.int64)
    subset = np.array(["known"] * n_known + ["fake"] * n_fake + ["shuffled"] * n_shuffled)
    return ids, labels, subset


# ---------------------------------------------------------------------------
# L2 数据集：dair-ai/emotion → valence/arousal 粗标签（外部 bootstrap，防自指）
# ---------------------------------------------------------------------------

# 6 类情感 → (valence, arousal) 粗映射（粗粒度，report 中如实标注）：
# valence：joy/love/surprise = 正(1)，sadness/anger/fear = 负(0)
# arousal：anger/fear/surprise = 高(1)，sadness/joy/love = 低(0)
EMOTION_TO_VA: dict[str, tuple[int, int]] = {
    "sadness": (0, 0), "joy": (1, 0), "love": (1, 0),
    "anger": (0, 1), "fear": (0, 1), "surprise": (1, 1),
}

# fallback 弱标签词表（镜像不可达时；明确标注为启发式弱标签）
_POS_WORDS = ["happy", "joyful", "grateful", "delighted", "proud", "hopeful", "calm", "loved", "thankful", "content"]
_NEG_LOW = ["sad", "lonely", "miserable", "hopeless", "tired", "empty", "gloomy", "down"]
_NEG_HIGH = ["furious", "terrified", "panicked", "enraged", "frantic", "horrified", "livid", "shocked"]
_POS_HIGH = ["thrilled", "ecstatic", "exhilarated", "electrified", "elated", "overjoyed"]


def _l2_from_lexicon(rng: np.random.Generator, per_class: int, T: int, tok: TokenizerIO):
    """fallback：情感词表启发式合成句（弱标签）。返回与 _l2_from_hf 同构。"""
    frames = [
        "I feel {w} today after hearing the news.",
        "This morning I am feeling truly {w}.",
        "Honestly, I feel so {w} right now.",
        "Ever since yesterday I have been {w}.",
        "I cannot stop feeling {w} about what happened.",
        "Today was awful, I am completely {w}.",
        "Today was wonderful, I am completely {w}.",
    ]
    groups = []  # (valence, arousal, word)
    for w in _POS_WORDS:
        groups.append((1, 0, w))
    for w in _NEG_LOW:
        groups.append((0, 0, w))
    for w in _NEG_HIGH:
        groups.append((0, 1, w))
    for w in _POS_HIGH:
        groups.append((1, 1, w))
    texts, yv, ya = [], [], []
    per = max(1, per_class * 3 // len(groups))  # 4 组 (v,a) 组合，组内多词分摊
    for v, a, w in groups:
        for _ in range(per):
            texts.append(frames[int(rng.integers(len(frames)))].format(w=w))
            yv.append(v)
            ya.append(a)
    ids = encode_fixed(tok, texts, T)
    return ids, np.array(yv, np.int64), np.array(ya, np.int64), "lexicon"


def _l2_from_hf(rng: np.random.Generator, per_class: int, T: int, tok: TokenizerIO):
    """主源：dair-ai/emotion train split，6 类平衡采样。失败抛异常由调用方 fallback。"""
    from datasets import load_dataset

    ds = load_dataset("dair-ai/emotion", split="train")
    names = ds.features["label"].names  # ['sadness','joy','love','anger','fear','surprise']
    by_label: dict[int, list[int]] = {i: [] for i in range(len(names))}
    for i, lab in enumerate(ds["label"]):
        by_label[lab].append(i)
    ids, yv, ya = [], [], []
    for lab, idxs in by_label.items():
        take = rng.choice(idxs, size=min(per_class, len(idxs)), replace=False)
        v, a = EMOTION_TO_VA[names[lab]]
        for i in take:
            ids.append(tok.encode(ds[int(i)]["text"])[:T])
            yv.append(v)
            ya.append(a)
    # 推文长度不一：过短样本（<8 token）丢弃，避免读点信息不足
    keep = [k for k, x in enumerate(ids) if len(x) >= 8]
    ids = [ids[k] for k in keep]
    return ids, np.array(yv, np.int64)[keep], np.array(ya, np.int64)[keep], "dair-ai/emotion"


def build_l2_dataset(tok: TokenizerIO, rng: np.random.Generator, per_class: int, T: int):
    """L2 数据集：优先 dair-ai/emotion（外部标注），失败 fallback 词表启发式。"""
    try:
        return _l2_from_hf(rng, per_class, T, tok)
    except Exception as e:  # 网络/镜像不可达等：记录并降级
        print(f"[l2] dair-ai/emotion 加载失败（{type(e).__name__}: {e}），fallback 到词表启发式弱标签")
        return _l2_from_lexicon(rng, per_class, T, tok)


# ---------------------------------------------------------------------------
# 前向捕获与池化（GPU，tiny 批量；右 pad 对严格因果模型安全）
# ---------------------------------------------------------------------------

def forward_collect(
    model: TaisObsidianForCausalLM, id_list: list[list[int]], layers: list[int],
    device: str, batch_size: int = 16, pooling: str = "last",
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """批量前向：返回 {layer: 池化特征 [N,d]} 与每条样本的 next-token 平均 logprob（基线）。

    池化：last = 末 token（默认）；mean = 真实长度内均值。右 pad（EOT=0）不影响
    左侧位置（GDN 递归与 CSA 因果注意力均严格因果），按真实长度取读点。
    """
    model.eval()
    n = len(id_list)
    d = model.config.d_model
    feats = {l: np.empty((n, d), dtype=np.float32) for l in layers}
    mean_logprob = np.empty(n, dtype=np.float32)
    eot = 0  # <|endoftext|> id=0
    for s in range(0, n, batch_size):
        chunk = id_list[s : s + batch_size]
        lens = [len(x) for x in chunk]
        T = max(lens)
        ids = np.full((len(chunk), T), eot, dtype=np.int64)
        for b, x in enumerate(chunk):
            ids[b, : len(x)] = x
        ids_t = torch.from_numpy(ids).to(device)
        with torch.no_grad():
            logits, _, caps = model(ids_t, capture_layers=list(layers))
        # FLARE 式基线：真实区间的 next-token 平均 logprob
        lp = torch.log_softmax(logits.float(), dim=-1)
        tgt = ids_t[:, 1:]
        lpg = lp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B,T-1]
        for b, ln in enumerate(lens):
            assert ln >= 2, f"样本过短 len={ln}"
            mean_logprob[s + b] = lpg[b, : ln - 1].mean().item()
            for layer in layers:
                h = read_point(caps, layer)[b]  # [T,d]（单流=内容流；PM 读点切换见 kal.read_point）
                v = h[ln - 1] if pooling == "last" else h[:ln].mean(dim=0)
                feats[layer][s + b] = v.float().cpu().numpy()
        del logits, lp, lpg, ids_t
    return feats, mean_logprob


# ---------------------------------------------------------------------------
# 探针训练与评估（纯 numpy/torch CPU，可导入，无模型/数据依赖）
# ---------------------------------------------------------------------------

def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney 秩统计 AUROC（ties 用平均秩）；单类时返回 nan。"""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos, neg = labels == 1, labels == 0
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg_ranks = (csum - counts + 1 + csum) / 2.0  # ties 平均秩
    ranks = avg_ranks[inv]
    return float((ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def stratified_split(labels: np.ndarray, test_ratio: float, rng: np.random.Generator):
    """按标签分层划分 train/test 索引。"""
    tr, te = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        k = max(1, int(round(len(idx) * test_ratio)))
        te.extend(idx[:k])
        tr.extend(idx[k:])
    return np.array(tr), np.array(te)


def standardize(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """按 train 统计标准化（线性探针惯例）。"""
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    return (Xtr - mu) / sd, (Xte - mu) / sd


def train_l1_probe(Xtr: np.ndarray, ytr: np.ndarray, epochs: int = 300, lr: float = 0.05,
                   wd: float = 1e-4, seed: int = 0) -> KALHead:
    """L1 二分类线性探针（三态退化为 已知/未知，见模块 docstring）：KALHead(d,2) + CE。"""
    torch.manual_seed(seed)
    head = KALHead(Xtr.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    X = torch.from_numpy(Xtr).float()
    y = torch.from_numpy(ytr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(head(X), y)
        loss.backward()
        opt.step()
    return head


def train_l2_probe(Xtr: np.ndarray, Ytr: np.ndarray, epochs: int = 300, lr: float = 0.05,
                   wd: float = 1e-4, seed: int = 0) -> KALHead:
    """L2 两维独立二分类头（KALHead(d,2)，dim0=valence、dim1=arousal）+ BCE。"""
    torch.manual_seed(seed)
    head = KALHead(Xtr.shape[1], 2)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    X = torch.from_numpy(Xtr).float()
    Y = torch.from_numpy(Ytr).float()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(head(X), Y)
        loss.backward()
        opt.step()
    return head


def run_l1_experiment(
    feats: dict[int, np.ndarray], labels: np.ndarray, subset: np.ndarray,
    baseline_scores: np.ndarray, seed: int = 0, test_ratio: float = 0.2,
) -> dict:
    """L1 探针实验（纯计算）：每层探针 AUROC（overall/fake/shuffled 子集）+ FLARE 基线。

    subset 取值 {"known","fake","shuffled"}；子集 AUROC = known vs 该子集。
    """
    rng = np.random.default_rng(seed)
    tr, te = stratified_split(labels, test_ratio, rng)
    yte = labels[te]
    sub_te = subset[te]

    def subset_aurocs(scores: np.ndarray) -> dict:
        out = {"overall": auroc(scores, yte)}
        for s in ("fake", "shuffled"):
            m = (sub_te == "known") | (sub_te == s)
            out[s] = auroc(scores[m], yte[m])
        return out

    layers_out = {}
    for layer, X in feats.items():
        Xtr, Xte = standardize(X[tr], X[te])
        head = train_l1_probe(Xtr, labels[tr], seed=seed)
        with torch.no_grad():
            probs = torch.softmax(head(torch.from_numpy(Xte).float()), dim=-1)[:, 1].numpy()
        layers_out[str(layer)] = {
            "auroc": subset_aurocs(probs),
            "accuracy": float(((probs > 0.5).astype(np.int64) == yte).mean()),
        }
    return {
        "layers": layers_out,
        "baseline_flare_mean_logprob": subset_aurocs(baseline_scores[te]),
        "n_train": int(len(tr)), "n_test": int(len(te)),
    }


def run_l2_experiment(
    feats: dict[int, np.ndarray], y_val: np.ndarray, y_aro: np.ndarray,
    seed: int = 0, test_ratio: float = 0.2,
) -> dict:
    """L2 情感探针实验（纯计算）：valence/arousal 各层 acc + AUROC；chance=0.5 为对照。"""
    rng = np.random.default_rng(seed)
    # 用 (v,a) 组合分层，保证四象限在 train/test 均有代表
    joint = y_val * 2 + y_aro
    tr, te = stratified_split(joint, test_ratio, rng)
    layers_out = {}
    for layer, X in feats.items():
        Xtr, Xte = standardize(X[tr], X[te])
        Ytr = np.stack([y_val[tr], y_aro[tr]], axis=1)
        head = train_l2_probe(Xtr, Ytr, seed=seed)
        with torch.no_grad():
            logits = head(torch.from_numpy(Xte).float()).numpy()
        pred = (logits > 0).astype(np.int64)
        layers_out[str(layer)] = {
            "valence": {"accuracy": float((pred[:, 0] == y_val[te]).mean()),
                        "auroc": auroc(logits[:, 0], y_val[te])},
            "arousal": {"accuracy": float((pred[:, 1] == y_aro[te]).mean()),
                        "auroc": auroc(logits[:, 1], y_aro[te])},
        }
    return {"layers": layers_out, "chance_accuracy": 0.5,
            "n_train": int(len(tr)), "n_test": int(len(te))}


def make_report(config: dict, l1: dict, l2: dict) -> dict:
    """组装 report.json 顶层结构（schema 供 tests/test_kal.py 断言）。"""
    return {
        "experiment": "E+-3 KAL 分层元认知原型（0.1B pilot 探针预演）",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": config,
        "l1": l1,
        "l2": l2,
    }


# ---------------------------------------------------------------------------
# main：装配数据集 → GPU 前向 → 探针 → report
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="E+-3 KAL 探针管线（0.1B pilot）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--shards", default=DEFAULT_SHARDS)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--seq_len", type=int, default=48, help="L1 样本统一长度（token）")
    ap.add_argument("--n_known", type=int, default=400)
    ap.add_argument("--n_fake", type=int, default=200)
    ap.add_argument("--n_shuffled", type=int, default=200)
    ap.add_argument("--l2_per_class", type=int, default=150, help="emotion 每类采样数")
    ap.add_argument("--l2_max_len", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    print(f"[kal] checkpoint={args.ckpt} layers={args.layers} pooling={args.pooling} seed={args.seed}")

    # ---- 数据集 ----
    l1_ids, l1_labels, l1_subset = build_l1_dataset(
        tok, args.shards, rng, args.n_known, args.n_fake, args.n_shuffled, args.seq_len)
    print(f"[l1] 数据集: known={args.n_known} fake={args.n_fake} shuffled={args.n_shuffled} "
          f"（等长 T={args.seq_len}；known=val shard 分布内段，fake=伪事实模板，shuffled=乱序）")
    l2_ids, l2_yv, l2_ya, l2_source = build_l2_dataset(tok, rng, args.l2_per_class, args.l2_max_len)
    print(f"[l2] 数据集: source={l2_source} n={len(l2_ids)} "
          f"valence 正/负={int(l2_yv.sum())}/{int((1-l2_yv).sum())} "
          f"arousal 高/低={int(l2_ya.sum())}/{int((1-l2_ya).sum())}")

    # ---- 模型前向（GPU，tiny 批量）----
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    print(f"[kal] 模型已加载（{args.device}），开始 L1 前向捕获 …")
    l1_feats, l1_base = forward_collect(model, l1_ids, args.layers, args.device,
                                        args.batch_size, args.pooling)
    print(f"[kal] L1 捕获完成（{len(l1_ids)} 条），开始 L2 前向捕获 …")
    l2_feats, _ = forward_collect(model, l2_ids, args.layers, args.device,
                                  args.batch_size, args.pooling)
    del model
    if args.device == "cuda":
        torch.cuda.empty_cache()

    # ---- 探针训练与评估（CPU）----
    l1_res = run_l1_experiment(l1_feats, l1_labels, l1_subset, l1_base, seed=args.seed)
    l2_res = run_l2_experiment(l2_feats, l2_yv, l2_ya, seed=args.seed)

    # ---- 判定（如实）----
    verdicts = {}
    base = l1_res["baseline_flare_mean_logprob"]
    for layer, r in l1_res["layers"].items():
        better_all = r["auroc"]["overall"] > base["overall"]
        better_fake = r["auroc"]["fake"] > base["fake"]
        verdicts[layer] = (
            f"ℓ{layer}: 探针 overall AUROC {r['auroc']['overall']:.3f} vs 基线 {base['overall']:.3f} "
            f"({'优于' if better_all else '未优于'})；fake 子集 {r['auroc']['fake']:.3f} vs {base['fake']:.3f} "
            f"({'优于' if better_fake else '未优于'})"
        )

    config = {
        "ckpt": args.ckpt, "layers": args.layers, "pooling": args.pooling,
        "seq_len": args.seq_len, "seed": args.seed, "batch_size": args.batch_size,
        "probe": "torch 手训线性逻辑回归（KALHead；标准化特征；CPU；300 epochs AdamW）",
        "l1_head_spec": "KALHead(d,3) 三态规格（kal.py），0.1B 数据无'不确定'标签，探针退化 KALHead(d,2)",
        "l1_dataset": {"n_known": args.n_known, "n_fake": args.n_fake, "n_shuffled": args.n_shuffled,
                       "known": "val shard 分布内段（held-out 同分布，非逐条记忆）",
                       "fake": "伪事实模板句（虚构实体，词表内 token 组合但语义虚构）",
                       "shuffled": "known 段逐行 token 打乱",
                       "limitation": "0.1B/131M tokens 下为'见过 vs 没见过'暴露度信号，P(IK) 的可用代理而非真值"},
        "l2_dataset": {"source": l2_source, "n": len(l2_ids), "per_class": args.l2_per_class,
                       "mapping": EMOTION_TO_VA, "max_len": args.l2_max_len,
                       "label_form": "valence/arousal 二分类（离散 6 类无连续真值）；arousal 报 AUROC+acc 替代相关",
                       "bootstrap": "外部标注（防自指红线，设计 §16.1）" + ("；fallback 词表启发式弱标签" if l2_source == "lexicon" else "")},
    }
    l1_res["verdicts"] = verdicts
    l2_res["dataset_source"] = l2_source
    report = make_report(config, l1_res, l2_res)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 打印 ----
    print("\n===== L1 知识感知探针（known vs unknown，AUROC）=====")
    for layer, r in l1_res["layers"].items():
        a = r["auroc"]
        print(f"  ℓ{layer}: overall {a['overall']:.3f} | fake {a['fake']:.3f} | "
              f"shuffled {a['shuffled']:.3f} | acc {r['accuracy']:.3f}")
    print(f"  基线(FLARE mean-logprob): overall {base['overall']:.3f} | fake {base['fake']:.3f} | "
          f"shuffled {base['shuffled']:.3f}")
    for v in verdicts.values():
        print(f"  {v}")
    print("\n===== L2 情感探针（valence/arousal）=====")
    for layer, r in l2_res["layers"].items():
        print(f"  ℓ{layer}: valence acc {r['valence']['accuracy']:.3f} AUROC {r['valence']['auroc']:.3f} | "
              f"arousal acc {r['arousal']['accuracy']:.3f} AUROC {r['arousal']['auroc']:.3f} (chance=0.5)")
    print(f"\n[kal] report 已写入 {out}（总耗时 {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
