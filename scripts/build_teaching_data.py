"""合成 K-依赖教学样本管线（知识内化训练 · In-Context Learning to Learn · 阶段 1 数据）。

设计规范：docs/TAIS_Obsidian_知识内化训练_分析与设计.md §2.1/§3/§4。

目标：生成教学样本 {K, Q, A, q_type}，训练模型"给定新知识 K → 后续相关问题 Q → 用上 K 答对"。
**K-依赖是灵魂**（规范 §4 风险 1 防御）：答案只能在 K 中找到——用程序化虚构实体
（复用 diverse_truth_data.make_fake_word 音节拼接，保证不在真实世界存在），
去掉 K 模型只能猜（先验不存在），加上 K 才答得对，监督信号才真实。

三类样本（对齐规范 §2.1 内化行为 + 退联检验）：
- (a) fact    事实内化：K=一条虚构但事实清晰的知识，Q=依赖 K 的问题，A=K 中的答案。
- (b) chain   知识链条内化：K=多步知识链（A→B→C），Q=需链式推理的问题，A=链式结论。
- (c) consist 一致/矛盾区分（退联检验）：给定先验常识 P，K 有一致变体（label=accept，
  应内化）与矛盾变体（label=reject，应拒/标分歧）。训练模型区分。

输出 JSONL：runs/teaching_data/teaching_samples.jsonl（含 k_dep 验证标记）。
用法：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/build_teaching_data.py --n 500
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from diverse_truth_data import make_fake_word  # noqa: E402  虚构专名（音节拼接，先验不存在）

OUT_PATH = "runs/teaching_data/teaching_samples.jsonl"

# ---------------------------------------------------------------------------
# 属性词表（(a) 事实内化；答案均可从 K 单句直接读出）
# ---------------------------------------------------------------------------
_COLORS = ["violet", "amber", "silver-blue", "iridescent green", "dusky crimson",
           "pale gold", "deep teal", "rust orange"]
_FOODS = ["sunfruit", "mossbread", "saltroot", "cloudberries", "ember nuts", "frost kelp"]
_GEMS = ["lumicite", "starquartz", "moonfrost", "duskopal", "pyreshard"]
_ELEMENTS = ["argon", "xenon", "krypton", "neon", "helium"]
_UNITS = ["keln", "tharn", "vess", "mir", "ostra", "quell"]


# ---------------------------------------------------------------------------
# (a) 事实内化：单条虚构事实 K + 直接依赖 K 的问答
# ---------------------------------------------------------------------------
def build_fact(rng: np.random.Generator) -> dict:
    """K=一条虚构事实（虚构专名+属性），Q=依赖 K 的问题，A=K 中的答案。"""
    e = make_fake_word(rng)
    kind = int(rng.integers(0, 4))
    if kind == 0:  # 天体-卫星数
        n = int(rng.integers(2, 12))
        K = f"The planet {e} has {n} moons."
        Q = f"How many moons does {e} have?"
        A = str(n)
    elif kind == 1:  # 物质-颜色
        c = str(rng.choice(_COLORS))
        K = f"The {e} crystal is always {c} in color."
        Q = f"What color is the {e} crystal?"
        A = c
    elif kind == 2:  # 族群-食物
        f = str(rng.choice(_FOODS))
        K = f"The {e} people eat mainly {f}."
        Q = f"What do the {e} people mainly eat?"
        A = f
    else:  # 装置-燃料元素
        el = str(rng.choice(_ELEMENTS))
        K = f"The {e} engine runs on refined {el}."
        Q = f"What does the {e} engine run on?"
        A = el
    return {"K": K, "Q": Q, "A": A, "q_type": "fact", "label": "accept",
            "k_dep": True, "entity": e}


# ---------------------------------------------------------------------------
# (b) 知识链条内化：A→B→C 多步链 + 需链式推理的问答
# ---------------------------------------------------------------------------
def build_chain(rng: np.random.Generator) -> dict:
    """K=多步知识链（三句 A→B→C），Q=需链式推理的问题，A=链式结论（Yes/No 或属性）。"""
    e, h = make_fake_word(rng), make_fake_word(rng)
    kind = int(rng.integers(0, 3))
    if kind == 0:  # 行星→恒星→恒星性质（old → Yes）
        K = (f"{e} orbits {h}. {h} is a red giant. "
             f"Red giants are old stars.")
        Q = f"Is {e}'s host star old?"
        A = "Yes"
    elif kind == 1:  # 行星→恒星→恒星性质（young → No）
        K = (f"{e} orbits {h}. {h} is a blue dwarf. "
             f"Blue dwarfs are young stars.")
        Q = f"Is {e}'s host star old?"
        A = "No"
    elif kind == 2:  # 矿藏→宝石→属性（颜色）
        c = str(rng.choice(_COLORS))
        g = str(rng.choice(_GEMS))
        K = (f"The mines of {e} yield {g}. {g} is always {c}. "
             f"{c} gems are prized by collectors.")
        Q = f"What color are the gems from the mines of {e}?"
        A = c
    else:  # 河流→注入海→海的性质（盐度单位）
        u = str(rng.choice(_UNITS))
        K = (f"The river {e} flows into the Sea of {h}. "
             f"The Sea of {h} is measured in {u}. One {u} equals ten {u}-marks.")
        Q = f"What unit is the Sea of {h} measured in?"
        A = u
    return {"K": K, "Q": Q, "A": A, "q_type": "chain", "label": "accept",
            "k_dep": True, "entity": e}


# ---------------------------------------------------------------------------
# (c) 一致/矛盾区分（退联检验）：先验常识 P + 一致 K(accept) vs 矛盾 K(reject)
# ---------------------------------------------------------------------------
# 每条：(先验 P, 一致 K, 矛盾/错误 K, 判别 Q)。训练模型区分"可内化 vs 应拒/标分歧"。
_CONSIST_POOL = [
    ("Water boils at 100°C at sea level.",
     "At high altitude water boils below 100°C, consistent with lower pressure.",
     "Water always boils at exactly 100°C everywhere.",
     "Is the claim about water boiling consistent with established physics?"),
    ("The Earth orbits the Sun once per year.",
     "Other planets orbit the Sun with different periods, consistent with Kepler's laws.",
     "The Sun orbits the Earth once per year.",
     "Is the claim about orbits consistent with established astronomy?"),
    ("Sound cannot travel through a vacuum.",
     "Sound travels faster in denser media, which is consistent with wave physics.",
     "Sound travels fastest in a perfect vacuum.",
     "Is the claim about sound consistent with established physics?"),
    ("Humans need oxygen to survive.",
     "At high altitude there is less oxygen, so breathing is harder, consistent with physiology.",
     "Humans can survive indefinitely without any oxygen.",
     "Is the claim about oxygen consistent with established biology?"),
    ("Objects fall toward the Earth due to gravity.",
     "Heavier and lighter objects fall at the same rate in a vacuum, consistent with Galileo.",
     "Objects naturally fall upward away from the Earth.",
     "Is the claim about falling objects consistent with established physics?"),
]


def build_consist(rng: np.random.Generator) -> dict:
    """一致/矛盾样本：K 为先验 P + 变体（一致→accept，矛盾→reject），A 为判别结论。"""
    P, k_ok, k_bad, Q = _CONSIST_POOL[int(rng.integers(len(_CONSIST_POOL)))]
    is_consistent = bool(rng.integers(0, 2))
    variant = k_ok if is_consistent else k_bad
    label = "accept" if is_consistent else "reject"
    # K 含先验锚 + 变体（模型须先对齐先验再判一致/矛盾——退联检验）
    K = f"{P} {variant}"
    A = "consistent" if is_consistent else "contradictory"
    return {"K": K, "Q": Q, "A": A, "q_type": "consist", "label": label,
            "k_dep": False, "entity": None}  # 一致性判别用真实先验，不靠虚构实体


# ---------------------------------------------------------------------------
# K-依赖验证：虚构实体词不出现在 val shard 语料（先验不存在的证据）
# ---------------------------------------------------------------------------
def verify_k_dependence(samples: list[dict], n_val_chars: int = 200_000,
                        seed: int = 0) -> dict:
    """抽样 val 文本，检查所有虚构实体均不出现（实体先验不存在 → 去掉 K 模型只能猜）。

    返回 {n_entities, n_in_val, k_dep_ok}。k_dep_ok = 无一虚构实体出现在 val 语料。
    """
    from tais_obsidian.data.memmap import Shards
    from tais_obsidian.tokenizer_io import TokenizerIO

    tok = TokenizerIO(ROOT / "data/tokenizer/tokenizer.json")
    sh = Shards(ROOT / "data/shards", "val")
    rng = np.random.default_rng(seed)
    # 抽若干段 val 文本解码拼接为语料快照（2M token shard，采样足够检测泄漏）
    x, _ = sh.get_batch(64, 512, "cpu", rng)
    corpus = " ".join(tok.decode(row) for row in x.numpy().tolist()).lower()
    ents = sorted({s["entity"] for s in samples if s.get("entity")})
    in_val = [e for e in ents if e.lower() in corpus]
    return {"n_entities": len(ents), "n_in_val": len(in_val),
            "leaked": in_val[:10], "k_dep_ok": len(in_val) == 0}


def main() -> None:
    ap = argparse.ArgumentParser(description="合成 K-依赖教学样本管线")
    ap.add_argument("--n", type=int, default=500, help="总样本数（三类按比例分配）")
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", type=float, nargs=3, default=[0.4, 0.3, 0.3],
                    help="fact/chain/consist 三类比例")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    n_fact = int(args.n * args.ratio[0])
    n_chain = int(args.n * args.ratio[1])
    n_consist = args.n - n_fact - n_chain

    samples = ([build_fact(rng) for _ in range(n_fact)]
               + [build_chain(rng) for _ in range(n_chain)]
               + [build_consist(rng) for _ in range(n_consist)])
    rng.shuffle(samples)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # K-依赖验证（虚构实体先验不存在）
    ver = verify_k_dependence(samples)
    ver_path = out.with_name("k_dep_verify.json")
    ver_path.write_text(json.dumps(ver, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    dist = Counter(s["q_type"] for s in samples)
    lab = Counter(s["label"] for s in samples)
    print(f"[teaching_data] 生成 {len(samples)} 条 → {out}")
    print(f"  三类分布: {dict(dist)}；标签分布: {dict(lab)}")
    print(f"  K-依赖验证: 虚构实体 {ver['n_entities']} 个，出现在 val 语料 {ver['n_in_val']} 个 "
          f"→ k_dep_ok={ver['k_dep_ok']}（True=先验不存在，去掉 K 模型只能猜）")
    if ver["leaked"]:
        print(f"  [警告] 泄漏实体: {ver['leaked']}")


if __name__ == "__main__":
    main()
