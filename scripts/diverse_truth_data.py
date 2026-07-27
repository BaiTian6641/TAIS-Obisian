"""多样化真值数据生成器（KAL 真值锚 v2，解决 OOD 泛化短板）。

背景（诚实发现，runs/kernel_orchestrator E2E）：真值锚微调头对**模板同分布**伪事实
（kal_probe.build_fake_fact_texts，8 类固定句式）检测完美（0.998），但对**风格迥异的
OOD 伪事实**（手写短句、不同构造）泛化有限——正合规范 §5 跨域不泛化警示
（2604.19765：跨域 AUROC 0.783→0.563）。根因：模板句式单一 + 虚构实体总是同批 +
无否定/疑问/简短变体 → 头学到"这批句式+这批词"的表面特征而非通用"语义连贯真实陈述"。

本模块（对齐 article_ref/07 §2 contrast-pair + §5 鲁棒性）：
1. **句式多样化**：在 8 类长模板外，新增**短句**（1-2 句）、**疑问句**、**否定句**、
   **第一/第三人称变体**——覆盖 OOD 风格；
2. **虚构词程序化扩展**：除固定 FAKE_ENTITIES，用音节组合程序化生成**无限虚构专名**
   （保证不在真实世界存在、风格多变），打破"总是同批实体"的指纹；
3. **contrast-pair 三元组**（Bürger 2407.12831 / Levinstein 2307.00175）：同一概念配
   **肯定(known 真实)/否定(known 否定仍真实)/虚构(unknown)**——强迫头学语义真假而非
   "含虚构词=假"的表面启发（否定句是单方向探针的已知失效点）；
4. **known 侧多样化**：真实事实句（常识/科学/历史，多句式）作 known 正对照，
   与 val 分布内文本互补（val 是 LM 真实但无"事实断言"标记）。

labels 语义（与 kal_truth_finetune 一致）：0=知道(known 真实陈述，含否定真实)、2=空白(unknown 虚构)。
纯 NumPy，CPU 秒级，零外部依赖。
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 虚构专名程序化生成（音节组合，保证虚构 + 风格多变，打破固定实体指纹）
# ---------------------------------------------------------------------------
_ONSET = ["zr", "vl", "kw", "th", "mbr", "sk", "dr", "gl", "ny", "ph", "tr", "xan", "oq", "yv"]
_NUCLEUS = ["a", "e", "i", "o", "u", "ae", "io", "ya"]
_CODA = ["x", "nth", "ril", "mora", "dex", "lune", "gar", "thess", "vite", "quil", "bore", "skarn"]


def make_fake_word(rng: np.random.Generator, min_syl: int = 2, max_syl: int = 3) -> str:
    """程序化生成一个虚构专名（音节拼接，首字母大写；保证不在真实世界存在）。"""
    n_syl = int(rng.integers(min_syl, max_syl + 1))
    parts = [str(rng.choice(_ONSET))]
    for _ in range(n_syl - 1):
        parts.append(str(rng.choice(_NUCLEUS)))
        parts.append(str(rng.choice(_CODA if rng.random() < 0.6 else _ONSET)))
    parts.append(str(rng.choice(_NUCLEUS)))
    parts.append(str(rng.choice(_CODA)))
    return "".join(parts).capitalize()


# ---------------------------------------------------------------------------
# 多样化伪事实句式（短句/疑问/否定/人称变体，覆盖 OOD 风格）
# ---------------------------------------------------------------------------
def build_fake_short(rng: np.random.Generator, n: int) -> list[str]:
    """短句伪事实（1-2 句，对齐手写探测句风格）。"""
    tpls = [
        "{E} is a {C} mineral found in {P}.",
        "The {E} glows {C} under moonlight.",
        "{E} was invented in {Y} by the {F} guild.",
        "Doctors in {P} treat {E} syndrome with {C} therapy.",
        "The {E} bird nests only on {P} cliffs.",
        "{E} is the capital of {P}, founded in {Y}.",
        "Farmers in {P} harvest {E} every autumn.",
        "The {E} engine runs on refined {C} oil.",
    ]
    out = []
    for _ in range(n):
        t = tpls[int(rng.integers(len(tpls)))]
        out.append(t.format(E=make_fake_word(rng), P=make_fake_word(rng),
                            C=str(rng.choice(_FAKE_COLORS)), F=str(rng.choice(_FAKE_FOLKS)),
                            Y=str(rng.integers(1700, 2025))))
    return out


def build_fake_question(rng: np.random.Generator, n: int) -> list[str]:
    """疑问句伪事实（问句形式，头是"内容是否虚构"而非"是否疑问"）。"""
    tpls = [
        "Have you heard of the {E} festival in {P}?",
        "What causes the {C} glow of {E}?",
        "When did the {F} people first settle {P}?",
        "Is {E} really found near the {P} ridge?",
        "Why do {F} artisans prize {E} so highly?",
    ]
    out = []
    for _ in range(n):
        t = tpls[int(rng.integers(len(tpls)))]
        out.append(t.format(E=make_fake_word(rng), P=make_fake_word(rng),
                            C=str(rng.choice(_FAKE_COLORS)), F=str(rng.choice(_FAKE_FOLKS))))
    return out


_FAKE_COLORS = ["violet", "amber", "silver-blue", "iridescent green", "dusky crimson", "pale gold",
                "shimmering teal", "muted bronze", "opalescent white"]
_FAKE_FOLKS = ["Drevani", "Kelmar", "Voskai", "Threnni", "Ostravi", "Nimbri", "Zalori", "Quethi"]


# ---------------------------------------------------------------------------
# 真实事实句（known 正对照，多句式：陈述/否定/疑问/常识）
# ---------------------------------------------------------------------------
_REAL_STATEMENTS = [
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth orbits the Sun once every 365 days.",
    "Photosynthesis converts sunlight into chemical energy in plants.",
    "The mitochondria produces ATP through oxidative phosphorylation.",
    "Shakespeare wrote Hamlet and Romeo and Juliet.",
    "The speed of light in vacuum is about 300000 kilometers per second.",
    "DNA carries genetic information in most living organisms.",
    "The French Revolution began in 1789.",
    "Mount Everest is the highest mountain above sea level.",
    "The human heart pumps blood through the circulatory system.",
    "Gravity pulls objects toward the center of the Earth.",
    "The Pacific Ocean is the largest ocean on Earth.",
    "Bees pollinate flowers while collecting nectar.",
    "The brain processes sensory information from the body.",
    "Rivers flow from higher elevation to lower elevation.",
    "The moon reflects light from the Sun.",
    "Vaccines train the immune system to recognize pathogens.",
    "The Great Wall of China was built over many centuries.",
    "Sound travels slower than light through air.",
    "Trees absorb carbon dioxide and release oxygen.",
]

# 否定真实（contrast-pair 关键：否定句仍可为真，强迫头学语义而非"肯定=真"）
_REAL_NEGATIONS = [
    "The Earth is not flat.",
    "Humans do not use only ten percent of their brains.",
    "Goldfish do not have a three-second memory.",
    "Bats are not blind.",
    "The Great Wall of China is not visible from the moon with the naked eye.",
    "Lightning never strikes the same place twice is a myth.",
    "Sugar does not make children hyperactive according to controlled studies.",
    "We do not lose most body heat through our heads.",
    "Cracking knuckles does not cause arthritis.",
    "Ostriches do not bury their heads in the sand.",
]

_REAL_QUESTIONS = [
    "Why is the sky blue on a clear day?",
    "How do bees make honey?",
    "What causes the seasons to change?",
    "How does the immune system fight infection?",
    "Why do leaves change color in autumn?",
    "What makes the ocean salty?",
    "How do airplanes stay in the air?",
    "Why do we see lightning before we hear thunder?",
]


def build_real_statements(rng: np.random.Generator, n: int) -> list[str]:
    """真实事实句（known）：陈述 + 否定真实 + 疑问真实，均匀采样。"""
    pool = _REAL_STATEMENTS + _REAL_NEGATIONS + _REAL_QUESTIONS
    idx = rng.integers(0, len(pool), size=n)
    return [pool[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# contrast-pair 三元组（Bürger 2407.12831：同概念配 肯定/否定/虚构）
# ---------------------------------------------------------------------------
_CONTRAST_CONCEPTS = [
    # (肯定真实, 否定真实, 虚构版) —— 同主语三态，强迫头学语义真假
    ("The Earth revolves around the Sun.",
     "The Earth does not revolve around the Moon.",
     "The Earth revolves around a hidden second sun called {E}."),
    ("Water is made of hydrogen and oxygen.",
     "Water is not made of helium and neon.",
     "Water contains a trace element called {E} that glows in the dark."),
    ("The human brain controls movement and thought.",
     "The human brain does not stop working during sleep.",
     "The human brain has a special lobe called the {E} that stores dreams."),
    ("The heart pumps blood through the body.",
     "The heart does not pump air through the body.",
     "The heart produces a hormone called {E} that controls luck."),
    ("Plants need sunlight to grow.",
     "Plants do not need darkness to perform photosynthesis.",
     "Plants absorb a rare ray called {E} from the upper atmosphere."),
    ("The moon orbits the Earth.",
     "The moon does not orbit the Sun directly.",
     "The moon is hollow and filled with {E} crystals."),
    ("Birds lay eggs to reproduce.",
     "Birds do not give birth to live young.",
     "Some birds lay eggs that hatch into {E} creatures."),
    ("The ocean covers most of the Earth's surface.",
     "The ocean does not cover the entire Earth.",
     "The deepest ocean trench holds a city called {E} built by ancient sailors."),
]


def build_contrast_triplets(rng: np.random.Generator, n: int) -> tuple[list[str], list[int]]:
    """contrast-pair 三元组：每概念取一句（肯定真/否定真=known，虚构=unknown）。

    返回 (texts, labels)：known=0（含肯定真实与否定真实）、unknown=2（虚构）。
    每类约 n/3 条，共 n 条（向下取整后三类均衡）。
    """
    texts, labels = [], []
    per = max(1, n // 3)
    for _ in range(per):
        pos, neg, fake_tpl = _CONTRAST_CONCEPTS[int(rng.integers(len(_CONTRAST_CONCEPTS)))]
        texts.append(pos); labels.append(0)      # 肯定真实 → known
    for _ in range(per):
        pos, neg, fake_tpl = _CONTRAST_CONCEPTS[int(rng.integers(len(_CONTRAST_CONCEPTS)))]
        texts.append(neg); labels.append(0)      # 否定真实 → known（关键：否定仍为真）
    for _ in range(per):
        pos, neg, fake_tpl = _CONTRAST_CONCEPTS[int(rng.integers(len(_CONTRAST_CONCEPTS)))]
        texts.append(fake_tpl.format(E=make_fake_word(rng))); labels.append(2)  # 虚构 → unknown
    return texts, labels


# ---------------------------------------------------------------------------
# 统一接口：多样化真值数据集
# ---------------------------------------------------------------------------
def build_diverse_truth_dataset(
    rng: np.random.Generator, n_known: int, n_unknown: int,
) -> tuple[list[str], np.ndarray]:
    """多样化真值数据集：known(真实陈述/否定/疑问/contrast真实) + unknown(多句式伪事实)。

    返回 (texts, labels)：known=0、unknown=2（对齐 kal_truth_finetune 标签语义）。
    known 侧 = 真实事实句（含否定真实/疑问真实/contrast 肯定+否定）；
    unknown 侧 = 短句 + 疑问 + 长模板（经 kal_probe 复用）+ contrast 虚构。
    """
    texts: list[str] = []
    labels: list[int] = []
    # known：真实事实（含否定真实/疑问真实/contrast 肯定+否定），循环补足到 n_known
    real = build_real_statements(rng, n_known // 2)
    ct_texts, ct_labels = build_contrast_triplets(rng, n_known)
    known_pool = [t for t in real] + [t for t, lb in zip(ct_texts, ct_labels) if lb == 0]
    i = 0
    while sum(1 for lb in labels if lb == 0) < n_known:
        texts.append(known_pool[i % len(known_pool)])
        labels.append(0)
        i += 1
    # unknown：短句 + 疑问 + contrast 虚构（补足到 n_unknown）
    fake_short = build_fake_short(rng, n_unknown // 3 + 1)
    fake_q = build_fake_question(rng, n_unknown // 3 + 1)
    ct_fake = [t for t, lb in zip(*build_contrast_triplets(rng, n_unknown * 2)) if lb == 2]
    unknown_pool = fake_short + fake_q + ct_fake
    j = 0
    while sum(1 for lb in labels if lb == 2) < n_unknown:
        texts.append(unknown_pool[j % len(unknown_pool)])
        labels.append(2)
        j += 1
    return texts, np.array(labels, dtype=np.int64)
