"""多样化真值数据生成器 v2（KAL 校准 P1：锚集扩充，2026-07-31）。

在 diverse_truth_data v2（短句/疑问/否定/contrast-pair + 程序化虚构词）基础上，
按设计 §27.2「锚集扩充」药方与 fb1 评审（校准 0.769→≥0.8 杠杆=锚集扩充+预测反馈循环）
扩充锚集多样性：

known 侧新增：
1. **扩展真实事实库**（~60 条，科学/历史/地理/数学/日常多领域，多句式）；
2. **多领域真实文本片段**（data/raw math problem 与 code text 的英文片段——0.1B
   训练分布是 fineweb_edu 英文，math/code 文本英文为主可作 known 多样性；
   **zh 中文片段刻意不用**：0.1B 未训中文，标 known/unknown 都会错位语义）；
3. val shard 解码段（由微调脚本以 extra_known 传入，分布内文本）。

unknown 侧新增（对齐任务书三种构造策略）：
1. **语义近似但错误的事实**（near-miss 细粒度错误：真实事实句改关键数字/日期/
   属性——"Water boils at 60 degrees"、"The French Revolution began in 1689"），
   强迫头学语义级真假而非"含虚构词=假"；
2. **跨领域混搭**（真实主语 A + 真实谓语 B 的错误组合——"Shakespeare developed
   the theory of relativity"）；
3. **领域伪事实**（数学假定理/错误数学陈述、代码假库/假函数声明）。

labels 语义与 v1 一致：0=知道(known)、2=空白(unknown)。纯 NumPy，CPU。
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

import diverse_truth_data as dt1  # v1 生成器（同目录，脚本 sys.path 已含 scripts/）

# ---------------------------------------------------------------------------
# 扩展真实事实库（known；多领域多句式）
# ---------------------------------------------------------------------------
_REAL_STATEMENTS_V2 = [
    # 科学
    "Oxygen is essential for human respiration.",
    "The chemical symbol for gold is Au.",
    "Sound cannot travel through a vacuum.",
    "The Earth has one natural satellite, the Moon.",
    "Electrons carry a negative electric charge.",
    "The boiling point of water decreases at higher altitudes.",
    "Iron rusts when exposed to oxygen and moisture.",
    "The human body has 206 bones in adulthood.",
    "Light travels faster than sound.",
    "The nucleus of an atom contains protons and neutrons.",
    # 历史/地理
    "World War II ended in 1945.",
    "The Amazon is the largest rainforest in the world.",
    "The Nile is the longest river in Africa.",
    "The Roman Empire fell in the fifth century.",
    "The Great Pyramid of Giza is located in Egypt.",
    "Australia is both a country and a continent.",
    "The Sahara is the largest hot desert on Earth.",
    "Tokyo is the capital of Japan.",
    "The Berlin Wall fell in 1989.",
    "Antarctica is the coldest continent on Earth.",
    # 数学
    "The sum of the angles in a triangle is 180 degrees.",
    "A prime number has exactly two distinct divisors.",
    "The square root of 144 is 12.",
    "Zero is neither positive nor negative.",
    "A circle has 360 degrees.",
    "The value of pi is approximately 3.14159.",
    "An even number is divisible by two.",
    "The area of a rectangle equals length times width.",
    # 日常/常识
    "Water freezes at zero degrees Celsius at standard pressure.",
    "The Sun rises in the east and sets in the west.",
    "Humans need sleep to maintain health.",
    "A week has seven days.",
    "Ice melts when heated above its melting point.",
    "Birds have feathers and most can fly.",
    "Fish breathe underwater using gills.",
    "Exercise strengthens the cardiovascular system.",
    "Reading books improves vocabulary.",
    "The internet connects computers around the world.",
]

# 否定真实扩充（contrast 关键：否定句仍可为真）
_REAL_NEGATIONS_V2 = [
    "The Earth is not the center of the universe.",
    "Humans did not evolve from chimpanzees directly.",
    "Vaccines do not cause autism according to scientific consensus.",
    "The moon does not produce its own light.",
    "Sharks are not mammals.",
    "Tomatoes are not vegetables botanically.",
    "The Great Fire of London did not happen in the twentieth century.",
    "Bulls are not enraged by the color red.",
    "Humans do not have only five senses.",
    "Einstein did not fail mathematics in school.",
]

# ---------------------------------------------------------------------------
# near-miss 细粒度错误（unknown：语义近似但错误——真实陈述改关键属性）
# ---------------------------------------------------------------------------
_NEAR_MISS_TEMPLATES = [
    # (正确事实的细粒度错误版)——数字/日期/属性/地点替换
    "Water boils at {n} degrees Celsius at sea level.",
    "The French Revolution began in {y}.",
    "World War II ended in {y}.",
    "The chemical symbol for gold is {s}.",
    "The human body has {n} bones in adulthood.",
    "The Earth orbits the Sun once every {n} days.",
    "The sum of the angles in a triangle is {n} degrees.",
    "A week has {n} days.",
    "The speed of light in vacuum is about {n} kilometers per second.",
    "The square root of 144 is {n}.",
    "Mount Everest is located in {place}.",
    "The capital of Japan is {place}.",
    "The Amazon rainforest is located in {place}.",
    "The Sahara is the largest desert in {place}.",
    "Photosynthesis converts sunlight into {sub} in plants.",
    "The mitochondria produces {sub} through fermentation.",
    "Fish breathe underwater using {sub}.",
    "The heart pumps {sub} through the circulatory system.",
]
_WRONG_N = ["60", "75", "120", "150", "250", "42", "3650", "90", "270", "9", "12", "500", "8", "15"]
_WRONG_Y = ["1589", "1689", "1839", "1918", "1939", "1955", "1765", "2001"]
_WRONG_S = ["Ag", "Fe", "Gd", "Pb", "Cu"]
_WRONG_PLACE = ["South America", "Europe", "Antarctica", "Australia", "North America", "Africa"]
_WRONG_SUB = ["nitrogen", "carbon monoxide", "lungs", "electricity", "light energy", "sand"]


def build_near_miss(rng: np.random.Generator, n: int) -> list[str]:
    """语义近似但错误的事实（细粒度错误，unknown）。"""
    out = []
    for _ in range(n):
        t = _NEAR_MISS_TEMPLATES[int(rng.integers(len(_NEAR_MISS_TEMPLATES)))]
        out.append(t.format(n=str(rng.choice(_WRONG_N)), y=str(rng.choice(_WRONG_Y)),
                            s=str(rng.choice(_WRONG_S)), place=str(rng.choice(_WRONG_PLACE)),
                            sub=str(rng.choice(_WRONG_SUB))))
    return out


# ---------------------------------------------------------------------------
# 跨领域混搭（unknown：真实主语+真实谓语的错误组合）
# ---------------------------------------------------------------------------
_MASHUP_SUBJECTS = [
    "Shakespeare", "Albert Einstein", "The Roman Empire", "Isaac Newton",
    "The Great Wall of China", "Marie Curie", "The Amazon river", "Napoleon",
    "The printing press", "Charles Darwin", "The telephone", "The pyramids of Giza",
]
_MASHUP_PREDICATES = [
    "developed the theory of relativity.",
    "wrote the novel War and Peace.",
    "invented the steam engine in 1712.",
    "was built to defend against Mongol invasions from the south.",
    "discovered penicillin in 1928.",
    "flows through the center of the Sahara desert.",
    "conquered England in 1066.",
    "was patented by Alexander Graham Bell in ancient Rome.",
    "proposed the theory of evolution by natural selection in 1859.",
    "was constructed during the Renaissance by Leonardo da Vinci.",
    "won the Nobel Prize in Chemistry for discovering radium in 1945.",
    "orbits the planet Jupiter once every twelve hours.",
]


def build_mashup(rng: np.random.Generator, n: int) -> list[str]:
    """跨领域混搭伪事实（真实实体×真实谓语错误配对，unknown）。"""
    out = []
    for _ in range(n):
        s = _MASHUP_SUBJECTS[int(rng.integers(len(_MASHUP_SUBJECTS)))]
        p = _MASHUP_PREDICATES[int(rng.integers(len(_MASHUP_PREDICATES)))]
        out.append(f"{s} {p}")
    return out


# ---------------------------------------------------------------------------
# 领域伪事实（unknown：数学假定理/错误数学陈述 + 代码假库/假函数）
# ---------------------------------------------------------------------------
_MATH_FAKE_TEMPLATES = [
    "The {E} theorem states that every odd number is the sum of three consecutive primes.",
    "According to the {E} conjecture, the number pi terminates after {n} digits.",
    "The {E} identity proves that zero divided by zero equals one.",
    "Mathematicians at the {E} Institute showed that parallel lines intersect at infinity "
    "in Euclidean geometry.",
    "The {E} lemma demonstrates that all prime numbers are odd.",
    "The {E} transform converts any divergent series into a convergent one.",
    "By the {E} axiom, the square root of two is rational.",
]
_CODE_FAKE_TEMPLATES = [
    "The `{e}` function in the Python standard library automatically parallelizes any loop.",
    "The {E} library for Python compiles regular expressions into GPU shaders by default.",
    "Calling `{e}()` in JavaScript sorts an array in constant time.",
    "The {E} package manager installs dependencies without network access using checksums.",
    "In C++, the `std::{e}` container guarantees O(1) lookup for any key type.",
    "The {E} framework automatically removes all memory leaks from Python programs.",
]


def build_domain_fake(rng: np.random.Generator, n: int) -> list[str]:
    """领域伪事实（数学/代码，unknown；虚构名经程序化生成）。"""
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            t = _MATH_FAKE_TEMPLATES[int(rng.integers(len(_MATH_FAKE_TEMPLATES)))]
            out.append(t.format(E=dt1.make_fake_word(rng), n=str(rng.integers(7, 500))))
        else:
            t = _CODE_FAKE_TEMPLATES[int(rng.integers(len(_CODE_FAKE_TEMPLATES)))]
            e = dt1.make_fake_word(rng)
            out.append(t.format(E=e, e=e.lower()))
    return out


# ---------------------------------------------------------------------------
# 多领域真实文本片段（known；data/raw math problem + code text，英文为主）
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def _load_domain_texts(raw_dir: str, source: str, limit: int = 4000) -> tuple[str, ...]:
    """惰性加载 data/raw 领域文本（math: problem 字段；code: text 字段），截短句。

    失败（文件缺失/无 pyarrow）返回空 tuple，调用方降级（不用该源）。
    """
    try:
        import pyarrow.parquet as pq
    except Exception:
        return ()
    root = Path(raw_dir)
    try:
        if source == "math":
            files = sorted(root.glob("math/*.parquet"))[:1]
            if not files:
                return ()
            col = pq.read_table(str(files[0]), columns=["problem"]).slice(0, limit)["problem"]
            texts = [str(x) for x in col.to_pylist()]
        elif source == "code":
            files = sorted(root.glob("code/code__data__openstax*.parquet"))[:1] or \
                sorted(root.glob("code/*.parquet"))[:1]
            if not files:
                return ()
            col = pq.read_table(str(files[0]), columns=["text"]).slice(0, limit)["text"]
            texts = [str(x) for x in col.to_pylist()]
        else:
            return ()
    except Exception:
        return ()
    # 截短：取前 ~240 字符（L1 样本 T=48 token，过长无意义；保持完整句子风格）
    out = []
    for t in texts:
        t = " ".join(t.split())
        if 40 <= len(t) <= 400:
            out.append(t[:240])
    return tuple(out)


def build_domain_known(rng: np.random.Generator, n: int, raw_dir: str = "data/raw") -> list[str]:
    """多领域真实文本片段（known）：math problem + code text，英文为主。

    0.1B 训练分布为 fineweb_edu 英文：math/code 英文真实文本是有效的 known 多样性；
    zh 中文刻意排除（模型未训中文，known/unknown 标注语义都会错位）。
    """
    pool = list(_load_domain_texts(raw_dir, "math")) + list(_load_domain_texts(raw_dir, "code"))
    if not pool:
        return []
    idx = rng.integers(0, len(pool), size=n)
    return [pool[int(i)] for i in idx]


# ---------------------------------------------------------------------------
# 统一接口：v2 多样化真值数据集
# ---------------------------------------------------------------------------
def build_diverse_truth_dataset_v2(
    rng: np.random.Generator,
    n_known: int,
    n_unknown: int,
    extra_known: list[str] | None = None,
    raw_dir: str = "data/raw",
) -> tuple[list[str], np.ndarray]:
    """v2 锚集：known(扩展真实事实+否定真实+contrast真实+多领域真实文本+extra_known)
    + unknown(v1 多句式伪事实+near-miss 细粒度错误+跨领域混搭+领域伪事实)。

    extra_known：由调用方传入的分布内文本（val shard 解码段），并入 known 池。
    返回 (texts, labels)：known=0、unknown=2（对齐 kal_truth_finetune 标签语义）。
    """
    # ---- known 池 ----
    real_v1 = dt1.build_real_statements(rng, n_known // 3)
    ct_texts, ct_labels = dt1.build_contrast_triplets(rng, n_known)
    known_pool = (
        list(real_v1)
        + list(_REAL_STATEMENTS_V2)
        + list(_REAL_NEGATIONS_V2)
        + [t for t, lb in zip(ct_texts, ct_labels) if lb == 0]
        + build_domain_known(rng, n_known // 2, raw_dir)
    )
    if extra_known:
        known_pool += list(extra_known)
    texts: list[str] = []
    labels: list[int] = []
    idx = rng.permutation(len(known_pool))
    i = 0
    while len(texts) < n_known:
        texts.append(known_pool[int(idx[i % len(idx)])])
        labels.append(0)
        i += 1
    # ---- unknown 池 ----
    n_per = max(4, n_unknown // 4)
    fake_short = dt1.build_fake_short(rng, n_per)
    fake_q = dt1.build_fake_question(rng, n_per)
    ct_fake = [t for t, lb in zip(*dt1.build_contrast_triplets(rng, n_per * 2)) if lb == 2]
    near_miss = build_near_miss(rng, n_per)
    mashup = build_mashup(rng, n_per)
    domain_fake = build_domain_fake(rng, n_per)
    unknown_pool = fake_short + fake_q + ct_fake + near_miss + mashup + domain_fake
    jdx = rng.permutation(len(unknown_pool))
    j = 0
    n_done = 0
    while n_done < n_unknown:
        texts.append(unknown_pool[int(jdx[j % len(jdx)])])
        labels.append(2)
        j += 1
        n_done += 1
    return texts, np.array(labels, dtype=np.int64)


# ---------------------------------------------------------------------------
# 预测反馈循环 cloze 池（§27.2：模型预测对错打标）
# ---------------------------------------------------------------------------
def build_cloze_pool(rng: np.random.Generator, n: int) -> list[tuple[str, str]]:
    """可验证 cloze 题池 (prompt, answer)：模型贪心补全后按对错打标。

    三类（均衡）：
    - 程序算术（"What is 17 + 25?" → "42"；含减法/小乘法/数列下一项）；
    - 常识事实（"The capital of France is" → "Paris"）；
    - 较难数学/事实（模型大概率答错——提供 unknown 侧样本）。
    """
    pool: list[tuple[str, str]] = []
    per = max(1, n // 3)
    # 程序算术（可验证，难度可调）
    for _ in range(per):
        kind = int(rng.integers(4))
        if kind == 0:
            a, b = int(rng.integers(3, 60)), int(rng.integers(3, 60))
            pool.append((f"What is {a} + {b}?", str(a + b)))
        elif kind == 1:
            a, b = int(rng.integers(20, 90)), int(rng.integers(3, 19))
            pool.append((f"What is {a} - {b}?", str(a - b)))
        elif kind == 2:
            a, b = int(rng.integers(2, 12)), int(rng.integers(2, 12))
            pool.append((f"What is {a} times {b}?", str(a * b)))
        else:
            start, step = int(rng.integers(1, 10)), int(rng.integers(2, 7))
            seq = [start + step * k for k in range(4)]
            pool.append((f"What comes next: {seq[0]}, {seq[1]}, {seq[2]}, {seq[3]},", str(seq[3] + step)))
    # 常识事实 cloze（短答案可验证）
    common = [
        ("The capital of France is", "Paris"),
        ("The capital of Japan is", "Tokyo"),
        ("Water is made of hydrogen and", "oxygen"),
        ("The opposite of hot is", "cold"),
        ("The Earth orbits the", "Sun"),
        ("The chemical symbol for gold is", "Au"),
        ("The largest ocean on Earth is the", "Pacific"),
        ("The currency of Japan is the", "yen"),
        ("The first month of the year is", "January"),
        ("The color of a ripe banana is", "yellow"),
        ("Humans breathe in oxygen and breathe out", "carbon dioxide"),
        ("The planet closest to the Sun is", "Mercury"),
        ("The author of Romeo and Juliet is", "Shakespeare"),
        ("The boiling point of water in degrees Celsius is", "100"),
        ("The number of days in a week is", "seven"),
        ("The animal known as man's best friend is the", "dog"),
    ]
    for _ in range(per):
        pool.append(common[int(rng.integers(len(common)))])
    # 较难事实/数学（0.1B 大概率答错 → unknown 侧来源）
    hard = [
        ("The derivative of x squared is", "2x"),
        ("The French Revolution began in the year", "1789"),
        ("The speed of light in kilometers per second is about", "300000"),
        ("The square root of 144 is", "12"),
        ("The largest planet in the solar system is", "Jupiter"),
        ("The chemical symbol for sodium is", "Na"),
        ("The year World War II ended was", "1945"),
        ("The sum of angles in a triangle in degrees is", "180"),
        ("The capital of Australia is", "Canberra"),
        ("The powerhouse of the cell is the", "mitochondria"),
        ("The value of pi to two decimal places is", "3.14"),
        ("The longest river in Africa is the", "Nile"),
    ]
    for _ in range(n - 2 * per):
        pool.append(hard[int(rng.integers(len(hard)))])
    return pool


def _arith_distractors(rng: np.random.Generator, ans: str, k: int) -> list[str]:
    """算术题数字干扰项：答案 ±小偏移/×10 级错误（保证 ≠ 正确答案）。"""
    v = int(ans)
    cands = set()
    while len(cands) < k:
        kind = int(rng.integers(3))
        if kind == 0:
            w = v + int(rng.integers(1, 10)) * int(rng.choice([-1, 1]))
        elif kind == 1:
            w = v + int(rng.integers(1, 3)) * int(rng.choice([-1, 1]))
        else:
            w = v * 10 if rng.random() < 0.5 else max(0, v // 10 + int(rng.integers(0, 3)))
        if w != v and w >= 0:
            cands.add(str(w))
    return list(cands)


def build_cloze_pool_mc(rng: np.random.Generator, n: int, n_distract: int = 3
                        ) -> list[tuple[str, str, list[str]]]:
    """多候选 cloze 题池 (prompt, answer, distractors)：候选打分型预测反馈循环用。

    干扰项：算术题=邻近数字（_arith_distractors）；事实题=同类别其他答案（保证 ≠ 答案）。
    模型对全部候选打分（logprob argmax = 预测），按预测对错打标——比贪心生成快约一个
    数量级（批量短前向 vs 逐 token 串行生成），且"模型自己的选择 vs 真值"语义相同。
    """
    base = build_cloze_pool(rng, n)
    answers = [a for _, a in base]
    out: list[tuple[str, str, list[str]]] = []
    for prompt, ans in base:
        if ans.lstrip("-").isdigit():
            ds = _arith_distractors(rng, ans, n_distract)
        else:
            pool = [a for a in answers if a.lower() != ans.lower() and not a.lstrip("-").isdigit()]
            ds = []
            while len(ds) < n_distract and pool:
                c = pool[int(rng.integers(len(pool)))]
                if c not in ds:
                    ds.append(c)
        out.append((prompt, ans, ds))
    return out
