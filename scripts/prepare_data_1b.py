"""1B 模型训练数据集准备：多领域混合 ~10B tokens → uint16 shard（Colab 友好）。

复用现有 32k BPE tokenizer（data/tokenizer/tokenizer.json，**不重训**），
按分领域配比流式构建混合语料并落盘 uint16 shard。目标运行环境 Google Colab
（磁盘有限、会话会断）：全部 shard 原子写（*.part→rename），进度落
_progress.json 支持断点续跑（已完成的源跳过，未完成的源从头重收，重复少量
可接受）。

配比（默认 target_tokens=10B，可用 --mix 覆盖）：
  - 英文主力 fineweb-edu **sample-100BT**   ~73%（7.3B，原生流式；10BT 不够 10B 目标）
  - 数学   NuminaMath-CoT + FineMath-4+     ~12%（1.2B，两子源 4:8 混合）
  - 代码   HuggingFaceTB/cosmopedia         ~10%（1.0B，HTTP GET→parquet）
  - 中文   epfml/FineWeb2-HQ cmn_Hani       ~5% （0.5B，HTTP GET→parquet，--no_zh 关闭）

数学子源 4:8：NuminaMath-CoT（~430M tokens，CoT 解题格式）先收 math 配额的 1/3，
剩余 2/3 由 FineMath-4+（finemath-4plus/ 子集，64 片 ~9.6B tokens，HfApi 探明）
补足；numina 若提前耗尽，finemath 自动补满整个 math 配额。

访问要点（沿用 0p5b 实测结论，2026-07-30/31）：
  * fineweb-edu 原生提供 parquet，load_dataset 流式直连可用（本机 ~37k tok/s）；
  * 其余源因 hf-mirror 不暴露 auto-converted parquet 分支、且 huggingface_hub
    元数据 HEAD 校验在直连/镜像下均失败，故一律 **HTTP GET 直接下载 parquet
    分片 → load_dataset("parquet", 本地文件) 流式**（HEAD/GET 已验证 200）；
  * 断流重试 12 次指数退避；parquet 分片本地缓存于 data/raw/（--raw_dir 可改，
    已下载的分片重跑时直接复用，不下第二遍）。

Colab 用法（会话断了重跑同一命令即可续）：
  !python scripts/prepare_data_1b.py --target_tokens 10B --out /content/drive/MyDrive/shards_1b
  建议 --raw_dir 指向本地盘（/content/raw，parquet 缓存），--out 指 Google Drive
  （shard 产物，~20GB/10B tokens）；断连后已完成源被 _progress.json 跳过。

预计耗时（本机 2026-07-31 冒烟实测外推，假设 Colab 网络与本机同档）：
  - fineweb-edu sample-100BT 流式：爬升后稳定 ~0.6–1.4M tok/s → 7.3B 约 1.5–3.5h（全程瓶颈）；
  - 本地 parquet 各源（tokenize 限速）：numina ~0.4M / finemath ~1.3M / code ~1.2M / zh ~1.0M tok/s
    → math 1.2B ≈ 30min、code 1.0B ≈ 15min、zh 0.5B ≈ 10min；
  - parquet 下载 ~12GB（numina 5 片 + finemath ~6 片 + code ~17 片 + zh ~4 片），6–14MB/s ≈ 0.5h，与流式交错；
  - 合计 ~3–5 小时量级，Colab 单次会话（12h）可完成；断连重跑损失 ≤ 当前源（已完成源跳过，
    未完成源从头重收，其残留 shard 保留在盘上——stats 按磁盘实测，重复量如实反映）。

val 集独立：从各源开头各取 val_tokens/N 凑成 val，其余全部进 train。
文档边界统一插 <|endoftext|>(id 0)，与 prepare_data.py 一致。
词表硬约束：所有 token id 必须 < 32768；全部 shard 写完后 numpy 全量扫描
max_id 并 assert。

用法：
  python scripts/prepare_data_1b.py --target_tokens 30M --val_tokens 2M --out data/shards_1b_smoke  # 冒烟
  python scripts/prepare_data_1b.py --target_tokens 10B                # 全量 ~10B
  python scripts/prepare_data_1b.py --target_tokens 10B --no_zh       # 省中文源
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EOT = "<|endoftext|>"
VOCAB_LIMIT = 32768  # 32768–32772 为保留特殊 token，语料 token id 必须 < 32768

# ---------------------------------------------------------------------------
# 数据混合配比（键 → 默认占比）。可用 --mix "fineweb_edu=0.73,math=0.12,..." 覆盖。
# ---------------------------------------------------------------------------
DEFAULT_MIX = {
    "fineweb_edu": 0.73,   # 英文高质量主力（sample-100BT）
    "math": 0.12,          # NuminaMath-CoT : FineMath-4+ = 4:8
    "code": 0.10,          # cosmopedia
    "zh": 0.05,            # FineWeb2-HQ cmn_Hani
}
NUMINA_SHARE_OF_MATH = 4 / 12  # math 配额中 NuminaMath 占 4/(4+8)

# 各源 parquet 分片下载上限（按 10B 配额 ×1.2 余量 + 实测单片产出估算）
SHARD_DL_CAP = {
    "math_numina": 5,     # NuminaMath train 全 5 片（~1.2GB，~430M tokens > 400M 配额）
    "math_finemath": 10,  # finemath-4plus 单片 ~286MB / 估算 ~150M tok → 10 片 ≈ 1.5B（配额 0.8B×1.2）
    "code": 24,           # cosmopedia 实测 ~60M tok/片（0p5b 5 片出 300M）→ 24 片 ≈ 1.4B（配额 1.0B×1.2）
    "zh": 6,              # cmn_Hani 实测单片 ≥150M tok（0p5b 1 片出 150M）→ 6 片 ≈ 0.9B（配额 0.5B×1.2）
}

RAW_DIR = ROOT / "data" / "raw"          # parquet 原始下载缓存（--raw_dir 可改）
TOK_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
PROGRESS_NAME = "_progress.json"


# ---------------------------------------------------------------------------
# HTTP GET 直接下载（绕过 huggingface_hub 元数据 HEAD 校验；已验证 200）
# ---------------------------------------------------------------------------
def http_download(repo: str, fname: str, out_dir: Path, tag: str) -> Path:
    """把仓库 blob 直接 GET 到本地，返回路径；已存在则跳过。*.part 原子落盘。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = fname.replace("/", "__")
    dst = out_dir / f"{tag}__{safe}"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_name(dst.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    tmp.replace(dst)
    sz = dst.stat().st_size / 1e6
    print(f"    [dl] {tag}::{Path(fname).name} {sz:.1f}MB 用时 {time.time()-t0:.0f}s")
    return dst


def list_parquet(repo: str, prefix: str, cap: int) -> list[str]:
    """用 HfApi 列出 prefix 下的 parquet 分片（仅列名，不下载）。

    分片按"子目录（config）轮询"选取：cosmopedia 等多 config 数据集若直接取
    排序前 cap 个会全部落在同一个 config（字母序），导致领域单一。
    """
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(repo, repo_type="dataset")
    pq = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if len(pq) <= cap:
        return pq
    groups: dict[str, list[str]] = {}
    for f in pq:
        groups.setdefault(str(Path(f).parent), []).append(f)
    if len(groups) <= 1:
        return pq[:cap]
    picked: list[str] = []
    gkeys = sorted(groups)
    i = 0
    while len(picked) < cap and any(groups[k] for k in gkeys):
        k = gkeys[i % len(gkeys)]
        if groups[k]:
            picked.append(groups[k].pop(0))
        i += 1
    return picked


# ---------------------------------------------------------------------------
# 各源文档流：统一产出 (text) 迭代器
# ---------------------------------------------------------------------------
def stream_fineweb_edu():
    """英文主力：fineweb-edu sample-100BT 原生流式（10BT 样本不足以支撑 10B 目标）。"""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-100BT", split="train", streaming=True)
    for row in ds:
        t = row.get("text", "")
        if t:
            yield t


def stream_local_parquet(repo: str, prefix: str, cap: int, tag: str, pick):
    """HTTP GET 下载 parquet 分片 → 本地流式，pick(row)->text 抽取文本。"""
    from datasets import load_dataset
    fnames = list_parquet(repo, prefix, cap)
    if not fnames:
        raise RuntimeError(f"{tag}: {repo} 下未找到 {prefix}*.parquet")
    print(f"    [{tag}] 计划使用 {len(fnames)} 个分片")
    for fn in fnames:
        local = http_download(repo, fn, RAW_DIR / tag, tag)
        ds = load_dataset("parquet", data_files=str(local), split="train", streaming=True)
        for row in ds:
            t = pick(row)
            if t:
                yield t


def _pick_text(r):
    return r.get("text", "")


def stream_math_numina():
    """数学子源 A：NuminaMath-CoT，problem+solution 拼成 CoT 文档。"""
    def pick(r):
        p, s = r.get("problem", ""), r.get("solution", "")
        if not p:
            return ""
        return f"Problem:\n{p}\n\nSolution:\n{s}" if s else p
    yield from stream_local_parquet(
        # tag 沿用 "math"：与 0p5b 的 data/raw/math/ 缓存同名，已下载分片直接复用
        "AI-MO/NuminaMath-CoT", "data/train-", SHARD_DL_CAP["math_numina"], "math", pick
    )


def stream_math_finemath():
    """数学子源 B：FineMath-4+（finemath-4plus/ 子集，HfApi 探明 64 片，质量最高档）。"""
    yield from stream_local_parquet(
        "HuggingFaceTB/finemath", "finemath-4plus/", SHARD_DL_CAP["math_finemath"],
        "math_finemath", _pick_text
    )


def stream_code():
    """代码：cosmopedia。多 config 子目录轮询取片。"""
    yield from stream_local_parquet(
        "HuggingFaceTB/cosmopedia", "data/", SHARD_DL_CAP["code"], "code", _pick_text
    )


def stream_zh():
    """中文：FineWeb2-HQ cmn_Hani。"""
    yield from stream_local_parquet(
        "epfml/FineWeb2-HQ", "cmn_Hani/", SHARD_DL_CAP["zh"], "zh", _pick_text
    )


SOURCE_FN = {
    "fineweb_edu": stream_fineweb_edu,
    "code": stream_code,
    "zh": stream_zh,
    # "math" 为双子源，main 中特判
}


# ---------------------------------------------------------------------------
# ShardWriter（uint16 <u2 原始字节；*.part→rename 原子写，断连不留半个 shard）
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int, split: str):
        from tais_obsidian.data.memmap import write_shard
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.split = split
        self.buf = np.empty(shard_size, dtype="<u2")
        self.filled = 0
        self.count = self._next_idx()   # 断点续跑：接在已有 shard 后面编号
        self.total = 0
        self._write = write_shard

    def _next_idx(self) -> int:
        n = 0
        for p in self.out_dir.glob(f"{self.split}_*.bin"):
            try:
                n = max(n, int(p.stem.split("_")[-1]) + 1)
            except ValueError:
                pass
        return n

    def push(self, tokens: np.ndarray) -> None:
        pos = 0
        while pos < tokens.size:
            take = min(self.shard_size - self.filled, tokens.size - pos)
            self.buf[self.filled:self.filled + take] = tokens[pos:pos + take]
            self.filled += take
            pos += take
            if self.filled == self.shard_size:
                self.flush()
        self.total += tokens.size

    def flush(self) -> None:
        if self.filled == 0:
            return
        path = self.out_dir / f"{self.split}_{self.count:03d}.bin"
        tmp = path.with_name(path.name + ".part")
        self._write(tmp, self.buf[:self.filled])
        os.replace(tmp, path)  # 原子 rename
        print(f"    [shard] {path.name}: {self.filled/1e6:.1f}M tokens", flush=True)
        self.count += 1
        self.filled = 0


def parse_target(s: str) -> int:
    s = s.strip().upper()
    if s.endswith("B"):
        return int(float(s[:-1]) * 1e9)
    if s.endswith("M"):
        return int(float(s[:-1]) * 1e6)
    if s.endswith("K"):
        return int(float(s[:-1]) * 1e3)
    return int(s)


def parse_mix(s: str) -> dict[str, float]:
    mix = {}
    for kv in s.split(","):
        k, v = kv.split("=")
        mix[k.strip()] = float(v)
    return mix


# ---------------------------------------------------------------------------
# 断点续跑进度：源名 → {"tokens": n, "done": bool}；每源完成即落盘（原子写）
# ---------------------------------------------------------------------------
def load_progress(out_dir: Path) -> dict:
    p = out_dir / PROGRESS_NAME
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[resume] {PROGRESS_NAME} 损坏，忽略（全部源重收）")
    return {}


def save_progress(out_dir: Path, prog: dict) -> None:
    p = out_dir / PROGRESS_NAME
    tmp = p.with_name(p.name + ".part")
    tmp.write_text(json.dumps(prog, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------
def main() -> None:
    global RAW_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=str, default="10B", help="如 30M / 10B")
    ap.add_argument("--val_tokens", type=str, default="20M")
    ap.add_argument("--shard_tokens", type=int, default=50_000_000, help="~50M token/片")
    ap.add_argument("--mix", type=str, default="", help="覆盖配比 fineweb_edu=0.73,math=0.12,code=0.1,zh=0.05")
    ap.add_argument("--no_zh", action="store_true", help="关闭中文源，其配额转给英文主力")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "shards_1b")
    ap.add_argument("--raw_dir", type=Path, default=RAW_DIR, help="parquet 下载缓存目录（Colab 可指本地盘）")
    args = ap.parse_args()
    RAW_DIR = args.raw_dir

    target = parse_target(args.target_tokens)
    val_total = parse_target(args.val_tokens)
    mix = parse_mix(args.mix) if args.mix else dict(DEFAULT_MIX)
    if args.no_zh and "zh" in mix:
        mix["fineweb_edu"] = mix.get("fineweb_edu", 0) + mix.pop("zh")
    # 归一化
    tot_w = sum(mix.values())
    mix = {k: v / tot_w for k, v in mix.items()}

    print(f"[plan] 目标 {target/1e9:.2f}B tokens（val {val_total/1e6:.0f}M），输出 {args.out}")
    for k, v in mix.items():
        print(f"    {k:12s} {v*100:5.1f}%  ≈ {target*v/1e6:7.0f}M tokens")

    # tokenizer 复用（不重训）
    from tais_obsidian.tokenizer_io import TokenizerIO
    tok = TokenizerIO(TOK_PATH)
    assert tok.vocab_size < 65536, f"vocab {tok.vocab_size} 超出 uint16"
    eot = tok.eot_id
    print(f"[tokenizer] 复用 {TOK_PATH}（vocab={tok.vocab_size}, eot={eot}，不重训）")

    args.out.mkdir(parents=True, exist_ok=True)
    stats_path = args.out / "_stats.txt"
    val_writer = ShardWriter(args.out, val_total, "val")
    train_writer = ShardWriter(args.out, args.shard_tokens, "train")

    # ---- 断点续跑：已完成源跳过 ----
    progress = load_progress(args.out)
    done_srcs = {k for k, v in progress.items() if v.get("done")}
    if done_srcs:
        print(f"[resume] 已完成源（跳过）: {sorted(done_srcs)}")
        if val_writer.count or train_writer.count:
            print(f"[resume] 已有 val shard ×{val_writer.count} / train shard ×{train_writer.count}，"
                  f"新 shard 续编号；未完成源将从头重收（重复少量可接受）")

    per_source_val = val_total // max(1, len(mix))
    source_stats: dict[str, int] = {k: int(progress[k]["tokens"]) for k in done_srcs}

    def tokenize_docs(docs: list[str]) -> np.ndarray:
        encs = tok.encode_batch(docs)
        n = sum(len(e) + 1 for e in encs)
        out = np.empty(n, dtype="<u2")
        pos = 0
        for ids in encs:
            out[pos:pos + len(ids)] = ids
            pos += len(ids)
            out[pos] = eot
            pos += 1
        return out

    def run_stream(label: str, fn, val_left: int, train_left: int) -> int:
        """从 fn 文档流收集 val_left + train_left tokens 写入 writer，返回 train 实收。

        断流重试：长任务中网络抖动/SSL EOF 难免，流中断后重建流继续收集（已写入
        writer 的 tokens 保留）。重建流从源开头重读，会引入少量重复文档（占比
        <0.1%，预训练语料可接受）；重试间隔指数退避（上限 300s）。
        """
        retries = 0
        max_retries = 12
        batch: list[str] = []
        done = 0
        t_src = time.time()
        stream = None
        while train_left > 0:
            if stream is None:
                try:
                    stream = iter(fn())
                except Exception as e:  # noqa: BLE001
                    retries += 1
                    if retries > max_retries:
                        print(f"    [{label}] 流创建重试 {max_retries} 次仍失败（放弃该流）: "
                              f"{type(e).__name__}: {str(e)[:200]}")
                        break
                    wait = min(300, 15 * retries)
                    print(f"    [{label}] 流创建失败({retries}/{max_retries})，{wait}s 后重试: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    time.sleep(wait)
                    continue
            try:
                text = next(stream)
                retries = 0  # 正常取到文档，重置退避
            except StopIteration:
                print(f"    [{label}] 语料流耗尽，提前结束（实收 {done/1e6:.1f}M）")
                break
            except Exception as e:  # noqa: BLE001
                retries += 1
                if retries > max_retries:
                    print(f"    [{label}] 重试 {max_retries} 次仍失败（放弃该流）: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    break
                wait = min(300, 15 * retries)
                print(f"    [{label}] 流中断({retries}/{max_retries})，{wait}s 后重建流续收: "
                      f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                time.sleep(wait)
                stream = None
                continue
            batch.append(text)
            if len(batch) >= 128:
                arr = tokenize_docs(batch)
                batch = []
                if val_left > 0:
                    take = min(val_left, arr.size)
                    val_writer.push(arr[:take])
                    val_left -= take
                    arr = arr[take:]
                if arr.size:
                    take = min(train_left, arr.size)
                    train_writer.push(arr[:take])
                    train_left -= take
                    done += take
                    el = time.time() - t_src
                    print(f"    [{label}] train 已收 {done/1e6:.1f}M / 还需 {train_left/1e6:.1f}M  "
                          f"({done/el/1e3:.0f}k tok/s)", flush=True)
        if batch:  # 残余
            arr = tokenize_docs(batch)
            if val_left > 0:
                take = min(val_left, arr.size)
                val_writer.push(arr[:take])
                arr = arr[take:]
            if arr.size and train_left > 0:
                take = min(train_left, arr.size)
                train_writer.push(arr[:take])
                done += take
        return done

    t_start = time.time()
    for src, weight in mix.items():
        if src in done_srcs:
            print(f"\n[src:{src}] 已完成（_progress.json），跳过 —— 已收 "
                  f"{progress[src]['tokens']/1e6:.1f}M")
            continue
        src_target = int(target * weight)
        val_q = per_source_val
        train_q = max(0, src_target - val_q)
        print(f"\n[src:{src}] 目标 {src_target/1e6:.0f}M（val {val_q/1e6:.1f}M + train {train_q/1e6:.1f}M）")
        t_src = time.time()
        if src == "math":
            # 双子源 4:8——numina 先收 math 配额的 1/3，finemath 补足剩余（含 numina 短缺）
            numina_q = max(0, int(src_target * NUMINA_SHARE_OF_MATH) - val_q)
            done_n = run_stream("math/numina", stream_math_numina, val_q, numina_q)
            done_f = run_stream("math/finemath", stream_math_finemath, 0, train_q - done_n)
            done = done_n + done_f
            print(f"    [src:math] numina {done_n/1e6:.1f}M + finemath {done_f/1e6:.1f}M "
                  f"= {done/1e6:.1f}M")
        else:
            done = run_stream(src, SOURCE_FN[src], val_q, train_q)
        source_stats[src] = done + val_q if src != "math" else done + val_q
        # 每源完成即落进度（原子写）——断连重跑时跳过
        progress[src] = {
            "tokens": int(source_stats[src]),
            "done": True,
            "elapsed_min": round((time.time() - t_src) / 60, 1),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_progress(args.out, progress)
        print(f"    [progress] {src} 完成 → {PROGRESS_NAME}", flush=True)

    val_writer.flush()
    train_writer.flush()

    # ---- max_id 全量扫描（10B 数据第一道关：token id 必须 < 32768）----
    print("\n[check] 全量扫描各 shard max token id ...")
    shard_paths = sorted(args.out.glob("train_*.bin")) + sorted(args.out.glob("val_*.bin"))
    overall_max = 0
    per_shard: list[tuple[str, int]] = []
    for p in shard_paths:
        a = np.memmap(p, dtype="<u2", mode="r")
        m = int(a.max()) if a.size else 0
        per_shard.append((p.name, m))
        overall_max = max(overall_max, m)
    for name, m in per_shard:
        print(f"    {name}: max_id={m}")
    print(f"[check] overall max_id = {overall_max}（上限 {VOCAB_LIMIT}）")
    assert overall_max < VOCAB_LIMIT, (
        f"max_id {overall_max} ≥ {VOCAB_LIMIT}：语料混入保留特殊 token，需排查！")

    # ---- 汇总（totals 按磁盘实际 shard 计，断点续跑重复也如实反映）----
    train_bins = sorted(args.out.glob("train_*.bin"))
    val_bins = sorted(args.out.glob("val_*.bin"))
    train_total = sum(p.stat().st_size for p in train_bins) // 2
    val_total_disk = sum(p.stat().st_size for p in val_bins) // 2
    total = train_total + val_total_disk
    el = time.time() - t_start
    lines = [
        f"# 1B 数据集 stats  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"target={target}  val_target={val_total}  shard_tokens={args.shard_tokens}",
        f"mix={mix}",
        f"train_total={train_total}  val_total={val_total_disk}  total={total}  （按磁盘 shard 实测）",
        f"train_shards={len(train_bins)}  val_shards={len(val_bins)}",
        f"max_id={overall_max} (<{VOCAB_LIMIT} OK)",
        f"elapsed_min={el/60:.1f}（本次运行；断点续跑不含已完成源耗时）",
        "per_source_tokens:",
    ]
    for k, v in source_stats.items():
        tag = "（此前已完成）" if k in done_srcs else ""
        lines.append(f"  {k}={v} ({v/1e6:.1f}M){tag}")
    stats_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[done] train {train_total/1e6:.1f}M + val {val_total_disk/1e6:.1f}M "
          f"= {total/1e6:.1f}M tokens，{len(train_bins)}+{len(val_bins)} shards（磁盘实测）")
    print(f"[done] 各源 tokens: " + ", ".join(f"{k}={v/1e6:.1f}M" for k, v in source_stats.items()))
    print(f"[done] max_id={overall_max} < {VOCAB_LIMIT} ✓")
    print(f"[done] 本次耗时 {el/60:.1f} min；stats → {stats_path}")
    if total < target * 0.9:
        print(f"[warn] 实际 {total/1e6:.0f}M < 目标 {target/1e6:.0f}M 的 90%")


if __name__ == "__main__":
    main()
