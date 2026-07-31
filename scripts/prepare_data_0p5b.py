"""0.5B 模型训练数据集准备：多领域混合 ~3B tokens → uint16 shard。

复用现有 32k BPE tokenizer（data/tokenizer/tokenizer.json，**不重训**），
按 OLMo 课程思路的分领域配比流式构建混合语料并落盘 uint16 shard。

配比（默认 target_tokens=3B，可用 --mix 覆盖）：
  - 英文主力 fineweb-edu sample-10BT   ~70%  （原生流式）
  - 数学   AI-MO/NuminaMath-CoT        ~15%  （HTTP GET→parquet 本地流式）
  - 代码   HuggingFaceTB/cosmopedia    ~10%  （HTTP GET→parquet；the-stack-v2 gated 401 替代）
  - 中文   epfml/FineWeb2-HQ cmn_Hani  ~5%   （HTTP GET→parquet，可选 --no_zh 关闭）

访问要点（2026-07-30 实测）：
  * fineweb-edu 原生提供 parquet，load_dataset 流式直连可用（~37k tok/s）；
  * 其余三源因 hf-mirror 不暴露 auto-converted parquet 分支、且 huggingface_hub
    元数据 HEAD 校验在本机直连/镜像下均失败，故改用 **HTTP GET 直接下载 parquet
    分片 → load_dataset("parquet", 本地文件) 流式**（HEAD/GET 已验证 200，5.8–14MB/s）；
  * the-stack-v2 为 gated（blob 401 需申请+token），匿名环境不可用 → 代码源改用
    cosmopedia（开放 parquet，Apache 2.0，选型文档已列）。

val 集独立：从各源开头各取 val_tokens/4 凑成 val，其余全部进 train。
文档边界统一插 <|endoftext|>(id 0)，与 prepare_data.py 一致。

用法：
  python scripts/prepare_data_0p5b.py --target_tokens 100M            # 小规模验证管线
  python scripts/prepare_data_0p5b.py --target_tokens 3B              # 全量 ~3B
  python scripts/prepare_data_0p5b.py --target_tokens 3B --no_zh     # 省中文源
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EOT = "<|endoftext|>"

# ---------------------------------------------------------------------------
# 数据混合配比（键 → 默认占比）。可用 --mix "fineweb_edu=0.7,math=0.15,..." 覆盖。
# ---------------------------------------------------------------------------
DEFAULT_MIX = {
    "fineweb_edu": 0.70,   # 英文高质量主力
    "math": 0.15,          # NuminaMath-CoT
    "code": 0.10,          # cosmopedia（the-stack-v2 gated 替代）
    "zh": 0.05,            # FineWeb2-HQ cmn_Hani
}

# 各源 parquet 分片下载上限（防止单源下爆磁盘；够用即可，不够再调）
SHARD_DL_CAP = {
    "math": 6,     # NuminaMath train 5 分片（~1.2GB）全要
    "code": 24,    # cosmopedia 多 config，按需
    "zh": 3,       # FineWeb2-HQ cmn_Hani 单片 ~1.2GB，3 片约够 150M tok
}

RAW_DIR = ROOT / "data" / "raw"          # parquet 原始下载缓存
TOK_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"


# ---------------------------------------------------------------------------
# HTTP GET 直接下载（绕过 huggingface_hub 元数据 HEAD 校验；已验证 200）
# ---------------------------------------------------------------------------
def http_download(repo: str, fname: str, out_dir: Path, tag: str) -> Path:
    """把仓库 blob 直接 GET 到本地，返回路径；已存在则跳过。带续传/校验大小。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = fname.replace("/", "__")
    dst = out_dir / f"{tag}__{safe}"
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
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
    """英文主力：原生流式（直连可达）。"""
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
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


def stream_math():
    """数学：NuminaMath-CoT，problem+solution 拼成 CoT 文档。"""
    def pick(r):
        p, s = r.get("problem", ""), r.get("solution", "")
        if not p:
            return ""
        return f"Problem:\n{p}\n\nSolution:\n{s}" if s else p
    yield from stream_local_parquet(
        "AI-MO/NuminaMath-CoT", "data/train-", SHARD_DL_CAP["math"], "math", pick
    )


def stream_code():
    """代码：cosmopedia（the-stack-v2 gated 401 替代）。多 config 取前面若干分片。"""
    def pick(r):
        return r.get("text", "")
    # cosmopedia 分 config 子目录；取 web_samples_v1 + auto_math_text 等混合
    yield from stream_local_parquet(
        "HuggingFaceTB/cosmopedia", "data/", SHARD_DL_CAP["code"], "code", pick
    )


def stream_zh():
    """中文：FineWeb2-HQ cmn_Hani。"""
    def pick(r):
        return r.get("text", "")
    yield from stream_local_parquet(
        "epfml/FineWeb2-HQ", "cmn_Hani/", SHARD_DL_CAP["zh"], "zh", pick
    )


SOURCE_FN = {
    "fineweb_edu": stream_fineweb_edu,
    "math": stream_math,
    "code": stream_code,
    "zh": stream_zh,
}


# ---------------------------------------------------------------------------
# ShardWriter（与 prepare_data.py 同格式：uint16 <u2 原始字节）
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, out_dir: Path, shard_size: int):
        from tais_obsidian.data.memmap import write_shard
        self.out_dir = out_dir
        self.shard_size = shard_size
        self.buf = np.empty(shard_size, dtype="<u2")
        self.filled = 0
        self.count = 0
        self.total = 0
        self._write = write_shard

    def push(self, tokens: np.ndarray, split: str) -> None:
        pos = 0
        while pos < tokens.size:
            take = min(self.shard_size - self.filled, tokens.size - pos)
            self.buf[self.filled:self.filled + take] = tokens[pos:pos + take]
            self.filled += take
            pos += take
            if self.filled == self.shard_size:
                self.flush(split)
        self.total += tokens.size

    def flush(self, split: str) -> None:
        if self.filled == 0:
            return
        path = self.out_dir / f"{split}_{self.count:03d}.bin"
        self._write(path, self.buf[:self.filled])
        print(f"    [shard] {path.name}: {self.filled/1e6:.1f}M tokens")
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
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=str, default="3B", help="如 100M / 3B")
    ap.add_argument("--val_tokens", type=str, default="10M")
    ap.add_argument("--shard_tokens", type=int, default=50_000_000, help="~50M token/片")
    ap.add_argument("--mix", type=str, default="", help="覆盖配比 fineweb_edu=0.7,math=0.15,code=0.1,zh=0.05")
    ap.add_argument("--no_zh", action="store_true", help="关闭中文源，其配额转给英文主力")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "shards_0p5b")
    args = ap.parse_args()

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
    val_writer = ShardWriter(args.out, val_total)
    train_writer = ShardWriter(args.out, args.shard_tokens)

    # 每个源先取 val 配额的一部分，再取 train 配额
    per_source_val = val_total // max(1, len(mix))
    source_stats: dict[str, int] = {}

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

    t_start = time.time()
    for src, weight in mix.items():
        src_target = int(target * weight)
        # 该源 val 配额 + train 配额
        val_left = per_source_val
        train_left = src_target - per_source_val
        if train_left < 0:
            train_left = 0
        print(f"\n[src:{src}] 目标 {src_target/1e6:.0f}M（val {val_left/1e6:.1f}M + train {train_left/1e6:.1f}M）")
        fn = SOURCE_FN[src]
        # 断流重试：长任务（十几小时）中网络抖动/SSL EOF 难免，流中断后重建流继续
        # 收集（已写入 writer 的 tokens 保留）。重建流从源开头重读，会引入少量重复
        # 文档（占比 <0.1%，预训练语料可接受）；重试间隔指数退避（上限 300s）。
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
                        print(f"    [src:{src}] 流创建重试 {max_retries} 次仍失败（跳过该源）: "
                              f"{type(e).__name__}: {str(e)[:200]}")
                        break
                    wait = min(300, 15 * retries)
                    print(f"    [{src}] 流创建失败({retries}/{max_retries})，{wait}s 后重试: "
                          f"{type(e).__name__}: {str(e)[:120]}", flush=True)
                    time.sleep(wait)
                    continue
            try:
                text = next(stream)
                retries = 0  # 正常取到文档，重置退避
            except StopIteration:
                print(f"    [{src}] 语料流耗尽，提前结束")
                break
            except Exception as e:  # noqa: BLE001
                retries += 1
                if retries > max_retries:
                    print(f"    [src:{src}] 重试 {max_retries} 次仍失败（跳过该源）: "
                          f"{type(e).__name__}: {str(e)[:200]}")
                    break
                wait = min(300, 15 * retries)
                print(f"    [{src}] 流中断({retries}/{max_retries})，{wait}s 后重建流续收: "
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
                    val_writer.push(arr[:take], "val")
                    val_left -= take
                    arr = arr[take:]
                if arr.size:
                    take = min(train_left, arr.size)
                    train_writer.push(arr[:take], "train")
                    train_left -= take
                    done += take
                    el = time.time() - t_src
                    print(f"    [{src}] train 已收 {done/1e6:.1f}M / 还需 {train_left/1e6:.1f}M  "
                          f"({done/el/1e3:.0f}k tok/s)", flush=True)
        if batch:  # 残余
            arr = tokenize_docs(batch)
            if val_left > 0:
                take = min(val_left, arr.size)
                val_writer.push(arr[:take], "val")
                arr = arr[take:]
            if arr.size and train_left > 0:
                take = min(train_left, arr.size)
                train_writer.push(arr[:take], "train")
        source_stats[src] = src_target - train_left

    val_writer.flush("val")
    train_writer.flush("train")

    # ---- 汇总 ----
    total = val_writer.total + train_writer.total
    el = time.time() - t_start
    lines = [
        f"# 0.5B 数据集 stats  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"target={target}  val_target={val_total}  shard_tokens={args.shard_tokens}",
        f"mix={mix}",
        f"train_total={train_writer.total}  val_total={val_writer.total}  total={total}",
        f"train_shards={train_writer.count}  val_shards={val_writer.count}",
        f"elapsed_min={el/60:.1f}",
        "per_source_tokens:",
    ]
    for k, v in source_stats.items():
        lines.append(f"  {k}={v} ({v/1e6:.1f}M)")
    stats_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"[done] train {train_writer.total/1e6:.1f}M + val {val_writer.total/1e6:.1f}M "
          f"= {total/1e6:.1f}M tokens，{train_writer.count}+{val_writer.count} shards")
    print(f"[done] 各源 tokens: " + ", ".join(f"{k}={v/1e6:.1f}M" for k, v in source_stats.items()))
    print(f"[done] 总耗时 {el/60:.1f} min；stats → {stats_path}")
    if total < target * 0.9:
        print(f"[warn] 实际 {total/1e6:.0f}M < 目标 {target/1e6:.0f}M 的 90%")


if __name__ == "__main__":
    main()
