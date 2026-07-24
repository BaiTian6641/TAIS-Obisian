"""数据准备：流式读取 FineWeb-Edu → 自训 32k BPE tokenizer → uint16 shard 落盘。

流程：
1. 探测 huggingface.co 连通性，失败则切 HF_ENDPOINT=https://hf-mirror.com；
2. 流式累计 ~300MB 文本训练 BPE tokenizer（vocab 32768，byte_fallback，
   special tokens <|endoftext|>(0) <|im_start|>(1) <|im_end|>(2)）→ data/tokenizer/tokenizer.json；
3. 继续 tokenize 至目标 token 数（默认 120M），文档间插 <|endoftext|>，
   先取 ~2M 作 val，其余写 train shards（uint16，~20M token/片）。

若 FineWeb-Edu 完全不可达，fallback 到 wikitext-103 并在输出中明确标注偏差。
用法：python scripts/prepare_data.py [--target_tokens N] [--val_tokens N] [--tok_corpus_mb MB]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

# Windows 控制台 UTF-8
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EOT = "<|endoftext|>"
SPECIALS = [EOT, "<|im_start|>", "<|im_end|>"]


def probe(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except Exception:
        return False


def pick_endpoint() -> None:
    """HF 不可达时切镜像（须在 import datasets 之前设置）。"""
    if probe("https://huggingface.co"):
        print("[endpoint] huggingface.co 直连可达")
    else:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[endpoint] huggingface.co 不可达，切换 HF_ENDPOINT=https://hf-mirror.com")


def stream_fineweb_edu():
    """流式产出 fineweb-edu sample-10BT 文档文本。"""
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if text:
            yield text


def stream_wikitext():
    """fallback 语料：wikitext-103（标注偏差用）。"""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    for row in ds:
        text = row.get("text", "")
        if text and text.strip():
            yield text


def train_tokenizer(texts: list[str], vocab_size: int, out_path: Path) -> None:
    """用累计语料训练 BPE tokenizer 并保存。"""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(byte_fallback=True))
    # 先隔离 special tokens，再走 ByteLevel（GPT-2 风格字节覆盖）
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(Regex("|".join(re.escape(s) for s in SPECIALS)), behavior="isolated"),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIALS,  # 按顺序占 id 0/1/2
        byte_fallback=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(iter(texts), trainer)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(out_path))
    # 校验特殊符号 id
    check = Tokenizer.from_file(str(out_path))
    assert check.token_to_id(EOT) == 0, f"{EOT} id 应为 0，实际 {check.token_to_id(EOT)}"
    print(f"[tokenizer] 训练完成，vocab={check.get_vocab_size()} → {out_path}")


class ShardWriter:
    """累计 token 缓冲，满 shard_size 即落盘一个 shard。"""

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
            self.buf[self.filled : self.filled + take] = tokens[pos : pos + take]
            self.filled += take
            pos += take
            if self.filled == self.shard_size:
                self.flush(split)
        self.total += tokens.size

    def flush(self, split: str) -> None:
        if self.filled == 0:
            return
        path = self.out_dir / f"{split}_{self.count:03d}.bin"
        self._write(path, self.buf[: self.filled])
        print(f"[shard] {path.name}: {self.filled/1e6:.1f}M tokens")
        self.count += 1
        self.filled = 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=int, default=120_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--tok_corpus_mb", type=float, default=300.0)
    ap.add_argument("--vocab_size", type=int, default=32768)
    ap.add_argument("--shard_tokens", type=int, default=20_000_000)
    ap.add_argument("--out", type=Path, default=ROOT / "data")
    args = ap.parse_args()

    t0 = time.time()
    pick_endpoint()

    from tais_obsidian.data.memmap import write_shard  # noqa: F401  (确认包可导入)

    tok_path = args.out / "tokenizer" / "tokenizer.json"
    shards_dir = args.out / "shards"
    target_train = args.target_tokens - args.val_tokens

    # ---- 语料流（带一次镜像重试 + wikitext 兜底）----
    corpus_note = "HuggingFaceFW/fineweb-edu (sample-10BT)"
    try:
        stream = iter(stream_fineweb_edu())
        first = next(stream)  # 试探首条，失败在累计前就切换
        stream = _chain_first(first, stream)
    except Exception as e:  # noqa: BLE001
        print(f"[corpus] fineweb-edu 直连/当前 endpoint 失败: {e!r}")
        if os.environ.get("HF_ENDPOINT") != "https://hf-mirror.com":
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            print("[corpus] 改 HF_ENDPOINT=https://hf-mirror.com 重试（注：已 import 的 huggingface_hub 常量可能已固定，必要时重跑脚本）")
            try:
                stream = iter(stream_fineweb_edu())
                first = next(stream)
                stream = _chain_first(first, stream)
            except Exception as e2:  # noqa: BLE001
                print(f"[corpus] 镜像仍失败: {e2!r} → fallback wikitext-103（偏差！）")
                corpus_note = "Salesforce/wikitext wikitext-103-raw-v1（fallback，偏离 fineweb-edu 计划）"
                stream = stream_wikitext()
        else:
            corpus_note = "Salesforce/wikitext wikitext-103-raw-v1（fallback，偏离 fineweb-edu 计划）"
            stream = stream_wikitext()

    print(f"[corpus] 语料来源: {corpus_note}")

    # ---- 阶段 1：累计 tokenizer 训练语料 ----
    corpus: list[str] = []
    corpus_bytes = 0
    tok_bytes_target = int(args.tok_corpus_mb * 1024 * 1024)
    for text in stream:
        corpus.append(text)
        corpus_bytes += len(text.encode("utf-8"))
        if corpus_bytes >= tok_bytes_target:
            break
    print(f"[corpus] tokenizer 训练语料: {len(corpus)} 篇 / {corpus_bytes/1024**2:.0f} MB，耗时 {time.time()-t0:.0f}s")

    if not tok_path.exists():
        train_tokenizer(corpus, args.vocab_size, tok_path)
    else:
        print(f"[tokenizer] 已存在，跳过训练: {tok_path}")

    from tais_obsidian.tokenizer_io import TokenizerIO

    tok = TokenizerIO(tok_path)
    assert tok.vocab_size == args.vocab_size, f"vocab {tok.vocab_size} != {args.vocab_size}"
    eot = tok.eot_id

    # ---- 阶段 2：tokenize 全部语料 → shards ----
    def tokenize_docs(docs: list[str]) -> np.ndarray:
        """批量编码，文档间插 eot，返回 uint16 数组。"""
        encs = tok.encode_batch(docs)
        n = sum(len(e) + 1 for e in encs)
        out = np.empty(n, dtype="<u2")
        pos = 0
        for ids in encs:
            out[pos : pos + len(ids)] = ids
            pos += len(ids)
            out[pos] = eot
            pos += 1
        return out

    val_writer = ShardWriter(shards_dir, args.val_tokens)  # val 只写一片
    train_writer = ShardWriter(shards_dir, args.shard_tokens)
    val_left = args.val_tokens
    train_left = target_train

    t_tok = time.time()
    batch_docs: list[str] = []
    n_docs = 0

    def drain(docs: list[str]) -> None:
        nonlocal val_left, train_left, n_docs
        if not docs:
            return
        arr = tokenize_docs(docs)
        n_docs += len(docs)
        if val_left > 0:
            take = min(val_left, arr.size)
            val_writer.push(arr[:take], "val")
            val_left -= take
            arr = arr[take:]
        if arr.size and train_left > 0:
            take = min(train_left, arr.size)
            train_writer.push(arr[:take], "train")
            train_left -= take

    # 先把 tokenizer 语料 tokenize 进数据集（val 从头取，避免浪费已下载内容），
    # 再继续流式补足剩余 train 配额。
    for i in range(0, len(corpus), 128):
        drain(corpus[i : i + 128])
    print(f"[tokenize] tokenizer 语料已入库: val 余 {val_left/1e6:.2f}M, train 余 {train_left/1e6:.1f}M")

    while train_left > 0:
        try:
            text = next(stream)
        except StopIteration:
            print("[corpus] 语料流耗尽，提前结束")
            break
        batch_docs.append(text)
        if len(batch_docs) >= 128:
            drain(batch_docs)
            batch_docs = []
            done = target_train - train_left
            el = time.time() - t_tok
            rate = done / el if el > 0 else 0
            print(f"[tokenize] train {done/1e6:.1f}M / {target_train/1e6:.0f}M tokens "
                  f"({rate/1e6:.2f}M tok/s), 已处理 {n_docs} 篇")
    drain(batch_docs)

    val_writer.flush("val")
    train_writer.flush("train")

    total = val_writer.total + train_writer.total
    el = time.time() - t0
    print(f"[done] 语料={corpus_note}")
    print(f"[done] val {val_writer.total/1e6:.2f}M tokens, train {train_writer.total/1e6:.2f}M tokens, "
          f"合计 {total/1e6:.2f}M，总耗时 {el/60:.1f} min")
    if total < 80_000_000:
        print(f"[warn] 总 token 数 {total/1e6:.1f}M < 80M 下限")


def _chain_first(first, rest):
    yield first
    yield from rest


if __name__ == "__main__":
    main()
