# 临时探测：①本地 tokenize 速率 ②NuminaMath/FineWeb2-HQ/stack 单分片 GET 下载+读列名
import os, sys, time, urllib.request, tempfile
os.environ.pop("HF_ENDPOINT", None)
sys.path.insert(0, "src")
from tais_obsidian.tokenizer_io import TokenizerIO
from datasets import load_dataset

# ① 本地 tokenize 速率
tok = TokenizerIO("data/tokenizer/tokenizer.json")
docs = ["The quick brown fox jumps over the lazy dog. " * 50] * 512
t0 = time.time()
encs = tok.encode_batch(docs)
n = sum(len(e) for e in encs)
dt = time.time() - t0
print(f"[tokenize] {n/1e3:.0f}k tok / {dt:.2f}s = {n/dt/1e6:.2f}M tok/s (本地 CPU)")

# ② 三源各下一小片验证列名
def get_first(repo, fname, tag, maxbytes=25*1024*1024):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
    tmp = os.path.join(tempfile.gettempdir(), tag + ".parquet")
    if not os.path.exists(tmp):
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(tmp, "wb") as f:
            got = 0
            while got < maxbytes:
                c = r.read(1 << 20)
                if not c: break
                f.write(c); got += len(c)
    # 注意：部分 parquet 无法读（截断），需完整文件。改用整下小文件。
    return tmp, url

# 整下小文件才能 parquet 读——改为验证 FineWeb2-HQ cmn_Hani 第一个分片（通常较小）
def dl_full(repo, fname, tag):
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{fname}"
    tmp = os.path.join(tempfile.gettempdir(), tag + ".parquet")
    if os.path.exists(tmp) and os.path.getsize(tmp) > 0:
        pass
    else:
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                c = r.read(1 << 20)
                if not c: break
                f.write(c)
        print(f"  下载 {tag}: {os.path.getsize(tmp)/1e6:.1f}MB 用时 {time.time()-t0:.0f}s")
    ds = load_dataset("parquet", data_files=tmp, split="train", streaming=True)
    row = next(iter(ds))
    print(f"  [{tag}] cols={list(row.keys())}")
    for k, v in row.items():
        s = str(v); print(f"      {k}: len={len(s)} | {s[:70]!r}")

print("==== FineWeb2-HQ cmn_Hani ====")
try: dl_full("epfml/FineWeb2-HQ", "cmn_Hani/000_00000.parquet", "fw2hq")
except Exception as e: print("  ERR:", type(e).__name__, str(e)[:200])
print("==== the-stack-v2 Python ====")
try: dl_full("bigcode/the-stack-v2", "data/Python/train-00000-of-00009.parquet", "stack_py")
except Exception as e: print("  ERR:", type(e).__name__, str(e)[:200])
