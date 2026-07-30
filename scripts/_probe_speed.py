# 临时探测：①fineweb-edu 流式 token 速率 ②the-stack-v2 语言目录 ③各源 GET 速率
import os, sys, time
os.environ.pop("HF_ENDPOINT", None)
sys.path.insert(0, "src")
from datasets import load_dataset
from huggingface_hub import HfApi

# ① fineweb-edu 流式速率（用官方 token_count 字段估算）
print("==== fineweb-edu 流式速率 ====")
ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
t0 = time.time(); n = 0; toks = 0; bytes_ = 0
for row in ds:
    toks += int(row.get("token_count") or 0)
    bytes_ += len(row["text"].encode("utf-8"))
    n += 1
    if n >= 300:
        break
dt = time.time() - t0
print(f"  {n} 篇 / {toks/1e3:.0f}k tokens / {bytes_/1e6:.1f}MB文本  用时 {dt:.1f}s")
print(f"  => {toks/dt/1e3:.0f}k tok/s,  {bytes_/1e6/dt:.1f} MB文本/s;  单篇均值 {toks/n:.0f} tok")

# ② the-stack-v2 语言目录（找主流语言）
print("==== the-stack-v2 语言目录（采样主流）====")
api = HfApi()
files = api.list_repo_files("bigcode/the-stack-v2", repo_type="dataset")
langs = sorted({f.split("/")[1] for f in files if f.startswith("data/")})
want = ["Python", "JavaScript", "TypeScript", "Java", "C++", "C", "Go", "Rust", "Markdown", "Jupyter Notebook"]
print("  总语言数:", len(langs))
print("  关注语言是否在列:", {w: (w in langs) for w in want})
# 每语言文件数
from collections import Counter
cnt = Counter(f.split("/")[1] for f in files if f.startswith("data/") and f.endswith(".parquet"))
for w in want:
    if w in cnt:
        print(f"    {w}: {cnt[w]} 分片")
