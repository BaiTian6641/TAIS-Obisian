# 临时探测：本地下载 parquet → 本地流式读取，验证列名与 Unicode 文件名处理
import os, sys, glob
sys.path.insert(0, "src")
from datasets import load_dataset

# 用之前下载的 NuminaMath 部分文件？不完整。改为：直接验证机制——把任意 parquet 用 load_dataset("parquet") 流式读
# 先确认 stack-v2 Python 分片的 blob 命名（是否 URL 编码 C++）
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files("bigcode/the-stack-v2", repo_type="dataset")
for pat in ["data/C++/", "data/Python/", "data/Go/"]:
    fs = [f for f in files if f.startswith(pat) and f.endswith(".parquet")]
    print(f"{pat}: {len(fs)} -> {fs[:2]}")

# 验证 load_dataset("parquet") 本地流式（用现有任意 parquet）
# 找一个已下载的 HF 缓存 parquet
cache = os.path.expanduser("~/.cache/huggingface")
pq = glob.glob(cache + "/**/*.parquet", recursive=True)
print("\n缓存 parquet 数:", len(pq))
if pq:
    f = pq[0]
    print("用", os.path.basename(f), "验证本地流式")
    try:
        ds = load_dataset("parquet", data_files=f, split="train", streaming=True)
        row = next(iter(ds))
        print("  OK cols:", list(row.keys()))
    except Exception as e:
        print("  ERR:", type(e).__name__, str(e)[:200])
