# 临时探测：确认各数据集在镜像下的可达方式与真实列名/文件路径
import os, sys, traceback
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import HfApi

api = HfApi()
for repo in ["AI-MO/NuminaMath-CoT", "epfml/FineWeb2-HQ", "bigcode/the-stack-v2"]:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
        pq = [f for f in files if f.endswith(".parquet")]
        print(f"[OK] {repo}: {len(files)} files, parquet={len(pq)}; 示例: {pq[:4]}")
    except Exception as e:
        print(f"[FAIL-list] {repo}: {type(e).__name__}: {str(e)[:200]}")

print("---- 尝试 datasets 流式 ----")
from datasets import load_dataset

def try_stream(tag, *args, **kw):
    try:
        ds = load_dataset(*args, streaming=True, **kw)
        row = next(iter(ds))
        print(f"[OK-stream] {tag}: cols={list(row.keys())}")
        for k, v in row.items():
            s = str(v)
            print(f"      {k}: len={len(s)} | {s[:80]!r}")
    except Exception as e:
        print(f"[FAIL-stream] {tag}: {type(e).__name__}: {str(e)[:240]}")

# NuminaMath：显式 data_files parquet 通配
try_stream("numina data_files", "AI-MO/NuminaMath-CoT", data_files="data/train-*.parquet", split="train")
# FineWeb2-HQ：config=cmn_Hani
try_stream("fineweb2hq cmn_Hani", "epfml/FineWeb2-HQ", "cmn_Hani", split="train")
# the-stack-v2：尝试 default
try_stream("stackv2 default", "bigcode/the-stack-v2", split="train")
