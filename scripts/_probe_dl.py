# 临时探测：hf_hub_download 能否在镜像下下载 parquet 分片
import os, time, traceback
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from huggingface_hub import hf_hub_download

def probe(repo, fname):
    t = time.time()
    try:
        p = hf_hub_download(repo, fname, repo_type="dataset")
        sz = os.path.getsize(p) / 1e6
        print(f"[OK] {repo} :: {fname} -> {sz:.1f}MB  ({time.time()-t:.0f}s)  {p}")
        return p
    except Exception as e:
        print(f"[FAIL] {repo} :: {fname}: {type(e).__name__}: {str(e)[:240]}")

# 小规模：NuminaMath 第一个训练分片（~几十 MB）
p = probe("AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet")
# 验证能读
if p:
    import pyarrow.parquet as pq
    tb = pq.read_table(p)
    print("  NuminaMath 分片行数:", tb.num_rows, "列:", tb.column_names)
    row = tb.slice(0, 1).to_pylist()[0]
    for k, v in row.items():
        s = str(v)
        print(f"    {k}: len={len(s)} | {s[:80]!r}")
