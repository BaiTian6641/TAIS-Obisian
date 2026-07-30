# 临时探测：huggingface_hub 网络细节 + 直连 vs 镜像对照
import os, traceback, urllib.request
ep = os.environ.get("HF_ENDPOINT", "<未设置>")
print("HF_ENDPOINT =", ep)
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import get_session

repo, fname = "AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet"

# 1) HEAD 该文件的 resolve URL（镜像 vs 直连）
for tag, base in [("mirror", "https://hf-mirror.com"), ("direct", "https://huggingface.co")]:
    url = f"{base}/datasets/{repo}/resolve/main/{fname}"
    try:
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "hf_hub"})
        with urllib.request.urlopen(req, timeout=20) as r:
            print(f"[HEAD-{tag}] {r.status} len={r.headers.get('Content-Length')} loc={r.headers.get('Location','-')[:60]}")
    except Exception as e:
        print(f"[HEAD-{tag}] ERR: {type(e).__name__}: {str(e)[:160]}")

# 2) hf_hub_download 完整 traceback
print("---- hf_hub_download traceback ----")
try:
    p = hf_hub_download(repo, fname, repo_type="dataset")
    print("OK ->", p)
except Exception:
    traceback.print_exc()
