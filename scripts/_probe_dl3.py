# 临时探测：抓取 huggingface_hub 元数据 HEAD 失败的底层异常
import os, traceback
os.environ.pop("HF_ENDPOINT", None)  # 直连
from huggingface_hub.utils._http import get_session
from huggingface_hub.constants import ENDPOINT
print("ENDPOINT =", ENDPOINT)
repo, fname = "AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet"
url = f"{ENDPOINT}/datasets/{repo}/resolve/main/{fname}"
sess = get_session()
try:
    r = sess.head(url, allow_redirects=True, timeout=20)
    print("HEAD status:", r.status_code)
    print("headers:", dict(list(r.headers.items())[:8]))
    r.raise_for_status()
    print("raise_for_status OK")
except Exception as e:
    print("EXC:", type(e).__name__, str(e)[:300])
    traceback.print_exc()
