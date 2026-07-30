# 临时探测：HTTP GET 流式下载 parquet（绕过 huggingface_hub 元数据 HEAD）
import os, time, urllib.request
os.environ.pop("HF_ENDPOINT", None)

repo, fname = "AI-MO/NuminaMath-CoT", "data/train-00000-of-00005.parquet"
out = "data/raw/_test_numina.parquet"
os.makedirs("data/raw", exist_ok=True)

for tag, base in [("direct", "https://huggingface.co"), ("mirror", "https://hf-mirror.com")]:
    url = f"{base}/datasets/{repo}/resolve/main/{fname}"
    t = time.time()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r, open(out, "wb") as f:
            got = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if got > 40 * 1024 * 1024:  # 只下 40MB 验证
                    break
        dt = time.time() - t
        print(f"[GET-{tag}] 已下载 {got/1e6:.1f}MB  用时 {dt:.0f}s  速度 {got/1e6/dt:.1f}MB/s")
        break
    except Exception as e:
        print(f"[GET-{tag}] ERR: {type(e).__name__}: {str(e)[:200]}")
