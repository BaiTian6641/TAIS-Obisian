# 临时探测：the-stack-v2 401 后，找可匿名访问的代码语料备选源
import os, urllib.request, json
os.environ.pop("HF_ENDPOINT", None)

def head(repo, fname=""):
    url = f"https://huggingface.co/api/datasets/{repo}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            d = json.loads(r.read())
        gating = d.get("gating", False)
        sib = d.get("siblings") or []
        pq = [s["rfilename"] for s in sib if s["rfilename"].endswith(".parquet")]
        print(f"[{'GATED' if gating else 'open'}] {repo}: parquet={len(pq)}; {pq[:3]}")
        return not gating
    except Exception as e:
        print(f"[ERR] {repo}: {type(e).__name__} {str(e)[:120]}")
        return False

for repo in [
    "bigcode/the-stack-v2",
    "bigcode/the-stack-smol",
    "bigcode/the-stack-march-sample",
    "codeparrot/codeparrot-clean",
    "Fraser/python-state-changes",
    "OpenCoder-LLM/opc-annealing-corpus",
    "HuggingFaceTB/cosmopedia",
]:
    head(repo)
