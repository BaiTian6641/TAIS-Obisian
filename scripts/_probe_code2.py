# 临时探测：代码备选源的真实数据文件结构与可流式性
import os, sys
os.environ.pop("HF_ENDPOINT", None)
from huggingface_hub import HfApi
from datasets import load_dataset

api = HfApi()
for repo in ["bigcode/the-stack-smol", "codeparrot/codeparrot-clean", "OpenCoder-LLM/opc-annealing-corpus", "HuggingFaceTB/cosmopedia"]:
    try:
        files = api.list_repo_files(repo, repo_type="dataset")
        data_files = [f for f in files if not f.startswith(".") and f != "README.md"]
        print(f"\n[{repo}] {len(files)} files; 数据文件示例: {data_files[:5]}")
    except Exception as e:
        print(f"\n[{repo}] list ERR: {type(e).__name__} {str(e)[:120]}")

# 尝试流式 stack-smol
print("\n==== 尝试流式 ====")
for repo, cfg in [("bigcode/the-stack-smol", None), ("HuggingFaceTB/cosmopedia", None)]:
    try:
        ds = load_dataset(repo, split="train", streaming=True)
        row = next(iter(ds))
        print(f"[OK-stream] {repo}: cols={list(row.keys())}")
    except Exception as e:
        print(f"[FAIL-stream] {repo}: {type(e).__name__} {str(e)[:160]}")
