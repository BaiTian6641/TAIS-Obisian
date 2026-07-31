#!/usr/bin/env bash
# TAIS Obsidian 新机环境初始化（Linux/Windows Git Bash 通用）
# 用法：bash scripts/cloud/setup_env.sh [cu128|cu126|cpu]   （默认 cu128）
set -euo pipefail
CUDA_TAG="${1:-cu128}"
cd "$(dirname "$0")/../.."   # 仓库根

echo "== [1/5] Python 3.12 + uv venv =="
if ! command -v uv >/dev/null 2>&1; then
  echo "安装 uv…"; curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -d .venv ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

echo "== [2/5] torch（$CUDA_TAG）=="
if [ "$CUDA_TAG" = "cpu" ]; then
  uv pip install torch --index-url https://download.pytorch.org/whl/cpu
else
  # Blackwell(sm_120) 必须 cu128+；Hopper/Ada 可 cu126
  uv pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "== [3/5] 项目依赖 =="
uv pip install -e .
uv pip install datasets huggingface_hub tokenizers tensorboard pytest

echo "== [4/5] 环境自检 =="
python - <<'EOF'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "avail", torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name} {p.total_memory/1e9:.1f}GB sm_{p.major}{p.minor}")
    if p.major >= 12 and not torch.version.cuda.startswith("12.8") and not torch.version.cuda.startswith("13"):
        print("  ⚠️ Blackwell(sm_120+) 需要 cu128+ wheel，请重装")
EOF
python scripts/check_env.py || true

echo "== [5/5] 数据就位检查 =="
python - <<'EOF'
from pathlib import Path
for p, hint in [("data/shards_0p5b", "0.5B 3B tokens（缺则跑 sync 脚本或 prepare_data_0p5b.py）"),
                ("data/shards", "0.1B 120M tokens"),
                ("data/tokenizer/tokenizer.json", "32k BPE")]:
    print(("OK  " if Path(p).exists() else "MISS") + f"  {p}  {'' if Path(p).exists() else hint}")
EOF
echo "setup_env 完成。下一步见 scripts/cloud/README.md"
