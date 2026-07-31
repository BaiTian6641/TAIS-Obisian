#!/usr/bin/env bash
# 从旧工作站同步数据到新机（在新机上运行；需要能 SSH 到旧机）
# 用法：OLD=user@old-host OLD_DIR=/c/Users/Tass/Documents/TAIS-Obisian bash scripts/cloud/sync_from_workstation.sh [minimal|full]
set -euo pipefail
OLD="${OLD:?请设 OLD=user@old-host（旧机 SSH）}"
OLD_DIR="${OLD_DIR:-/c/Users/Tass/Documents/TAIS-Obisian}"
MODE="${1:-minimal}"   # minimal=代码+0.5B数据+tokenizer+0.1B数据+关键checkpoint；full=再加全部 runs/data/raw
cd "$(dirname "$0")/../.."

RSYNC=(rsync -avzP --protect-args)
echo "== 代码与文档（含 .git）=="
"${RSYNC[@]}" "$OLD:$OLD_DIR/src" "$OLD:$OLD_DIR/configs" "$OLD:$OLD_DIR/scripts" \
  "$OLD:$OLD_DIR/tests" "$OLD:$OLD_DIR/docs" "$OLD:$OLD_DIR/article_ref" .
"${RSYNC[@]}" "$OLD:$OLD_DIR/AGENTS.md" "$OLD:$OLD_DIR/MEMORY.md" "$OLD:$OLD_DIR/pyproject.toml" .
# .git 可选（大但保住历史）：取消下行注释
# "${RSYNC[@]}" "$OLD:$OLD_DIR/.git" .

echo "== 数据（tokenizer + 0.1B + 0.5B shards）=="
"${RSYNC[@]}" "$OLD:$OLD_DIR/data/tokenizer" "$OLD:$OLD_DIR/data/shards" "$OLD:$OLD_DIR/data/shards_0p5b" data/

echo "== 关键 checkpoint（0.1B 底座/校准/统一）=="
for ck in pilot_0p1b_gdn2_10k pilot_0p1b_gdn2_10k_kaltruth_v2 pilot_0p1b_gdn2_10k_unified; do
  "${RSYNC[@]}" "$OLD:$OLD_DIR/checkpoints/$ck" checkpoints/
done
mkdir -p runs && "${RSYNC[@]}" "$OLD:$OLD_DIR/runs/kal_truth_v2" "$OLD:$OLD_DIR/runs/niah_length_scan" runs/ || true

if [ "$MODE" = "full" ]; then
  echo "== full：data/raw + 全部 runs =="
  "${RSYNC[@]}" "$OLD:$OLD_DIR/data/raw" data/
  "${RSYNC[@]}" "$OLD:$OLD_DIR/runs" .
fi
echo "同步完成。校验步骤见 scripts/cloud/README.md（数据完整性一节）。"
