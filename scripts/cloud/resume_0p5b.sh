#!/usr/bin/env bash
# 0.5B 预训练启动（新机）：自动选卡 + micro batch 标定 + 断点续训
# 用法：
#   bash scripts/cloud/resume_0p5b.sh                 # 自动选最大显存单卡，从头或续训
#   MICRO=24 ACCUM=6 bash scripts/cloud/resume_0p5b.sh # 手动覆盖 batch（保持全局 ~131k tok/step）
#   RESUME=0 bash scripts/cloud/resume_0p5b.sh        # 强制从头（忽略 latest.pt）
set -euo pipefail
cd "$(dirname "$0")/../.."
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# 1) 选卡：取显存最大的 GPU（勿依赖 nvidia-smi 序 ≠ torch 序）
DEV=$(python - <<'EOF'
import torch
best, mem = 0, -1
for i in range(torch.cuda.device_count()):
    m = torch.cuda.get_device_properties(i).total_memory
    if m > mem: best, mem = i, m
print(best)
EOF
)
echo "[resume_0p5b] 选用物理卡 cuda:$DEV（$(CUDA_VISIBLE_DEVICES=$DEV python -c 'import torch;print(torch.cuda.get_device_name(0))')）"
export CUDA_VISIBLE_DEVICES=$DEV

# 2) batch 标定（可用 MICRO/ACCUM 覆盖）；显存 ≥48GB 建议 MICRO=32 ACCUM=4，≥80GB 可 MICRO=48 ACCUM=3
MICRO="${MICRO:-16}"; ACCUM="${ACCUM:-8}"
echo "[resume_0p5b] micro $MICRO × accum $ACCUM × seq 1024 = $((MICRO*ACCUM*1024)) tok/step"

# 3) 续训检测
CKPT="checkpoints/pilot_0p5b_gdn2/latest.pt"
RESUME_ARG=""
if [ "${RESUME:-1}" = "1" ] && [ -f "$CKPT" ]; then
  RESUME_ARG="--resume $CKPT"; echo "[resume_0p5b] 续训自 $CKPT"
else
  echo "[resume_0p5b] 从头训练（max_steps 22900 ≈ 3B tokens）"
fi

# 4) 启动（nohup 防 ssh 断连；日志 logs_train_0p5b.txt）
# 注意：max_steps 22900 按 131k tok/step 口径；改 MICRO/ACCUM 保持乘积 128 即口径不变
nohup python -u -m tais_obsidian.train --config configs/pilot_0p5b_gdn2.json \
  --micro_batch "$MICRO" --grad_accum "$ACCUM" $RESUME_ARG \
  > logs_train_0p5b.txt 2>&1 &
echo "[resume_0p5b] PID $! — 监控：tail -f logs_train_0p5b.txt"
echo "[resume_0p5b] ⚠️ GPU 纪律：训练期间勿在此卡跑其他进程；bench 只在空闲时做"
