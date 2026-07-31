# scripts/cloud/ — 新机（云端）迁移与续训指南

> 面向新机上接手的 agent。前置：先读仓库根 `MEMORY.md`（迁移状态总档）与 `AGENTS.md`。
> 三步走：**① 环境 → ② 同步 → ③ 续训**。全程任何训练启动后遵守 GPU 纪律（MEMORY.md §1.6）。

## ① 环境（setup_env.sh）

```bash
bash scripts/cloud/setup_env.sh cu128   # Blackwell(sm_120) 必须 cu128；Hopper/Ada 可 cu126；纯 CPU 调试用 cpu
```
做什么：装 uv → 建 .venv（Python 3.12）→ 装 torch + `pip install -e .` + datasets/tokenizers/tensorboard/pytest → GPU 自检 → 数据就位检查。
验收：打印 torch 版本与 GPU 列表无 ⚠️；`python -m pytest tests/test_gdn.py -q` 抽测应绿。

## ② 同步（sync_from_workstation.sh，在新机上跑）

```bash
OLD=user@old-host OLD_DIR=/c/Users/Tass/Documents/TAIS-Obisian bash scripts/cloud/sync_from_workstation.sh minimal
# full 模式追加 data/raw（3.5GB parquet 缓存）+ 全部 runs/ 报告
```
同步内容（minimal）：src/configs/scripts/tests/docs/article_ref + AGENTS.md/MEMORY.md/pyproject.toml + data/{tokenizer,shards,shards_0p5b} + 3 个关键 checkpoint（gdn2_10k 扩窗底座 / kaltruth_v2 校准 / unified 统一）+ 2 份关键 runs 报告。
若旧机不便 SSH：改为旧机 `git add -A && git commit` 推远端，新机 clone（代码），数据单独走 rsync/scp/U 盘。

**数据完整性校验**（同步后必做）：
```bash
source .venv/bin/activate && python - <<'EOF'
import sys; sys.path.insert(0,'src')
from tais_obsidian.data.memmap import Shards
from tais_obsidian.tokenizer_io import TokenizerIO
import numpy as np
tr = Shards('data/shards_0p5b','train')
assert abs(tr.total - 2989999999) < 1e6, f'train tokens {tr.total} 不等于 2.99B'
tok = TokenizerIO('data/tokenizer/tokenizer.json')
x,_ = tr.get_batch(1, 64, 'cpu', np.random.default_rng(0))
print('shards OK:', f'{tr.total/1e9:.2f}B tokens;', '解码:', tok.decode(x[0].tolist())[:60])
EOF
```

## ③ 续训 0.5B（resume_0p5b.sh）

```bash
bash scripts/cloud/resume_0p5b.sh                  # 自动选最大显存卡 + micro 16×accum 8
MICRO=32 ACCUM=4 bash scripts/cloud/resume_0p5b.sh # ≥48GB 显存建议（保持乘积 128 → 步数口径不变）
```
做什么：自动选卡（按显存，避开 nvidia-smi 序 ≠ torch 序的坑）→ 检测 `checkpoints/pilot_0p5b_gdn2/latest.pt` 存在则续训 → nohup 后台启动（日志 logs_train_0p5b.txt）。
监控：`tail -f logs_train_0p5b.txt`；正常信号 = loss 从 ~10.6 降、gnorm <20、2-3 分钟内出 step 1。
**标定义务**：新卡第一次跑请记录 tok/s 与峰值显存并写回 MEMORY.md §1.3 表（22900 步 ÷ tok/s = ETA）。

## 后续任务（按 MEMORY.md §3 队列）
1. 0.5B 训练完成 → val 终值记录 + `python -m tais_obsidian.generate --ckpt checkpoints/pilot_0p5b_gdn2/final --prompt "..."` 抽验。
2. RoPE 扩容+NTK → 256K（方案要点见 MEMORY.md §3 任务②，代码待写）。
3. 记忆层读出/寻址训练；④ 0.5B KAL 复测；⑤ 1.5B 规划。

## 多卡说明
- `scripts/train_dp.py` 目前写死**双卡**单进程手动 DP（Windows 无 NCCL 的遗产）；Linux 新机 ≥2 卡建议改用 torchrun+NCCL DDP（需小幅改 train.py 支持 init_process_group，或继续用 train_dp.py 双卡模式）。
- 单大卡（≥48GB）最简单：micro 32×accum 4 起步标定。

## 常见坑（旧机实录）
1. torch 设备序 ≠ nvidia-smi 序 → 一律用 `torch.cuda.get_device_name(i)` 确认。
2. HF 直连不稳：`HF_ENDPOINT=https://hf-mirror.com`；prepare_data_0p5b.py 已内置 HTTP GET 直下 + 断流重试。
3. 训练循环改动后必须先 `--max_steps 3` 冒烟。
4. Windows 控制台乱码是 GBK 代码页问题，脚本已 `reconfigure(encoding='utf-8')`，日志文件本身是 UTF-8。
