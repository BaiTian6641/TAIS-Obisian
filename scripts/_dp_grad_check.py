"""一次性校验脚本（_tmp）：双卡加权梯度 vs 单卡全 batch 梯度等价性。

同一批 8 条序列：单卡 backward 得参考梯度；DP 路径 6(master)+2(worker) 分片、
按 token 占比加权 backward、worker 梯度搬回 master 相加。逐参数比较相对误差。
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import build_model_config, chunked_ce

dev_m, dev_w = "cuda:1", "cuda:0"
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
cfg = dict(seed=42, micro_batch=8, seq_len=512)
model_cfg = build_model_config(cfg)
ref = TaisObsidianForCausalLM(model_cfg).to(dev_m)
master = TaisObsidianForCausalLM(model_cfg).to(dev_m)
worker = TaisObsidianForCausalLM(model_cfg).to(dev_w)
with torch.no_grad():
    for p_r, p_m, p_w in zip(ref.parameters(), master.parameters(), worker.parameters()):
        p_m.copy_(p_r)
        p_w.copy_(p_r.to(dev_w))

shards = Shards("data/shards", "train")
rng = np.random.default_rng(123)
x, y = shards.get_batch(8, 512, dev_m, rng)  # 固定一批 8 条
xw, yw = x[6:].to(dev_w), y[6:].to(dev_w)

# 参考：单卡 8 条全 batch（fp32 无 autocast，排除 bf16/不同 batch shape 的数值噪声，
# 严格校验"分片加权求和 = 全 batch 梯度"的数学等价性）
with torch.autocast("cuda", enabled=False):
    logits, _ = ref(x)
    loss_ref = chunked_ce(logits.float(), y)
loss_ref.backward()

# DP：master 6 条（权重 6/8）+ worker 2 条（权重 2/8）
with torch.autocast("cuda", enabled=False):
    logits, _ = master(x[:6])
    loss_m = chunked_ce(logits.float(), y[:6])
(loss_m * (6 / 8)).backward()
with torch.autocast("cuda", enabled=False):
    logits, _ = worker(xw)
    loss_w = chunked_ce(logits.float(), yw)
(loss_w * (2 / 8)).backward()
with torch.no_grad():
    for p_m, p_w in zip(master.parameters(), worker.parameters()):
        if p_w.grad is not None:
            p_m.grad.add_(p_w.grad.to(dev_m))

loss_dp = loss_m.item() * 6 / 8 + loss_w.item() * 2 / 8
print(f"loss 单卡 {loss_ref.item():.6f} vs DP 加权 {loss_dp:.6f}（Δ={abs(loss_ref.item()-loss_dp):.2e}）")

worst = 0.0
n_checked = 0
for (name, p_r), p_m in zip(ref.named_parameters(), master.parameters()):
    if p_r.grad is None:
        continue
    diff = (p_r.grad - p_m.grad).abs().max().item()
    scale = p_r.grad.abs().max().item() + 1e-12
    rel = diff / scale
    worst = max(worst, rel)
    n_checked += 1
    if rel > 1e-3:
        print(f"  [偏差大] {name}: rel={rel:.2e}")
print(f"梯度逐参数最大相对误差 {worst:.2e}（{n_checked} 个参数，阈值 1e-3）")
print("PASS" if worst < 1e-3 else "FAIL")
