"""一次性实测：torch.compile 对自研 0.5B（GDN-2+三级栈）训练步的加速效果（llm conda 环境）。

对照 eager vs torch.compile(default) 的 micro-step 吞吐。显存控制在 ~8GB（micro 8），
与后台单卡训练（PRO 4000，~12GB）共存不 OOM。
"""
import sys, time, json
sys.path.insert(0, "src")
import numpy as np
import torch

from tais_obsidian.train import build_model_config, chunked_ce
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.data.memmap import Shards

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

cfg = json.load(open("configs/pilot_0p5b_gdn2.json", encoding="utf-8"))
model = TaisObsidianForCausalLM(build_model_config(cfg)).cuda().train()
sh = Shards("data/shards_0p5b", "train")
rng = np.random.default_rng(0)
MICRO, SEQ = 8, 1024


def step(m):
    x, y = sh.get_batch(MICRO, SEQ, "cuda", rng)
    with torch.autocast("cuda", torch.bfloat16):
        logits, _ = m(x)
        loss = chunked_ce(logits, y)
    loss.backward()
    m.zero_grad(set_to_none=True)
    return loss.item()


def bench(m, n=10, tag=""):
    for _ in range(3):
        step(m)
    torch.cuda.synchronize()
    t0 = time.time()
    losses = [step(m) for _ in range(n)]
    torch.cuda.synchronize()
    el = time.time() - t0
    print(f"[{tag}] {MICRO*SEQ*n/el/1e3:.2f}k tok/s  loss {np.mean(losses):.4f}  "
          f"peak {torch.cuda.max_memory_allocated()/1e9:.1f}GB", flush=True)


bench(model, tag="eager")
cmodel = torch.compile(model)
print("[compile] 首次编译中（可能数分钟）…", flush=True)
bench(cmodel, tag="compiled")
print("DONE")
