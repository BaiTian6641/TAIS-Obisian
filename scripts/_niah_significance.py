"""NIAH 显著性检验：bounded vs 无界 vs GDN-1，多 seed 降方差。"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import eval_retrieval_niah as en  # noqa: E402

CKPTS = {
    "gdn1": "checkpoints/pilot_0p1b_gdn1/final",
    "gdn2_unbounded_10k": "checkpoints/pilot_0p1b_gdn2_10k/final",
    "gdn2_bounded_7k": "checkpoints/_gdn2_bounded_step7000_eval",
    "gdn2_bounded_10k": "checkpoints/pilot_0p1b_gdn2_bounded_10k/final",
}

tok = TokenizerIO("data/tokenizer/tokenizer.json")
res = {}
for tag, path in CKPTS.items():
    m = TaisObsidianForCausalLM.from_pretrained(path, "cuda", strict=False).eval()
    accs = []
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        accs.append(en.eval_retrieval(m, tok, rng, 8, 200, "cuda"))
    res[tag] = accs
    print(f"{tag}: accs={[f'{a:.3f}' for a in accs]}  mean={np.mean(accs):.3f}  std={np.std(accs):.3f}")
    del m
    torch.cuda.empty_cache()

print()
u, b10, b7 = (np.mean(res[k]) for k in ["gdn2_unbounded_10k", "gdn2_bounded_10k", "gdn2_bounded_7k"])
print(f"无界10k {u:.3f} vs bounded10k {b10:.3f}（Δ={b10-u:+.3f}）")
print(f"bounded7k {b7:.3f} vs bounded10k {b10:.3f}（Δ={b10-b7:+.3f}）")
print(f"GDN-1 {np.mean(res['gdn1']):.3f}（基线）")
