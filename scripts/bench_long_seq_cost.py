"""长 seq 成本实测（fb1 P1 扩窗配套）：三级栈各分支随 T 的复杂度实测 + 解析估计。

结论先行（本脚本实测验证的解析模型）：
- **滑窗分支（L0）**：注意力 mask 限 w=tri_window（512），FLOPs O(T·w·D) **线性**；但当前
  SDPA 走 math 后端（任意 bool mask 无 flash），会物化 [B,n_q,T,T] scores——**显存随 T²**，
  4096 以上 micro=1 也会爆（8192 时 scores 即 1.6GB×2）。这是显存瓶颈而非算力瓶颈
  （→ 256K 训练须先把滑窗改分块/ring 实现或稀疏 kernel，见 256K 计划文档）。
- **CSA 分支（L1）**：压缩 S=T/stride(4) 条目；indexer 打分 [T,S] 与压缩注意力 logits [T,S]
  均 **随 T² 增长**（常数 1/4）——选择检索是"全因果条目打分 + top-k"，不打分就无法 top-k。
  这是 V4/NSA 设计的固有二次项（生产靠低维 indexer + fp8 + kernel 摊薄，非线性化）。
- **HCA 分支（L2）**：gist S2=T/128 条目 dense 注意力 [T,S2]，随 T²（常数 1/128，可忽略至很长）。
- **GDN-2**：递归 chunked，O(T) 严格线性（长上下文的成本锚点）。
- 256K 推算：CSA 打分矩阵 = 256K×64K = 16.4G 元素/层（bf16 33GB 仅打分矩阵！）——**纯密集
  实现 256K 不可行，须先落地分块打分/top-k 两阶段或 indexer-only 粗选**。上限建议见计划文档。

用法：
  CUDA_VISIBLE_DEVICES=1 python scripts/bench_long_seq_cost.py \
      --ckpt checkpoints/pilot_0p1b_gdn2_10k_ctx4k/final [--lengths 512 1024 2048 4096]
输出：控制台表 + runs/long_seq_cost/report.json（随代码库版本存档，供 256K 计划引用）。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402


def analytic_counts(cfg, T: int) -> dict:
    """各分支每 A 层打分矩阵元素数（注意力/打分的主导项；投影/MLP 为 O(T) 不列）。"""
    w = cfg.tri_window
    s_csa = T // cfg.tri_csa_stride
    s_hca = T // cfg.tri_hca_stride
    return {
        "window_scores": T * min(w, T),          # 线性（常数 w）——但 math SDPA 物化 T×T
        "csa_scores": T * s_csa,                 # 二次（常数 1/4）
        "hca_scores": T * s_hca,                 # 二次（常数 1/128）
        "csa_entries": s_csa,
        "hca_entries": s_hca,
    }


@torch.no_grad()
def bench_length(model, T: int, device: str, iters: int, warmup: int) -> dict:
    """micro=1 整段前向计时（bf16 autocast，no_grad）。"""
    x = torch.randint(0, model.config.vocab_size, (1, T), device=device)
    for _ in range(warmup):
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            model(x)
        if device == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    med = sorted(ts)[len(ts) // 2]
    mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0
    return {"sec": med, "tok_per_s": T / med, "peak_mem_gb": round(mem, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description="长 seq 成本实测（三级栈分支复杂度）")
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_gdn2_10k/final")
    ap.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                    help="≤4096 安全（math SDPA 物化 T×T scores，8192 约 3.2GB 峰值）")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--out", default="runs/long_seq_cost/report.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device).eval()
    cfg = model.config
    n_a = sum(1 for t in cfg.layer_types if t == "A")
    print(f"[bench] {args.ckpt} max_seq={cfg.max_seq} scaling={getattr(cfg, 'rope_scaling', 'none')} "
          f"A层×{n_a} lengths={args.lengths}")

    rows = []
    prev = None
    for T in args.lengths:
        assert T <= cfg.max_seq, f"T={T} 超 max_seq={cfg.max_seq}（先用 extend_context.py 扩窗）"
        r = bench_length(model, T, args.device, args.iters, args.warmup)
        a = analytic_counts(cfg, T)
        exp_t = math.log2(r["sec"] / prev["sec"]) if prev else float("nan")  # 实测时间指数
        row = {"T": T, **r, "counts": a, "time_exponent_vs_prev": round(exp_t, 2)}
        rows.append(row)
        print(f"  T={T:6d} | {r['sec']*1e3:8.1f}ms | {r['tok_per_s']/1e3:6.1f}k tok/s | "
              f"峰值 {r['peak_mem_gb']:5.2f}GB | 时间指数 ×2^{row['time_exponent_vs_prev'] if prev else float('nan'):.2f} | "
              f"CSA {a['csa_scores']/1e6:.1f}M 元素 / HCA {a['hca_scores']/1e6:.1f}M")
        prev = r

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ckpt": args.ckpt, "max_seq": cfg.max_seq,
        "rope_scaling": getattr(cfg, "rope_scaling", "none"),
        "rope_scale": getattr(cfg, "rope_scale", 1.0),
        "n_a_layers": n_a, "lengths": args.lengths, "rows": rows,
        "notes": "window/CSA/HCA 打分矩阵元素数见 counts；时间指数>1 即超线性（二次分支主导）",
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] report → {out}")


if __name__ == "__main__":
    main()
