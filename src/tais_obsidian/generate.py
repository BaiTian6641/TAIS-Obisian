"""推理：加载 final/ checkpoint，逐 token 生成（attn KV cache + GDN state 传递，不重算全前缀）。

用法：
  python -m tais_obsidian.generate --ckpt checkpoints/pilot_0p1b/final --prompt "The capital of France is" \
      --max_new_tokens 100 --temperature 0.8 --top_k 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.tokenizer_io import TokenizerIO


@torch.no_grad()
def generate(
    model: TaisObsidianForCausalLM,
    tok: TokenizerIO,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: str,
) -> tuple[str, float]:
    ids = tok.encode(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    t0 = time.time()
    with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
        logits, cache = model(x)  # prefill：一次性过全前缀
        t_prefill = time.time() - t0
        out_ids: list[int] = []
        t1 = time.time()
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].float()
            if temperature <= 0:
                nxt = int(next_logits.argmax(-1).item())
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits = next_logits.masked_fill(next_logits < v[:, -1:], float("-inf"))
                probs = torch.softmax(next_logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            out_ids.append(nxt)
            if nxt == tok.eot_id:
                break
            x = torch.tensor([[nxt]], dtype=torch.long, device=device)
            logits, cache = model(x, cache)  # 增量：只算新 token，复用 cache
        dt = time.time() - t1
    n = len(out_ids)
    tok_s = n / dt if dt > 0 else 0.0
    print(f"[gen] prefill {len(ids)} tokens {t_prefill*1e3:.0f}ms；生成 {n} tokens，{tok_s:.1f} tok/s")
    return tok.decode(out_ids), tok_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b/final")
    ap.add_argument("--tokenizer", default="data/tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, device).eval()
    tok = TokenizerIO(args.tokenizer)
    text, _ = generate(
        model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device
    )
    print("=" * 60)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
