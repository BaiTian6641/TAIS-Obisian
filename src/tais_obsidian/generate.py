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

# ipykernel OutStream 无 reconfigure 方法（notebook import 即炸），hasattr 守卫
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.tokenizer_io import TokenizerIO

# 仓库根目录（本文件 src/tais_obsidian/generate.py 上两级）
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_TOKENIZER = _REPO_ROOT / "data" / "tokenizer" / "tokenizer.json"


def resolve_tokenizer_path(ckpt_dir: str | Path, tokenizer: str | None = None) -> Path:
    """解析 tokenizer.json 路径：显式 --tokenizer > <ckpt>/tokenizer.json > data/tokenizer/tokenizer.json。

    训练结束会把 tokenizer 复制进 final 目录（train.py copy_tokenizer_to_final），
    使权重产物自包含（HF 上传/下载后直接推理）；缺省依次回退，找不到则报清晰错误。
    """
    if tokenizer is not None:
        p = Path(tokenizer)
        if not p.exists():
            raise FileNotFoundError(f"--tokenizer 指定的文件不存在: {p}")
        return p
    for cand in (Path(ckpt_dir) / "tokenizer.json", _DEFAULT_TOKENIZER):
        if cand.exists():
            return cand
    raise FileNotFoundError(
        f"找不到 tokenizer.json：{Path(ckpt_dir) / 'tokenizer.json'} 与 {_DEFAULT_TOKENIZER} 均不存在；"
        f"请显式指定 --tokenizer"
    )


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
    # 越界防护：tokenizer 词表（32773，含特殊 token）可大于模型 vocab_size（32768），
    # 越界 id 会在 embedding 处炸出难懂的 CUDA assert——此处先报清晰错误并列出越界 id。
    bad = sorted({i for i in ids if not 0 <= i < model.config.vocab_size})
    assert not bad, (
        f"prompt 编码出越界 token id {bad}（模型 vocab_size={model.config.vocab_size}）："
        f"tokenizer 与模型词表不匹配，请检查 --tokenizer 是否与训练时一致"
    )
    # 长度守卫：prefill 超过 max_seq 会越出 RoPE 缓存行数（广播 RuntimeError），提前报清晰错误
    if len(ids) > model.config.max_seq:
        raise ValueError(
            f"prompt 长度 {len(ids)} tokens 超过模型 max_seq={model.config.max_seq}；"
            f"请缩短 prompt 或使用扩窗 checkpoint"
        )
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
            # 位置守卫：即将前向的 token 绝对位置达 max_seq 时 RoPE 缓存越界——
            # 提前停止并打印警告，而非放任 RuntimeError 广播错
            pos_next = len(ids) + len(out_ids) - 1
            if pos_next >= model.config.max_seq:
                print(f"[gen] 警告：cache 位置 {pos_next} 已达 max_seq={model.config.max_seq}，"
                      f"停止增量生成（已生成 {len(out_ids)} tokens）")
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
    ap.add_argument("--tokenizer", default=None,
                    help="tokenizer.json 路径；缺省依次尝试 <ckpt>/tokenizer.json → data/tokenizer/tokenizer.json")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max_new_tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, device).eval()
    tok = TokenizerIO(resolve_tokenizer_path(args.ckpt, args.tokenizer))
    text, _ = generate(
        model, tok, args.prompt, args.max_new_tokens, args.temperature, args.top_k, device
    )
    print("=" * 60)
    print(args.prompt + text)


if __name__ == "__main__":
    main()
