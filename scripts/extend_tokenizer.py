"""特殊 token 扩容（阶段 E+-2）：向已训 tokenizer 追加 5 个特殊 token。

决策记录（实施计划 §7.5 E+-2）：采用**扩展**而非重训——既有 token id 全部不变，
`data/shards` 的 120M tokens 保持有效；新 token 追加到词表尾部：

    <|recall|>(32768) <|blank|>(32769) <|gist|>(32770) <|ref|>(32771) <|box|>(32772)

tokenizers 库的 added-token trie 在 pre-tokenizer 之前匹配 special token，
故无需改动训练时的 Split 规则。原文件备份为 tokenizer.v32768.bak.json。

注意：本脚本**不改** `ModelConfig.vocab_size` 默认值（32768）——32776（8 倍数）仅在
显式配置时启用（E+-3 起的实验 config），避免影响既有对照 run。

用法：python scripts/extend_tokenizer.py [--tokenizer data/tokenizer/tokenizer.json]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]

NEW_SPECIALS = ["<|recall|>", "<|blank|>", "<|gist|>", "<|ref|>", "<|box|>"]
ANCHOR = "<|endoftext|>"  # 既有 id 0，扩容后必须仍为 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", type=Path, default=ROOT / "data" / "tokenizer" / "tokenizer.json")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))
    base_size = tok.get_vocab_size()
    print(f"[tok] 当前词表大小: {base_size}（{args.tokenizer}）")

    if tok.token_to_id(NEW_SPECIALS[0]) is not None:
        print(f"[tok] {NEW_SPECIALS[0]} 已存在（id={tok.token_to_id(NEW_SPECIALS[0])}），跳过扩容（幂等）")
    else:
        backup = args.tokenizer.with_name("tokenizer.v32768.bak.json")
        if not backup.exists():
            shutil.copy2(args.tokenizer, backup)
            print(f"[tok] 原文件已备份 → {backup}")
        added = tok.add_special_tokens(NEW_SPECIALS)
        print(f"[tok] 追加 {added} 个 special token: {NEW_SPECIALS}")

    # ---- 校验 ----
    assert tok.token_to_id(ANCHOR) == 0, f"{ANCHOR} id 漂移！应为 0，实际 {tok.token_to_id(ANCHOR)}"
    for i, s in enumerate(NEW_SPECIALS):
        tid = tok.token_to_id(s)
        assert tid == base_size + i, f"{s} id 应为 {base_size + i}，实际 {tid}"
        enc = tok.encode(s, add_special_tokens=False)
        assert enc.ids == [tid], f"encode({s!r}) 应为 [{tid}]，实际 {enc.ids}（added-token trie 未命中？）"
    tok.save(str(args.tokenizer))
    print(f"[tok] 校验通过，已保存：vocab_size={tok.get_vocab_size()}；"
          f"模型侧启用时 ModelConfig.vocab_size 应设为 32776（8 倍数，含 padding）")


if __name__ == "__main__":
    main()
