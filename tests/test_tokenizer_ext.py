"""tokenizer 扩容（E+-2）回归测试：5 个新 special 的 id、编码行为与既有 id 兼容性。

依赖 data/tokenizer/tokenizer.json（先运行 scripts/extend_tokenizer.py）；
文件不存在时 skip（数据不入库，其他环境可无数据跑纯代码测试）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
BAK = ROOT / "data" / "tokenizer" / "tokenizer.v32768.bak.json"

NEW_SPECIALS = ["<|recall|>", "<|blank|>", "<|gist|>", "<|ref|>", "<|box|>"]

pytestmark = pytest.mark.skipif(not TOK.exists(), reason="data/tokenizer 未就绪（需先跑 prepare_data + extend_tokenizer）")


def test_extended_tokenizer() -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOK))
    assert tok.get_vocab_size() == 32773, f"扩容后词表应为 32773，实际 {tok.get_vocab_size()}"
    assert tok.token_to_id("<|endoftext|>") == 0, "EOT id 漂移"
    for i, s in enumerate(NEW_SPECIALS):
        assert tok.token_to_id(s) == 32768 + i, f"{s} id 应为 {32768 + i}"
        assert tok.encode(s, add_special_tokens=False).ids == [32768 + i], f"encode({s}) 未命中 added-token trie"
    # 夹杂普通文本：special 独立成 id，其余走原 BPE
    ids = tok.encode("让我回想一下<|recall|>再继续", add_special_tokens=False).ids
    assert 32768 in ids and ids.count(32768) == 1
    # 与扩容前备份对比：普通文本编码必须逐点一致（既有 id 未受影响）
    if BAK.exists():
        old = Tokenizer.from_file(str(BAK))
        for text in ("The capital of France is Paris.", "黑曜石框架的混合架构实验", "1 + 1 = 2"):
            assert old.encode(text, add_special_tokens=False).ids == tok.encode(text, add_special_tokens=False).ids, f"既有 id 兼容性破坏: {text!r}"
    print("test_tokenizer_ext 全部通过。")
