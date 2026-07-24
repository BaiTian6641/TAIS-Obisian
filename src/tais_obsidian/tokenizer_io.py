"""tokenizer 加载与 encode/decode 封装（基于 tokenizers 库）。"""
from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer


class TokenizerIO:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.tok = Tokenizer.from_file(str(self.path))

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    @property
    def eot_id(self) -> int:
        return self.tok.token_to_id("<|endoftext|>")

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)
