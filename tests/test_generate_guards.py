"""generate.py 守卫回归测试：越界 id 清晰报错、超长 prompt 清晰报错、cache pos 达 max_seq 停止。

背景：tokenizer 词表 32773 > 模型 vocab_size 32768，特殊 token 字面量会越界（embedding CUDA
assert 难定位）；RoPE 缓存只有 max_seq 行，prompt/增量超界会炸广播 RuntimeError。三处守卫
把这三类崩溃转为清晰报错/受控停止。
用法：python -m pytest tests/test_generate_guards.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.generate import generate
from tais_obsidian.model.model import TaisObsidianForCausalLM

VOCAB = 512
MAX_SEQ = 64


def tiny_model() -> TaisObsidianForCausalLM:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=VOCAB,
        d_model=256,
        n_layer=4,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        n_v_heads=4,
        n_qk_heads=2,
        mlp_hidden=688,
        max_seq=MAX_SEQ,
        check_0p1b_params=False,
    )
    return TaisObsidianForCausalLM(cfg).eval()


class StubTok:
    """最小 tokenizer 桩：encode 返回固定 id 序列；eot_id=-1 保证生成不被 eot 提前截断。"""

    def __init__(self, ids: list[int]):
        self._ids = ids
        self.eot_id = -1

    def encode(self, text: str) -> list[int]:
        return list(self._ids)

    def decode(self, ids: list[int]) -> str:
        return "<stub>"


def test_out_of_vocab_ids_clear_error() -> None:
    """越界 id（≥vocab_size）：AssertionError 列出全部越界 id 与 vocab_size（非 CUDA assert）。"""
    model = tiny_model()
    tok = StubTok([1, 2, 600, 700])  # 600/700 ≥ 512 越界
    with pytest.raises(AssertionError) as ei:
        generate(model, tok, "x", 4, 0.0, 0, "cpu")
    msg = str(ei.value)
    assert "600" in msg and "700" in msg and str(VOCAB) in msg, f"错误信息应列出越界 id 与 vocab_size: {msg}"
    print(f"[guard] 越界报错：{msg}")


def test_prompt_exceeds_max_seq() -> None:
    """prompt 长度 > max_seq：ValueError 报清晰错误（非 RoPE 缓存越界 RuntimeError）。"""
    model = tiny_model()
    tok = StubTok([1] * (MAX_SEQ + 1))  # 65 > 64
    with pytest.raises(ValueError, match="max_seq"):
        generate(model, tok, "x", 4, 0.0, 0, "cpu")
    print("[guard] 超长 prompt 报错通过")


def test_generation_stops_at_max_seq(capsys) -> None:
    """增量生成：cache 位置达 max_seq 前停止并打印警告（不放任 RuntimeError）。"""
    model = tiny_model()
    tok = StubTok([1] * (MAX_SEQ - 4))  # prompt 60，余量 4 个位置
    text, _ = generate(model, tok, "x", 50, 0.0, 0, "cpu")  # 要求 50 个，只能放 4+1 个
    out = capsys.readouterr().out
    assert "已达 max_seq" in out, f"应打印 max_seq 停止警告: {out}"
    # prefill 60（位置 0..59）→ 位置 60/61/62/63 可增量 4 步；第 5 个采样后 pos=64 触顶停止
    assert "已生成 5 tokens" in out, f"停止时已生成数应为 5（4 步增量 + 1 次触顶采样）: {out}"
    assert text == "<stub>"
    print(f"[guard] max_seq 停止：{[ln for ln in out.splitlines() if 'max_seq' in ln]}")


def main() -> None:
    test_out_of_vocab_ids_clear_error()
    test_prompt_exceeds_max_seq()
    # test_generation_stops_at_max_seq 依赖 pytest 的 capsys 注入，请经 pytest 运行
    print("test_generate_guards 全部通过（max_seq 停止用例需经 pytest 运行）。")


if __name__ == "__main__":
    main()
