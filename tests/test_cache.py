"""推理 cache 一致性测试：整段前向 vs prefill + 逐 token 增量前向，logits 应一致。

同时验证 save_pretrained/from_pretrained 往返一致。
用法：python tests/test_cache.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM


def tiny_cfg(attn_only: bool = False) -> ModelConfig:
    return ModelConfig(
        vocab_size=512,
        d_model=256,
        n_layer=4,
        n_q_heads=4,
        n_kv_heads=2,
        head_dim=64,
        n_v_heads=4,
        n_qk_heads=2,
        mlp_hidden=688,
        max_seq=128,
        attn_only=attn_only,
        check_0p1b_params=False,
    )


def check_cache(model: TaisObsidianForCausalLM, device: str, tag: str) -> None:
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (2, 33), device=device)
    with torch.no_grad():
        logits_full, _ = model(ids)
        # prefill 前 17 个，再逐 token 生成剩余 16 个
        logits_pre, cache = model(ids[:, :17])
        steps = [logits_pre]
        for i in range(17, 33):
            logits_i, cache = model(ids[:, i : i + 1], cache)
            steps.append(logits_i)
        logits_inc = torch.cat(steps, dim=1)
    d = (logits_full - logits_inc).abs().max().item()
    print(f"[{tag}] 整段 vs 增量: max diff {d:.2e}")
    assert d < 1e-4, d


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for attn_only in (False, True):
        torch.manual_seed(42)
        model = TaisObsidianForCausalLM(tiny_cfg(attn_only)).to(device).eval()
        check_cache(model, device, f"attn_only={attn_only}")
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(tmp)
            model2 = TaisObsidianForCausalLM.from_pretrained(tmp, device)
            ids = torch.randint(0, 512, (1, 16), device=device)
            with torch.no_grad():
                o1 = model(ids)[0]
                o2 = model2(ids)[0]
            d = (o1 - o2).abs().max().item()
            rel = d / o1.abs().max().item()
            print(f"[attn_only={attn_only}] save/load 往返: max diff {d:.2e}, 相对 {rel:.2e}")
            assert rel < 1e-2, rel  # bf16 存储的相对误差量级
    print("test_cache 全部通过。")


def test_cache_consistency_and_saveload() -> None:
    """pytest 收集入口：与 main() 等价。"""
    main()


if __name__ == "__main__":
    main()
