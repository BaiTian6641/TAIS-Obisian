"""export_final 全链回归：latest.pt → 导出 save_pretrained 目录 → from_pretrained → 生成 1 token。

覆盖：ckpt["model_cfg"] 恢复结构、tokenizer.json 随产物复制、无 model_cfg 时 --config 回退
与双缺报错。链路即 Colab 训练 → 上传 HF → 下载推理的最后一公里。
用法：python -m pytest tests/test_export_final.py -q
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from export_final import export_final  # noqa: E402
from tais_obsidian.config import ModelConfig  # noqa: E402
from tais_obsidian.generate import generate  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.train import build_optimizer, save_checkpoint  # noqa: E402

REPO_TOKENIZER = ROOT / "data" / "tokenizer" / "tokenizer.json"


def tiny_cfg() -> ModelConfig:
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
        check_0p1b_params=False,
    )


class StubTok:
    def __init__(self, ids: list[int]):
        self._ids = ids
        self.eot_id = -1

    def encode(self, text: str) -> list[int]:
        return list(self._ids)

    def decode(self, ids: list[int]) -> str:
        return "<stub>"


def _make_latest_pt(path: Path, with_model_cfg: bool = True) -> None:
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(tiny_cfg())
    opt = build_optimizer(model, {"lr": 1e-3, "weight_decay": 0.0})
    rng = np.random.default_rng(0)
    save_checkpoint(path, model, opt, 7, {"lr": 1e-3}, rng)
    if not with_model_cfg:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        del ckpt["model_cfg"]
        torch.save(ckpt, path)


def test_export_from_latest_pt_full_chain(tmp_path: Path) -> None:
    """latest.pt（含 model_cfg）→ export → from_pretrained → generate 1 token 全链。"""
    latest = tmp_path / "latest.pt"
    _make_latest_pt(latest)
    out = export_final(latest, tmp_path / "final")
    # 产物齐全：config.json + model.safetensors（+ tokenizer.json 随产物，仓库 tokenizer 存在时）
    assert (out / "config.json").exists() and (out / "model.safetensors").exists()
    if REPO_TOKENIZER.exists():
        assert (out / "tokenizer.json").exists(), "tokenizer.json 应随权重产物复制"
    # from_pretrained 结构还原 + 权重一致 + 生成 1 token
    m2 = TaisObsidianForCausalLM.from_pretrained(out)
    assert m2.config == tiny_cfg(), "config 应从 ckpt['model_cfg'] 完整恢复"
    text, tok_s = generate(m2, StubTok([1, 2, 3]), "x", 1, 0.0, 0, "cpu")
    assert text == "<stub>" and tok_s >= 0
    print("[export] latest.pt → final → from_pretrained → generate 1 token 全链通过")


def test_export_config_fallback_and_missing(tmp_path: Path) -> None:
    """无 model_cfg 的极旧 latest.pt：--config JSON 回退可导出；双缺则报错退出。"""
    latest = tmp_path / "old.pt"
    _make_latest_pt(latest, with_model_cfg=False)
    # 双缺：无 model_cfg 且无 --config → SystemExit
    with pytest.raises(SystemExit):
        export_final(latest, tmp_path / "final_bad")
    # --config 回退：经 build_model_config 还原结构（字段与 tiny_cfg 对齐）
    cfg_json = tmp_path / "cfg.json"
    cfg_json.write_text(json.dumps({
        "vocab_size": 512, "d_model": 256, "n_layer": 4, "n_q_heads": 4, "n_kv_heads": 2,
        "head_dim": 64, "n_v_heads": 4, "n_qk_heads": 2, "mlp_hidden": 688, "max_seq": 128,
        "check_0p1b_params": False,
    }), encoding="utf-8")
    out = export_final(latest, tmp_path / "final_ok", config=cfg_json)
    m2 = TaisObsidianForCausalLM.from_pretrained(out)
    assert m2.config.d_model == 256 and m2.config.n_layer == 4
    print("[export] --config 回退 + 双缺报错路径通过")


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_export_from_latest_pt_full_chain(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_export_config_fallback_and_missing(Path(d))
    print("test_export_final 全部通过。")


if __name__ == "__main__":
    main()
