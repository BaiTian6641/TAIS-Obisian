"""save_pretrained/from_pretrained 逐字段往返 + tokenizer.json 回退加载回归测试。

覆盖：非默认尺寸（0.5B 风格字段值）+ rope_scaling(YaRN) 字段 + kernel_enabled 权重共存
（attach_kernel 同步 config 标志的回归，test_kal_gdn2_truth.py:49-53 记录过的坑）
+ generate.resolve_tokenizer_path 的 <ckpt>/tokenizer.json → data/tokenizer 回退链。
用法：python -m pytest tests/test_save_load_sizes.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.generate import resolve_tokenizer_path
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import copy_tokenizer_to_final

REPO_TOKENIZER = ROOT / "data" / "tokenizer" / "tokenizer.json"


def nondefault_cfg() -> ModelConfig:
    """全字段非默认值（0.5B 风格的显式配置写法）：任何字段透传丢失都会在往返中暴露。"""
    return ModelConfig(
        vocab_size=33000,          # 非 32768 默认
        d_model=256,
        n_layer=4,
        block_pattern=["G2", "A", "G2", "A"],
        n_q_heads=8,
        n_kv_heads=2,
        head_dim=32,
        rope_theta=50000.0,        # 非 10000 默认
        rope_scaling="yarn",       # 非 none 默认
        rope_scale=4.0,
        rope_original_max_seq=128,
        n_v_heads=8,
        n_qk_heads=4,
        conv_kernel=3,             # 非 4 默认
        gdn_decay_g_min=-3.0,      # 非 -5.0 默认
        mlp_hidden=1024,
        rms_eps=1e-5,              # 非 1e-6 默认
        max_seq=512,
        grad_checkpoint=False,     # 非 True 默认
        check_0p1b_params=False,
        pm_stream=1,
        pm_constrain=True,
        pm_sk_t_max=10,            # 非 20 默认
        tri_window=256,
        tri_csa_stride=4,
        tri_csa_topk=32,
        tri_hca_stride=64,
        tri_use_indexer=True,
        tri_index_heads=2,
        tri_index_dim=16,
        kernel_enabled=True,       # 挂内核（bf16 权重随 state_dict 往返）
        kernel_dg_dim=128,         # 非 256 默认
        kernel_dg_topk=16,         # 非 32 默认
        kernel_sense_layers=[0, 2],
        manifold_dim=32,           # 非 64 默认
    )


def test_nondefault_sizes_roundtrip(tmp_path: Path) -> None:
    """非默认尺寸 + rope_scaling 字段：config 逐字段相等 + 全部权重 bf16 往返一致。"""
    torch.manual_seed(0)
    model = TaisObsidianForCausalLM(nondefault_cfg())
    out = tmp_path / "final"
    model.save_pretrained(out)
    m2 = TaisObsidianForCausalLM.from_pretrained(out)  # strict 默认 True
    # config 逐字段往返（dataclass __eq__ 全字段比较；任何透传/落盘丢失都会暴露）
    assert m2.config == model.config
    # 权重往返：from_pretrained 载入的是 bf16 值，逐键与 fp32→bf16 截断值比较
    sd1, sd2 = model.state_dict(), m2.state_dict()
    assert set(sd1) == set(sd2)
    for k in sd1:
        assert torch.equal(sd2[k], sd1[k].to(torch.bfloat16).to(torch.float32)), f"权重不一致: {k}"
    # kernel 权重随 state_dict 往返（kernel_enabled=True 时 from_pretrained 自动建内核）
    assert m2.kernel is not None
    print(f"[roundtrip] {len(sd1)} 键 + config 全字段（含 yarn/kernel/非默认尺寸）往返一致")


def test_attach_kernel_sets_enabled_flag(tmp_path: Path) -> None:
    """attach_kernel() 幂等且同步 config.kernel_enabled=True（修 kernel 权重+标志 false 的 strict 崩溃）。"""
    torch.manual_seed(0)
    cfg = nondefault_cfg()
    cfg.kernel_enabled = False
    model = TaisObsidianForCausalLM(cfg)
    assert model.kernel is None and model.config.kernel_enabled is False
    model.attach_kernel()
    assert model.kernel is not None
    assert model.config.kernel_enabled is True, "attach_kernel 必须同步 config 标志（否则存取不一致）"
    model.attach_kernel()  # 幂等：重复调用不炸
    # 存取闭环：config.json 记 True → from_pretrained 建内核 → kernel.* 键 strict 载入
    out = tmp_path / "final"
    model.save_pretrained(out)
    m2 = TaisObsidianForCausalLM.from_pretrained(out)
    assert m2.kernel is not None and m2.config.kernel_enabled is True
    print("[attach_kernel] 标志同步 + 幂等 + strict 存取闭环通过")


def test_tokenizer_fallback_resolution(tmp_path: Path) -> None:
    """resolve_tokenizer_path：显式 > <ckpt>/tokenizer.json > data/tokenizer/tokenizer.json。"""
    # 显式路径优先（且校验存在性）
    explicit = tmp_path / "tok.json"
    explicit.write_text("{}", encoding="utf-8")
    assert resolve_tokenizer_path(tmp_path, str(explicit)) == explicit
    with pytest.raises(FileNotFoundError):
        resolve_tokenizer_path(tmp_path, str(tmp_path / "missing.json"))
    # <ckpt>/tokenizer.json 次优
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert resolve_tokenizer_path(ckpt_dir, None) == ckpt_dir / "tokenizer.json"
    # 回退仓库默认（data/tokenizer/tokenizer.json 存在时）
    (ckpt_dir / "tokenizer.json").unlink()
    if REPO_TOKENIZER.exists():
        assert resolve_tokenizer_path(ckpt_dir, None) == REPO_TOKENIZER
    # copy_tokenizer_to_final：把仓库 tokenizer 复制进产物目录（随权重上传）
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    if REPO_TOKENIZER.exists():
        copy_tokenizer_to_final(final_dir)
        assert (final_dir / "tokenizer.json").exists()
        assert resolve_tokenizer_path(final_dir, None) == final_dir / "tokenizer.json"
    print("[tokenizer] 解析回退链 + 随产物复制通过")


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_nondefault_sizes_roundtrip(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_attach_kernel_sets_enabled_flag(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_tokenizer_fallback_resolution(Path(d))
    print("test_save_load_sizes 全部通过。")


if __name__ == "__main__":
    main()
