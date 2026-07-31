"""checkpoint/resume 端到端回归：step/loss/lr 精确接续 + model_cfg 不一致报错路径。

用 CPU 小模型保证确定性（fp32 无 autocast，无 CUDA 非确定算子）：
同一随机序列下，不间断训 5 步的末 2 步 loss 必须与"3 步存档 → resume 续训 2 步"逐 bit 一致
（权重/优化器矩/numpy batch 采样 RNG 全恢复）。另有 model_cfg 关键字段错配拒绝续训路径。
用法：python -m pytest tests/test_resume_e2e.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import (
    build_optimizer,
    chunked_ce,
    lr_at,
    save_checkpoint,
    set_lr,
    validate_resume_model_cfg,
)

DEVICE = "cpu"  # 确定性：CPU fp32 同进程逐 bit 可复现
CFG = dict(lr=1e-3, warmup=2, max_steps=5, decay_frac=0.4, weight_decay=0.0,
           grad_clip=1.0, micro_batch=2, seq_len=32)


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


def make_batch(data: torch.Tensor, rng: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """模拟 Shards.get_batch：numpy rng 采偏移取窗（resume 后偏移序列接续是关键验证点）。"""
    seq = CFG["seq_len"]
    offs = rng.integers(0, data.numel() - seq - 1, size=CFG["micro_batch"])
    x = torch.stack([data[o : o + seq] for o in offs])
    y = torch.stack([data[o + 1 : o + seq + 1] for o in offs])
    return x, y


def train_steps(model, opt, data, rng, start: int, n: int) -> list[float]:
    """镜像 train.py 主循环核心（set_lr → 前向 → clip → step），CPU 路径无 autocast。"""
    losses = []
    model.train()
    for step in range(start, start + n):
        set_lr(opt, lr_at(step, CFG), CFG)
        opt.zero_grad(set_to_none=True)
        x, y = make_batch(data, rng)
        logits, _ = model(x)
        loss = chunked_ce(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
        opt.step()
        losses.append(loss.item())
    return losses


def _fresh_model_opt_data():
    """固定种子重建 data/模型/优化器（两次调用产出逐 bit 一致的初始状态）。"""
    torch.manual_seed(42)
    data = torch.randint(0, 512, (4096,), dtype=torch.long)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(DEVICE)
    opt = build_optimizer(model, {"lr": CFG["lr"], "weight_decay": CFG["weight_decay"]})
    return data, model, opt


def test_resume_step_loss_lr_continuation(tmp_path: Path) -> None:
    """3 步存档 → resume 续训 2 步：step 接续、loss 与不间断训练一致、优化器矩恢复。"""
    ckpt_path = tmp_path / "latest.pt"

    # —— 不间断参考：训满 5 步，记录末 2 步 loss ——
    data, model, opt = _fresh_model_opt_data()
    rng = np.random.default_rng(7)
    train_steps(model, opt, data, rng, 0, 3)
    save_checkpoint(ckpt_path, model, opt, 3, CFG, rng)
    ref_losses = train_steps(model, opt, data, rng, 3, 2)  # 参考 continuation（步 3、4）

    # —— resume 路径：全新模型/优化器从断点恢复，续训 2 步（语义同 train.py main 的 resume 块）——
    data2, model2, opt2 = _fresh_model_opt_data()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    validate_resume_model_cfg(ckpt.get("model_cfg"), model2.config)  # 一致：不应抛
    assert ckpt["step"] == 3, "断点 step 应为 3"
    model2.load_state_dict(ckpt["model"])
    opt2.load_state_dict(ckpt["opt"])
    start_step = ckpt["step"]
    torch.set_rng_state(ckpt["rng"]["torch"])
    rng2 = np.random.default_rng()
    rng2.bit_generator.state = ckpt["rng"]["numpy"]
    assert start_step == 3

    resumed_losses = train_steps(model2, opt2, data2, rng2, start_step, 2)
    assert resumed_losses == ref_losses, (
        f"resume 后续训 loss 应与不间断训练逐 bit 一致：{resumed_losses} ≠ {ref_losses}"
    )
    # 优化器矩接续：Adam step 计数应累计到 5（若 resume 丢优化器状态则重新从 1 计）
    adam_step = {float(s["step"]) for st in opt2.state_dict()["state"].values() for s in [st]}
    assert adam_step == {5.0}, f"Adam step 计数应为 5，实际 {adam_step}"
    print(f"[resume] step 3→5 接续，loss {resumed_losses} 与不间断训练一致")


def test_validate_resume_model_cfg_mismatch() -> None:
    """model_cfg 关键字段错配 → SystemExit 且打印差异字段；一致/缺失字段的兼容路径不炸。"""
    cur = tiny_cfg()
    good = {k: getattr(cur, k) for k in ("vocab_size", "d_model", "n_layer", "n_q_heads",
                                         "n_kv_heads", "head_dim", "n_v_heads", "n_qk_heads",
                                         "mlp_hidden", "max_seq", "rope_scaling", "rope_scale",
                                         "gdn_decay_g_min")}
    validate_resume_model_cfg(good, cur)  # 一致：静默通过

    bad = dict(good, d_model=512, max_seq=2048)
    with pytest.raises(SystemExit) as ei:
        validate_resume_model_cfg(bad, cur)
    msg = str(ei.value)
    assert "d_model" in msg and "512" in msg and "max_seq" in msg, f"差异字段表应列出错配字段: {msg}"
    print(f"[resume] 错配拒绝信息：{msg.splitlines()[1:]}")

    validate_resume_model_cfg(None, cur)  # 极旧 ckpt 无 model_cfg：降级警告，不抛


def main() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_resume_step_loss_lr_continuation(Path(d))
    test_validate_resume_model_cfg_mismatch()
    print("test_resume_e2e 全部通过。")


if __name__ == "__main__":
    main()
