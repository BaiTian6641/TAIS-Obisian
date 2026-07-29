"""KAL 真值锚微调后 GDN-2 10k checkpoint 验证（CUDA）。

前置：先跑 scripts/kal_truth_finetune_gdn2.py 产出 checkpoints/pilot_0p1b_gdn2_10k_kaltruth。
验证（对齐实现要求）：
- 微调后 checkpoint 加载 kernel is not None（attach_kernel 已随 state_dict 存入）；
- 真值 AUROC（known vs fake，score=logit[0]−logit[2]）overall ≥ 0.8——certainty 可作元认知门控；
- known/fake 文本的 P(known) 方向正确（known 高、fake 低，certainty 语义正确）；
- 主干未被污染：微调后模型对 val 文本的 next-token loss 与微调前一致（detach 主干红线）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import kal_probe as kp  # noqa: E402
from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

ORIG_CKPT = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k" / "final"
TUNED_DIR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_kaltruth"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
SHARDS = ROOT / "data" / "shards"
# 微调脚本读点扫描默认候选，选 report 中记录的层（缺省 10 末 GDN 层）
READ_LAYER = 10

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")

_DEVICE = "cuda"


@pytest.fixture(scope="module")
def tok():
    return TokenizerIO(str(TOK))


@pytest.fixture(scope="module")
def tuned_model():
    if not TUNED_DIR.exists():
        pytest.skip("微调 checkpoint 未产出（先跑 scripts/kal_truth_finetune_gdn2.py）")
    # 微调 checkpoint 的 config.kernel_enabled=False（10k 训练时未挂内核），但
    # save_pretrained 存入了 attach_kernel 后的内核权重 → strict=True 会因 kernel.*
    # 多余键报错。用 strict=False 载入（主干权重严格匹配，kernel.* 多余键忽略），
    # 再显式 attach_kernel() 挂载内核头（其权重未被载入——需从 state_dict 手动恢复）。
    # 更简单做法：先 attach_kernel 再 load_state_dict(strict=True)。
    from safetensors.torch import load_file
    from tais_obsidian.config import ModelConfig
    cfg = ModelConfig.from_json(TUNED_DIR / "config.json")
    m = TaisObsidianForCausalLM(cfg)
    m.attach_kernel()  # 先挂载内核（strict 载入 kernel.* 键的前提）
    sd = load_file(str(TUNED_DIR / "model.safetensors"))
    m.load_state_dict(sd, strict=True)
    m = m.to(_DEVICE)
    m.eval()
    return m


@torch.no_grad()
def _auroc_on_l1(model, tok, layer, n_eval=120):
    """真值 AUROC：known(val) vs fake（kal_probe 模板），score = logit[0]−logit[2]。"""
    ids, labels_np, subset = kp.build_l1_dataset(
        tok, str(SHARDS), np.random.default_rng(999), n_eval, n_eval // 2, 0, 48)
    feats, _ = kp.forward_collect(model, ids, [layer], _DEVICE, 16, "last")
    h = torch.from_numpy(feats[layer]).to(_DEVICE)
    logits = model.kernel.kal_l1(h).float()
    scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
    known_binary = (labels_np == 1).astype(np.int64)
    fake_mask = (subset == "known") | (subset == "fake")
    return kp.auroc(scores, known_binary), kp.auroc(scores[fake_mask], known_binary[fake_mask])


@torch.no_grad()
def _p_known(model, tok, texts, layer):
    ids = kp.encode_fixed(tok, texts, 48)
    feats, _ = kp.forward_collect(model, ids, [layer], _DEVICE, 16, "last")
    h = torch.from_numpy(feats[layer]).to(_DEVICE)
    probs = torch.softmax(model.kernel.kal_l1(h).float(), dim=-1)
    return probs[:, 0].cpu().numpy()


@torch.no_grad()
def _val_next_token_loss(model, tok, n=16, T=48):
    """val 文本 next-token 平均 CE loss（主干未被污染的回归度量）。

    Shards.get_batch 返回 (x, y)，y=x 右移一位，两者均 [n, T]；next-token CE
    直接在 logits 与 y 间计算（logits 末位预测越界 token 不计，取前 T-1 位对齐）。
    """
    val = Shards(str(SHARDS), "val")
    rng = np.random.default_rng(123)
    x, y = val.get_batch(n, T, "cpu", rng)
    ids = x.to(_DEVICE)
    tgt = y.to(_DEVICE)
    logits, _ = model(ids)
    lp = torch.log_softmax(logits.float(), dim=-1)  # [n, T, V]
    # 位置 i 的 logit 预测位置 i+1（即 y[i]）；对齐到 y 的全部 T 位（y 已是移位目标）
    ce = -lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [n, T]
    return float(ce.mean().item())


def test_kernel_mounted(tuned_model):
    """微调后 checkpoint 加载 kernel is not None（内核权重已随 state_dict 存入）。"""
    assert tuned_model.kernel is not None, "kernel 为 None——attach_kernel 权重未随 checkpoint 存取"
    assert hasattr(tuned_model.kernel, "kal_l1")


def test_truth_auroc(tuned_model, tok):
    """真值 AUROC overall ≥ 0.8（certainty 可作元认知门控）。"""
    overall, fake = _auroc_on_l1(tuned_model, tok, READ_LAYER)
    print(f"\n[test] 真值 AUROC overall {overall:.3f} | fake {fake:.3f}")
    assert overall >= 0.8, f"overall AUROC {overall:.3f} 未达 0.8"


def test_certainty_direction(tuned_model, tok):
    """known 文本 P(known) 高、fake 文本 P(known) 低（certainty 语义正确）。"""
    import diverse_truth_data as dt
    rng = np.random.default_rng(777)
    known_texts = dt.build_real_statements(rng, 8)
    fake_texts = kp.build_fake_fact_texts(rng, 8)
    pk_known = _p_known(tuned_model, tok, known_texts, READ_LAYER)
    pk_fake = _p_known(tuned_model, tok, fake_texts, READ_LAYER)
    print(f"\n[test] known 文本 P(known) 均值 {pk_known.mean():.3f} | "
          f"fake 文本 P(known) 均值 {pk_fake.mean():.3f}")
    assert pk_known.mean() > 0.5, f"known 文本 P(known) {pk_known.mean():.3f} 应 > 0.5"
    assert pk_fake.mean() < 0.5, f"fake 文本 P(known) {pk_fake.mean():.3f} 应 < 0.5"
    assert pk_known.mean() > pk_fake.mean(), "known 应高于 fake"


def test_backbone_unpolluted(tuned_model, tok):
    """主干未被污染：微调后 val next-token loss 与微调前（原 10k checkpoint）一致。"""
    orig = TaisObsidianForCausalLM.from_pretrained(ORIG_CKPT, _DEVICE)
    orig.eval()
    loss_orig = _val_next_token_loss(orig, tok)
    loss_tuned = _val_next_token_loss(tuned_model, tok)
    print(f"\n[test] val next-token loss 微调前 {loss_orig:.5f} | 微调后 {loss_tuned:.5f}")
    # detach 主干红线：loss 应几乎逐位一致（容差 1e-3，bf16 加载数值抖动）
    assert abs(loss_orig - loss_tuned) < 1e-3, (
        f"主干被污染：loss 漂移 {abs(loss_orig - loss_tuned):.5f} ≥ 1e-3")
