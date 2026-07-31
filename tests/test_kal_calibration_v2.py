"""KAL 真值锚微调 v2（锚集扩充+预测反馈循环）GDN-2 10k checkpoint 验证（CUDA）。

前置：先跑 scripts/kal_truth_finetune_v2.py 产出 checkpoints/pilot_0p1b_gdn2_10k_kaltruth_v2。
验证（校准 P1 目标：双口径稳定 ≥0.8，非卡边）：
- 微调后 checkpoint 加载 kernel is not None（attach_kernel 权重随 state_dict 存入）；
- 真值 AUROC（known vs fake，score=logit[0]−logit[2]）双口径（测试 n_eval=400 + 脚本
  n_eval=200）×3 评估 seed：均值 ≥0.8 且最低 seed ≥0.78（稳定性，非单 seed 卡边）；
- known/fake 文本 P(known) 方向正确（certainty 语义）；
- 主干未被污染：val next-token loss 与微调前一致（detach 主干红线）。
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
TUNED_DIR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_kaltruth_v2"
TOK = ROOT / "data" / "tokenizer" / "tokenizer.json"
SHARDS = ROOT / "data" / "shards"
READ_LAYER = 10  # kaltruth v1 读点扫描选定的末 GDN 层（v2 report 同层）

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="需 CUDA")

_DEVICE = "cuda"
_EVAL_SEEDS = [999, 1999, 2999]  # 与 v2 脚本 EVAL_SEEDS 一致


@pytest.fixture(scope="module")
def tok():
    return TokenizerIO(str(TOK))


@pytest.fixture(scope="module")
def tuned_model():
    if not TUNED_DIR.exists():
        pytest.skip("v2 微调 checkpoint 未产出（先跑 scripts/kal_truth_finetune_v2.py）")
    # 记忆坑 1：config.kernel_enabled=False 但 state_dict 含 kernel.* 键——
    # 先 attach_kernel() 再 load_state_dict(strict=True)。
    from safetensors.torch import load_file
    from tais_obsidian.config import ModelConfig
    cfg = ModelConfig.from_json(TUNED_DIR / "config.json")
    m = TaisObsidianForCausalLM(cfg)
    m.attach_kernel()
    sd = load_file(str(TUNED_DIR / "model.safetensors"))
    m.load_state_dict(sd, strict=True)
    m = m.to(_DEVICE)
    m.eval()
    return m


@torch.no_grad()
def _auroc_on_l1(model, tok, layer, n_eval=400, eval_seed=999):
    """真值 AUROC：known(val) vs fake（kal_probe 模板），score = logit[0]−logit[2]。

    模块级缓存：同一 (n_eval, eval_seed) 不重复前向（双口径×3seed 多测试复用）。
    """
    key = (n_eval, eval_seed)
    if key in _AUROC_CACHE:
        return _AUROC_CACHE[key]
    ids, labels_np, subset = kp.build_l1_dataset(
        tok, str(SHARDS), np.random.default_rng(eval_seed), n_eval, n_eval // 2, 0, 48)
    feats, _ = kp.forward_collect(model, ids, [layer], _DEVICE, 8, "last")
    h = torch.from_numpy(feats[layer]).to(_DEVICE)
    logits = model.kernel.kal_l1(h).float()
    scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
    known_binary = (labels_np == 1).astype(np.int64)
    fake_mask = (subset == "known") | (subset == "fake")
    res = kp.auroc(scores, known_binary), kp.auroc(scores[fake_mask], known_binary[fake_mask])
    _AUROC_CACHE[key] = res
    return res


_AUROC_CACHE: dict = {}


@torch.no_grad()
def _p_known(model, tok, texts, layer):
    ids = kp.encode_fixed(tok, texts, 48)
    feats, _ = kp.forward_collect(model, ids, [layer], _DEVICE, 8, "last")
    h = torch.from_numpy(feats[layer]).to(_DEVICE)
    probs = torch.softmax(model.kernel.kal_l1(h).float(), dim=-1)
    return probs[:, 0].cpu().numpy()


@torch.no_grad()
def _val_next_token_loss(model, tok, n=16, T=48):
    """val 文本 next-token 平均 CE loss（主干未被污染的回归度量；get_batch 已右移对齐）。"""
    val = Shards(str(SHARDS), "val")
    rng = np.random.default_rng(123)
    x, y = val.get_batch(n, T, "cpu", rng)
    logits, _ = model(x.to(_DEVICE))
    lp = torch.log_softmax(logits.float(), dim=-1)
    ce = -lp.gather(-1, y.to(_DEVICE).unsqueeze(-1)).squeeze(-1)
    return float(ce.mean().item())


def test_kernel_mounted(tuned_model):
    """v2 checkpoint 加载 kernel is not None（内核权重已随 state_dict 存入）。"""
    assert tuned_model.kernel is not None, "kernel 为 None——attach_kernel 权重未随 checkpoint 存取"
    assert hasattr(tuned_model.kernel, "kal_l1")


def test_truth_auroc_v2(tuned_model, tok):
    """真值 AUROC 测试口径（n_eval=400）≥ 0.8——校准 P1 目标（v1 为 0.79~0.80 卡边）。"""
    overall, fake = _auroc_on_l1(tuned_model, tok, READ_LAYER)
    print(f"\n[test] v2 真值 AUROC overall {overall:.3f} | fake {fake:.3f}")
    assert overall >= 0.8, f"overall AUROC {overall:.3f} < 0.8（校准 P1 未达标）"


def test_truth_auroc_stability(tuned_model, tok):
    """3 评估 seed 稳定性：均值 ≥0.8 且最低 seed ≥0.78（双口径达标的稳定性判据）。"""
    ovs = []
    for es in _EVAL_SEEDS:
        ov, _ = _auroc_on_l1(tuned_model, tok, READ_LAYER, eval_seed=es)
        ovs.append(ov)
    mean, worst = float(np.mean(ovs)), float(np.min(ovs))
    print(f"\n[test] 3 seed overall AUROC {[round(v,3) for v in ovs]} "
          f"均值 {mean:.3f} 最低 {worst:.3f}")
    assert mean >= 0.8, f"3 seed 均值 {mean:.3f} < 0.8"
    assert worst >= 0.78, f"最低 seed {worst:.3f} < 0.78（卡边不稳定）"


def test_truth_auroc_script_n200(tuned_model, tok):
    """脚本口径（n_eval=200）×3 seed：均值 ≥0.8 且最低 seed ≥0.78（双口径的另一口径）。"""
    ovs = []
    for es in _EVAL_SEEDS:
        ov, _ = _auroc_on_l1(tuned_model, tok, READ_LAYER, n_eval=200, eval_seed=es)
        ovs.append(ov)
    mean, worst = float(np.mean(ovs)), float(np.min(ovs))
    print(f"\n[test] 脚本口径 n200 3 seed overall {[round(v,3) for v in ovs]} "
          f"均值 {mean:.3f} 最低 {worst:.3f}")
    assert mean >= 0.8, f"n200 3 seed 均值 {mean:.3f} < 0.8"
    assert worst >= 0.78, f"n200 最低 seed {worst:.3f} < 0.78（卡边不稳定）"


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
    """主干未被污染：v2 微调后 val next-token loss 与微调前（原 10k checkpoint）一致。"""
    orig = TaisObsidianForCausalLM.from_pretrained(ORIG_CKPT, _DEVICE)
    orig.eval()
    loss_orig = _val_next_token_loss(orig, tok)
    loss_tuned = _val_next_token_loss(tuned_model, tok)
    print(f"\n[test] val next-token loss 微调前 {loss_orig:.5f} | 微调后 {loss_tuned:.5f}")
    assert abs(loss_orig - loss_tuned) < 1e-3, (
        f"主干被污染：loss 漂移 {abs(loss_orig - loss_tuned):.5f} ≥ 1e-3")
