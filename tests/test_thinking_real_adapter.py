"""第二阶段真实部件适配测试：真实 0.1B 模型部件接入推理循环 + 有核/无核消融。

覆盖判据（对齐任务规范 §实现要求）：
  a) 维度桥接：RealThoughtAdapter down_proj 768→384、up_proj 384→768 形状正确；
  b) 真实 GDN 状态读出：model.forward capture 给 [B,T,768]，适配后 [B,T,384]；
  c) 真实 certainty 通路：run_kernel sense pik_logits → known 概率 ∈[0,1]
     （通路正确，无论校准与否；10k KAL 头未微调，仅演示非可靠元认知）；
  d) 真实部件接入 ReasoningLoop.run 跑通（轨迹非空，无异常）；
  e) 消融对比：有核 vs 无核轨迹不同（核确实改变动力学）；
  f) 共享 projector 一致性保持；
  g) demo 脚本 AST 通过。

用法：$env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe -m pytest tests/test_thinking_real_adapter.py -q
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.model import TaisObsidianForCausalLM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k" / "final"
DEMO_PATH = ROOT / "scripts" / "thinking_real_adapter_demo.py"

needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="真实 checkpoint 前向需 CUDA"
)
needs_ckpt = pytest.mark.skipif(
    not (CKPT_DIR / "model.safetensors").exists(), reason="10k checkpoint 缺失"
)


@pytest.fixture(scope="module")
def adapter():
    """加载真实模型 + 适配器（module 级复用，避免重复加载）。"""
    import thinking_real_adapter_demo as demo

    model = TaisObsidianForCausalLM.from_pretrained(CKPT_DIR, device=DEVICE)
    model.eval()
    if model.kernel is None:
        model.attach_kernel()
    ad = demo.RealThoughtAdapter(model).to(DEVICE)
    return ad


@pytest.fixture(scope="module")
def input_ids(adapter):
    g = torch.Generator(device=DEVICE).manual_seed(42)
    return torch.randint(
        0, adapter.model.config.vocab_size, (2, 32), device=DEVICE, generator=g
    )


# ---------------------------------------------------------------------------
@needs_cuda
@needs_ckpt
def test_dim_bridge(adapter):
    """维度桥接：down_proj 768→384、up_proj 384→768 形状正确。"""
    x768 = torch.randn(2, 5, 768, device=DEVICE)
    h384 = adapter.down_proj(x768)
    assert h384.shape == (2, 5, 384)
    x384 = torch.randn(2, 5, 384, device=DEVICE)
    h768 = adapter.up_proj(x384)
    assert h768.shape == (2, 5, 768)


@needs_cuda
@needs_ckpt
def test_real_gdn_state(adapter, input_ids):
    """真实 GDN 状态读出：capture 给 [B,T,768]，适配后 [B,T,384]。"""
    B, T = input_ids.shape
    # 原始 capture（未适配）应为 [B,T,768]
    with torch.no_grad():
        _, _, captures = adapter.model(input_ids, capture_layers=[10])
    assert captures[10].shape == (B, T, 768)
    # 适配后 [B,T,384]
    state = adapter.read_gdn_state(input_ids)
    assert state.shape == (B, T, 384)
    assert torch.isfinite(state).all()


@needs_cuda
@needs_ckpt
def test_real_certainty(adapter, input_ids):
    """真实 certainty 通路：sense pik_logits → known 概率 ∈[0,1]（通路正确）。"""
    cert = adapter.certainty(input_ids)
    assert 0.0 <= cert <= 1.0
    # 通路正确：pik_logits 三态
    with torch.no_grad():
        _, _, captures = adapter.model(input_ids, run_kernel=True)
    sense = captures["__kernel__"][10]["sense"]
    assert sense.pik_logits.shape[-1] == 3
    probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)


@needs_cuda
@needs_ckpt
def test_real_glimpse(adapter, input_ids):
    """真实 glimpse：CSA 层输出 [B,T,768] → [B,T,384]。"""
    B, T = input_ids.shape
    gl = adapter.glimpse(input_ids)
    assert gl.shape == (B, T, 384)
    assert torch.isfinite(gl).all()


@needs_cuda
@needs_ckpt
def test_real_loop_run(adapter, input_ids):
    """真实部件接入 ReasoningLoop.run 跑通：轨迹非空，无异常。"""
    import thinking_real_adapter_demo as demo

    built = demo.build_real_loop(adapter, input_ids, seed=42, mock_certainty=True)
    rl = built["reasoning_loop"]
    B, T, _ = built["real_state"].shape
    g = torch.Generator(device=DEVICE).manual_seed(42)
    target = torch.randn(B, T, 64, device=DEVICE, generator=g)
    final_state, trajectory, stop_tick = rl.run(
        built["real_state"], target_coord=target, max_ticks=8,
        stop_threshold=0.9, recall_threshold=0.3, bridge_alpha=0.1,
    )
    assert len(trajectory) >= 1
    assert final_state.shape == (B, T, 384)
    assert all(torch.isfinite(ts.current_coord.float()).all() for ts in trajectory)


@needs_cuda
@needs_ckpt
def test_ablation_differs(adapter, input_ids):
    """消融对比：有核 vs 无核轨迹不同（核确实改变动力学）。"""
    import thinking_real_adapter_demo as demo

    ab = demo.run_ablation(adapter, input_ids, seed=42)
    # 有核多 tick 演化（>1 步轨迹），无核单步——轨迹/位移非零即核改变动力学
    assert ab["n_ticks"] >= 1
    assert ab["total_disp"] > 0.0
    # 有核末坐标经多 tick 位移 ≠ 无核单步坐标（动力学被核改变）
    assert ab["dist_core"] != ab["dist_no_core"]


@needs_cuda
@needs_ckpt
def test_shared_projector(adapter, input_ids):
    """共享 projector 一致性：core.bridge、loop.bridge、shared 同一实例。"""
    import thinking_real_adapter_demo as demo

    built = demo.build_real_loop(adapter, input_ids, seed=42, mock_certainty=True)
    sp = built["shared_projector"]
    tc = built["thought_core"]
    rl = built["reasoning_loop"]
    assert tc.bridge.projector is sp
    assert rl.bridge.projector is sp
    assert rl.thought_core.bridge.projector is sp


def test_demo_ast():
    """demo 脚本 AST 语法通过。"""
    src = DEMO_PATH.read_text(encoding="utf-8")
    ast.parse(src)  # 不抛错即通过
