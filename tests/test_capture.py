"""hidden-state 捕获 API（capture_layers）测试：形状、数值、增量路径与向后兼容。

用法：python tests/test_capture.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM

# 一个 G 层 + 一个 A 层（tiny_cfg 层型为 G,G,G,A）
LAYERS_IDX = [0, 3]


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
        check_0p1b_params=False,  # grad_checkpoint 保持默认 True
    )


def check_backward_compat(model: TaisObsidianForCausalLM, device: str) -> None:
    """默认调用（capture_layers=None）仍严格返回 (logits, cache) 二元组。"""
    ids = torch.randint(0, 512, (1, 8), device=device)
    with torch.no_grad():
        out = model(ids)
        out3 = model(ids, capture_layers=[])
    assert isinstance(out, tuple) and len(out) == 2, f"默认返回应为二元组，实际长度 {len(out)}"
    assert len(out3) == 3 and out3[2] == {}, "空 capture_layers 应返回三元组 + 空 captures"
    print("[compat] 默认调用返回二元组，向后兼容。")


def check_capture_matches_hooks(model: TaisObsidianForCausalLM, device: str) -> None:
    """捕获全部 4 层：形状 [B,T,d]，数值与 register_forward_hook 参考逐点一致。"""
    types = [layer.type for layer in model.layers]
    # GDN 系（"G"/"G2"）+ 注意力（"A"）；GDN-2 切换后 tiny 默认 G2G2G2A
    assert set(types) <= {"G", "G2", "A"} and "A" in types, f"tiny 配置应覆盖 GDN 系/A 层，实际 {types}"
    torch.manual_seed(0)
    ids = torch.randint(0, 512, (2, 24), device=device)
    refs: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(model.layers):

        def make_hook(idx: int):
            def hook(module, args, output):  # Block 返回 (x, new_state)
                refs[idx] = output[0]

            return hook

        handles.append(layer.register_forward_hook(make_hook(i)))
    with torch.no_grad():
        _, _, captures = model(ids, capture_layers=list(range(model.config.n_layer)))
    for h in handles:
        h.remove()
    assert set(captures) == set(range(model.config.n_layer)), f"捕获层索引不符: {sorted(captures)}"
    for i in range(model.config.n_layer):
        cap, ref = captures[i], refs[i]
        assert cap.shape == (2, 24, model.config.d_model), (i, cap.shape)
        d = (cap - ref).abs().max().item()
        print(f"[capture] layer {i} ({model.layers[i].type}) vs hook: max diff {d:.2e}")
        assert d < 1e-6, (i, d)


def check_capture_incremental(model: TaisObsidianForCausalLM, device: str) -> None:
    """增量路径：prefill 17 token 带捕获 + 带 cache 单 token 前向带捕获，形状与数值正确。"""
    torch.manual_seed(1)
    ids = torch.randint(0, 512, (2, 18), device=device)
    with torch.no_grad():
        _, _, caps_full = model(ids, capture_layers=LAYERS_IDX)
        _, cache, caps_pre = model(ids[:, :17], capture_layers=LAYERS_IDX)
        _, _, caps_tok = model(ids[:, 17:18], cache, capture_layers=LAYERS_IDX)
    for i in LAYERS_IDX:
        assert caps_pre[i].shape == (2, 17, model.config.d_model), (i, caps_pre[i].shape)
        assert caps_tok[i].shape == (2, 1, model.config.d_model), (i, caps_tok[i].shape)
        d = (caps_full[i][:, 17:18] - caps_tok[i]).abs().max().item()
        print(f"[incremental] layer {i} ({model.layers[i].type}) 整段 vs 增量: max diff {d:.2e}")
        assert d < 1e-4, (i, d)


def check_capture_grad_checkpoint(model: TaisObsidianForCausalLM, device: str) -> None:
    """训练路径（grad checkpoint 生效）：捕获仍正确，且反传不受影响。"""
    assert model.config.grad_checkpoint, "tiny_cfg 应保持 grad_checkpoint=True"
    torch.manual_seed(2)
    ids = torch.randint(0, 512, (2, 16), device=device)
    model.train()
    logits, _, caps_train = model(ids, capture_layers=LAYERS_IDX)
    for i in LAYERS_IDX:
        assert caps_train[i].shape == (2, 16, model.config.d_model), (i, caps_train[i].shape)
    logits.float().sum().backward()  # checkpoint 反传路径不受捕获影响
    model.eval()
    model.zero_grad()
    with torch.no_grad():
        _, _, caps_eval = model(ids, capture_layers=LAYERS_IDX)
    for i in LAYERS_IDX:
        d = (caps_train[i].detach() - caps_eval[i]).abs().max().item()
        print(f"[grad_ckpt] layer {i} ({model.layers[i].type}) train(ckpt) vs eval: max diff {d:.2e}")
        assert d < 1e-6, (i, d)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
    model = TaisObsidianForCausalLM(tiny_cfg()).to(device).eval()
    check_backward_compat(model, device)
    check_capture_matches_hooks(model, device)
    check_capture_incremental(model, device)
    check_capture_grad_checkpoint(model, device)
    print("test_capture 全部通过。")


def test_capture_api() -> None:
    """pytest 收集入口：与 main() 等价。"""
    main()


if __name__ == "__main__":
    main()
