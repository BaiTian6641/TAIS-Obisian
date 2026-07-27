"""ITI 干预头单元测试（KAL 执行通道，规范 §5）。

判据（iti_head docstring / Braun 2505.22637 红线）：
- 方向派生：from_kal_l1 方向 = W[know]−W[blank]（diff-in-means），归一化；
- α 有界：alpha_frac 钳制到 max_alpha_frac（防过强崩溃）；
- steer 语义：alpha=0 不动、alpha>0 沿方向 shift、reverse 取反；
- 红线：绝不把空白 steer 成知道（造假）；fail-closed（无 kal_l1 → RuntimeError）；
- 人效代理：小 α steer 不大幅改变 hidden 整体方向（cos 相似度高）。
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tais_obsidian.model.iti_head import ITIHead, make_iti_from_kernel
from tais_obsidian.model.kal import make_l1_head
from tais_obsidian.model.tais_kernel import TAISKernel

D = 64


def _kal_l1():
    head = make_l1_head(D)
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.weight[0, 0] = 1.0   # know 轴 = e0
        head.proj.weight[2, 1] = 1.0   # blank 轴 = e1
    return head


def test_direction_from_kal_l1() -> None:
    head = _kal_l1()
    iti = ITIHead.from_kal_l1(head)
    # 方向 = W[0]−W[2] = e0−e1，归一化后 = (e0−e1)/√2
    expected = F.normalize(torch.tensor([1.0, -1.0] + [0.0] * (D - 2)), dim=0)
    assert torch.allclose(iti.direction, expected, atol=1e-5)
    assert abs(iti.direction.norm().item() - 1.0) < 1e-5


def test_alpha_zero_no_steer() -> None:
    iti = ITIHead.from_kal_l1(_kal_l1())
    h = torch.randn(2, 5, D)
    out = iti.steer(h, alpha_frac=0.0)
    assert torch.equal(out, h), "alpha=0 应不动"


def test_alpha_bounded() -> None:
    iti = ITIHead.from_kal_l1(_kal_l1(), max_alpha_frac=0.2)
    h = torch.randn(1, 4, D)
    # 请求超大 alpha_frac=10 → 钳制到 0.2
    out = iti.steer(h, alpha_frac=10.0)
    res_norm = h.norm(dim=-1).mean()
    shift = (out - h).norm(dim=-1).mean()
    # shift ≈ 0.2 × res_norm（钳制后），非 10×
    assert shift < 0.3 * res_norm, f"α 须钳制有界，shift={shift:.2f} res_norm={res_norm:.2f}"


def test_steer_direction_and_reverse() -> None:
    iti = ITIHead.from_kal_l1(_kal_l1(), max_alpha_frac=0.5)
    h = torch.zeros(1, 1, D)
    out_pos = iti.steer(h, alpha_frac=0.1, reverse=False)  # h 全零 → res_norm=0，alpha=0
    # 全零残差 norm=0 → alpha=0 → 不动（边界）
    assert torch.allclose(out_pos, h)
    # 非零 hidden：正向 shift 应沿 +direction，反向沿 −direction
    h2 = torch.ones(1, 1, D)
    d_pos = iti.steer(h2, 0.1, reverse=False) - h2
    d_neg = iti.steer(h2, 0.1, reverse=True) - h2
    assert torch.dot(d_pos.flatten(), iti.direction) > 0, "正向应沿 +direction"
    assert torch.dot(d_neg.flatten(), iti.direction) < 0, "反向应沿 −direction"


def test_make_iti_fail_closed() -> None:
    class _NoKal:
        kernel = None
    with pytest.raises(RuntimeError):
        make_iti_from_kernel(_NoKal().kernel)


def test_make_iti_from_kernel() -> None:
    kern = TAISKernel(D)
    iti = make_iti_from_kernel(kern)
    assert isinstance(iti, ITIHead)
    assert abs(iti.direction.norm().item() - 1.0) < 1e-4


def test_steer_preserves_global_direction() -> None:
    # 小 α steer 不应大幅改变 hidden 整体方向（人效代理：cos 相似度高）
    iti = ITIHead.from_kal_l1(_kal_l1(), max_alpha_frac=0.1)
    h = torch.randn(1, 8, D) * 10  # 大 norm
    out = iti.steer(h, alpha_frac=0.05)
    cos = F.cosine_similarity(h.flatten(), out.flatten(), dim=0)
    assert cos > 0.99, f"小 α steer 应保整体方向（人效代理），cos={cos:.4f}"


def test_invalid_max_alpha() -> None:
    with pytest.raises(ValueError):
        ITIHead(torch.ones(D), max_alpha_frac=0.0)
    with pytest.raises(ValueError):
        ITIHead(torch.ones(D), max_alpha_frac=1.5)
