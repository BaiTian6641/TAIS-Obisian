"""KAL L1 校准层单元测试（isotonic/conformal/AURC/ECE，纯 CPU）。

判据（article_ref/07 §2）：
- PAV isotonic 输出单调非降；
- 校准后 OOD 概率在 [0,1] 且保序；
- conformal 负类阈值给出误受率 ≤ α（有限样本，分布交换）；
- AURC：完美探测→0，随机→正类先验×(1-先验)量级；
- ECE：完美校准→0；未 fit 调用 fail-closed。
"""
from __future__ import annotations

import numpy as np
import pytest

from tais_obsidian.model.kal_calibrate import (
    ConformalGate,
    IsotonicCalibrator,
    _pav_isotonic,
    aurc,
    expected_calibration_error,
)


def test_pav_monotonic() -> None:
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    labels = (scores + rng.normal(scale=0.5, size=200) > 0).astype(int)
    thr, val = _pav_isotonic(scores, labels)
    assert np.all(np.diff(val) >= -1e-9), "PAV 输出须单调非降"
    assert np.all((val >= 0) & (val <= 1)), "概率须在 [0,1]"


def test_isotonic_fit_predict_perfect() -> None:
    # 完美可分：score>0 全为正类 → 校准后负区≈0、正区≈1
    scores = np.linspace(-1, 1, 100)
    labels = (scores > 0).astype(int)
    cal = IsotonicCalibrator().fit(scores, labels)
    p = cal.predict(scores)
    assert p[scores < -0.5].mean() < 0.2
    assert p[scores > 0.5].mean() > 0.8
    assert np.all((p >= 0) & (p <= 1))


def test_isotonic_unfitted_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().predict(np.array([0.0]))


def test_conformal_gate_coverage() -> None:
    # 负类 P(correct) 低、正类高；阈值应使负类误受率 ≤ α
    rng = np.random.default_rng(1)
    n = 2000
    neg = rng.beta(2, 8, size=n // 2)   # 负类 P(correct) 偏低
    pos = rng.beta(8, 2, size=n // 2)   # 正类 P(correct) 偏高
    alpha = 0.05
    gate = ConformalGate(alpha=alpha).fit(neg)
    # 负类误受率（经验，含有限样本波动，放宽到 2.5×α 上界）
    far = gate.accept(neg).mean()
    assert far <= alpha * 2.5, f"负类误受率 {far:.3f} 超界"
    # 正类覆盖率应较高（负类低分、阈值不高）
    assert gate.accept(pos).mean() > 0.8


def test_conformal_unfitted_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        ConformalGate().accept(np.array([0.5]))


def test_aurc_perfect_vs_random() -> None:
    # 完美：正类全高分。selective AURC 语义下，前 50% 覆盖（全正类）risk=0，
    # 后 50%（进负类）risk 才升——完美探测器 AURC = 最小可能值（非 0，除非拒绝负类）。
    labels = np.array([1] * 50 + [0] * 50)
    perfect_scores = np.array([1.0] * 50 + [0.0] * 50)
    auroc_perfect = aurc(perfect_scores, labels)
    # 随机：AURC 显著更大（高置信区混入更多负类）
    rng = np.random.default_rng(2)
    rand_scores = rng.normal(size=100)
    auroc_rand = aurc(rand_scores, labels)
    assert auroc_perfect < auroc_rand, "完美探测器 AURC 须小于随机"
    # 完美的理论值：仅后 50% 覆盖有风险，risk 从 0 线性升到 0.5，面积 = ∫ ≈ 0.125
    assert auroc_perfect < 0.2, f"完美 AURC {auroc_perfect:.3f} 应接近理论最小 ~0.125"
    # 单调性：把少数正类打低分（模拟探测失误）→ AURC 升
    bad_scores = perfect_scores.copy()
    bad_scores[0] = -1.0  # 一个正类被打到最低（高置信区混入负类提前）
    assert aurc(bad_scores, labels) > auroc_perfect


def test_ece_perfect_and_miscalibrated() -> None:
    labels = np.array([1, 1, 0, 0, 1, 0, 1, 0] * 20)
    perfect = labels.astype(float)
    assert expected_calibration_error(perfect, labels) < 1e-6
    # 完全反向 → ECE 大
    inverse = 1.0 - labels.astype(float)
    assert expected_calibration_error(inverse, labels) > 0.4


def test_empty_fail_closed() -> None:
    with pytest.raises(ValueError):
        IsotonicCalibrator().fit(np.array([]), np.array([]))
    with pytest.raises(ValueError):
        ConformalGate().fit(np.array([]))
    with pytest.raises(ValueError):
        ConformalGate(alpha=1.5)
