"""KAL L1 校准层（isotonic regression + conformal 拒答阈值 + AURC）。

设计依据（article_ref/07_kal_math_engineering_spec.md §2，逐条已核实）：
- **裸 logit 阈值不可审计**——内生头输出的是分数非概率，直接设阈值无覆盖保证。
- **isotonic regression**（Kossen 2406.15927，SEP）：非参数单调映射 score→P(correct)，
  OOD 下比 Platt/temperature 稳（探针常过自信）。纯 NumPy PAV（Pool Adjacent Violators）。
- **conformal quantile 拒答阈值**（Mohri & Hashimoto 2402.10978）：用校准集分位数定阈值，
  给**有限样本覆盖保证** P(accept 且错误) ≤ α——"诚实降级 / 记忆暂不可用"红线的数学基础。
- **AURC**（Su 2603.21172）：risk-coverage 曲线下面积，比单纯 AUROC 更贴合部署
  （探针价值=在高覆盖下保持低风险）。

红线：本模块只做**校准/阈值**，不改探针权重（探针冻结、只读 hidden state——
监测/执行分置 + 防 Goodhart：模型不可通过校准层反向影响自身表征）。
纯 NumPy/torch，零新依赖（对齐项目"纯 PyTorch、Windows 原生"纪律，不引 sklearn/scipy）。
"""
from __future__ import annotations

import numpy as np


def _pav_isotonic(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators 拟合单调非降 isotonic 映射 score→P(label=1)。

    标准 PAV：按 score 升序后逐点入栈，每入栈一个块，若栈顶块均值 > 新块均值
    （违反单调非降）则向前加权合并，直至栈内单调。返回 (thresholds, values)：
    各块的右端 score 节点与块均值，供 np.interp 线性插值查表。
    labels: 1=正类（known/correct），0=负类。score 越大应越倾向正类。
    """
    order = np.argsort(scores, kind="stable")
    s = scores[order].astype(np.float64)
    y = labels[order].astype(np.float64)
    n = len(s)
    # 栈：每块记录（块加权和, 块权重, 块右端 score）
    blk_sum: list[float] = []
    blk_w: list[float] = []
    blk_end: list[float] = []
    for j in range(n):
        cur_sum = y[j]
        cur_w = 1.0
        cur_end = s[j]
        # 向前合并所有违反单调性的栈顶块（栈顶均值 > 当前均值）
        while blk_sum and (blk_sum[-1] / blk_w[-1]) > (cur_sum / cur_w):
            cur_sum += blk_sum.pop()
            cur_w += blk_w.pop()
            blk_end.pop()  # 当前块右端保持为 s[j]（合并后块的右端）
        blk_sum.append(cur_sum)
        blk_w.append(cur_w)
        blk_end.append(cur_end)
    thresholds = np.array(blk_end, dtype=np.float64)
    values = np.array([blk_sum[i] / blk_w[i] for i in range(len(blk_sum))], dtype=np.float64)
    return thresholds, values


class IsotonicCalibrator:
    """score→P(correct) 的 isotonic 校准器（保序回归，非参数单调）。

    fit(scores, labels)：labels 1=correct/known；predict(scores)→P(correct)。
    """

    def __init__(self) -> None:
        self._thr: np.ndarray | None = None
        self._val: np.ndarray | None = None

    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "IsotonicCalibrator":
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        if scores.size == 0:
            raise ValueError("空校准集（fail-closed）")
        self._thr, self._val = _pav_isotonic(scores, labels)
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._thr is None:
            raise RuntimeError("calibrator 未 fit（fail-closed）")
        scores = np.asarray(scores, dtype=np.float64)
        # 分段线性插值（边界外取端点值）
        return np.interp(scores, self._thr, self._val)


class ConformalGate:
    """conformal 拒答门：给定目标误受率 α，用校准集负类分数分位数定阈值。

    语义：score ≥ threshold → "知道"（accept），否则 → "空白/回想"（reject）。
    保证（分布交换性假设下）：P(accept 且实际错误) ≤ α（有限样本边际覆盖）。
    """

    def __init__(self, alpha: float = 0.05):
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha 须在 (0,1)")
        self.alpha = alpha
        self.threshold_: float | None = None

    def fit(self, neg_scores: np.ndarray) -> "ConformalGate":
        """用校准集**负类**（实际为 unknown/fake）的校准后 P(correct) 定阈值。

        neg_scores：已知为负类的样本经 isotonic 校准后的 P(correct)。
        阈值 = 负类 P(correct) 的 ceil((n+1)(1-α)/n) 分位数 → 误受率 ≤ α。
        """
        neg_scores = np.asarray(neg_scores, dtype=np.float64)
        if neg_scores.size == 0:
            raise ValueError("空负类校准集（fail-closed）")
        n = neg_scores.size
        # conformal 分位数级别（有限样本修正）
        level = min(1.0, np.ceil((n + 1) * (1.0 - self.alpha)) / n)
        self.threshold_ = float(np.quantile(neg_scores, level, method="higher"))
        return self

    def accept(self, p_correct: np.ndarray) -> np.ndarray:
        """p_correct ≥ threshold → True(accept/知道)，否则 False(reject/空白)。"""
        if self.threshold_ is None:
            raise RuntimeError("gate 未 fit（fail-closed）")
        return np.asarray(p_correct) >= self.threshold_


def aurc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Risk-Coverage 曲线下面积（AURC，越低越好；selective prediction 指标）。

    语义（Geifman & El-Yaniv selective prediction）：探测器对每个样本给置信度 score
    与预测（此处统一"预测为正类"）。按置信度**降序**逐步扩大"接受"覆盖：
    coverage_k = k/n，risk_k = 前 k 个最自信样本中**预测错误**（实际为负类）的比例。
    AURC = risk 对 coverage 的曲线面积（梯形积分）。
    完美探测器（正类全高分、负类全低分）：前 n_pos 全对 risk=0，AURC→0；
    随机探测器 AURC ≈ 负类比例 × 0.5 量级。labels: 1=正类（accept 正确），0=负类。
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(scores)
    if n == 0:
        return float("nan")
    order = np.argsort(-scores, kind="stable")  # 置信度降序
    y = labels[order]
    errors = (y == 0).astype(np.float64)  # 预测为正类时，实际负类=错误
    cum_err = np.cumsum(errors)
    ks = np.arange(1, n + 1)
    risks = cum_err / ks          # 前 k 中的错误率
    coverages = ks / n
    # 梯形积分（起点 coverage=0, risk=0）
    coverages = np.concatenate([[0.0], coverages])
    risks = np.concatenate([[0.0], risks])
    return float(np.trapezoid(risks, coverages))


def expected_calibration_error(p_correct: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """ECE（等频分箱）：校准后概率与实际正确率的平均绝对偏差。"""
    p_correct = np.asarray(p_correct, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(p_correct)
    if n == 0:
        return float("nan")
    order = np.argsort(p_correct, kind="stable")
    bins = np.array_split(order, n_bins)
    ece = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf = p_correct[b].mean()
        acc = labels[b].mean()
        ece += (len(b) / n) * abs(acc - conf)
    return float(ece)
