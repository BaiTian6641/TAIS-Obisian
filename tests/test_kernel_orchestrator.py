"""内核 ↔ 运行时编排层单元测试（sense→route→inject 闭环，CPU）。

判据（kernel_orchestrator.py docstring / 接口计划 §1 / KAL 规范 §2）：
- sense 空白门：KAL 判空白 → should_recall=True + 诚实降级 message + 不 route 不 inject；
- 非空白：route 取 top-k 块 → inject 写 pm_pre（vector 加法）；
- fail-closed：缺页/namespace 不匹配的载荷被丢弃（n_page_faults 自增）；
- 校准门接入：挂 calibrator+gate 时按校准概率判定；
- 梯度隔离：route/inject detach 主干（MoE-RL 红线）。
"""
from __future__ import annotations

import numpy as np
import torch

from tais_obsidian.model.kal_calibrate import ConformalGate, IsotonicCalibrator
from tais_obsidian.model.tais_kernel import BlockPayload, TAISKernel
from tais_obsidian.runtime import (
    BlockSpec,
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    make_orchestrator,
)

D = 32
NS = ("m1", 0, 1, "bf16", 10000.0)


def _kernel_with_pik(know_logit: float, blank_logit: float) -> TAISKernel:
    """构造内核并把 KAL L1 头 bias 调到指定 logits（控制 sense 输出）。

    TAISKernel 直接持 kal_l1（KALHead(d,3)，proj=Linear(d,3)）；把 weight 置零、
    bias 设为指定 logits，使 sense() 对任意输入都输出固定 P(IK) logits。
    """
    k = TAISKernel(D)
    head = k.kal_l1
    with torch.no_grad():
        head.proj.weight.zero_()
        head.proj.bias.zero_()
        head.proj.bias[0] = know_logit    # 知道
        head.proj.bias[2] = blank_logit   # 空白
    return k


def _bus_with_blocks(vectors: dict[str, torch.Tensor], ns=NS) -> MemoryBus:
    pt = PageTable()
    bs = BlockStore()
    for bid, vec in vectors.items():
        pt.register(BlockSpec(block_id=bid, route_key=bid, namespace=ns,
                              compiled_kind="steering", factual_recall=False))
        bs.put(bid, BlockPayload(block_id=bid, compiled_kind="steering", vector=vec, layer_ns=ns))
    pager = Pager(bs, pt)
    return MemoryBus(pt, bs, pager)


def test_blank_gate_triggers_recall_no_inject() -> None:
    # KAL 判空白（blank_logit 高）→ 应 recall + 诚实降级 + 不注入
    k = _kernel_with_pik(know_logit=-2.0, blank_logit=2.0)
    bus = _bus_with_blocks({"b1": torch.ones(D)})
    orch = make_orchestrator(k, bus)
    pm = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm.clone())
    assert out.decision.should_recall is True
    assert out.decision.is_blank is True
    assert out.decision.message != "", "空白须给诚实降级文案"
    assert out.injected_pm is None and out.n_injected == 0


def test_nonblank_route_and_inject_vector() -> None:
    # KAL 判知道（know_logit 高）→ route 取块 + vector 注入 pm_pre
    k = _kernel_with_pik(know_logit=3.0, blank_logit=-3.0)
    vec = torch.ones(D)
    bus = _bus_with_blocks({"b1": vec, "b2": torch.zeros(D)})
    orch = make_orchestrator(k, bus)
    pm_out = torch.zeros(1, 4, D)
    pm_pre = torch.zeros(1, 4, D)
    query = torch.zeros(1, 2, D)
    cand_vecs = torch.zeros(1, 2, D)
    out = orch.orchestrate(pm_out, pm_pre, query=query, candidate_vecs=cand_vecs,
                           candidate_ids=["b1", "b2"], k=1, namespace=NS)
    assert out.decision.should_recall is False
    assert out.n_injected == 1
    assert out.injected_pm is not None
    # vector 注入 = pm_pre + α·vec（单次加法）；注入后应非全零
    assert out.injected_pm.abs().sum() > 0
    assert out.routed_block_ids, "应有路由到的块 ID"


def test_page_fault_fail_closed() -> None:
    # 候选含未注册块 → Pager 缺页 fail-closed 丢弃（n_page_faults>0）
    k = _kernel_with_pik(know_logit=3.0, blank_logit=-3.0)
    bus = _bus_with_blocks({"b1": torch.ones(D)})
    orch = make_orchestrator(k, bus)
    pm = torch.zeros(1, 4, D)
    query = torch.zeros(1, 2, D)
    cand = torch.zeros(1, 2, D)
    # "ghost" 未注册 → 取载荷时缺页
    top_ids, payloads = orch.route_blocks(query, cand, ["b1", "ghost"], k=2, namespace=NS)
    assert "ghost" in top_ids or "b1" in top_ids
    assert all(p.block_id != "ghost" for p in payloads), "缺页块须被丢弃"
    assert bus.pager.page_faults >= 1


def test_calibrated_gate() -> None:
    # 挂校准器+conformal 门：用校准概率判定
    k = _kernel_with_pik(know_logit=1.0, blank_logit=-1.0)  # 裸 score=2
    bus = _bus_with_blocks({"b1": torch.ones(D)})
    # 校准器：score 越高 P(correct) 越高；门：负类（低分）阈值
    scores = np.linspace(-3, 3, 100)
    labels = (scores > 0).astype(int)
    cal = IsotonicCalibrator().fit(scores, labels)
    gate = ConformalGate(alpha=0.1).fit(cal.predict(scores[labels == 0]))
    orch = make_orchestrator(k, bus, calibrator=cal, gate=gate)
    dec = orch.sense_gate(torch.zeros(1, 4, D))
    # score=2>0 → 校准 P 高 → accept（非空白）
    assert dec.is_blank is False
    assert 0.0 <= dec.p_correct <= 1.0


def test_no_candidates_skips_route() -> None:
    # 非空白但无候选 → 仅 sense 门，不注入
    k = _kernel_with_pik(know_logit=3.0, blank_logit=-3.0)
    bus = _bus_with_blocks({"b1": torch.ones(D)})
    orch = make_orchestrator(k, bus)
    pm = torch.zeros(1, 4, D)
    out = orch.orchestrate(pm, pm.clone())
    assert out.decision.should_recall is False
    assert out.n_injected == 0 and out.injected_pm is None
