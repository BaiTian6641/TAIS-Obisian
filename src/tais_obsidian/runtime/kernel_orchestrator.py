"""内核 ↔ 运行时编排层（sense → route → inject 端到端闭环）。

设计依据（逐条对齐）：
- 接口与实现计划 v1.0 §1：主干一次前向内调 TAIS 内核读 PM-stream，内核经 Bus 调运行时
  取块、回填注入——本类是该桥的**编排者**：把 M1 内核（tais_kernel）的 sense/route/inject
  与 M4 运行时（MemoryBus/Pager/PageTable/BlockStore）+ M5 注入器（injection.Injector）
  + KAL 校准层（kal_calibrate）串成完整回路。
- KAL 数学规范（article_ref/07 §2/§5）：L1 P(IK) 经 isotonic 校准 → conformal 拒答门——
  检测到**知识空白**时**诚实降级**（声明"记忆暂不可用"/触发回想），而非盲目 route 注入；
  非空白才进 route 取候选块。这是"空白检测→回想/拒答"的 FLARE/MeCo 式闭环。
- 红线落实：
  * 监测/执行分置：sense 只读（GDN 层 PM-stream），inject 写（CSA 残差前 PM-stream）——不同层；
  * 梯度隔离：route/inject 全程 detach 主干（MoE-RL 红线，内核 detach_input 默认开）；
  * fail-closed：namespace 不匹配 / 缺页 / 校准未 fit → 拒答或丢弃，绝不静默注入；
  * 诚实降级：KAL 判空白 → 返回 RecallDecision(should_recall=True)，由调用方走回想/文本 RAG 回退。

数据流（一次前向一个读点）：
  pm_out ──sense──▶ SenseOut(KAL logits)
     └─calibrate──▶ p_correct ──conformal──▶ 空白? ─是─▶ RecallDecision(recall, 诚实降级)
     │空白? 否
     └─route──▶ indexer 对候选块打分 ──Bus──▶ top-k 块 ──Pager fail-closed──▶ payloads
     └─inject──▶ pm_pre + Σ α·payload（injector 按载体路由）

纯编排，不新增可学习参数；KAL/indexer 权重来自内核（已训/已 warmup），校准器须先 fit。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..model.tais_kernel import BlockPayload, TAISKernel
from .bus import MemoryBus


@dataclass
class RecallDecision:
    """一次 sense 的判定结果（诚实降级信号）。"""

    p_correct: float              # KAL L1 校准后的"知道"概率（末 token）
    is_blank: bool                # conformal 门判定为知识空白
    should_recall: bool           # 是否应触发回想/拒答（= is_blank）
    message: str = ""             # 诚实降级文案（空白时非空）


@dataclass
class OrchestrateOut:
    """编排一次闭环的输出。"""

    decision: RecallDecision
    injected_pm: torch.Tensor | None = None   # 注入后的 pm_pre（空白/无载荷时为 None）
    n_injected: int = 0                        # 实际注入的载荷数
    n_page_faults: int = 0                     # 本次取块的缺页数（fail-closed 丢弃数）
    routed_block_ids: list[str] = field(default_factory=list)


class KernelOrchestrator:
    """内核 ↔ 运行时编排者：sense → (空白门) → route → inject。

    - ``kernel``：TAISKernel（KAL/indexer 已训；sense 只读、inject 写，监测/执行分置）。
    - ``bus``：MemoryBus（PageTable+BlockStore+Pager，fail-closed 取载荷）。
    - ``injector``：injection.Injector（M5 注入器，按载体路由 KV/mem/vector）。
    - ``calibrator/gate``：kal_calibrate 的 IsotonicCalibrator + ConformalGate（须先 fit）。
    """

    def __init__(
        self,
        kernel: TAISKernel,
        bus: MemoryBus,
        injector=None,
        calibrator=None,
        gate=None,
        blank_message: str = "该部分记忆暂不可用（KAL 检测到知识空白，诚实降级）",
    ):
        self.kernel = kernel
        self.bus = bus
        self.injector = injector
        self.calibrator = calibrator
        self.gate = gate
        self.blank_message = blank_message

    # ------------------------------------------------------------------
    def sense_gate(self, pm_out: torch.Tensor) -> RecallDecision:
        """sense + 校准 + conformal 空白门（只读，零副作用）。

        pm_out [B,T,d]（GDN 层输出 PM-stream）。取末 token 的 KAL L1 logits →
        校准概率 → conformal 判定。未挂校准器/门时退化为裸 logit（标注未校准）。
        """
        # sense 只读（监测）：detach 防梯度泄漏（红线：sense 不该携带/泄漏 autograd 图）
        with torch.no_grad():
            sense = self.kernel.sense(pm_out)
        logits = sense.pik_logits.detach()  # [B,T,3]
        # 末 token 的"知道 vs 空白"对数几率（与 train/eval 口径一致）
        last = logits[:, -1, :].float()  # [B,3]
        raw_score = (last[:, 0] - last[:, 2]).cpu().numpy()  # [B]
        if self.calibrator is not None and self.gate is not None:
            import numpy as np

            p_corr = self.calibrator.predict(raw_score)
            accept = self.gate.accept(np.asarray(p_corr))
            p0 = float(p_corr[0])
            is_blank = not bool(accept[0])
        else:
            # 未校准退化：裸 score<0 视为空白（标注未校准，部署应挂校准层）
            p0 = float(raw_score[0])
            is_blank = bool(raw_score[0] < 0.0)
        return RecallDecision(
            p_correct=p0,
            is_blank=is_blank,
            should_recall=is_blank,
            message=self.blank_message if is_blank else "",
        )

    # ------------------------------------------------------------------
    def route_blocks(
        self,
        query: torch.Tensor,
        candidate_vecs: torch.Tensor,
        candidate_ids: list[str],
        k: int,
        namespace,
    ) -> tuple[list[str], list]:
        """route + Bus 取块（fail-closed）。

        query [B,Tq,d]（当前思考段），candidate_vecs [B,Tk,d]（候选块表示）。
        indexer 打分 → 压平取每候选最高分 → Bus top-k → Pager fail-closed 取载荷。
        返回 (top_block_ids, payloads)。载荷缺页/namespace 不匹配项已被丢弃。
        """
        if k <= 0 or not candidate_ids:
            return [], []
        # indexer 对候选集合打分 [B,Tq,Tk] → 对 query 维取 max 作该候选相关分
        scores = self.kernel.route_candidates(query, candidate_vecs, k=None, detach_input=True)
        cand_score = scores[0].max(dim=0).values  # [Tk]
        # 与候选数对齐（Tk 可能 ≠ len(candidate_ids)，取小者）
        n = min(len(candidate_ids), cand_score.shape[0])
        score_list = cand_score[:n].detach().cpu().tolist()
        top_ids = self.bus.route_to_blocks(score_list, candidate_ids[:n], k)
        payloads = self.bus.fetch_payloads(top_ids, namespace)
        return top_ids, payloads

    # ------------------------------------------------------------------
    def orchestrate(
        self,
        pm_out: torch.Tensor,
        pm_pre: torch.Tensor,
        query: torch.Tensor | None = None,
        candidate_vecs: torch.Tensor | None = None,
        candidate_ids: list[str] | None = None,
        k: int = 4,
        namespace=None,
        alphas: list[float] | None = None,
    ) -> OrchestrateOut:
        """端到端闭环：sense 空白门 →（非空白）route 取块 → inject。

        - 空白（should_recall）：不 route 不 inject，返回 decision（诚实降级）。
        - 非空白：route 取候选块载荷 → kernel.inject 写 pm_pre。
        缺 query/candidates 时跳过 route（仅 sense 门 + 可选直接注入空载荷）。
        """
        decision = self.sense_gate(pm_out)
        if decision.should_recall:
            return OrchestrateOut(decision=decision, injected_pm=None, n_injected=0)

        payloads: list = []
        top_ids: list[str] = []
        n_faults_before = self.bus.pager.page_faults if hasattr(self.bus, "pager") else 0
        if query is not None and candidate_vecs is not None and candidate_ids:
            top_ids, payloads = self.route_blocks(
                query, candidate_vecs, candidate_ids, k, namespace)
        n_faults = (self.bus.pager.page_faults - n_faults_before) if hasattr(self.bus, "pager") else 0

        if not payloads:
            return OrchestrateOut(decision=decision, injected_pm=None, n_injected=0,
                                  n_page_faults=n_faults, routed_block_ids=top_ids)
        # 设备/dtype 对齐（防御加固）：BlockStore 载荷可能在 CPU，pm_pre 在 CUDA——
        # 注入前把 vector/entries 对齐到 pm_pre 设备与 dtype，避免 RuntimeError。
        payloads = [self._align_payload(p, pm_pre) for p in payloads]
        injected = self.kernel.inject(pm_pre, payloads, alphas=alphas, injector=self.injector)
        return OrchestrateOut(decision=decision, injected_pm=injected,
                              n_injected=len(payloads), n_page_faults=n_faults,
                              routed_block_ids=top_ids)

    # ------------------------------------------------------------------
    @staticmethod
    def _align_payload(payload: BlockPayload, ref: torch.Tensor) -> BlockPayload:
        """把载荷的 vector/entries 对齐到参考张量 ref 的 device/dtype（注入前防御）。"""
        vec = payload.vector
        if vec is not None and (vec.device != ref.device or vec.dtype != ref.dtype):
            vec = vec.to(device=ref.device, dtype=ref.dtype)
        entries = payload.entries
        if entries is not None:
            entries = tuple(
                e.to(device=ref.device, dtype=ref.dtype) if isinstance(e, torch.Tensor) else e
                for e in entries)
        if vec is payload.vector and entries is payload.entries:
            return payload
        return BlockPayload(
            block_id=payload.block_id, compiled_kind=payload.compiled_kind,
            vector=vec, entries=entries, layer_ns=payload.layer_ns,
            signature=payload.signature)


def make_orchestrator(kernel, bus, injector=None, calibrator=None, gate=None, **kw) -> KernelOrchestrator:
    """工厂函数。"""
    return KernelOrchestrator(kernel, bus, injector=injector, calibrator=calibrator, gate=gate, **kw)
