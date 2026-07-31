"""TAIS Obsidian 运行时服务包（M4 运行时骨架）。

设计依据（接口与实现计划 v1.0 §1 包结构；部件实现详细计划 Part C）：
- ``runtime/`` 只放 **数据/算法/IO**（🟡 运行时服务，非权重），与 ``model/`` 包
  （前向可微、随 state_dict 存取）严格分离；两者经 TAIS Memory Bus 通信。
- 辅助损失梯度隔离红线：本包无任何 autograd，不触碰 ``model/``，纯数据/算法。

子模块：
- pagetable  页表 Block Spec（SQLite 元数据，Part C3）
- blockstore 块载荷 L0/L1/L2 分页存储（usage_weighted 淘汰，非 LRU）
- pager      缺页处理 + namespace 五元组校验 + fail-closed 回退
- bus        TAIS Memory Bus（M1 内核调用桥）
- ca3_ppr    CA3 PPR 联想检索（HippoRAG 式 Personalized PageRank）
- ca1_gate   CA1 巩固门（验证门 + 信念漂移监测，规则骨架）
- state_ckpt GDN 状态 save/restore（🔧 关键工程缺口，自研）
"""
from .blockstore import BlockStore
from .bus import MemoryBus
from .ca1_gate import (
    RE_VERIFY,
    CA1Gate,
    EvidenceWeights,
    SourceCredibilityTracker,
    ca1_gate,
    evidence_aware_consensus,
)
from .ca3_ppr import ca3_ppr
from .kernel_orchestrator import KernelOrchestrator, OrchestrateOut, RecallDecision, make_orchestrator
from .pagetable import KNOWN_KINDS, BlockSpec, PageTable
from .pager import Pager
from .safety import SafetyPipeline, make_safety_pipeline, sign_block, verify_signature
from .state_ckpt import restore_state, save_state, states_equal

__all__ = [
    "BlockSpec",
    "KNOWN_KINDS",
    "PageTable",
    "BlockStore",
    "Pager",
    "MemoryBus",
    "KernelOrchestrator",
    "OrchestrateOut",
    "RecallDecision",
    "make_orchestrator",
    "CA1Gate",
    "ca1_gate",
    "RE_VERIFY",
    "EvidenceWeights",
    "evidence_aware_consensus",
    "SourceCredibilityTracker",
    "ca3_ppr",
    "save_state",
    "restore_state",
    "states_equal",
    "SafetyPipeline",
    "make_safety_pipeline",
    "sign_block",
    "verify_signature",
]
