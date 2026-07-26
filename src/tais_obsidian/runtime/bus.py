"""TAIS Memory Bus（M1 内核 ↔ 运行时桥，🟡 运行时服务）。

设计依据：
- 接口与实现计划 v1.0 §1：``model/`` 与 ``runtime/`` 经 **TAIS Memory Bus** 通信——
  主干一次前向内调 TAIS 内核读 PM-stream，内核经 Bus 调运行时取块、回填注入。
- 本类为薄封装：持 PageTable + BlockStore + Pager 引用，面向纯 python 数据（list/dict），
  依赖极轻，不引入 autograd（梯度隔离红线）。
"""
from __future__ import annotations


class MemoryBus:
    """内存总线：路由打分 → 候选块 ID → fail-closed 取载荷。

    - ``route_to_blocks``：按 route_scores 取 top-k 候选 block_id（M1 route() 输出喂这里）。
    - ``fetch_payloads``：对候选 ID 逐个经 Pager fail-closed 取载荷（namespace 校验）。
    """

    def __init__(self, pagetable, blockstore, pager):
        self.pagetable = pagetable
        self.blockstore = blockstore
        self.pager = pager

    def route_to_blocks(self, route_scores: list[float], keys: list[str], k: int) -> list[str]:
        """按打分取 top-k 候选 block_id（降序）。分数与 key 等长；k<=0 返回空。"""
        if k <= 0 or not keys:
            return []
        order = sorted(range(len(keys)), key=lambda i: route_scores[i], reverse=True)
        return [keys[i] for i in order[:k]]

    def fetch_payloads(self, block_ids: list[str], namespace) -> list:
        """对候选 ID 逐个 fail-closed 取载荷；缺页/namespace 不匹配项被丢弃。"""
        out = []
        for bid in block_ids:
            p = self.pager.fetch(bid, namespace)
            if p is not None:
                out.append(p)
        return out
