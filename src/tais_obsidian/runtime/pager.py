"""缺页处理 + namespace 校验 + fail-closed 回退（🟡 运行时逻辑）。

设计依据：
- 接口与实现计划 v1.0 §4：Pager 缺页处理 + namespace 校验（模型/层/压缩矩阵版本/
  dtype/RoPE 五元组）+ **fail-closed** 回退（重算/文本 RAG）。
- 设计文档"诚实降级"红线：缺页/namespace 不匹配时明确返回 None（该部分记忆暂不可用），
  **绝不静默注入**，也绝不把异常抛给调用方。
"""
from __future__ import annotations

_NS_FIELDS = ("model_id", "layer_idx", "compressor_version", "dtype", "rope_theta")


def namespace_ok(expected, given) -> bool:
    """namespace 五元组逐字段比对。接受 dict 或定长五元组（按 _NS_FIELDS 序）。

    任一字段不匹配 → False（fail-closed）。宽容输入、严格判定。
    """
    exp = _as_dict(expected)
    got = _as_dict(given)
    if exp is None or got is None:
        return False
    return all(exp.get(f) == got.get(f) for f in _NS_FIELDS)


def _as_dict(ns) -> dict | None:
    if isinstance(ns, dict):
        return ns
    if isinstance(ns, (tuple, list)) and len(ns) == len(_NS_FIELDS):
        return dict(zip(_NS_FIELDS, ns))
    return None


class Pager:
    """缺页处理器：包一层 BlockStore + PageTable，做 namespace 校验与用量记账。

    fail-closed：namespace 不匹配 → 返回 None、``page_faults`` 自增、**不抛异常**、
    绝不注入。命中则返回载荷并经 PageTable 更新 usage_count。
    """

    def __init__(self, blockstore, pagetable=None):
        self._bs = blockstore
        self._pt = pagetable
        self.page_faults: int = 0

    def fetch(self, block_id: str, namespace):
        """按 block_id + namespace 取载荷。

        返回载荷；namespace 不匹配或缺页 → None（fail-closed，page_faults 自增）。
        namespace 期望取自页表 BlockSpec（若挂 PageTable 且有记录），否则以传入
        namespace 自证（骨架简化；M5 起 namespace 必须来自页表/载荷头，不可来自调用方）。
        """
        spec = self._pt.get(block_id) if self._pt is not None else None
        payload = self._bs.get(block_id)
        if payload is None:
            self.page_faults += 1
            return None
        # 校验：页表存有 namespace 时以其为准；否则要求调用方给定与载荷侧一致（骨架用传入值）
        expected = spec.namespace if (spec is not None and spec.namespace) else namespace
        if not namespace_ok(expected, namespace):
            self.page_faults += 1
            return None
        if self._pt is not None:
            self._pt.update_usage(block_id)
        return payload
