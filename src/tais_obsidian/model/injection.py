"""注入闭环（M5）：统一注入器——KV 拼接（blockpath）+ 记忆层查询 + 向量加法。

设计依据（必须逐条对齐）：
- 接口与实现计划 v1.0 §5.1 / 部件实现详细计划 Part D：注入点接受 KV/记忆层/向量
  三载荷；**载体能力边界**——token 寻址载体（kv/mem_entry/gist）能事实召回，位置不变
  向量（icv/steering/concept_slot）只能 steer 行为。
- M1 骨架纪律：token 寻址载体在 `TAISKernel.inject()` 里 fail-closed 拒绝，由本模块
  （M5）接通——KV/gist 走 blockpath 的 namespace 校验 + tri_attention.inject_hca_entries
  （HCA 区前置拼接，设计 §17.3 块注入原生落点）；mem_entry 走 memlayer.MemoryLayer 查询。
- 设计 §11.1/§17.3：注入即"读自己写的东西"（HCA 区）消除前缀偏差；namespace 五元组
  fail-closed（任一字段不匹配即拒注，走重算/文本 RAG 回退）。
- 监测/执行分置：注入写 CSA 残差前 PM-stream（紧邻检索层），探针读 GDN 输出层——不同层。

纪律：
- 本模块只负责"把载荷按载体类型路由到正确注入路径"；namespace 校验复用 blockpath
  check_namespace（fail-closed，NamespaceMismatchError 由调用方捕获走回退）。
- 纯 PyTorch，Windows 原生，秒级 CPU 测试。
"""
from __future__ import annotations

import torch

from .blockpath import NamespaceMismatchError, check_namespace
from .memlayer import MemoryLayer
from .tais_kernel import ADDRESSED_KINDS, VECTOR_KINDS, BlockPayload

# KV/gist 载体（走 blockpath → tri_attention HCA 拼接）
_KV_KINDS: frozenset = frozenset({"kv", "gist"})
# 记忆层载体（走 memlayer 查询）
_MEM_KINDS: frozenset = frozenset({"mem_entry"})


class Injector:
    """统一注入器：把 BlockPayload 按载体类型路由到对应注入路径。

    - KV/gist：namespace 校验 → 返回待 HCA 拼接的 (k,v) 条目（实际拼入由
      tri_attention.inject_hca_entries 完成，调用方持有 mixer state）；
    - mem_entry：写入/查询 memlayer（delta 规则，分布内）；
    - 向量（icv/steering/concept_slot）：PM-stream 单次加法（steer 行为，一次加法）。
    """

    def __init__(self, memory_layer: MemoryLayer | None = None):
        self.memory_layer = memory_layer

    # ------------------------------------------------------------------
    def inject(self, payload: BlockPayload, namespace: dict | None = None):
        """按载体类型路由注入。返回注入结果（类型依载体而异）或 None。

        fail-closed：未知载体 → None（不静默注入）；namespace 不匹配 →
        抛 NamespaceMismatchError 由调用方走回退（重算/文本 RAG）。
        """
        kind = payload.compiled_kind
        if kind in _KV_KINDS:
            return self._inject_kv(payload, namespace)
        if kind in _MEM_KINDS:
            return self._inject_mem(payload)
        if kind in VECTOR_KINDS:
            return self._inject_vector(payload)
        return None  # 未知载体 fail-closed

    # ------------------------------------------------------------------
    def _inject_kv(self, payload: BlockPayload, namespace: dict | None):
        """KV/gist：namespace 校验后返回待 HCA 拼接的 (k,v) 条目。"""
        if payload.entries is None:
            raise ValueError(f"KV 载体 {payload.compiled_kind} 缺 entries (k,v) 载荷")
        if namespace is not None and payload.layer_ns:
            # 五元组校验（fail-closed；不匹配抛 NamespaceMismatchError 走回退）
            check_namespace(payload.layer_ns if isinstance(payload.layer_ns, dict)
                            else _ns_tuple_to_dict(payload.layer_ns), namespace)
        return payload.entries  # (k,v)，交由 tri_attention.inject_hca_entries 拼入 HCA 区

    def _inject_mem(self, payload: BlockPayload):
        """mem_entry：delta 写入记忆层（分布内）或按键查询读出。"""
        if self.memory_layer is None:
            raise RuntimeError("未挂 MemoryLayer，无法注入 mem_entry")
        if payload.entries is not None:
            k, v = payload.entries  # delta 写入
            self.memory_layer.write(k, v)
            return True
        if payload.vector is not None:
            return self.memory_layer.query(payload.vector)  # 查询读出
        raise ValueError("mem_entry 需 entries(k,v) 写入或 vector(k) 查询")

    def _inject_vector(self, payload: BlockPayload):
        """向量（icv/steering/concept_slot）：返回单次加法载荷（steer 行为）。"""
        if payload.vector is None:
            raise ValueError(f"向量载体 {payload.compiled_kind} 缺 vector 载荷")
        return payload.vector  # 调用方做 pm_pre + alpha·vector（一次加法，零上下文开支）


def _ns_tuple_to_dict(ns: tuple) -> dict:
    """namespace 五元组（tuple）→ dict（按 blockpath 字段序）。"""
    fields = ("model_id", "layer_idx", "compressor_version", "dtype", "rope_theta")
    if len(ns) != len(fields):
        raise NamespaceMismatchError(f"namespace 元组长度 {len(ns)} ≠ 5")
    return dict(zip(fields, ns))


def make_injector(memory_layer: MemoryLayer | None = None) -> Injector:
    """工厂函数。"""
    return Injector(memory_layer)
