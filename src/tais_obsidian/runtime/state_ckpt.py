"""GDN 状态 save/restore（🔧 关键工程缺口，自研）。

设计依据：
- 接口与实现计划 v1.0 §4 / 部件实现详细计划 M4：**state_ckpt 自研 GDN 状态持久化**。
  现有引擎（llama.cpp）的 slot API **不保存 SSM/DeltaNet 状态**（discussion #24043），
  W-State（运行时读写 GDN 循环状态）路径在主流引擎是空白——本模块为自研补缺口。
- GDN 层无 KV cache，其"状态"为循环 recurrence state（张量字典），须能字节级往返。

功能：``save_state``/``restore_state`` 用 torch.save/torch.load 到 BytesIO 做字节往返；
``states_equal`` 做容差比对（M4 退出标准：state 往返 < 1e-5）。
"""
from __future__ import annotations

from io import BytesIO

import torch


def save_state(state: dict) -> bytes:
    """把 {name: Tensor} 的 GDN 循环状态序列化为 bytes（torch.save → BytesIO）。"""
    buf = BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def restore_state(payload: bytes) -> dict:
    """从 bytes 反序列化状态字典（torch.load ← BytesIO）。

    weights_only=False：状态为可信自研产物（非外部输入），含张量字典。
    """
    buf = BytesIO(payload)
    return torch.load(buf, weights_only=False)


def states_equal(a: dict, b: dict, tol: float = 1e-5) -> bool:
    """逐张量容差比对两个状态字典（键集合一致且每键 allclose(tol)）。"""
    if set(a.keys()) != set(b.keys()):
        return False
    for k in a:
        ta, tb = a[k], b[k]
        if not isinstance(ta, torch.Tensor) or not isinstance(tb, torch.Tensor):
            if ta != tb:
                return False
            continue
        if ta.shape != tb.shape:
            return False
        if not torch.allclose(ta.float(), tb.float(), atol=tol, rtol=tol):
            return False
    return True
