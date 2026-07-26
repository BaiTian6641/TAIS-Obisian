"""M5 注入闭环单元测试：MemoryLayer、Injector、内核 inject 集成、state_ckpt 集成。

判据（接口与实现计划 v1.0 §5 / 部件实现详细计划 Part D / M5 退出标准）：
- MemoryLayer：查询命中、delta 写入分布内、门控遗忘；
- Injector：KV/gist namespace 校验 fail-closed、mem_entry delta 写入/查询、向量单次加法；
- 内核 inject()：token 寻址载体传入 injector 后接通（不再 fail-closed）、向量仍单次加法、
  不给 injector 时仍 fail-closed（不静默注入）；
- state_ckpt：含 MemoryLayer 状态的 dict 往返 <1e-5。
"""
from __future__ import annotations

import pytest
import torch

from tais_obsidian.model.blockpath import NamespaceMismatchError
from tais_obsidian.model.injection import make_injector
from tais_obsidian.model.memlayer import make_memory_layer
from tais_obsidian.model.tais_kernel import BlockPayload, make_kernel
from tais_obsidian.runtime import restore_state, save_state, states_equal

D = 32
KD = 16   # 记忆层 key_dim
NS = {"model_id": "m1", "layer_idx": 0, "compressor_version": "v1", "dtype": "bf16", "rope_theta": 10000.0}
NS_BAD = dict(NS, layer_idx=999)


# ---------------- MemoryLayer ----------------

def test_memlayer_query_shape_and_topk() -> None:
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    q = torch.randn(2, 3, KD)
    out = ml.query(q, topk=4)
    assert out.shape == (2, 3, D)


def test_memlayer_delta_write_changes_readback() -> None:
    """delta 写入改变读出：写 (k,v) 后，对 k 的读出应接近 v（分布内写入）。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k = torch.randn(KD)
    v = torch.randn(D)
    kn = torch.nn.functional.normalize(k, dim=-1)
    before = kn @ ml.state
    ml.write(k, v, beta=1.0)
    after = kn @ ml.state
    # 写入后读出更接近 v（先擦除旧关联再写入）
    assert (after - v).norm() < (before - v).norm()


def test_memlayer_forget_gates_state() -> None:
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    ml.write(torch.randn(KD), torch.randn(D))
    assert ml.state.abs().sum() > 0
    ml.forget(0.0)
    assert ml.state.abs().sum() == 0


# ---------------- Injector ----------------

def test_injector_kv_namespace_pass_and_fail_closed() -> None:
    inj = make_injector()
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)
    p = BlockPayload(block_id="b", compiled_kind="kv", entries=(k, v), layer_ns=tuple(NS.values()))
    # namespace 匹配 → 返回待拼接 (k,v)
    assert inj.inject(p, namespace=NS) == (k, v)
    # namespace 不匹配 → 抛 NamespaceMismatchError（fail-closed，调用方走回退）
    with pytest.raises(NamespaceMismatchError):
        inj.inject(p, namespace=NS_BAD)


def test_injector_kv_missing_entries_raises() -> None:
    inj = make_injector()
    p = BlockPayload(block_id="b", compiled_kind="kv")
    with pytest.raises(ValueError):
        inj.inject(p, namespace=NS)


def test_injector_mem_write_and_query() -> None:
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    inj = make_injector(ml)
    k, v = torch.randn(KD), torch.randn(D)
    # delta 写入
    pw = BlockPayload(block_id="w", compiled_kind="mem_entry", entries=(k, v))
    assert inj.inject(pw) is True
    assert ml.state.abs().sum() > 0
    # 查询读出
    pq = BlockPayload(block_id="q", compiled_kind="mem_entry", vector=k)
    out = inj.inject(pq)
    assert out.shape == (D,)


def test_injector_vector_single_add() -> None:
    inj = make_injector()
    vec = torch.ones(D)
    p = BlockPayload(block_id="v", compiled_kind="icv", vector=vec)
    assert torch.equal(inj.inject(p), vec)  # 返回单次加法载荷


def test_injector_unknown_kind_fail_closed() -> None:
    inj = make_injector()
    # BlockPayload 拒收未知 kind（__post_init__），故此处直接测 injector 分支
    p = BlockPayload(block_id="x", compiled_kind="lora")  # lora 在 ADDRESSED 但 injector 未实现
    assert inj.inject(p) is None  # 未实现载体 fail-closed 返回 None


# ---------------- 内核 inject() 集成 ----------------

def test_kernel_inject_addressed_with_injector() -> None:
    """token 寻址载体传入 injector 后接通（不再 fail-closed）。"""
    kernel = make_kernel(D, dg_dim=32, dg_topk=4)
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    inj = make_injector(ml)
    pm = torch.zeros(1, 2, D)
    k, v = torch.randn(KD), torch.randn(D)
    p = BlockPayload(block_id="m", compiled_kind="mem_entry", entries=(k, v))
    out = kernel.inject(pm, [p], injector=inj)
    # PM-stream 未被 mem 注入修改（mem 走记忆层，不走 PM-stream 加法）
    assert torch.equal(out, pm)
    assert ml.state.abs().sum() > 0  # mem 写入生效


def test_kernel_inject_addressed_without_injector_still_fail_closed() -> None:
    """不给 injector 时 token 寻址载体仍 fail-closed（不静默注入）。"""
    kernel = make_kernel(D, dg_dim=32, dg_topk=4)
    p = BlockPayload(block_id="k", compiled_kind="kv")
    with pytest.raises(NotImplementedError):
        kernel.inject(torch.zeros(1, 1, D), [p])


def test_kernel_inject_vector_still_single_add_with_injector() -> None:
    kernel = make_kernel(D, dg_dim=32, dg_topk=4)
    inj = make_injector(make_memory_layer(n_slots=8, key_dim=KD, value_dim=D))
    pm = torch.zeros(1, 2, D)
    vec = torch.ones(D)
    p = BlockPayload(block_id="v", compiled_kind="steering", vector=vec)
    out = kernel.inject(pm, [p], alphas=[3.0], injector=inj)
    assert torch.allclose(out, pm + 3.0 * vec.view(1, 1, D))


# ---------------- state_ckpt 集成（含记忆层状态） ----------------

def test_state_ckpt_with_memlayer_state() -> None:
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    ml.write(torch.randn(KD), torch.randn(D))
    state = {"mem_state": ml.state.clone(), "gdn_h": torch.randn(2, 4)}
    back = restore_state(save_state(state))
    assert states_equal(state, back, tol=1e-5)
