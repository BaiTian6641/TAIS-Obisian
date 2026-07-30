"""动态词表（concept_slot）真实启用装配 demo：Kaplan 真实提取 → 注册 → HRL 检索 → 注入闭环。

打通"部件就绪但未真实启用"的缺口：此前 extract_fn 只在测试里用 lambda mock
（lambda text: torch.ones(D)），本脚本用**真实模型 Kaplan 内词典提取**装配
orchestrator.dynamic_vocab，演示并验证 concept_slot 全生命周期：

  检测词表摩擦（虚构专名/OOV 概念，高熵+高共现+低 P(IK)）
    → Kaplan 真实提取（末 token @ ℓ3 detokenized hidden，一次前向，no_grad 只读）
    → concept_slot 注册（PageTable 元数据 + BlockStore 向量载荷）
    → HRL route_graph 入图（orchestrator 内部）→ associative_recall（CA3 PPR）检索命中
    → 内核 inject 向量路径注入可用（位置不变向量 steer，非事实查表）

红线：concept_slot = 位置不变向量（factual_recall=False，载体能力边界）；提取 no_grad 只读
（监测/执行分置）；第 0 级输入侧零风险，输出侧不升格（Over-Tokenized）。

用法：$env:CUDA_VISIBLE_DEVICES="0"; .venv/Scripts/python.exe scripts/dynamic_vocab_real_demo.py
"""
from __future__ import annotations

import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.dyn_vocab import make_dynamic_vocab
from tais_obsidian.model.kaplan_extract import DEFAULT_KAPLAN_LAYER, make_kaplan_extract_fn
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.runtime import (
    BlockStore,
    MemoryBus,
    PageTable,
    Pager,
    make_orchestrator,
)

CKPT = "checkpoints/pilot_0p1b_gdn2_10k/final"
NS = ("m1", 0, 1, "bf16", 10000.0)  # namespace 五元组（与块/载荷一致）


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 68)
    print("动态词表 concept_slot 真实启用装配 demo（Kaplan 真实提取，非 mock）")
    print("=" * 68)

    # 1) 加载真实模型 + 真实 Kaplan extract_fn（替代测试里的 lambda mock）
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, device).eval()
    d_model = model.config.d_model
    extract_fn = make_kaplan_extract_fn(model)  # 默认 ℓ3（pilot detokenize 最强层）
    print(f"\n[1] 模型加载：d_model={d_model}，{model.config.n_layer} 层；"
          f"Kaplan 提取层 ℓ{DEFAULT_KAPLAN_LAYER}（末 token detokenized hidden，no_grad 只读）")

    # 2) 装配运行时：PageTable + BlockStore + Pager + Bus + orchestrator（dynamic_vocab 注入真实件）
    pt, bs = PageTable(), BlockStore()
    bus = MemoryBus(pt, bs, Pager(bs, pt))
    model.attach_kernel()  # 10k checkpoint kernel_enabled=False → 挂载内核（随机初始化，仅作注入路径）
    dyn = make_dynamic_vocab(pt, NS, extract_fn=extract_fn, blockstore=bs)
    orch = make_orchestrator(model.kernel, bus, dynamic_vocab=dyn)
    print(f"[2] orchestrator.dynamic_vocab 装配：{'真实 Kaplan extract_fn' if dyn.extract_fn is not None else 'None'}"
          f"（extract_fn 非 None 且非 mock）")

    # 3) 检测词表摩擦 → 触发 concept_slot 注册（虚构专名/OOV：高熵+高共现+低 P(IK)）
    concepts = ["Qeltharion", "Zorblax", "quantum entanglement"]
    print(f"\n[3] 词表摩擦检测 → concept_slot 注册（{len(concepts)} 个概念）：")
    for c in concepts:
        ok = orch.assess_vocab_friction(c, p_ik=0.10, next_token_entropy=0.90, repeat_cooccur=0.90)
        spec = pt.get(f"concept/{c}")
        payload = bs.get(f"concept/{c}")
        print(f"  · {c!r:26s} 触发={ok}  注册={'✓' if spec else '✗'}  "
              f"kind={spec.compiled_kind if spec else '-'}  factual_recall={spec.factual_recall if spec else '-'}  "
              f"载荷向量={'✓['+str(tuple(payload.vector.shape))+']' if payload is not None and payload.vector is not None else '✗'}")

    # 4) 闭环验证 A：注册（页表元数据 + BlockStore 载荷）
    print("\n[4] 闭环验证 A —— 注册：页表 + BlockStore")
    for c in concepts:
        spec = pt.get(f"concept/{c}")
        payload = bs.get(f"concept/{c}")
        assert spec is not None and spec.compiled_kind == "concept_slot", c
        assert spec.factual_recall is False, "concept_slot 必须是位置不变向量（非事实查表）"
        assert payload is not None and payload.vector is not None and payload.vector.shape == (d_model,), c
    print("  ✓ 全部 concept_slot 已注册（页表元数据 factual_recall=False + BlockStore [d_model] 载荷）")

    # 5) 闭环验证 B：HRL route_graph 入图 → associative_recall 可检索到
    print("\n[5] 闭环验证 B —— HRL route_graph 入图 + CA3 PPR 联想检索")
    graph = orch.route_graph
    for c in concepts:
        bid = f"concept/{c}"
        assert bid in graph, f"{bid} 未入 route_graph"
    print(f"  ✓ {len(concepts)} 个 concept_slot 已入 route_graph（节点）：{[f'concept/{c}' in graph for c in concepts]}")
    recalled = orch.associative_recall({f"concept/{concepts[0]}": 1.0})
    print(f"  · associative_recall（种子 concept/{concepts[0]}）可达节点：{sorted(recalled.keys())}")
    assert f"concept/{concepts[0]}" in recalled, "PPR 检索应命中 concept_slot 节点"
    print("  ✓ associative_recall 检索命中 concept_slot（动态词表 ↔ HRL 互动）")

    # 6) 闭环验证 C：注入可用（内核 inject 向量路径，位置不变向量 steer）
    print("\n[6] 闭环验证 C —— 内核 inject 向量路径注入可用")
    pm_pre = torch.zeros(1, 4, d_model, device=device)
    payload = bs.get(f"concept/{concepts[0]}")
    payload = type(payload)(
        block_id=payload.block_id, compiled_kind=payload.compiled_kind,
        vector=payload.vector.to(device), layer_ns=payload.layer_ns, signature=payload.signature)
    injected = model.kernel.inject(pm_pre, [payload], alphas=[1.0])
    delta = (injected - pm_pre).abs().sum().item()
    print(f"  ✓ concept_slot 经 inject() 注入（向量加法 steer），pm_pre 改变量 Σ|Δ|={delta:.3f} > 0")
    assert delta > 0, "注入应改变 pm_pre（向量 steer）"

    # 7) 真实 extract_fn 语义（vs mock 常数）：真实表征有语义、非塌缩常数
    print("\n[7] 真实 Kaplan extract_fn 语义（vs mock 常数塌缩）")
    import torch.nn.functional as F
    cos_sim = float(F.cosine_similarity(extract_fn("electron"), extract_fn("photon"), dim=0))
    cos_diff = float(F.cosine_similarity(extract_fn("electron"), extract_fn("democracy"), dim=0))
    mock_a, mock_b = torch.ones(d_model), torch.ones(d_model)
    cos_mock = float(F.cosine_similarity(mock_a, mock_b, dim=0))
    print(f"  · 真实：cos(electron,photon 同类)={cos_sim:.3f}  cos(electron,democracy 不同类)={cos_diff:.3f}"
          f"  → 同类更近（{cos_sim > cos_diff}）")
    print(f"  · mock 常数：cos(ones,ones)={cos_mock:.3f}（任意概念 cos≡1，无语义区分）→ 真实 extract_fn ≠ 常数")

    print("\n" + "=" * 68)
    print("concept_slot 闭环全部打通：检测→Kaplan真实提取→注册(页表+BlockStore)→HRL检索→注入")
    print("=" * 68)


if __name__ == "__main__":
    main()
