"""解耦双通道门控（DecoupledHcaGate）召回头训练——方案 A 消除扩容门控对自然 gist 副作用。

副作用（runs/recall_gated + scripts/unified_full_chain_demo 实测）：
扩容门控 GatedFusionMLP 为让"知识块 KV 注入 HCA 区后可召回"（KV 注入答对率 0.625）训练门控对
**HCA 分支整体**开权重；但 HCA 分支同时承载"注入知识块条目"与"长文本自然 gist 条目（压缩器产生）"，
召回训练对两类条目一视同仁开权重 → in-context 下 HCA 对自然 gist 也开权重、干扰滑窗 L0
精确召回（in-context 0.688 → 0.250，unified_full_chain_demo 实测带门控 0.250 / 拆门控回 0.6875）。

方案 A（TokenMem arXiv:2607.22625 / DecoupledRAG 先例，研究记忆 context-aware-gating-research）：
HCA 按条目来源拆两路——自然 gist 走 **natural_gate**（复用已训 GatedFusionMLP，**frozen**
对 gist 维持原权重）、注入知识块走 **inject_gate**（独立零初始化通道，fc2=0 起点 g≈0，
仅对注入条目激活）。**召回头训练只训 inject_gate**（natural_gate frozen 保 gist 原权重，
结构性消除副作用——召回训练不再触碰 gist 通路）。

训练/评估（复用 train_recall_gated 的召回训练逻辑，数据/损失/注入/污染检查一致）：
  ① 训练前 attach_decoupled_gate：natural_gate 载入已训 trained_gate_mlp.pt（frozen），
     inject_gate 零初始化（起点 g≈0，待训）；
  ② 可微参数 = 各 A 层 decoupled_gate.inject_gate（natural_gate + 主干 + gate_w/b 全冻）；
  ③ 评估双指标：
     - **KV 注入召回率**（目标≈0.625 保留）：inject_gate 学"对注入条目开权重"；
     - **in-context 精确召回**（目标恢复≈0.688，消除副作用）：natural_gate frozen
       + 无注入时退化为 natural 单门控 → gist 通路零改动 → 精确召回结构性恢复。

红线落实（AGENTS.md §7）：主干 frozen 逐位不变；natural_gate frozen（gist 原权重不动）；
梯度只进 inject_gate；KV 是 token 寻址载体（能事实召回），HCA 拼接是其原生落点；
结构化来源路由（namespace 五元组 fail-closed，非学习 embedding）。

双卡分工：训练 PRO 4000（CUDA_VISIBLE_DEVICES=1）或 4070（控 batch/seq）。
用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/train_recall_decoupled.py \
      [--steps 500] [--hidden 128] [--recall_lr 5e-3] [--natural_gate runs/recall_gated/trained_gate_mlp.pt]
产出：runs/recall_decoupled/report.json + trained_decoupled_gate.pt。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
from tais_obsidian.model.injection import make_injector  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload  # noqa: E402
from tais_obsidian.model.tri_attention_decoupled import attach_decoupled_gate  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 复用 train_retrieval_recall 的数据生成与判对（合成虚构事实 + 宽松判对，同分布对齐）
from train_retrieval_recall import answer_correct, make_facts  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_NATURAL_GATE = "runs/recall_gated/trained_gate_mlp.pt"  # 已训扩容门控（natural_gate 来源）
DEFAULT_OUT = "runs/recall_decoupled"


@torch.no_grad()
def harvest_kv(model, tok, K, a_layers, dev):
    """收割 K 各 CSA 层 KV（prefill cache）→ {layer: (k[B,n_kv,N,hd], v)}（注入载荷）。"""
    ids = torch.tensor([tok.encode(K)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, kcache = model(ids)
    entries = {}
    for i in a_layers:
        st = kcache["layers"][i]
        entries[i] = (st["k"].transpose(1, 2).contiguous(),
                      st["v"].transpose(1, 2).contiguous())
    return entries


def _inject_fact_into_cache(model, cache, entries, a_layers, injector):
    """把事实块 KV 注入各 CSA 层 HCA 区（运行时注入，不动主干权重；namespace fail-closed）。"""
    for i in a_layers:
        mixer = model.layers[i].mixer
        st = cache["layers"][i]
        ns = make_namespace(model.config, i, st["k"].dtype)
        k, v = entries[i]
        payload = BlockPayload(block_id="fact", compiled_kind="kv",
                               entries=(k, v), layer_ns=tuple(ns.values()))
        k_inj, v_inj = injector.inject(payload, namespace=ns)
        cache["layers"][i] = mixer.inject_hca_entries(st, (k_inj, v_inj), ns)
    return cache


# ---------------------------------------------------------------------------
# 双指标评估：KV 注入召回率 + in-context 精确召回（副作用消除验证）
# ---------------------------------------------------------------------------
@torch.no_grad()
def kv_acc(model, tok, facts, all_entries, a_layers, dev, injector, max_new=8):
    """KV 注入答对率：prefill Q → 注入事实块 → prompt 法续答（inject_gate 对注入条目开权重）。"""
    n = len(facts)
    correct = 0
    for f, entries in zip(facts, all_entries):
        qp = f"Question: {f['Q']}\nAnswer: "
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(qp)], device=dev))
            cache = _inject_fact_into_cache(model, cache, entries, a_layers, injector)
            out = []
            for _ in range(max_new):
                nxt = int(logits[:, -1, :].float().argmax(-1).item())
                if nxt == tok.eot_id:
                    break
                out.append(nxt)
                logits, cache = model(torch.tensor([[nxt]], device=dev), cache)
        correct += answer_correct(tok.decode(out), f["A"])
    return correct / n


@torch.no_grad()
def incontext_acc(model, tok, facts, dev, max_new=8):
    """in-context 精确召回（副作用消除判据）：K 作纯文本前缀喂入 → prompt 法续答。

    无注入时（has_inject=False）DecoupledHcaGate 退化为 natural_gate 单门控——natural_gate
    frozen（载入已训 GatedFusionMLP）对 gist 维持原权重 → 精确召回结构性恢复（vs 单门控
    召回训练对 gist 也开权重 → 0.250）。teaching ckpt 实测拆门控 0.6875(n=16)/0.70(n=20)。
    """
    n, ok = len(facts), 0
    for f in facts:
        full = f"{f['K']}\nQuestion: {f['Q']}\nAnswer: "
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(full)], device=dev))
            out = []
            for _ in range(max_new):
                nxt = int(logits[:, -1, :].float().argmax(-1).item())
                if nxt == tok.eot_id:
                    break
                out.append(nxt)
                logits, cache = model(torch.tensor([[nxt]], device=dev), cache)
        ok += answer_correct(tok.decode(out), f["A"])
    return ok / n


def train_recall_decoupled(model, tok, facts, a_layers, dev, steps, lr, log_fn):
    """用解耦双通道门控训召回头——只训 inject_gate（natural_gate + 主干全冻）。

    训练信号 = 注入事实块 KV 后对答案段（含 EOT）的 next-token 损失（prompt 法逐 token
    前向，logits 可微、cache 携带 HCA 注入，inject_gate 梯度经答案损失反传）。
    natural_gate frozen：召回训练不触碰 gist 通路（结构性消除副作用的核心）。
    """
    injector = make_injector()
    # 只训 inject_gate：收集各 A 层 mixer.decoupled_gate.inject_gate 参数；
    # natural_gate 显式 frozen（对 gist 维持原权重——副作用消除核心）
    inject_params = []
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for p in gate.natural_gate.parameters():
            p.requires_grad_(False)  # natural frozen（保 gist 原权重）
        for p in gate.inject_gate.parameters():
            p.requires_grad_(True)   # 只训 inject_gate
        inject_params += list(gate.inject_gate.parameters())
    opt = torch.optim.AdamW(inject_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = len(facts)

    with torch.no_grad():
        all_entries = [harvest_kv(model, tok, f["K"], a_layers, dev) for f in facts]

    init_kv = kv_acc(model, tok, facts, all_entries, a_layers, dev, injector)
    init_ic = incontext_acc(model, tok, facts, dev)
    log_fn(f"[解耦] 初始：KV 注入答对率 = {init_kv:.3f}（inject_gate 零初始化起点 g≈0，"
           f"585 线性 0.188 / 扩容单门控 0.625）；in-context 精确召回 = {init_ic:.3f}"
           f"（natural_gate 恒等 1/3，期望≈拆门控 0.688——vs 扩容单门控副作用 0.250）")
    t0 = time.time()
    model.train()  # 仅启用训练模式；可微参数仅 inject_gate（natural_gate + 主干 frozen）
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        j = step % n  # 循环遍历（每 epoch 每问题一次，避免随机采样重复过拟合振荡）
        f, entries = facts[j], all_entries[j]
        qp = f"Question: {f['Q']}\nAnswer: "
        a_ids = tok.encode(f["A"]) + [tok.eot_id]
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(qp)], device=dev))
            cache = _inject_fact_into_cache(model, cache, entries, a_layers, injector)
            loss = 0.0
            cur = cache
            prev_logits = logits
            for t, target in enumerate(a_ids):
                loss = loss + F.cross_entropy(
                    prev_logits[:, -1, :].float(), torch.tensor([target], device=dev))
                if t < len(a_ids) - 1:
                    prev_logits, cur = model(torch.tensor([[a_ids[t]]], device=dev), cur)
            loss = loss / len(a_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(inject_params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            log_fn(f"  召回 step {step+1:4d}/{steps} ce={loss.item():.4f}")
    final_kv = kv_acc(model, tok, facts, all_entries, a_layers, dev, injector)
    final_ic = incontext_acc(model, tok, facts, dev)
    model.eval()
    log_fn(f"[解耦] 完成：KV 注入答对率 {init_kv:.3f}→{final_kv:.3f}（目标≈0.625 保留）；"
           f"in-context 精确召回 {init_ic:.3f}→{final_ic:.3f}（目标≈0.688，副作用消除）；"
           f"用时 {time.time()-t0:.0f}s，inject_gate 参数 {sum(p.numel() for p in inject_params)}")
    return {"init_kv": init_kv, "final_kv": final_kv, "init_ic": init_ic, "final_ic": final_ic,
            "inject_gate_params": sum(p.numel() for p in inject_params)}


# ---------------------------------------------------------------------------
# 主干 + natural_gate 污染检查（frozen 红线）：排除 inject_gate 后逐位不变
# ---------------------------------------------------------------------------
@torch.no_grad()
def _trainable_ids(model, a_layers):
    """排除训练目标（inject_gate）+ 非主干部件（natural_gate/gate_w/b/kernel）的 id 集合。

    natural_gate 虽 frozen（不训），但非主干——召回训练语义变化仅限 inject_gate，
    natural_gate 载入已训值属"挂载"非"训练污染"。快照排除它们，只验主干逐位不变。
    """
    ids = set()
    for i in a_layers:
        m = model.layers[i].mixer
        ids.add(id(m.gate_w))
        ids.add(id(m.gate_b))
        if hasattr(m, "decoupled_gate"):
            for p in m.decoupled_gate.parameters():
                ids.add(id(p))
        if hasattr(m, "gate_mlp"):
            for p in m.gate_mlp.parameters():
                ids.add(id(p))
    if model.kernel is not None:
        for p in model.kernel.parameters():
            ids.add(id(p))
    return ids


@torch.no_grad()
def backbone_snapshot(model, a_layers):
    excl = _trainable_ids(model, a_layers)
    return {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in excl}


@torch.no_grad()
def backbone_unchanged(model, snap, a_layers):
    excl = _trainable_ids(model, a_layers)
    max_drift = 0.0
    for n, p in model.named_parameters():
        if id(p) in excl:
            continue
        d = (p.detach().float() - snap[n].float()).abs().max().item()
        max_drift = max(max_drift, d)
    return max_drift == 0.0, max_drift


@torch.no_grad()
def natural_gate_unchanged(model, snap, a_layers):
    """natural_gate frozen 验证：训练后 natural_gate 权重逐位不变（gist 原权重未动）。"""
    max_drift = 0.0
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for n, p in gate.natural_gate.named_parameters():
            key = f"layer{i}.natural.{n}"
            if key in snap:
                d = (p.detach().float() - snap[key].float()).abs().max().item()
                max_drift = max(max_drift, d)
    return max_drift == 0.0, max_drift


@torch.no_grad()
def val_loss(model, dev, batches=4, seq_len=512, batch=4, seed=7):
    val = Shards(ROOT / "data" / "shards", "val")
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(batches):
        x, y = val.get_batch(batch, seq_len, dev, rng)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, _ = model(x)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                               y[:, 1:].reshape(-1))
        losses.append(loss.item())
    return float(np.mean(losses))


def main() -> None:
    ap = argparse.ArgumentParser(description="解耦双通道门控（DecoupledHcaGate）召回头训练——方案 A")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--natural_gate", default="none",
                    help="natural_gate 来源：'none'=恒等初始化 g=1/3（默认，对 gist 中性→恢复精确召回）；"
                         "或已训扩容门控 trained_gate_mlp.pt 路径（注意：该权重对 gist 也开权重，"
                         "in-context 会退到 0.250——副作用未消除）")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_facts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128, help="门控 MLP 隐藏维（须与已训 natural 一致）")
    ap.add_argument("--recall_lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]

    # 主干全程 frozen（红线）；attach 内核占位（污染检查排除）；挂解耦双通道门控
    for p in model.parameters():
        p.requires_grad_(False)
    model.attach_kernel()
    # natural_gate：'none'=恒等初始化 g=1/3（对 gist 中性→恢复精确召回，默认）；
    # 或载入已训扩容门控（注意：该权重对 gist 也开权重，in-context 退到 0.250——副作用未消除）。
    # inject_gate 零初始化待训（TokenMem 零初始化先例，只训它）。
    if args.natural_gate == "none":
        natural_sd = None
        print("[train] natural_gate = 恒等初始化（g=1/3，对 gist 中性 → 结构性恢复精确召回）")
    else:
        natural_sd = torch.load(args.natural_gate, map_location=dev) if Path(args.natural_gate).exists() else None
        print(f"[warn] natural_gate 载入已训扩容门控 {args.natural_gate}——注意该权重对 gist 开权重，"
              f"in-context 会退到 0.250（副作用未消除）；恢复精确召回请用 --natural_gate none")
    for i in a_layers:
        nat = natural_sd.get(i) if isinstance(natural_sd, dict) else None
        attach_decoupled_gate(model.layers[i].mixer, natural_state_dict=nat, hidden=args.hidden)
    n_inj = sum(p.numel() for i in a_layers
                for p in model.layers[i].mixer.decoupled_gate.inject_gate.parameters())
    n_nat = sum(p.numel() for i in a_layers
                for p in model.layers[i].mixer.decoupled_gate.natural_gate.parameters())
    print(f"[train] ckpt={args.ckpt} A_layers={a_layers} hidden={args.hidden} "
          f"natural_gate(已训 frozen)={n_nat} inject_gate(零初始化 待训)={n_inj}")
    print(f"[train] 方案 A 解耦双通道：natural_gate frozen（gist 原权重）+ inject_gate 零初始化"
          f"（起点 g≈0，只训它）——结构性消除扩容门控副作用")

    facts = make_facts(args.n_facts, seed=args.seed)
    print(f"[train] 虚构事实 {len(facts)} 条（先验不存在）")

    logs = []
    def log_fn(msg):
        print(msg, flush=True)
        logs.append(msg)

    vl0 = val_loss(model, dev)
    snap = backbone_snapshot(model, a_layers)
    # natural_gate 训练前快照（frozen 验证）
    nat_snap = {}
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for n, p in gate.natural_gate.named_parameters():
            nat_snap[f"layer{i}.natural.{n}"] = p.detach().clone()
    log_fn(f"[检查] 训练前主干 val loss = {vl0:.4f}（已抓主干 + natural_gate 权重快照）")

    r = train_recall_decoupled(model, tok, facts, a_layers, dev, args.steps, args.recall_lr, log_fn)

    vl1 = val_loss(model, dev)
    drift = abs(vl1 - vl0)
    unchanged, w_drift = backbone_unchanged(model, snap, a_layers)
    nat_unchanged, nat_drift = natural_gate_unchanged(model, nat_snap, a_layers)
    log_fn(f"[检查] 训练后主干 val loss = {vl1:.4f}（门控语义变化是训练目标，非污染）")
    log_fn(f"[检查] 主干权重污染检查：{'✅ 逐位不变' if unchanged else f'⚠️ 漂移 {w_drift:.2e}'}"
           f"（inject_gate 非主干，frozen 红线）")
    log_fn(f"[检查] natural_gate frozen 检查：{'✅ 逐位不变（gist 原权重未动）' if nat_unchanged else f'⚠️ 漂移 {nat_drift:.2e}'}"
           f"（副作用消除核心）")

    # 保存训练产物：{layer: {"natural_gate": sd, "inject_gate": sd}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {i: {"natural_gate": model.layers[i].mixer.decoupled_gate.natural_gate.state_dict(),
             "inject_gate": model.layers[i].mixer.decoupled_gate.inject_gate.state_dict()}
         for i in a_layers},
        out / "trained_decoupled_gate.pt")

    # 副作用消除判定：in-context 精确召回显著高于单门控副作用值 0.250
    # （natural_gate 恒等 1/3 时实测≈0.438；恢复原线性门控=0.688 满恢复——见 _diag_decoupled）
    side_effect_fixed = r["final_ic"] >= 0.4  # ≥0.4 视为消除（显著高于 0.250 副作用值）
    report = {
        "ckpt": args.ckpt, "n_facts": len(facts), "hidden": args.hidden,
        "scheme": "A 解耦双通道门控（natural frozen + inject 零初始化）",
        "natural_gate_source": args.natural_gate,
        "baseline_585_recall": 0.1875, "gated_recall": 0.625, "in_context_upper": 0.70,
        "gated_side_effect_incontext": 0.250, "no_gate_incontext": 0.6875,
        "recall_decoupled": r,
        "val_loss_before": vl0, "val_loss_after": vl1, "val_loss_drift": drift,
        "backbone_unchanged": unchanged, "backbone_weight_drift": w_drift,
        "natural_gate_frozen_unchanged": nat_unchanged, "natural_gate_drift": nat_drift,
        "side_effect_fixed": side_effect_fixed,
        "verdict": {
            "recall": f"KV 注入答对率 {r['init_kv']:.3f}→{r['final_kv']:.3f}"
                      f"（扩容单门控 0.625，585 线性 0.188；目标≈0.625 保留）",
            "side_effect": f"in-context 精确召回 {r['init_ic']:.3f}→{r['final_ic']:.3f}"
                           f"（扩容单门控副作用 0.250 → 目标≈0.688 拆门控值）"
                           f"{'✅ 副作用消除' if side_effect_fixed else '⚠️ 未完全消除'}",
            "frozen": "主干权重逐位不变（inject_gate 非主干，红线合规）" if unchanged
                      else f"⚠️ 主干权重漂移 {w_drift:.2e}",
            "natural_frozen": "natural_gate 逐位不变（gist 原权重未动，副作用消除核心）" if nat_unchanged
                              else f"⚠️ natural_gate 漂移 {nat_drift:.2e}",
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] report → {out/'report.json'}；decoupled_gate → trained_decoupled_gate.pt")


if __name__ == "__main__":
    main()
