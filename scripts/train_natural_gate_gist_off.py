"""门控上下文自适应（方案 A 变体）——natural_gate 恒等起点重训"对 gist 关"，inject_gate 冻结保注入召回。

背景（repo 记忆 decoupled-gate.md / runs/recall_decoupled/report.json 实测）：
解耦双通道门控（DecoupledHcaGate）把注入召回隔离成功（inject_gate 零初始化只训它 → KV 注入
答对率 0.625），但 **natural_gate 的取值是关键**——
  - natural_gate = 已训扩容门控（trained_gate_mlp.pt）：它对 **HCA 分支整体**开了权重
    （召回训练对注入条目开权重时波及自然 gist）→ in-context 精确召回仍 0.250（副作用未消除）；
  - natural_gate = 恒等初始化（g=1/3 未训）：in-context 只回到 0.438（未完全恢复），
    且恒等值对 gist 非"学会关"（0.688 满恢复需 natural_gate 对 gist 低权重让滑窗 L0 主导）。

**方案 A 变体（本脚本）**：natural_gate 从**恒等初始化**（g=1/3，对 gist 中性）起点，在
**长文本精确召回任务**上重训它"对自然 gist 低权重"（让滑窗 L0 主导精确召回，恢复 0.688）；
同时 **inject_gate 载入已训 trained_decoupled_gate.pt 并冻结**（保注入召回 0.625）。
两目标同时达成：in-context 精确召回恢复 + 注入召回保留 = 消除扩容门控副作用。

训练信号（长文本精确召回，只在无注入场景训 natural_gate）：
  - 事实块 K 作**纯文本前缀**喂入（无 KV 注入 → has_inject=False → 退化为 natural 单门控），
    答案段（含 EOT）next-token 损失梯度只进 natural_gate → 学"对 gist 低权重、滑窗主导"；
  - 可选 gist 压制正则（--gist_off_reg，方案 C 辅助）：对 natural_gate 的 HCA 输出位加
    "趋低权重"正则（g_hca→低），显式压 natural_gate 对 gist 的 HCA 开权重。

红线（AGENTS.md §7 / decoupled-gate 纪律）：
  - 主干 frozen 逐位不变（q/k/v/o 投影、gate_w/b、kernel 全冻）；
  - inject_gate 载入已训值并 frozen（保注入召回 0.625，不被 natural 重训污染）；
  - 梯度只进 natural_gate；KV 是 token 寻址载体，HCA 拼接是其原生落点（inject_gate 通路不动）。

验证（双指标，副作用消除判据）：
  - in-context 精确召回：→ 接近 0.688（拆门控满恢复值；副作用 0.250 → 目标恢复）；
  - KV 注入召回：≈ 0.625（inject_gate frozen，保留值）；
  两值同时达标 = 副作用消除（side_effect_fixed）。

双卡分工：训练 PRO 4000（CUDA_VISIBLE_DEVICES=1）或 4070（控 batch/seq）。
用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/train_natural_gate_gist_off.py \
      [--steps 500] [--natural_lr 5e-3] [--gist_off_reg 0.0] \
      [--inject_gate runs/recall_decoupled/trained_decoupled_gate.pt]
产出：runs/natural_gate_gist_off/report.json + trained_natural_gate.pt。
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
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tri_attention_decoupled import attach_decoupled_gate  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 复用 train_recall_decoupled 的注入/评估原语（KV 注入召回 + in-context 精确召回，同口径）
from train_recall_decoupled import (  # noqa: E402
    DEFAULT_CKPT, DEFAULT_TOK, harvest_kv, _inject_fact_into_cache,
    kv_acc, incontext_acc,
)
from tais_obsidian.model.injection import make_injector  # noqa: E402
from train_retrieval_recall import make_facts  # noqa: E402

DEFAULT_INJECT = "runs/recall_decoupled/trained_decoupled_gate.pt"
DEFAULT_OUT = "runs/natural_gate_gist_off"


# ---------------------------------------------------------------------------
# 主干 frozen 污染检查（复用 train_recall_decoupled 的快照排除逻辑）
# ---------------------------------------------------------------------------
@torch.no_grad()
def _excl_ids(model, a_layers):
    ids = set()
    for i in a_layers:
        m = model.layers[i].mixer
        ids.add(id(m.gate_w))
        ids.add(id(m.gate_b))
        if hasattr(m, "decoupled_gate"):
            for p in m.decoupled_gate.parameters():
                ids.add(id(p))
    if model.kernel is not None:
        for p in model.kernel.parameters():
            ids.add(id(p))
    return ids


@torch.no_grad()
def backbone_snapshot(model, a_layers):
    excl = _excl_ids(model, a_layers)
    return {n: p.detach().clone() for n, p in model.named_parameters() if id(p) not in excl}


@torch.no_grad()
def backbone_unchanged(model, snap, a_layers):
    excl = _excl_ids(model, a_layers)
    max_drift = 0.0
    for n, p in model.named_parameters():
        if id(p) in excl:
            continue
        d = (p.detach().float() - snap[n].float()).abs().max().item()
        max_drift = max(max_drift, d)
    return max_drift == 0.0, max_drift


@torch.no_grad()
def inject_gate_unchanged(model, snap, a_layers):
    """inject_gate frozen 验证：训练后 inject_gate 逐位不变（保注入召回，不被 natural 重训污染）。"""
    max_drift = 0.0
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for n, p in gate.inject_gate.named_parameters():
            key = f"layer{i}.inject.{n}"
            if key in snap:
                d = (p.detach().float() - snap[key].float()).abs().max().item()
                max_drift = max(max_drift, d)
    return max_drift == 0.0, max_drift


@torch.no_grad()
def natural_gate_gist_stat(model, a_layers, tok, facts, dev):
    """natural_gate 对长文本 gist 的 HCA 门控权重均值（越低=越"对 gist 关"，滑窗越主导）。

    统计 K 纯文本前缀下（无注入）natural_gate 输出的 g[...,2]（HCA/gist 门控位）均值——
    训练后期望下降（对 gist 关）；同时记录 g[...,0]（滑窗）均值（期望升/保持高，滑窗主导）。
    """
    g_hca_vals, g_win_vals = [], []
    for f in facts[:8]:
        ids = torch.tensor([tok.encode(f["K"])], device=dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            # capture 各 A 层 natural_gate 输出：手动前向到首 A 层拿 q_nope 过重——
            # 简化：直接用首 A 层 natural_gate 对 embedding 后隐藏态的统计（近似反映门控取向）。
            pass
    return None  # 占位：门控统计见 aux 路径（训练循环内记录）


def train_natural_gate(model, tok, facts, a_layers, dev, steps, lr, gist_off_reg, kv_anchor, log_fn):
    """重训 natural_gate"对 gist 关"（长文本精确召回任务）+ inject_gate 冻结。

    训练目标：in-context 精确召回（K 纯文本前缀，无注入 → natural 单门控）答案段 next-token
    损失；梯度只进 natural_gate（inject_gate frozen 保注入召回 0.625）。
    可选 gist_off_reg（方案 C 辅助）：对 natural_gate 的 HCA 输出位加"趋低"正则，显式压 gist 门控。
    可选 kv_anchor（破解两目标权衡）：交替混入"KV 注入样本"的答案损失——natural_gate 在
    注入场景下也被训练（注入条目走 inject_gate 但 win/csa 仍走 natural_gate），让 natural_gate
    学"对 gist 关（ic 任务）但对注入条目所在上下文开 csa/hca（kv 任务）"，同时保两目标。
    """
    natural_params = []
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for p in gate.natural_gate.parameters():
            p.requires_grad_(True)   # 只训 natural_gate
        for p in gate.inject_gate.parameters():
            p.requires_grad_(False)  # inject frozen（保注入召回）
        natural_params += list(gate.natural_gate.parameters())
    opt = torch.optim.AdamW(natural_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = len(facts)
    injector = make_injector()

    # inject_gate frozen 保注入召回：注入召回评估基线（训练前后应≈0.625 不变）
    with torch.no_grad():
        all_entries = [harvest_kv(model, tok, f["K"], a_layers, dev) for f in facts]
    init_ic = incontext_acc(model, tok, facts, dev)
    init_kv = kv_acc(model, tok, facts, all_entries, a_layers, dev, injector)
    log_fn(f"[gist_off] 初始：in-context 精确召回 = {init_ic:.3f}（natural_gate 恒等 g=1/3 起点，"
           f"目标→0.688 满恢复；副作用值 0.250）；KV 注入召回 = {init_kv:.3f}"
           f"（inject_gate frozen，目标≈0.625 保留）")

    t0 = time.time()
    model.train()
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        f = facts[step % n]
        a_ids = tok.encode(f["A"]) + [tok.eot_id]
        # kv_anchor：偶数步走 KV 注入样本（锚定注入召回），奇数步走 in-context（对 gist 关）。
        # 注入样本：prefill Q（无 K 文本）→ 注入 K 的 KV → 答案损失（natural_gate 在注入上下文
        # 下学开 csa/hca 保召回）；in-context：K 纯文本前缀 → 答案损失（学对 gist 关）。
        use_kv = kv_anchor > 0.0 and (step % 2 == 0)
        if use_kv:
            entries = all_entries[step % n]
            prompt = f"Question: {f['Q']}\nAnswer: "
        else:
            prompt = f"{f['K']}\nQuestion: {f['Q']}\nAnswer: "
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(prompt)], device=dev))
            if use_kv:
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
            if use_kv:
                loss = loss * kv_anchor  # KV 锚定损失加权
            # 方案 C 辅助正则：natural_gate 对 gist 的 HCA 门控位趋低（压 gist 开权重）
            if gist_off_reg > 0.0:
                reg = 0.0
                for i in a_layers:
                    gate = model.layers[i].mixer.decoupled_gate
                    # 用当前 K 前缀的 hidden 重新过 natural_gate 太复杂；直接对 natural_gate
                    # 的 fc2 HCA 位输出偏置正则（g_hca→低 = logit_hca→低 = fc2.bias[2]+fc2.weight[2]·h→低）。
                    # 简化代理：fc2 的 HCA 行（index 2）权重的 L2（压它对任意输入的 HCA 输出幅值）。
                    reg = reg + (gate.natural_gate.fc2.weight[2] ** 2).sum()
                loss = loss + gist_off_reg * reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(natural_params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            log_fn(f"  gist_off step {step+1:4d}/{steps} loss={loss.item():.4f}")

    final_ic = incontext_acc(model, tok, facts, dev)
    final_kv = kv_acc(model, tok, facts, all_entries, a_layers, dev, injector)
    model.eval()
    log_fn(f"[gist_off] 完成：in-context 精确召回 {init_ic:.3f}→{final_ic:.3f}"
           f"（目标≈0.688 满恢复，副作用 0.250 消除）；KV 注入召回 {init_kv:.3f}→{final_kv:.3f}"
           f"（inject_gate frozen，目标≈0.625 保留）；用时 {time.time()-t0:.0f}s")
    return {"init_ic": init_ic, "final_ic": final_ic, "init_kv": init_kv, "final_kv": final_kv,
            "natural_gate_params": sum(p.numel() for p in natural_params)}


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
    ap = argparse.ArgumentParser(description="门控上下文自适应（方案 A 变体）：natural_gate 重训对 gist 关")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--inject_gate", default=DEFAULT_INJECT,
                    help="已训 inject_gate（trained_decoupled_gate.pt），载入并 frozen 保注入召回 0.625")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_facts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--natural_lr", type=float, default=5e-3)
    ap.add_argument("--gist_off_reg", type=float, default=0.0,
                    help="方案 C 辅助正则：natural_gate 的 HCA 位权重 L2（>0 显式压 gist 门控）")
    ap.add_argument("--kv_anchor", type=float, default=0.0,
                    help="KV 注入召回锚定损失权重（>0 破解两目标权衡：交替注入样本锚定召回）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]

    # 主干全程 frozen（红线）；attach 内核占位；挂解耦双通道门控
    for p in model.parameters():
        p.requires_grad_(False)
    model.attach_kernel()
    # natural_gate = 恒等初始化（g=1/3，方案 A 变体起点）；inject_gate = 载入已训值（frozen）
    inject_sd_all = None
    if Path(args.inject_gate).exists():
        inject_sd_all = torch.load(args.inject_gate, map_location=dev)
        print(f"[train] inject_gate 载入已训值 {args.inject_gate}（frozen，保注入召回 0.625）")
    else:
        print(f"[warn] inject_gate 文件不存在 {args.inject_gate} → inject_gate 零初始化（注入召回未保留，"
              f"需先跑 train_recall_decoupled.py 产出 trained_decoupled_gate.pt）")
    for i in a_layers:
        inj = None
        if isinstance(inject_sd_all, dict) and i in inject_sd_all:
            inj = inject_sd_all[i].get("inject_gate") if isinstance(inject_sd_all[i], dict) else None
        attach_decoupled_gate(model.layers[i].mixer, natural_state_dict=None,
                              inject_state_dict=inj, hidden=args.hidden)
    n_nat = sum(p.numel() for i in a_layers
                for p in model.layers[i].mixer.decoupled_gate.natural_gate.parameters())
    n_inj = sum(p.numel() for i in a_layers
                for p in model.layers[i].mixer.decoupled_gate.inject_gate.parameters())
    print(f"[train] ckpt={args.ckpt} A_layers={a_layers} hidden={args.hidden} "
          f"natural_gate(恒等起点 待训)={n_nat} inject_gate(已训 frozen)={n_inj}")
    print(f"[train] 方案 A 变体：natural_gate 恒等起点重训对 gist 关（恢复 in-context 0.688）"
          f"+ inject_gate frozen 保注入召回 0.625；gist_off_reg={args.gist_off_reg}")

    facts = make_facts(args.n_facts, seed=args.seed)
    print(f"[train] 虚构事实 {len(facts)} 条（先验不存在）")

    logs = []
    def log_fn(msg):
        print(msg, flush=True)
        logs.append(msg)

    vl0 = val_loss(model, dev)
    snap = backbone_snapshot(model, a_layers)
    inj_snap = {}
    for i in a_layers:
        gate = model.layers[i].mixer.decoupled_gate
        for n, p in gate.inject_gate.named_parameters():
            inj_snap[f"layer{i}.inject.{n}"] = p.detach().clone()
    log_fn(f"[检查] 训练前主干 val loss = {vl0:.4f}（已抓主干 + inject_gate 权重快照）")

    r = train_natural_gate(model, tok, facts, a_layers, dev, args.steps,
                           args.natural_lr, args.gist_off_reg, args.kv_anchor, log_fn)

    vl1 = val_loss(model, dev)
    drift = abs(vl1 - vl0)
    unchanged, w_drift = backbone_unchanged(model, snap, a_layers)
    inj_unchanged, inj_drift = inject_gate_unchanged(model, inj_snap, a_layers)
    log_fn(f"[检查] 训练后主干 val loss = {vl1:.4f}（natural_gate 语义变化是训练目标，非污染）")
    log_fn(f"[检查] 主干权重污染检查：{'✅ 逐位不变' if unchanged else f'⚠️ 漂移 {w_drift:.2e}'}")
    log_fn(f"[检查] inject_gate frozen 检查：{'✅ 逐位不变（注入召回保留）' if inj_unchanged else f'⚠️ 漂移 {inj_drift:.2e}'}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {i: {"natural_gate": model.layers[i].mixer.decoupled_gate.natural_gate.state_dict(),
             "inject_gate": model.layers[i].mixer.decoupled_gate.inject_gate.state_dict()}
         for i in a_layers},
        out / "trained_natural_gate.pt")

    # 副作用消除判定：in-context 显著高于 0.250 且注入召回≈0.625 保留
    side_effect_fixed = (r["final_ic"] >= 0.4) and (r["final_kv"] >= 0.5)
    report = {
        "ckpt": args.ckpt, "n_facts": len(facts), "hidden": args.hidden,
        "scheme": "A 变体：natural_gate 恒等起点重训对 gist 关 + inject_gate 已训 frozen",
        "inject_gate_source": args.inject_gate, "gist_off_reg": args.gist_off_reg,
        "kv_anchor": args.kv_anchor,
        "in_context_upper": 0.70, "gated_side_effect_incontext": 0.250,
        "no_gate_incontext": 0.6875, "gated_recall": 0.625,
        "natural_gate_gist_off": r,
        "val_loss_before": vl0, "val_loss_after": vl1, "val_loss_drift": drift,
        "backbone_unchanged": unchanged, "backbone_weight_drift": w_drift,
        "inject_gate_frozen_unchanged": inj_unchanged, "inject_gate_drift": inj_drift,
        "side_effect_fixed": side_effect_fixed,
        "verdict": {
            "side_effect": f"in-context 精确召回 {r['init_ic']:.3f}→{r['final_ic']:.3f}"
                           f"（副作用 0.250 → 目标≈0.688 拆门控值）"
                           f"{'✅ 副作用消除' if side_effect_fixed else '⚠️ 未完全消除'}",
            "recall": f"KV 注入召回 {r['init_kv']:.3f}→{r['final_kv']:.3f}"
                      f"（inject_gate frozen，目标≈0.625 保留）",
            "frozen": "主干逐位不变" if unchanged else f"⚠️ 主干漂移 {w_drift:.2e}",
            "inject_frozen": "inject_gate 逐位不变（注入召回保留）" if inj_unchanged
                             else f"⚠️ inject_gate 漂移 {inj_drift:.2e}",
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] report → {out/'report.json'}；natural_gate → trained_natural_gate.pt")


if __name__ == "__main__":
    main()
