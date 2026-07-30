"""HCA 门控扩容重训召回头——突破 585 线性门控容量瓶颈（召回 0.188 → 逼近 in-context 上界 0.70）。

背景（runs/retrieval_recall/report.json + memories/repo/retrieval-recall-training.md）：
原召回头训练（train_retrieval_recall.py）只训各 A 层线性门控 gate_w/gate_b（**585 参数**），
KV 注入答对率 0.000→0.188 即触顶（<< in-context 上界 0.70）。诊断：**线性标量门控容量瓶颈**——
学不会"对注入条目开权重"的复杂内容路由（表征/通路可用，纯门控表达力不足）。

本脚本把门控扩为 GatedFusionMLP（tri_attention_gated，Linear+GELU+Linear 小 MLP，~8.7k/层）
重训召回头（**只训门控 MLP，主干全冻**，监测/执行分置 + 防遗忘红线不变）。

复用 train_retrieval_recall 的召回训练逻辑（数据/损失/评估/注入/污染检查），仅：
  ① 训练前 attach_gated_fusion 给各 A 层 mixer 挂 GatedFusionMLP（恒等初始化 g=1/3）；
  ② 可微参数 = 各 A 层 gate_mlp（替代原 gate_w/gate_b）；
  ③ 污染检查把 gate_mlp 也排除出"主干"（与 gate_w/b 同，训练目标非主干）。

红线落实（AGENTS.md §7）：主干 frozen 逐位不变；梯度只进门控 MLP；KV 是 token 寻址载体
（能事实召回），HCA 拼接是其原生落点。恒等初始化保既有 checkpoint 行为不变（g 初始=1/3）。

双卡分工：训练 PRO 4000（CUDA_VISIBLE_DEVICES=1）或 4070（控 batch/seq）。
用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/train_recall_gated.py \
      [--steps 500] [--hidden 128] [--recall_lr 5e-3]
产出：runs/recall_gated/report.json + trained_gate_mlp.pt。
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
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
from tais_obsidian.model.injection import make_injector  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload  # noqa: E402
from tais_obsidian.model.tri_attention_gated import attach_gated_fusion  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 复用 train_retrieval_recall 的数据生成与判对（合成虚构事实 + 宽松判对，同分布对齐）
from train_retrieval_recall import answer_correct, make_facts  # noqa: E402

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_OUT = "runs/recall_gated"


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


def train_recall_gated(model, tok, facts, a_layers, dev, steps, lr, seed, log_fn):
    """用扩容 GatedFusionMLP 重训召回头（只训门控 MLP，主干全冻）。

    训练信号 = 注入事实块 KV 后对答案段（含 EOT）的 next-token 损失（prompt 法逐 token
    前向，logits 可微、cache 携带 HCA 注入，门控 MLP 梯度经答案损失反传）。
    """
    injector = make_injector()
    # 只训门控 MLP：收集各 A 层 mixer.gate_mlp 参数（主干 + gate_w/b 全冻）
    gate_params = []
    for i in a_layers:
        m = model.layers[i].mixer
        for p in m.gate_mlp.parameters():
            p.requires_grad_(True)
        gate_params += list(m.gate_mlp.parameters())
    opt = torch.optim.AdamW(gate_params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    n = len(facts)

    with torch.no_grad():
        all_entries = [harvest_kv(model, tok, f["K"], a_layers, dev) for f in facts]

    @torch.no_grad()
    def kv_acc(max_new_=8):
        """KV 注入答对率（评估：注入→prompt 法续答，no_grad）。"""
        correct = 0
        for f, entries in zip(facts, all_entries):
            qp = f"Question: {f['Q']}\nAnswer: "
            with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
                logits, cache = model(torch.tensor([tok.encode(qp)], device=dev))
                cache = _inject_fact_into_cache(model, cache, entries, a_layers, injector)
                out = []
                for _ in range(max_new_):
                    nxt = int(logits[:, -1, :].float().argmax(-1).item())
                    if nxt == tok.eot_id:
                        break
                    out.append(nxt)
                    logits, cache = model(torch.tensor([[nxt]], device=dev), cache)
            correct += answer_correct(tok.decode(out), f["A"])
        return correct / n

    init_acc = kv_acc()
    log_fn(f"[召回·扩容] 初始 KV 注入答对率 = {init_acc:.3f}"
           f"（基线 0，585 线性门控 0.188，in-context 上界≈0.70）")
    t0 = time.time()
    model.train()  # 仅启用训练模式；可微参数仅门控 MLP（主干 frozen）
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
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            log_fn(f"  召回 step {step+1:4d}/{steps} ce={loss.item():.4f}")
    final_acc = kv_acc()
    model.eval()
    log_fn(f"[召回·扩容] 完成：KV 注入答对率 {init_acc:.3f}→{final_acc:.3f}"
           f"（用时 {time.time()-t0:.0f}s，门控参数 {sum(p.numel() for p in gate_params)}）")
    return {"init_acc": init_acc, "final_acc": final_acc,
            "gate_params": sum(p.numel() for p in gate_params)}


# ---------------------------------------------------------------------------
# 主干污染检查（frozen 红线）：排除训练目标（gate_mlp + gate_w/b + 内核）后主干逐位不变
# ---------------------------------------------------------------------------
@torch.no_grad()
def _trainable_ids(model, a_layers):
    ids = set()
    for i in a_layers:
        m = model.layers[i].mixer
        ids.add(id(m.gate_w))
        ids.add(id(m.gate_b))
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
    ap = argparse.ArgumentParser(description="HCA 门控扩容（GatedFusionMLP）重训召回头")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_facts", type=int, default=16)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--hidden", type=int, default=128, help="GatedFusionMLP 隐藏维")
    ap.add_argument("--recall_lr", type=float, default=5e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]

    # 主干全程 frozen（红线）；attach 内核占位（污染检查排除）；挂 GatedFusionMLP（恒等初始化）
    for p in model.parameters():
        p.requires_grad_(False)
    model.attach_kernel()
    for i in a_layers:
        attach_gated_fusion(model.layers[i].mixer, hidden=args.hidden)
    n_mlp = sum(p.numel() for i in a_layers for p in model.layers[i].mixer.gate_mlp.parameters())
    print(f"[train] ckpt={args.ckpt} A_layers={a_layers} hidden={args.hidden} "
          f"门控扩容参数={n_mlp}（585 线性门控 → {n_mlp} MLP，恒等初始化 g=1/3）")

    facts = make_facts(args.n_facts, seed=args.seed)
    print(f"[train] 虚构事实 {len(facts)} 条（先验不存在）")

    logs = []
    def log_fn(msg):
        print(msg, flush=True)
        logs.append(msg)

    vl0 = val_loss(model, dev)
    snap = backbone_snapshot(model, a_layers)
    log_fn(f"[检查] 训练前主干 val loss = {vl0:.4f}（已抓主干权重快照）")

    r = train_recall_gated(model, tok, facts, a_layers, dev,
                           args.steps, args.recall_lr, args.seed, log_fn)

    vl1 = val_loss(model, dev)
    drift = abs(vl1 - vl0)
    unchanged, w_drift = backbone_unchanged(model, snap, a_layers)
    log_fn(f"[检查] 训练后主干 val loss = {vl1:.4f}（门控语义变化是训练目标，非污染）")
    log_fn(f"[检查] 主干权重污染检查：{'✅ 逐位不变' if unchanged else f'⚠️ 漂移 {w_drift:.2e}'}"
           f"（gate_mlp 非主干，frozen 红线）")

    # 保存训练产物
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({i: model.layers[i].mixer.gate_mlp.state_dict() for i in a_layers},
               out / "trained_gate_mlp.pt")
    report = {
        "ckpt": args.ckpt, "n_facts": len(facts), "hidden": args.hidden,
        "baseline_585_recall": 0.1875, "in_context_upper": 0.70,
        "recall_gated": r,
        "val_loss_before": vl0, "val_loss_after": vl1, "val_loss_drift": drift,
        "backbone_unchanged": unchanged, "backbone_weight_drift": w_drift,
        "verdict": {
            "recall": f"KV 注入答对率 {r['init_acc']:.3f}→{r['final_acc']:.3f}"
                      f"（585 线性门控 0.188，in-context 上界≈0.70）",
            "frozen": "主干权重逐位不变（gate_mlp 非主干，红线合规）" if unchanged
                      else f"⚠️ 主干权重漂移 {w_drift:.2e}",
        },
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] report → {out/'report.json'}；gate_mlp → trained_gate_mlp.pt")


if __name__ == "__main__":
    main()
