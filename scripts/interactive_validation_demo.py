"""交互式全链验证系统——确定性剧本 demo（四阶段）+ 共享原语库。

本脚本同时是 scripts/interactive_chat.py 与 tests/test_interactive_validation.py 的
**共享原语库**（不重复造轮子：全链原语经 _ilu 复用 active_inquiry_full_chain_demo，
统一 checkpoint 加载复用 build_unified_checkpoint.load_unified——gate_mlp 复挂坑
已处理）。

四阶段剧本（固定 seed，产出 report.json + validation_panel.png）：
  **Phase A 空白区拒答**：6 个虚构实体问题 → KAL certainty 分布 + Decline 诚实降级率
    （已知判据：certainty≈0 / Decline 16/16，见 unified_full_chain_demo）。
  **Phase B 教学与即时召回**：教同样 6 条事实（求知执行器路径：CrossVerifier
    model_embed 交叉验证 → KnowledgeBlockWriter 累积不覆盖写入 → KV 块收割）→
    写入率 → baseline vs KV 注入召回对照 + HRL 检索命中率（训练同款均值池化协议）。
    已知判据：KV 注入召回 0.625 / 检索 0.938（均 n=16；本 demo n=6 小样本，如实标注）。
  **Phase C CoT/流形探针**：4 条推理 prompt（2 数学 + 2 常识）→ 逐生成步 certainty
    轨迹（ℓ10 读点）+ 末 GDN 层 hidden 的 GridCodeProbe grid_score（token 序号 2D
    网格展开口径，阈值 0.3）+ ThoughtManifold project_3d 轨迹位移范数（与随机游走
    基线对照）。⚠️ 统一 checkpoint 未挂路径积分/流形训练（均为第二阶段独立 pilot
    模块，已训 projector 未存盘）——本阶段是**信号采集口径验证**，网格码预期不成立
    （grid_score<0.3 属预期，非回归）。
  **Phase D 睡眠固化**：Phase B 写入块 + 1 条冲突块 → InquirySleepConsolidation
    （CA1 门调速）→ PROMOTE/QUARANTINE/REJECT 裁决计数 + 逐块理由（与
    consolidator 内部同规则同参数复算）。

诚实边界（报告里照此口径，禁止臆造）：
  - 统一 ckpt 带门控 in-context≈0.25（门控副作用）；KV 注入召回 0.625（n=16）；
    检索 0.938（15/16）；certainty 虚构事实≈0、Decline 16/16；KAL AUROC 0.769。
  - 0.1B 自由对话文本质量差是已知现象——本系统验证**部件信号**，非聊天流畅度。
  - ThoughtManifold 为**未训练**固定随机投影器（pilot 已训 projector 未存盘），
    3D 轨迹统计仅作信号采集口径演示，不作强度判据。

双卡分工：本 demo/评估用计算卡（PRO 4000：CUDA_VISIBLE_DEVICES=1）。
用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/interactive_validation_demo.py
产出：runs/interactive_validation/report.json + runs/interactive_validation/validation_panel.png。
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util as _ilu
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # build_unified_checkpoint / active_inquiry_full_chain_demo
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.inquiry_branch import InquiryRouter  # noqa: E402
from tais_obsidian.model.inquiry_executor import (  # noqa: E402
    CrossVerifier,
    Evidence,
    InquiryExecutor,
    KnowledgeBlockWriter,
)
from tais_obsidian.model.manifold import ThoughtManifold  # noqa: E402
from tais_obsidian.model.path_integration import GridCodeProbe  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.runtime.ca1_gate import SourceCredibilityTracker  # noqa: E402
from tais_obsidian.sleep.consolidator import SleepConsolidator  # noqa: E402
from tais_obsidian.sleep.inquiry_consolidation import (  # noqa: E402
    InquirySleepConsolidation,
)
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

from build_unified_checkpoint import load_unified  # 统一 checkpoint 标准加载（复挂坑处理）  # noqa: E402

# 复用 active_inquiry_full_chain_demo 的全链原语（不重复造轮子；其内部再复用 internalization_e2e）
_spec = _ilu.spec_from_file_location("active_inquiry_full_chain_demo",
                                     ROOT / "scripts" / "active_inquiry_full_chain_demo.py")
_fc = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_fc)
read_certainty = _fc.read_certainty      # KAL P(known) 读出（ℓ10 末 token 均值）
harvest_kv_block = _fc.harvest_kv_block  # draft 文本块 → KV 块收割（运行时零梯度）
_make_facts = _fc._make_facts            # 虚构事实生成（teaching 训练分布对齐）
answer_baseline = _fc.answer_baseline    # 不注入基线（凭先验，应≈0）
answer_correct = _fc.answer_correct      # 宽松判对
answer_with_kv_inject = _fc.answer_with_kv_inject  # KV 注入答（用上 K）
continue_from = _fc.continue_from        # prompt 法续答
hidden = _fc.hidden                      # capture_layers 表征提取

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k_unified"
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_REPORT = "runs/interactive_validation/report.json"
DEFAULT_PANEL = "runs/interactive_validation/validation_panel.png"
READ_LAYER = 10  # KAL sense 读点（kaltruth 微调层，末 GDN 层，G2G2G2A×3 的 ℓ10）
GRID_THRESHOLD = 0.3  # GridCodeProbe 网格码成立阈值（Sargolini 谱系惯例）

# 已知判据口径（诚实边界；报告对照用，禁止臆造）
KNOWN_BASELINES = {
    "kv_recall_n16": 0.625,       # 已训扩容门控 KV 注入召回（n=16，train_recall_gated 协议）
    "retrieval_n16": 0.938,       # 已训 HRL indexer 检索命中率（15/16，均值池化协议）
    "incontext_gated": 0.25,      # 带门控 in-context（门控副作用；拆门控回 0.6875≈teaching）
    "decline_known": "16/16",     # 虚构事实 Decline 诚实降级（unified demo A 组）
    "certainty_fake": "≈0",       # 虚构事实 KAL P(known)
    "kal_auroc": 0.769,           # KAL ℓ10 真值 AUROC（kaltruth final=0.75945 如实保留）
}

# Phase C 推理 prompt（2 数学 + 2 常识；英文为主对齐预训练分布）
REASONING_PROMPTS = [
    {"tag": "math", "prompt": "What is 17 + 25? Answer with a number."},
    {"tag": "math", "prompt": "Compute 6 times 7. Answer with a number."},
    {"tag": "common", "prompt": "What is the capital of France?"},
    {"tag": "common", "prompt": "Water is made of hydrogen and which other element?"},
]


# ---------------------------------------------------------------------------
# 基础加载 / 执行器装配
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(ckpt: str = DEFAULT_CKPT, tok_path: str = DEFAULT_TOK,
                             dev: str = "cuda"):
    """加载统一 checkpoint（load_unified：kernel.* + gate_mlp.* 复挂坑处理）+ tokenizer。"""
    model = load_unified(ckpt, dev)
    model.eval()
    tok = TokenizerIO(tok_path)
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    return model, tok, a_layers


def make_embed_fn(model, tok, a_layers, dev):
    """模型首 CSA 层均值隐藏态作语义 embed（CrossVerifier/CA1 门先验一致性用，
    与 unified demo 同策略——监测/执行分置：只读 detach）。"""
    def model_embed(text: str) -> torch.Tensor:
        return hidden(model, tok, text, a_layers[0], dev)[0].mean(0).float()
    return model_embed


def make_executor(model, tok, a_layers, dev, store: BlockStore):
    """求知执行器（CrossVerifier 用 model_embed 语义验证；KnowledgeBlockWriter
    累积不覆盖版本化写入，namespace="inquiry" draft 隔离）。"""
    model_embed = make_embed_fn(model, tok, a_layers, dev)
    executor = InquiryExecutor(
        blockstore=store, verifier=CrossVerifier(embed_fn=model_embed),
        ask_fn=None, tool_fn=None,
        writer=KnowledgeBlockWriter(tier="L1"), namespace="inquiry")
    return executor, model_embed


def make_manifold(d_model: int, seed: int = 20260731) -> ThoughtManifold:
    """构造 ThoughtManifold（**未训练**固定随机投影器，固定 seed 保证可复现；
    pilot 已训 projector 未存盘——3D 轨迹统计仅作信号采集口径演示）。"""
    st = torch.random.get_rng_state()
    torch.manual_seed(seed)
    m = ThoughtManifold(d_model)
    torch.random.set_rng_state(st)
    return m


# ---------------------------------------------------------------------------
# /teach 路径：自由文本事实 → Q/A 推导（quiz 判对锚）
# ---------------------------------------------------------------------------
def derive_qa(fact_text: str) -> dict:
    """从自由文本事实推导 quiz 锚点（自动口径，如实标注）：
      K = 事实原文；Q = 复述提示；A = 末句（宽松判对锚——baseline 凭先验答不出末句，
      注入后模型可复述 K 内容命中）。显式 quiz 请用 "K | Q | A" 三段式（interactive_chat）。
    """
    text = fact_text.strip().rstrip("。.!！?？")
    if not text:
        raise ValueError("空事实文本")
    tail = re.split(r"[，,；;：:]", text)[-1].strip()
    answer = tail if len(tail) >= 2 else text
    import hashlib
    slug = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]  # 确定性（非内建 hash）
    return {"K": text, "Q": "请复述你刚学到的知识。", "A": answer, "entity": slug}


# ---------------------------------------------------------------------------
# 教学（求知执行器写入 → KV 收割）与召回评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def teach_facts(model, tok, facts, store, executor, router, dev, a_layers,
                demo_certainty: float = 0.55) -> list[dict]:
    """B 组可学习区演示路径（与 unified demo 同口径，如实标注 demo_certainty 占位）：
    逐条 求知路由（Ask/CallTool）→ 执行器（CrossVerifier 交叉验证 → 写入 draft 块）
    → 收割 KV 块（运行时零梯度，不动权重）。真实 KAL 对虚构事实判 certainty≈0
    （完全空白区 Decline），为演示**执行→写入→实时→固化**完整链，路由 certainty
    取可学习区中值 0.55 占位（正式应由 KAL 对"半熟"问题读出）。
    """
    taught = []
    for i, f in enumerate(facts):
        priority = 0.6 if i % 2 == 0 else 0.2
        decision = router.decide(demo_certainty, hrl_hit=False, priority=priority)
        executor.ask_fn = (lambda _K: (lambda q: _K))(f["K"])   # AskQuestion→用户给 K（mock）
        executor.tool_fn = (lambda _K: (lambda q: _K))(f["K"])  # CallTool→检索得 K（mock）
        got = executor(decision)  # 执行→CrossVerifier→写入（draft，累积不覆盖）
        kv = None
        if got:
            kv = harvest_kv_block(model, tok, store, f"fact/{f['entity']}", f["K"], a_layers, dev)
        taught.append({"fact": f, "written": bool(got), "action": decision.action.value, "kv": kv})
    return taught


@torch.no_grad()
def eval_recall(model, tok, taught, dev, a_layers, max_new: int = 8) -> dict:
    """对已写入事实逐条：HRL 检索 top-1（训练同款均值池化协议）+ baseline vs KV 注入对照。"""
    written = [t for t in taught if t["written"] and t["kv"] is not None]
    kvs = [t["kv"] for t in written]
    per_fact = []
    if written:
        cand_repr = torch.cat([kv["repr"] for kv in kvs], dim=1).to(dev)  # [1,N,d]
        for j, t in enumerate(written):
            f = t["fact"]
            # 检索判据协议对齐 train_retrieval_recall（均值池化 query×候选块 repr）
            q_repr = hidden(model, tok, f["Q"], a_layers[0], dev)[0].mean(0, keepdim=True).unsqueeze(0)
            scores = model.kernel.route_candidates(q_repr, cand_repr, k=None, detach_input=True)[0, -1]
            top = int(scores.topk(1).indices[0])
            g_base = answer_baseline(model, tok, f, dev, max_new)
            g_kv = answer_with_kv_inject(model, tok, f, t["kv"], a_layers, dev, max_new)
            per_fact.append({
                "entity": f["entity"], "Q": f["Q"], "A": f["A"],
                "retrieval_hit": bool(top == j),
                "baseline_ok": bool(answer_correct(g_base, f["A"])),
                "kv_ok": bool(answer_correct(g_kv, f["A"])),
                "baseline_gen": g_base, "kv_gen": g_kv,
            })
    n = max(len(per_fact), 1)
    return {
        "n_taught": len(taught), "n_written": len(written),
        "write_rate": len(written) / max(len(taught), 1),
        "retrieval_hit": sum(p["retrieval_hit"] for p in per_fact) / n,
        "acc_baseline": sum(p["baseline_ok"] for p in per_fact) / n,
        "acc_kv_inject": sum(p["kv_ok"] for p in per_fact) / n,
        "per_fact": per_fact,
        "retrieval_protocol": "train_retrieval_recall 同款（均值池化 query×候选块 repr）",
    }


# ---------------------------------------------------------------------------
# Phase A：空白区拒答
# ---------------------------------------------------------------------------
@torch.no_grad()
def phase_a_blank_decline(model, tok, facts, dev, router: InquiryRouter | None = None) -> dict:
    """虚构实体问题 → KAL certainty + 求知路由 Decline 诚实降级率。"""
    router = router or InquiryRouter()
    per_item = []
    for f in facts:
        cert = read_certainty(model, tok, f["Q"], dev, layer=READ_LAYER)
        decision = router.decide(cert, hrl_hit=False, priority=0.6)
        per_item.append({"Q": f["Q"], "certainty": cert,
                         "action": decision.action.value,
                         "decline_message": decision.ask_token if decision.action.value == "Decline" else None})
    certs = [p["certainty"] for p in per_item]
    n_decline = sum(1 for p in per_item if p["action"] == "Decline")
    return {
        "per_item": per_item,
        "certainties": certs,
        "certainty_mean": float(np.mean(certs)) if certs else 0.0,
        "certainty_min": float(min(certs)) if certs else 0.0,
        "certainty_max": float(max(certs)) if certs else 0.0,
        "n_decline": n_decline, "n_total": len(facts),
        "decline_rate": n_decline / max(len(facts), 1),
    }


# ---------------------------------------------------------------------------
# Phase C：CoT/流形探针
# ---------------------------------------------------------------------------
def _pik_known(model, h) -> float:
    """从 capture 的 hidden [B,T,d] 读 KAL P(known)（末 token 均值；只读 detach）。"""
    if isinstance(h, dict):
        h = h["content"]  # PM-stream 配置取内容流（kal.read_point 同语义）
    sense = model.kernel.sense(h)
    probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)
    return float(probs[:, 0].mean().item())


@torch.no_grad()
def certainty_trace(model, tok, prompt: str, dev, layer: int = READ_LAYER,
                    max_new: int = 16) -> tuple[list[float], str]:
    """逐生成步 certainty 轨迹（ℓ10 读点，argmax 续答，确定性）。返回 (轨迹, 生成文本)。"""
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    if ids.shape[1] > model.config.max_seq - max_new - 8:
        raise ValueError(f"prompt 长度 {ids.shape[1]} tokens 超出 max_seq={model.config.max_seq} 余量")
    certs, out = [], []
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache, caps = model(ids, capture_layers=[layer])
        certs.append(_pik_known(model, caps[layer]))
        for _ in range(max_new):
            nxt = int(logits[:, -1, :].float().argmax(-1).item())
            if nxt == tok.eot_id:
                break
            out.append(nxt)
            x = torch.tensor([[nxt]], device=dev)
            logits, cache, caps = model(x, cache, capture_layers=[layer])
            certs.append(_pik_known(model, caps[layer]))
    return certs, tok.decode(out)


def grid_positions_2d(n_tokens: int) -> torch.Tensor:
    """token 序号 → 2D 网格展开位置 [N,2]（口径：pos[i]=(i%W, i//W)，W=ceil(sqrt(N))，
    行主序蛇形不启用——如实标注：这是 token 序号的规则 2D 嵌入，非物理空间）。"""
    w = max(1, math.ceil(math.sqrt(n_tokens)))
    return torch.tensor([[i % w, i // w] for i in range(n_tokens)], dtype=torch.float32)


@torch.no_grad()
def probe_hidden_signals(model, tok, text: str, dev, manifold: ThoughtManifold,
                         layer: int = READ_LAYER, max_new: int = 16,
                         perm_seed: int = 0, n_perm: int = 16) -> dict:
    """对模型续答过程采集探针信号（只读，零副作用）：
      - KAL certainty（ℓ10 P(known)，对输入 text）；
      - 末 GDN 层 hidden 轨迹（text+续答）经 ThoughtManifold.project → project_3d
        的路径长度/位移范数/平均步长 + 随机游走基线（同行列随机置换的端点位移均值，
        位移/基线比值=有序性指标）；
      - GridCodeProbe grid_score（全部维度均值，位置口径=token 序号 2D 网格展开，
        阈值 0.3；统一 ckpt 未挂路径积分训练，网格码预期不成立——口径验证非强度判据）。
    """
    cert = read_certainty(model, tok, text, dev, layer=layer)
    ids = torch.tensor([tok.encode(text)], device=dev)
    if ids.shape[1] > model.config.max_seq - max_new - 8:
        raise ValueError(
            f"prompt 长度 {ids.shape[1]} tokens 超过 max_seq={model.config.max_seq} 余量"
            f"（须为续答留 {max_new}+8）；请缩短文本")
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        logits, cache = model(ids)
    gen = continue_from(model, tok, logits, cache, dev, max_new)
    full = text + gen
    h = hidden(model, tok, full, layer, dev)
    if isinstance(h, dict):
        h = h["content"]
    h = h[0].float().cpu()  # [T,d]，末 GDN 层 hidden 轨迹
    t = h.shape[0]
    # ① ThoughtManifold 3D 轨迹统计（未训练固定投影器口径）
    with torch.no_grad():
        coords = manifold.project(h)          # [T,manifold_dim]
        xyz = manifold.project_3d(coords)     # [T,3]
    steps = (xyz[1:] - xyz[:-1]).norm(dim=-1)  # [T-1]
    path_length = float(steps.sum()) if steps.numel() else 0.0
    displacement = float((xyz[-1] - xyz[0]).norm()) if t >= 2 else 0.0
    mean_step = float(steps.mean()) if steps.numel() else 0.0
    # 随机游走基线：同一轨迹点集随机置换后端点位移均值（破坏时序的对照）
    rng = np.random.default_rng(perm_seed)
    xyz_np = xyz.numpy()
    rand_disps = []
    for _ in range(max(n_perm, 1)):
        perm = rng.permutation(t)
        rand_disps.append(float(np.linalg.norm(xyz_np[perm[-1]] - xyz_np[perm[0]])))
    rand_mean = float(np.mean(rand_disps))
    # ② GridCodeProbe（token 序号 2D 网格展开口径；全维度均值）
    pos = grid_positions_2d(t)
    probe = GridCodeProbe()
    gs, top_dims, grid_hit = probe.probe(h, pos, top_k=8)
    return {
        "certainty": cert, "gen": gen, "n_tokens": t,
        "path_length": path_length, "displacement": displacement, "mean_step": mean_step,
        "random_walk_displacement_mean": rand_mean,
        "orderliness": displacement / (rand_mean + 1e-8),
        "grid_score": float(gs), "grid_hit": bool(grid_hit),
        "grid_threshold": GRID_THRESHOLD, "grid_top_dims": top_dims.tolist(),
        "grid_positions_note": "token 序号 2D 网格展开 pos[i]=(i%W,i//W)，W=ceil(sqrt(T))",
        "manifold_note": "未训练 ThoughtManifold 固定随机投影器（pilot 已训 projector 未存盘），"
                         "3D 统计仅作信号采集口径演示，不作强度判据",
    }


@torch.no_grad()
def phase_c_probes(model, tok, prompts: list[dict], dev, manifold: ThoughtManifold,
                   trace_new: int = 16, seed: int = 0) -> dict:
    """4 条推理 prompt：certainty 轨迹 + grid_score + 3D 轨迹位移（随机游走对照）。"""
    per_prompt = []
    for i, p in enumerate(prompts):
        certs, gen = certainty_trace(model, tok, p["prompt"], dev, max_new=trace_new)
        sig = probe_hidden_signals(model, tok, p["prompt"], dev, manifold,
                                   max_new=trace_new, perm_seed=seed + i)
        per_prompt.append({
            "tag": p["tag"], "prompt": p["prompt"], "gen": gen,
            "certainty_trace": certs,
            "certainty_start": certs[0] if certs else 0.0,
            "certainty_end": certs[-1] if certs else 0.0,
            "certainty_mean": float(np.mean(certs)) if certs else 0.0,
            "grid_score": sig["grid_score"], "grid_hit": sig["grid_hit"],
            "path_length": sig["path_length"], "displacement": sig["displacement"],
            "mean_step": sig["mean_step"],
            "random_walk_displacement_mean": sig["random_walk_displacement_mean"],
            "orderliness": sig["orderliness"], "n_tokens": sig["n_tokens"],
        })
    gs_mean = float(np.mean([p["grid_score"] for p in per_prompt])) if per_prompt else 0.0
    return {
        "per_prompt": per_prompt, "grid_score_mean": gs_mean,
        "grid_hit_any": any(p["grid_hit"] for p in per_prompt),
        "grid_threshold": GRID_THRESHOLD,
        "note": "统一 ckpt 未挂路径积分/流形训练——grid_score<0.3 属预期（口径验证非强度判据）；"
                "位置口径=token 序号 2D 网格展开；manifold=未训练固定投影器",
    }


# ---------------------------------------------------------------------------
# Phase D：睡眠固化（CA1 门裁决 + 逐块理由）
# ---------------------------------------------------------------------------
def make_cross_verify_fn(executor):
    """CrossVerifier 二次复核回调（CA1 门 v1.1 边缘带 RE_VERIFY 用）。

    对块内容重跑交叉验证：verified 且无不决冲突 → True。检索证据强度（usage/HRL
    命中计数）已在证据感知共识公式中计入，本回调提供第二路独立验证信号
    （绝不裸自我修正：复核来自 CrossVerifier 外部验证，非模型自我判断）。
    """
    def verify_fn(item) -> bool:
        ev = Evidence(content=str(item.content), source=item.source or "doc")
        verified, _consist, conflict = executor.verifier.verify(ev, executor._knowledge)
        return bool(verified) and not conflict
    return verify_fn


def _ca1_verdict_reason(item, verdict: str, salience_boost: int, rv: dict | None) -> str:
    """按 CA1 门规则分支（按序，前者优先）+ v1.1 边缘带补验证日志生成逐块理由。"""
    if item.belief_drift > 0.5:
        return (f"belief_drift={item.belief_drift:.2f}>0.5：冲突未决→QUARANTINE "
                f"保留双方标分歧（MemoryGraft 拦截，不静默覆盖；自适应不触碰漂移拦截）")
    eff_usage = item.usage_count + salience_boost
    if eff_usage < 10 or not item.regression_ok:
        return (f"有效 usage={eff_usage}<10 或回归未过"
                f"（regression_ok={item.regression_ok}）→REJECT（验证门）")
    if rv is not None:  # 进入边缘带 [0.62, 0.7) 的补验证路径
        if rv.get("passed") is None:
            return (f"teacher_consensus={rv['consensus_before']:.3f}∈[0.62,0.7) 边缘带，"
                    f"但无补验证回调→fail-closed REJECT（同旧行为）")
        if rv.get("passed"):
            return (f"边缘带 RE_VERIFY：consensus={rv['consensus_before']:.3f}∈[0.62,0.7) "
                    f"→ CrossVerifier 二次复核通过+有界加成 → {rv['consensus_after']:.3f}"
                    f"→{rv['verdict']}（补验证修复信源可信度边缘效应）")
        return (f"边缘带 RE_VERIFY：consensus={rv['consensus_before']:.3f}∈[0.62,0.7) "
                f"→ 二次复核未过（验证通过率摊薄至 {rv['consensus_after']:.3f}）→REJECT"
                f"（补验证不放水）")
    if verdict == "PROMOTE":
        return (f"usage={item.usage_count}≥10 且回归通过 且 "
                f"consensus={item.teacher_consensus:.3f}≥0.7→PROMOTE（一致快固化，同化）")
    return (f"teacher_consensus={item.teacher_consensus:.3f}<0.62 边缘带下沿：证据不足"
            f"→REJECT（弱证据仍弱更新，不进补验证带）")


def sleep_consolidate(store: BlockStore, model_embed, namespace: str = "inquiry",
                      usage_count: int = 12, saliency: float = 1.0,
                      regression_ok: bool = True, verify_fn=None, cred_tracker=None):
    """对 namespace 下 draft 块跑睡眠固化；返回 (ConsolidateReport, 逐块裁决理由 list)。

    v1.1 起逐块最终裁决以 consolidator 报告为唯一来源（report.verdicts /
    reverify_log），不再"同规则同参数复算"CA1 门（旧口径报告只有计数才需复算）：
      - 证据感知共识 = 0.85·先验一致性×信任度 + 0.10·usage/20 + 0.05·验证通过率；
      - consensus ∈ [0.62, 0.7) 边缘带 → RE_VERIFY 补验证（verify_fn=CrossVerifier
        二次复核；None 时 fail-closed 按 REJECT 落账，与旧行为一致）；
      - cred_tracker：信源可信度在线学习（None=静态 payload 值，向后兼容）。
    固化不改块 draft 标记（consolidate 只读+SHY 归一化 item 副本），重复调用幂等。
    """
    isc = InquirySleepConsolidation(embed_fn=model_embed, credibility_tracker=cred_tracker)
    salience_boost = int(round(max(0.0, saliency - 1.0) * 4.0))  # consolidator salience_scale=4.0
    # 逐块证据分量快照（理由展示用；最终裁决以 consolidator report.verdicts 为准）
    snapshots: dict = {}
    for tier in ("L0", "L1", "L2"):
        od = store._store.get(tier)  # 只读遍历（不触发 get 的 usage/recency 副作用）
        if od is None:
            continue
        for bid, payload in od.items():
            if not isinstance(payload, dict):
                continue
            if not str(bid).startswith(namespace + "/"):
                continue
            if not payload.get("draft", False):
                continue
            item = isc.adapter.to_w0item(bid, payload, None, saliency=saliency,
                                         usage_count=usage_count)
            item.regression_ok = item.regression_ok and regression_ok
            snapshots[bid] = (item, payload)
    consolidator = SleepConsolidator(reverify_fn=verify_fn)
    report = isc.consolidate_inquiry_blocks(
        store, consolidator, prior_knowledge=None, namespace=namespace,
        usage_count=usage_count, saliency=saliency, regression_ok=regression_ok)
    reverify_by_id = {e["block_id"]: e for e in report.reverify_log}
    per_block = []
    for bid, (item, payload) in snapshots.items():
        verdict = report.verdicts.get(bid, "REJECT")
        rv = reverify_by_id.get(bid)
        per_block.append({
            "block_id": bid, "verdict": verdict,
            "reason": _ca1_verdict_reason(item, verdict, salience_boost, rv),
            "content": payload.get("content", "")[:80],
            "source": payload.get("source"), "conflict": bool(payload.get("conflict", False)),
            "teacher_consensus": float(item.teacher_consensus),
            "belief_drift": float(item.belief_drift),
            "reverify": rv,
        })
    return report, per_block


@torch.no_grad()
def phase_d_sleep(model, tok, store, executor, model_embed, facts, dev, a_layers,
                  add_conflict: bool = True) -> dict:
    """Phase B 写入块（+1 条冲突块）→ 睡眠固化 CA1 门裁决计数 + 逐块理由。

    v1.1 自适应 CA1 门：接 CrossVerifier 补验证回调（边缘带 RE_VERIFY）+
    信源可信度在线学习 tracker（报告前后对照）。
    """
    conflict_block = None
    if add_conflict and facts:
        f0 = facts[0]
        wrong = f"The {f0['entity']} engine runs on refined WATER."  # 与 K 矛盾
        ev_bad = Evidence(content=wrong, source="web")  # web 弱可信度
        verified_b, consist_b, _ = executor.verifier.verify(ev_bad, executor._knowledge)
        ev_bad.verified = verified_b
        bid_bad = executor.writer.write(ev_bad, store, namespace="inquiry",
                                        conflict=True, consistency=consist_b)  # 强制标冲突
        conflict_block = {"block_id": bid_bad, "content": wrong, "conflict_flag": True}
    tracker = SourceCredibilityTracker()
    cred_before = dict(tracker.cred)
    report, per_block = sleep_consolidate(
        store, model_embed, verify_fn=make_cross_verify_fn(executor), cred_tracker=tracker)
    return {
        "n_clusters": report.n_clusters, "n_practiced": report.n_practiced,
        "n_promoted": report.n_promoted, "n_quarantined": report.n_quarantined,
        "n_rejected": report.n_rejected, "promoted_ids": report.promoted_ids,
        "per_block": per_block, "conflict_block": conflict_block,
        "n_reverified": report.n_reverified, "reverify_log": report.reverify_log,
        "credibility_before": cred_before, "credibility_after": dict(tracker.cred),
    }


# ---------------------------------------------------------------------------
# 报告装配 + 判据对照
# ---------------------------------------------------------------------------
def build_report(pa: dict, pb: dict, pc: dict, pd: dict, meta: dict) -> dict:
    """汇总四阶段 + 与已知判据口径对照（异常如实标注，禁止臆造）。"""
    cm = pa["certainty_mean"]
    dr = pa["decline_rate"]
    wr = pb["write_rate"]
    ab, ak, rh = pb["acc_baseline"], pb["acc_kv_inject"], pb["retrieval_hit"]
    gm = pc["grid_score_mean"]
    nv = pd["n_promoted"] + pd["n_quarantined"] + pd["n_rejected"]
    criteria = {
        "phase_a_certainty_low": {
            "value": cm, "expected": "≈0（虚构事实完全空白区）",
            "pass": bool(cm < 0.5),
            "note": "方向阈 0.5（unified demo pk_fake<0.5 口径；非 AUROC 强度）"},
        "phase_a_decline_rate": {
            "value": dr, "expected": "≈1.0（已知 16/16）",
            "pass": bool(dr >= 0.99),
            "note": "完全空白区 Decline 诚实降级红线"},
        "phase_b_write_rate": {
            "value": wr, "expected": "1.0（CrossVerifier 验证通过即写入）",
            "pass": bool(wr >= 0.99), "note": "累积不覆盖版本化写入"},
        "phase_b_baseline_zero": {
            "value": ab, "expected": "≈0（凭先验答虚构事实答不出）",
            "pass": bool(ab <= 0.2), "note": "宽松判对"},
        "phase_b_kv_recall": {
            "value": ak, "expected": "≈0.625（n=16 判据；本 demo n=6 小样本）",
            "pass": bool(ak > ab),
            "note": "小样本判据=注入>基线；0.625 是 n=16 强度，n=6 方差大不臆造复现"},
        "phase_b_retrieval": {
            "value": rh, "expected": "≥0.9（0.938=15/16 同协议）",
            "pass": bool(rh >= 0.9), "note": "训练同款均值池化协议"},
        "phase_c_grid_no_code": {
            "value": gm, "expected": "<0.3（统一 ckpt 未挂路径积分训练，网格码应不成立）",
            "pass": bool(gm < GRID_THRESHOLD),
            "note": "口径验证：grid_score≥0.3 才算网格码成立；<0.3 是预期非回归"},
        "phase_d_verdicts_present": {
            "value": nv, "expected": "≥1（固化有裁决产出）",
            "pass": bool(nv >= 1), "note": "PROMOTE/QUARANTINE/REJECT 总计"},
    }
    honest_notes = [
        f"n_facts={meta['n_facts']} 小样本：已知判据 0.625/0.938 均为 n=16 口径，本 demo 数值"
        f"与之有样本量偏差，判定以方向性（注入>基线、检索≥0.9）为准",
        "ThoughtManifold 为未训练固定随机投影器（pilot 已训 projector 未存盘）——Phase C "
        "3D 轨迹统计仅作信号采集口径演示，不作强度判据",
        "统一 ckpt 未挂路径积分训练（第二阶段独立 pilot 模块）——grid_score<0.3 属预期",
        "0.1B 自由生成文本质量差是已知现象——本系统验证部件信号，非聊天流畅度",
        "带门控 in-context≈0.25 是召回训练门控副作用（拆门控回 0.6875≈teaching），"
        "与本 demo 的 KV 注入召回（HCA 通路）是不同口径，不矛盾",
    ]
    return {
        "meta": meta, "known_baselines": KNOWN_BASELINES,
        "phase_a_blank_decline": pa, "phase_b_teach_recall": pb,
        "phase_c_probes": pc, "phase_d_sleep": pd,
        "criteria": criteria,
        "all_pass": all(c["pass"] for c in criteria.values()),
        "honest_notes": honest_notes,
    }


# ---------------------------------------------------------------------------
# 图表面板（2×2，IBM Carbon White theme 色板）
# ---------------------------------------------------------------------------
def make_panel(report: dict, out_png: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    _CJK = ["Microsoft YaHei", "Noto Sans CJK SC", "PingFang SC", "SimHei"]
    _installed = {f.name for f in font_manager.fontManager.ttflist}
    for _f in _CJK:
        if _f in _installed:
            plt.rcParams["font.sans-serif"] = [_f, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # Carbon 色板（对齐 scripts/_make_report_assets.py 指定子集）
    BLUE, GREEN, TEAL, MAGENTA, INK, LINE = (
        "#0f62fe", "#24a148", "#009d9a", "#d12771", "#161616", "#e0e0e0")

    pa, pb, pc, pd = (report["phase_a_blank_decline"], report["phase_b_teach_recall"],
                      report["phase_c_probes"], report["phase_d_sleep"])
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=150)
    fig.patch.set_facecolor("white")

    # ── A：certainty 分布（虚构实体，期望≈0）──
    ax = axes[0, 0]
    certs = pa["certainties"]
    ax.bar(range(len(certs)), certs, color=MAGENTA, width=0.6)
    for i, v in enumerate(certs):  # 数值标注（≈0 时柱不可见，须有数字才读得出）
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.axhline(0.5, color=INK, ls="--", lw=1)
    ax.text(len(certs) - 0.5, 0.53, "方向阈 0.5", fontsize=8, color=INK, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title("Phase A 空白区 KAL certainty 分布（虚构实体，期望≈0）", fontsize=11, color=INK)
    ax.set_xlabel("虚构问题 #", fontsize=9)
    ax.set_ylabel("P(known)", fontsize=9)
    ax.text(0.03, 0.90, f"Decline 诚实降级率 = {pa['decline_rate']:.2f}"
            f"（{pa['n_decline']}/{pa['n_total']}）",
            transform=ax.transAxes, fontsize=9, color=INK)

    # ── B：baseline vs KV 注入召回对照 ──
    ax = axes[0, 1]
    vals = [pb["acc_baseline"], pb["acc_kv_inject"]]
    bars = ax.bar(["不注入基线", "KV 注入召回"], vals, color=[BLUE, GREEN], width=0.5)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=10, color=INK)
    ax.axhline(KNOWN_BASELINES["kv_recall_n16"], color=GREEN, ls=":", lw=1)
    ax.text(1.45, KNOWN_BASELINES["kv_recall_n16"] + 0.02, "已训判据 0.625（n=16）",
            fontsize=8, color=GREEN, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Phase B 教学即时召回（写入率 {pb['write_rate']:.2f}，"
                 f"检索命中 {pb['retrieval_hit']:.3f}）", fontsize=11, color=INK)
    ax.set_ylabel("答对率", fontsize=9)

    # ── C：certainty 轨迹（4 条推理 prompt）──
    ax = axes[1, 0]
    colors = [BLUE, TEAL, MAGENTA, GREEN]
    for i, p in enumerate(pc["per_prompt"]):
        tr = p["certainty_trace"]
        ax.plot(range(len(tr)), tr, color=colors[i % 4], lw=1.6,
                marker="o", ms=2.5, label=f"P{i+1} {p['tag']}")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Phase C 推理 certainty 轨迹（ℓ{READ_LAYER} 读点；"
                 f"grid_score 均值 {pc['grid_score_mean']:.3f}<0.3 预期）",
                 fontsize=11, color=INK)
    ax.set_xlabel("生成步", fontsize=9)
    ax.set_ylabel("P(known)", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="best")

    # ── D：睡眠固化 CA1 门裁决计数 ──
    ax = axes[1, 1]
    vals = [pd["n_promoted"], pd["n_quarantined"], pd["n_rejected"]]
    bars = ax.bar(["PROMOTE", "QUARANTINE", "REJECT"], vals,
                  color=[GREEN, MAGENTA, INK], width=0.5)
    for r, v in zip(bars, vals):
        ax.text(r.get_x() + r.get_width() / 2, v + 0.05, str(v),
                ha="center", fontsize=10, color=INK)
    ax.set_title("Phase D 睡眠固化 CA1 门裁决计数", fontsize=11, color=INK)
    ax.set_ylabel("块数", fontsize=9)
    ax.set_ylim(0, max(vals + [1]) + 1)

    for ax in axes.flat:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(LINE)
        ax.tick_params(colors=INK, labelsize=8)
        ax.grid(axis="y", color=LINE, lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    fig.suptitle("TAIS Obsidian 0.1B 交互式全链验证面板（统一 checkpoint 已训强度）",
                 fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 主流程（四阶段剧本）
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="交互式全链验证系统——确定性四阶段剧本 demo")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--panel", default=DEFAULT_PANEL)
    ap.add_argument("--n_facts", type=int, default=6)
    ap.add_argument("--max_new", type=int, default=8, help="召回评估续答长度")
    ap.add_argument("--trace_new", type=int, default=16, help="Phase C 轨迹续答长度")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, tok, a_layers = load_model_and_tokenizer(args.ckpt, args.tokenizer, dev)
    manifold = make_manifold(model.config.d_model)
    facts = _make_facts(args.n_facts, seed=args.seed)
    print("=" * 70)
    print("【交互式全链验证 demo】统一 checkpoint 四阶段剧本（确定性 seed）")
    print("=" * 70)
    print(f"[demo] ckpt={args.ckpt} A_layers={a_layers} n_facts={len(facts)} seed={args.seed}")

    store = BlockStore()
    router = InquiryRouter()
    executor, model_embed = make_executor(model, tok, a_layers, dev, store)

    # ── Phase A 空白区拒答 ──
    print("\n" + "=" * 70)
    print("【Phase A 空白区拒答】虚构实体问题 → certainty 分布 + Decline 率")
    print("=" * 70)
    pa = phase_a_blank_decline(model, tok, facts, dev, router)
    for p in pa["per_item"]:
        print(f"  certainty={p['certainty']:.3f} → {p['action']}  Q: {p['Q']}")
    print(f"  → certainty 均值 {pa['certainty_mean']:.3f}（期望≈0）；"
          f"Decline {pa['n_decline']}/{pa['n_total']}（已知判据 16/16）")

    # ── Phase B 教学与即时召回 ──
    print("\n" + "=" * 70)
    print("【Phase B 教学与即时召回】求知执行器写入 → KV 收割 → baseline vs 注入")
    print("=" * 70)
    taught = teach_facts(model, tok, facts, store, executor, router, dev, a_layers)
    pb = eval_recall(model, tok, taught, dev, a_layers, args.max_new)
    print(f"  写入率 {pb['write_rate']:.2f}（{pb['n_written']}/{pb['n_taught']}）")
    print(f"  HRL 检索 top-1 命中率 = {pb['retrieval_hit']:.3f}（判据 ≥0.9，0.938=15/16）")
    print(f"  不注入基线 = {pb['acc_baseline']:.3f}（应≈0）")
    print(f"  KV 注入召回 = {pb['acc_kv_inject']:.3f}（n=16 判据 0.625；n={pb['n_written']} 小样本）")
    for p in pb["per_fact"][:3]:
        print(f"    [{p['entity']}] hit={p['retrieval_hit']} base={p['baseline_ok']} "
              f"kv={p['kv_ok']} | kv_gen: {p['kv_gen'][:40]!r}")

    # ── Phase C CoT/流形探针 ──
    print("\n" + "=" * 70)
    print("【Phase C CoT/流形探针】certainty 轨迹 + grid_score + 3D 轨迹位移")
    print("=" * 70)
    pc = phase_c_probes(model, tok, REASONING_PROMPTS, dev, manifold,
                        trace_new=args.trace_new, seed=args.seed)
    for i, p in enumerate(pc["per_prompt"]):
        print(f"  P{i+1}[{p['tag']}] cert {p['certainty_start']:.3f}→{p['certainty_end']:.3f} "
              f"| grid={p['grid_score']:.3f}（阈值0.3）| 位移 {p['displacement']:.2f} "
              f"vs 随机游走 {p['random_walk_displacement_mean']:.2f}"
              f"（有序性 {p['orderliness']:.2f}）")
    print(f"  grid_score 均值 {pc['grid_score_mean']:.3f} < 0.3：网格码不成立（预期，口径验证）")

    # ── Phase D 睡眠固化 ──
    print("\n" + "=" * 70)
    print("【Phase D 睡眠固化】draft 块 + 冲突块 → CA1 门裁决")
    print("=" * 70)
    pd = phase_d_sleep(model, tok, store, executor, model_embed, facts, dev, a_layers)
    print(f"  固化报告：分簇={pd['n_clusters']} 提取={pd['n_practiced']} "
          f"PROMOTE={pd['n_promoted']} QUARANTINE={pd['n_quarantined']} REJECT={pd['n_rejected']}"
          f"（边缘带补验证 {pd['n_reverified']} 块）")
    for b in pd["per_block"]:
        print(f"    [{b['verdict']}] {b['block_id']}（{b['source']}）: {b['reason']}")
    ca, cb = pd.get("credibility_after", {}), pd.get("credibility_before", {})
    if ca:
        delta = ", ".join(f"{k} {cb.get(k, 0):.2f}→{v:.2f}" for k, v in sorted(ca.items()))
        print(f"  信源可信度在线学习（本轮固化后 EMA）：{delta}")

    # ── 汇总 + 面板 ──
    meta = {"ckpt": args.ckpt, "tokenizer": args.tokenizer, "n_facts": len(facts),
            "seed": args.seed, "max_new": args.max_new, "trace_new": args.trace_new,
            "read_layer": READ_LAYER, "device": dev,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    report = build_report(pa, pb, pc, pd, meta)
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_panel(report, args.panel)

    print("\n" + "=" * 70)
    print("【判据对照汇总】（已知判据口径，异常如实标注）")
    print("=" * 70)
    for name, c in report["criteria"].items():
        print(f"  [{'✅' if c['pass'] else '⚠️'}] {name}: {c['value']}（期望 {c['expected']}）")
    print(f"  全部判据: {'✅' if report['all_pass'] else '⚠️ 有偏差（见 honest_notes）'}")
    print(f"[save] report → {rep}")
    print(f"[save] panel  → {args.panel}")


if __name__ == "__main__":
    main()
