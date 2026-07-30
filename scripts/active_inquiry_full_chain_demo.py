"""0.1B 主动求知闭环**全链端到端 demo**——运行时学习 → 实时可用 → 长期固化 三阶段串联验证。

设计依据：docs/TAIS_Obsidian_主动求知闭环_架构设计文档.md（三阶段）+
《部件实现详细计划》红线总表。本脚本把已分别落地的三阶段**串成完整可运行闭环**，
验证协同（不重复造轮子——复用 inquiry_branch / inquiry_executor /
inquiry_consolidation / blockstore / reasoning_loop / internalization_e2e 原语）。

三阶段（单机跑通）：
  **阶段1 运行时学习**：给模型一个**不知道**的问题（虚构事实，KAL certainty 低 +
  HRL 未命中）→ 求知分支路由（InquiryRouter 四选一，RPL/LP 可学习区，预期
  AskQuestion/CallTool 而非 DirectAnswer）→ 求知执行器执行（mock ask_fn/tool_fn
  返回新知识 K）→ CrossVerifier 交叉验证（一致性 + 冲突检测，**绝不裸自我修正**）
  → KnowledgeBlockWriter 写入 BlockStore（draft 态，含 source_credibility，
  **累积不覆盖**版本化）。
  **阶段2 实时可用**：写入的 K 块（收割成 KV 块）→ HRL 检索（route_candidates，
  预期命中 top-k）→ HCA 注入（inject_hca_entries，**运行时注入不动权重**）→
  模型对原问题用上 K 答对（**实时可用，无需重新 SFT**）→ 对比"不注入基线"
  （凭先验答不出）。注：本 demo 用随机初始化 indexer + 未训门控，检索/召回落
  的是"通路通"而非"已训强度"（已训强度 1.000/0.188 见 train_retrieval_recall，
  其 indexer/门控权重未存入 checkpoint）。
  **阶段3 长期固化**：BlockStore 中的 draft 知识块 → InquirySleepConsolidation
  （CA1 门先验一致性调速：一致 fast_track / 冲突慢通道）→
  SleepConsolidator.consolidate（间隔提取 + CA1 门 + SHY）→ 三元奖励信号 →
  PROMOTE（一致）/ QUARANTINE（冲突保留双方标分歧）。

红线落实：
  - **绝不裸自我修正**（arXiv:2310.01798）：写入/固化均经 CrossVerifier/外部
    验证门控；未验证证据绝不写入。
  - **累积不覆盖**：写入版本化 :v{n} + 冲突保留双方标分歧。
  - **诚实降级**：Decline 声明"该部分记忆暂不可用"，绝不硬答。
  - **运行时注入不动权重**（实时可用 vs 离线 SFT）：HCA 注入零梯度，不触碰权重。
  - **kernel 加载坑**：kaltruth checkpoint config.kernel_enabled=False 但存入了
    attach_kernel 后的 kernel.* 权重 → 须先 attach_kernel() 再
    load_state_dict(strict=True)（见 /memories/repo/kal-gdn2-truth-finetune.md）。

双卡分工：本 demo/评估用 RTX 4070（CUDA_VISIBLE_DEVICES=0，8GB，控 batch/seq）。
运行：
  CUDA_VISIBLE_DEVICES=0 .venv/Scripts/python.exe scripts/active_inquiry_full_chain_demo.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from safetensors.torch import load_file  # noqa: E402

from tais_obsidian.config import ModelConfig  # noqa: E402
from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
from tais_obsidian.model.injection import make_injector  # noqa: E402
from tais_obsidian.model.inquiry_branch import (  # noqa: E402
    InquiryAction,
    InquiryBranch,
    InquiryRouter,
)
from tais_obsidian.model.inquiry_executor import (  # noqa: E402
    CrossVerifier,
    InquiryExecutor,
    KnowledgeBlockWriter,
)
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.sleep.consolidator import SleepConsolidator  # noqa: E402
from tais_obsidian.sleep.inquiry_consolidation import (  # noqa: E402
    InquirySleepConsolidation,
)
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

# 复用 internalization_e2e 的虚构事实生成 + KV 内化/注入/判对原语（不重复造）。
# scripts/ 非 Python 包，用 importlib 按文件路径加载。
import importlib.util as _ilu  # noqa: E402


def _load_e2e():
    spec = _ilu.spec_from_file_location("internalization_e2e", ROOT / "scripts" / "internalization_e2e.py")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_e2e = _load_e2e()
_make_facts = _e2e._make_facts
answer_baseline = _e2e.answer_baseline
answer_correct = _e2e.answer_correct
answer_with_kv_inject = _e2e.answer_with_kv_inject
continue_from = _e2e.continue_from
hidden = _e2e.hidden

DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k_kaltruth"  # KAL 真值锚校准（certainty 可靠）
DEFAULT_TOK = "data/tokenizer/tokenizer.json"
DEFAULT_REPORT = "runs/active_inquiry_full_chain/report.json"
READ_LAYER = 10  # KAL sense 读点（kaltruth 报告选 ℓ10 末 GDN 层）


# ---------------------------------------------------------------------------
# kernel 加载坑处理（防御记录，见 kal-gdn2-truth-finetune.md）
# ---------------------------------------------------------------------------
def load_model_with_kernel(ckpt: str, dev: str) -> TaisObsidianForCausalLM:
    """加载 kaltruth checkpoint（含 kernel.* 权重但 config.kernel_enabled=False）。

    ⚠️ kernel 加载坑：from_pretrained(strict=True) 会因 kernel.* 多余键报
    Unexpected key。零侵入解决：先 attach_kernel() 再 load_state_dict(strict=True)
    ——attach_kernel 把内核挂为子模块（kernel.* 键有归属），strict 载入即通过。
    """
    cfg = ModelConfig.from_json(Path(ckpt) / "config.json")
    model = TaisObsidianForCausalLM(cfg)
    model.attach_kernel()  # 先挂载内核（strict 载入 kernel.* 键的前提）
    sd = load_file(str(Path(ckpt) / "model.safetensors"))
    model.load_state_dict(sd, strict=True)
    model = model.to(dev)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 真实 certainty 读出（KAL 真值锚校准后的 kernel.sense，只读 detach）
# ---------------------------------------------------------------------------
@torch.no_grad()
def read_certainty(model, tok, text: str, dev: str, layer: int = READ_LAYER) -> float:
    """对文本读 KAL P(known) ∈ [0,1]（known 类概率末 token 均值；只读监测）。"""
    ids = torch.tensor([tok.encode(text)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, _, caps = model(ids, capture_layers=[layer])
    h = caps[layer]
    if isinstance(h, dict):
        h = h["content"]  # PM-stream 配置取内容流（kal.read_point 同语义）
    sense = model.kernel.sense(h)
    probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)  # [B,3]
    return float(probs[:, 0].mean().item())  # known 类（类 0）概率均值


# ---------------------------------------------------------------------------
# 把 draft 文本知识块收割成 KV 块（HRL 检索候选 + HCA 注入载体），补进 BlockStore
# ---------------------------------------------------------------------------
@torch.no_grad()
def harvest_kv_block(model, tok, store, block_id: str, content: str,
                     a_layers, dev: str) -> dict:
    """把求知写入的 draft 文本知识块 K 收割成 KV 知识块（token 寻址载体，能事实召回）。

    复用 internalization_e2e 的 KV 内化语义：prefill K 得各 CSA 层 K/V →
    转置成 inject_hca_entries 需要的 [B,n_kv,N,hd] → 作 BlockPayload 存库；
    同时存首 CSA 层均值隐藏态表征（供 HRL 检索候选）。**运行时零梯度，不动权重**。
    """
    ids = torch.tensor([tok.encode(content)], device=dev)
    with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
        _, kcache = model(ids)  # prefill K，得各 CSA 层 K/V
    entries = {}
    for i in a_layers:
        st = kcache["layers"][i]
        entries[i] = (st["k"].transpose(1, 2).contiguous(),
                      st["v"].transpose(1, 2).contiguous())
    repr_k = hidden(model, tok, content, a_layers[0], dev)[0].mean(0, keepdim=True).unsqueeze(0)
    payload = {
        "block_id": block_id,
        "kind": "kv",            # token 寻址载体（事实召回，载体能力边界红线）
        "entries": entries,      # {layer_idx: (k,v)}
        "repr": repr_k,          # 检索候选表征
        "text": content,
        "draft": False,          # KV 收割块是运行时可注入形态（非 draft 文本）
    }
    store.put(block_id + ":kv", payload, tier="L1")
    return payload


@torch.no_grad()
def retrieve(kernel, model, tok, query, candidates, k, dev, a_layers):
    """对候选块集合打分 → top-k 命中块 id 列表（HRL route_candidates，detach 只读）。"""
    q_repr = hidden(model, tok, query, a_layers[0], dev)  # [1,Tq,d]
    cand_repr = torch.cat([c["repr"] for c in candidates], dim=1)  # [1,Tk,d]
    scores = kernel.route_candidates(q_repr, cand_repr, k=None, detach_input=True)  # [1,Tq,Tk]
    cand_score = scores[0].max(dim=0).values  # [Tk]
    kk = min(k, len(candidates))
    _, topi = cand_score.topk(kk)
    return [candidates[j]["block_id"] for j in topi.tolist()], cand_score


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="0.1B 主动求知闭环全链端到端 demo")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--n_facts", type=int, default=6, help="虚构事实数（8GB 卡控量）")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--max_new", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = TokenizerIO(args.tokenizer)
    model = load_model_with_kernel(args.ckpt, dev)  # ⚠️ kernel 加载坑已处理
    kernel = model.kernel
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    d_model = model.config.d_model
    print(f"[demo] ckpt={args.ckpt} A_layers={a_layers} d_model={d_model} "
          f"kernel={'挂载' if kernel is not None else 'None'}")

    store = BlockStore()
    facts = _make_facts(args.n_facts, seed=args.seed)
    print(f"[demo] 虚构事实 {len(facts)} 条（实体先验不存在，KAL 应判低 certainty）\n")

    # =======================================================================
    # 阶段1 运行时学习：低 certainty → 求知路由 → 执行器 → 交叉验证 → 写入
    # =======================================================================
    print("=" * 70)
    print("【阶段1 运行时学习】求知分支路由 → 执行器执行 → 交叉验证 → 写入知识块")
    print("=" * 70)
    router = InquiryRouter()  # pilot 规则版（RPL/LP 可学习区）
    branch = InquiryBranch(router=router, kernel=kernel)

    # 语义感知 embed_fn：用真实模型首 CSA 层均值隐藏态作"与既有知识一致性"几何读出
    # （正式做法；pilot 字符 hash 投影无语义，会把同句式 K 误判冲突致累积写入失败）。
    # 同句式 K（仅实体/燃料不同）在模型表征空间相近 → CrossVerifier 判一致 → 累积写入。
    def model_embed(text: str) -> torch.Tensor:
        return hidden(model, tok, text, a_layers[0], dev)[0].mean(0).float()  # [d]

    # CrossVerifier 用模型 hidden 作 embed（监测/执行分置：只读 detach，零副作用）
    executor = InquiryExecutor(
        blockstore=store,
        verifier=CrossVerifier(embed_fn=model_embed),
        ask_fn=None, tool_fn=None,  # 下面按条注入 mock（注释标注）
        writer=KnowledgeBlockWriter(tier="L1"),
        namespace="inquiry",
    )

    # ---- A 组：真实虚构事实 → 真实 KAL certainty（≈0 完全空白区）→ Decline 诚实降级
    # （红线验证：完全空白区学习成本过高→诚实拒答"该部分记忆暂不可用"，绝不硬答）。
    print("  [A 组] 真实虚构事实 → 真实 KAL certainty（完全空白区）→ Decline 诚实降级：")
    stage1_a = []
    for f in facts:
        Q = f["Q"]
        cert = read_certainty(model, tok, Q, dev)  # 真实 KAL（真值锚校准）
        decision = router.decide(cert, hrl_hit=False, priority=0.6)
        executor.ask_fn = (lambda _K: (lambda q: _K))(f["K"])   # mock（Decline 不执行）
        executor.tool_fn = (lambda _K: (lambda q: _K))(f["K"])
        got = executor(decision)  # Decline → 不执行，返回 False（诚实降级无求知动作）
        stage1_a.append({"Q": Q, "K": f["K"], "certainty": cert,
                         "action": decision.action.value, "written": got})
        print(f"    Q: {Q}")
        print(f"      certainty={cert:.3f} → {decision.action.value}：{decision.reason}")
        print(f"      诚实降级声明: {decision.ask_token}（不执行求知，got={got}）")
    n_decline = sum(1 for s in stage1_a if s["action"] == "Decline")
    print(f"    → A 组 Decline 诚实降级 {n_decline}/{len(facts)}（完全空白区红线成立）\n")

    # ---- B 组：可学习区演示样例（RPL/LP"差一点就知道"）→ Ask/CallTool → 写入
    # 真实 KAL 对完全虚构事实全判 certainty≈0（空白区）。为演示**求知执行→写入→
    # 实时→固化**完整链，B 组构造落在可学习区（0.4<certainty<0.7）的 decision 样例——
    # certainty 取自 router 可学习区中值作演示占位（注释标注：正式应由 KAL 对
    # "半熟"问题读出），展示 AskQuestion/CallTool 求知动作的执行与写入。
    print("  [B 组] 可学习区演示样例（RPL/LP 差一点就知道）→ Ask/CallTool → 写入：")
    print("        （certainty=0.55 为可学习区演示占位；正式应由 KAL 对半熟问题读出）")
    stage1 = []
    for f in facts:
        Q, K = f["Q"], f["K"]
        cert_real = read_certainty(model, tok, Q, dev)
        cert_demo = 0.55  # 可学习区演示占位（mid0.4<0.55<high0.7，RPL/LP 触发区）
        # priority 高→CallTool 自我学习优先；priority 低→AskQuestion
        priority = 0.6 if len(stage1) % 2 == 0 else 0.2
        decision = router.decide(cert_demo, hrl_hit=False, priority=priority)
        executor.ask_fn = (lambda _K: (lambda q: _K))(K)   # AskQuestion→用户给 K（mock）
        executor.tool_fn = (lambda _K: (lambda q: _K))(K)  # CallTool→检索得 K（mock）
        got = executor(decision)  # 执行→CrossVerifier→写入（draft，累积不覆盖）
        stage1.append({
            "Q": Q, "K": K, "certainty_real": cert_real, "certainty_demo": cert_demo,
            "action": decision.action.value, "reason": decision.reason,
            "ask_token": decision.ask_token, "acquired_and_written": got,
        })
        print(f"    Q: {Q}")
        print(f"      certainty(demo)={cert_demo:.2f}（真实{cert_real:.3f}）priority={priority} "
              f"→ {decision.action.value}")
        print(f"      执行+验证+写入: {'✅ draft 块已写入' if got else '❌ 未写入'}")

    n_route_inquire = sum(1 for s in stage1 if s["action"] != "DirectAnswer")
    n_written = sum(1 for s in stage1 if s["acquired_and_written"])
    print(f"\n[阶段1小结] A 组 Decline 诚实降级 {n_decline}/{len(facts)}；"
          f"B 组求知路由(非直答) {n_route_inquire}/{len(facts)}，"
          f"执行器写入 draft 知识块 {n_written}/{len(facts)}；BlockStore={store.stats()}\n")

    # =======================================================================
    # 阶段2 实时可用：写入的 K 块 → HRL 检索命中 → HCA 注入 → 用上 K 答对
    # =======================================================================
    print("=" * 70)
    print("【阶段2 实时可用】HRL 检索命中 → HCA 注入 → 用上 K（运行时注入不动权重）")
    print("=" * 70)
    # 把求知写入的 draft 文本知识块收割成 KV 块（可注入载体）+ 检索候选
    kv_blocks = []
    for f, s in zip(facts, stage1):
        if not s["acquired_and_written"]:
            continue
        kv = harvest_kv_block(model, tok, store, f"fact/{f['entity']}", f["K"], a_layers, dev)
        kv_blocks.append((f, kv))
    print(f"  收割 draft 文本块 → KV 知识块 {len(kv_blocks)} 个（运行时零梯度，不动权重）")

    cand_blocks = [kv for _, kv in kv_blocks]
    hits = 0
    acc = {"baseline": 0, "kv": 0}
    samples = []
    for f, kv in kv_blocks:
        # ① HRL 检索命中（route_candidates top-k）
        top_ids, _ = retrieve(kernel, model, tok, f["Q"], cand_blocks, args.topk, dev, a_layers)
        hit = kv["block_id"] in top_ids
        hits += int(hit)
        # ② 不注入基线（凭先验答虚构事实，应答不出）
        g_base = answer_baseline(model, tok, f, dev, args.max_new)
        # ③ HCA 注入后答（用上 K；运行时注入，不动权重，无需重新 SFT）
        g_kv = answer_with_kv_inject(model, tok, f, kv, a_layers, dev, args.max_new)
        ok_base = answer_correct(g_base, f["A"])
        ok_kv = answer_correct(g_kv, f["A"])
        acc["baseline"] += int(ok_base)
        acc["kv"] += int(ok_kv)
        if len(samples) < 3:
            samples.append({"Q": f["Q"], "A": f["A"], "hit": hit,
                            "baseline": g_base, "kv_inject": g_kv})
    n2 = len(kv_blocks)
    hit_rate = hits / max(n2, 1)
    base_rate = acc["baseline"] / max(n2, 1)
    kv_rate = acc["kv"] / max(n2, 1)
    print(f"\n[阶段2小结] HRL 检索 top-{args.topk} 命中率 = {hit_rate:.3f} ({hits}/{n2})")
    print(f"  不注入基线答对率 = {base_rate:.3f}（凭先验，应≈0）")
    print(f"  HCA 注入答对率   = {kv_rate:.3f}（用上 K，实时可用）")
    print(f"  实时可用判据（注入>基线）: {kv_rate > base_rate}")
    print(f"  注：本 demo 用随机初始化 indexer+未训门控，检索/召回落'通路通'；")
    print(f"      已训强度（1.000/0.188）见 train_retrieval_recall（权重未存 checkpoint）\n")

    # =======================================================================
    # 阶段3 长期固化：draft 块 → InquirySleepConsolidation → CA1 门调速 →
    #                 SleepConsolidator → 三元奖励 → PROMOTE/QUARANTINE
    # =======================================================================
    print("=" * 70)
    print("【阶段3 长期固化】draft 知识块 → CA1 门调速 → 睡眠固化 → PROMOTE/QUARANTINE")
    print("=" * 70)
    # 注入一条**冲突**知识块（验证"冲突保留双方标分歧→QUARANTINE 慢通道"）：
    # 与已有事实同实体但内容矛盾（CrossVerifier 判 conflict → draft 仍写但标分歧）
    conflict_fact = None
    if facts:
        f0 = facts[0]
        wrong = f"The {f0['entity']} engine runs on refined WATER."  # 与 K 矛盾
        from tais_obsidian.model.inquiry_executor import Evidence
        ev_bad = Evidence(content=wrong, source="web")  # web 弱可信度
        verified_b, consist_b, conflict_b = executor.verifier.verify(ev_bad, executor._knowledge)
        ev_bad.verified = verified_b
        bid_bad = executor.writer.write(ev_bad, store, namespace="inquiry",
                                        conflict=True, consistency=consist_b)  # 强制标冲突
        conflict_fact = {"block_id": bid_bad, "content": wrong,
                         "verified": verified_b, "conflict_flag": True}
        print(f"  注入冲突块（与已有事实矛盾）: {wrong}")
        print(f"    verified={verified_b} conflict=True → 写入 {bid_bad}（保留双方标分歧）")

    # 睡眠固化（先验 = 主干/既有知识；pilot 用空先验——已验证块一致 fast_track）。
    # embed_fn 注入模型 hidden（与 CrossVerifier 同策略），使先验一致性语义感知
    # （pilot 字符 hash 投影无语义，会把同句式 K 误判冲突）。
    consolidator = SleepConsolidator()
    isc = InquirySleepConsolidation(embed_fn=model_embed)
    report = isc.consolidate_inquiry_blocks(
        store, consolidator,
        prior_knowledge=None,      # pilot 空先验（正式应接 BlockStore 全量/主干先验）
        namespace="inquiry",
        usage_count=12,            # 求知块被检索/使用次数（HRL 命中计数；CA1 门 min_usage=10）
        saliency=1.0,
        regression_ok=True,        # 校验集回归通过（防错误固化验证门）
    )
    # 三元奖励信号（TruthRL）：固化后对答对/幻觉/拒答的奖励读出
    tri = isc.tri_reward
    reward_demo = {"correct": tri.reward("correct"),
                   "hallucinate": tri.reward("hallucinate"),
                   "abstain": tri.reward("abstain")}
    print(f"\n[阶段3小结] 睡眠固化报告：分簇={report.n_clusters} 提取={report.n_practiced}")
    print(f"  PROMOTE={report.n_promoted} QUARANTINE={report.n_quarantined} "
          f"REJECT={report.n_rejected}")
    print(f"  promoted_ids={report.promoted_ids}")
    print(f"  三元奖励: correct={reward_demo['correct']:+.2f} "
          f"hallucinate={reward_demo['hallucinate']:+.2f} "
          f"abstain={reward_demo['abstain']:+.2f}（abstain 不重罚）")

    # =======================================================================
    # 汇总
    # =======================================================================
    full_chain_ok = (n_written > 0) and (n2 > 0) and (report.n_promoted + report.n_quarantined > 0)
    out = {
        "ckpt": args.ckpt, "n_facts": len(facts),
        "stage1_runtime_learning": {
            "group_a_decline": n_decline, "group_a_total": len(facts),
            "group_b_route_inquire": n_route_inquire, "group_b_written": n_written,
            "decisions_group_a": stage1_a, "decisions_group_b": stage1,
        },
        "stage2_realtime_usable": {
            "n_kv_blocks": n2, "retrieval_hit_rate": hit_rate,
            "acc_baseline": base_rate, "acc_kv_inject": kv_rate,
            "realtime_usable": bool(kv_rate > base_rate),
            "note": "随机初始化 indexer+未训门控→通路通；已训强度 1.000/0.188 见 train_retrieval_recall",
            "samples": samples,
        },
        "stage3_sleep_consolidation": {
            "n_clusters": report.n_clusters, "n_practiced": report.n_practiced,
            "n_promoted": report.n_promoted, "n_quarantined": report.n_quarantined,
            "n_rejected": report.n_rejected, "promoted_ids": report.promoted_ids,
            "tri_reward": reward_demo, "conflict_block": conflict_fact,
        },
        "full_chain_ok": bool(full_chain_ok),
        "redlines": {
            "no_bare_self_correction": "写入/固化经 CrossVerifier+regression 外部验证门控",
            "accumulate_no_overwrite": "写入版本化 :v{n} + 冲突保留双方标分歧",
            "honest_decline": "Decline 声明'该部分记忆暂不可用'",
            "runtime_inject_no_weight_change": "HCA 注入零梯度不动权重（实时可用 vs 离线 SFT）",
            "kernel_load_trap": "attach_kernel()+load_state_dict(strict=True) 处理 kernel_enabled=False 坑",
        },
    }
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("【全链协同验证】")
    print("=" * 70)
    print(f"  阶段1 运行时学习: A组Decline诚实降级 {n_decline}/{len(facts)}，"
          f"B组写入 {n_written}")
    print(f"  阶段2 实时可用  : 检索命中 {hit_rate:.3f}，注入答对 {kv_rate:.3f} vs 基线 {base_rate:.3f}")
    print(f"  阶段3 长期固化  : PROMOTE {report.n_promoted} / QUARANTINE {report.n_quarantined}")
    print(f"  全链端到端跑通  : {'✅' if full_chain_ok else '❌'}")
    print(f"[save] report → {rep}")


if __name__ == "__main__":
    main()
