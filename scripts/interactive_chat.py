"""交互式全链验证 REPL——加载 0.1B 统一 checkpoint 与人对话、教学、探针、睡眠固化。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/interactive_chat.py
  非交互冒烟：echo -e "/help\n/teach <事实>\n/quiz\n/blocks\n/sleep\n/quit" | \
      CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/interactive_chat.py

对话循环（自由文本 = 对话轮）：
  模型续答（prefill + prompt 法 continue_from）+ 显示该轮 KAL certainty（ℓ10 P(known)）。
  0.1B 自由对话文本质量差是已知现象——本系统验证**部件信号**，非聊天流畅度。

命令：
  /teach <事实>            教学：求知执行器路径（CrossVerifier model_embed 交叉验证
                          → 累积不覆盖写入 draft 块）→ KV 收割 → 立即 baseline vs
                          KV 注入对照各一次并打印。支持 "K | Q | A" 三段式显式 quiz
                          锚点；单段自由文本自动推导 Q/A（口径见 derive_qa 注释）。
  /quiz                   对本次会话教过的全部事实逐条 baseline vs 注入对照。
  /probe <文本>           探针：KAL certainty + 末 GDN 层 hidden 轨迹 ThoughtManifold
                          project_3d 统计（轨迹长度/位移范数，随机游走基线对照）+
                          GridCodeProbe grid_score（token 序号 2D 网格展开口径）。
  /blocks                 列出 BlockStore 全部块（id/namespace/载体/版本）。
  /sleep                  对本次会话写入的 inquiry draft 块跑睡眠固化，打印 CA1 门
                          逐块裁决（PROMOTE/QUARANTINE/REJECT 及理由）。
  /help                   帮助。
  /quit                   退出。

所有信号逐轮追加写入 runs/interactive_validation/session_log.jsonl
（含时间戳/轮次/类型/数值）。红线：运行时注入零梯度不动权重；CrossVerifier
外部验证门控（绝不裸自我修正）；累积不覆盖版本化；诚实降级。
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # interactive_validation_demo（共享原语）
if hasattr(sys.stdout, "reconfigure"):  # Windows 终端中文输出
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stdin, "reconfigure"):  # 管道输入 UTF-8（Git Bash echo 中文事实）
    sys.stdin.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.inquiry_branch import InquiryRouter  # noqa: E402
from tais_obsidian.runtime.blockstore import BlockStore  # noqa: E402
from tais_obsidian.runtime.ca1_gate import SourceCredibilityTracker  # noqa: E402

import interactive_validation_demo as ivd  # 共享原语库（不重复造轮子）  # noqa: E402

DEFAULT_CKPT = ivd.DEFAULT_CKPT
DEFAULT_TOK = ivd.DEFAULT_TOK
DEFAULT_LOG = "runs/interactive_validation/session_log.jsonl"

HELP_TEXT = """命令一览：
  /teach <事实>          教学一条新知识（可带 quiz 锚点："K | Q | A" 三段式）
  /quiz                  对本次会话教过的全部事实做 baseline vs KV 注入对照
  /probe <文本>          CoT/流形探针（certainty + 3D 轨迹统计 + grid_score）
  /blocks                列出 BlockStore 全部块（id/namespace/载体/版本）
  /sleep                 睡眠固化本会话 inquiry draft 块，打印 CA1 门逐块裁决
  /help                  本帮助
  /quit                  退出
  其余自由文本 = 对话（模型续答 + 显示 KAL certainty）"""


class Session:
    """交互会话状态：模型/分词器/块库/已教事实/日志。"""

    def __init__(self, ckpt: str, tok_path: str, dev: str, log_path: str):
        self.dev = dev
        print(f"[init] 加载统一 checkpoint: {ckpt}（load_unified：gate_mlp 复挂坑处理）")
        self.model, self.tok, self.a_layers = ivd.load_model_and_tokenizer(ckpt, tok_path, dev)
        self.manifold = ivd.make_manifold(self.model.config.d_model)
        self.max_seq = self.model.config.max_seq
        self.store = BlockStore()
        self.router = InquiryRouter()
        self.executor, self.model_embed = ivd.make_executor(
            self.model, self.tok, self.a_layers, dev, self.store)
        self.taught: list[dict] = []  # [{fact, written, action, kv}]
        # v1.1 信源可信度在线学习（会话级，跨多次 /sleep 累积 EMA）
        self.cred_tracker = SourceCredibilityTracker()
        self.turn = 0
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fp = open(self.log_path, "a", encoding="utf-8")
        self.log("session_start", ckpt=ckpt, tokenizer=tok_path, device=dev,
                 a_layers=self.a_layers, max_seq=self.max_seq)
        print(f"[init] 就绪：A_layers={self.a_layers} max_seq={self.max_seq} "
              f"d_model={self.model.config.d_model}；日志 → {self.log_path}")

    # ------------------------------------------------------------------
    def log(self, type_: str, **fields) -> None:
        rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"),
               "turn": self.turn, "type": type_, **fields}
        self._log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._log_fp.flush()

    def close(self) -> None:
        self.log("session_end")
        self._log_fp.close()

    # ------------------------------------------------------------------
    # 自由对话轮
    # ------------------------------------------------------------------
    @torch.no_grad()
    def chat(self, text: str) -> None:
        tok, model, dev = self.tok, self.model, self.dev
        prompt = f"User: {text}\nAssistant:"
        n_ids = len(tok.encode(prompt))
        max_new = 32
        if n_ids > self.max_seq - max_new - 8:
            print(f"  ⚠️ 输入过长（{n_ids} tokens）：超过 max_seq={self.max_seq} 余量"
                  f"（续答需 {max_new}+8）——请缩短输入（超长 prefill 会越出 RoPE 缓存）")
            self.log("chat_rejected", reason="prompt_too_long", n_tokens=n_ids)
            return
        cert = ivd.read_certainty(model, tok, text, dev)
        with torch.autocast("cuda", torch.bfloat16, enabled=(dev == "cuda")):
            logits, cache = model(torch.tensor([tok.encode(prompt)], device=dev))
        gen = ivd.continue_from(model, tok, logits, cache, dev, max_new)
        print(f"  模型续答: {gen!r}")
        print(f"  KAL certainty（ℓ{ivd.READ_LAYER} P(known)）= {cert:.3f}")
        self.log("chat", user=text, gen=gen, certainty=cert)

    # ------------------------------------------------------------------
    # /teach
    # ------------------------------------------------------------------
    def cmd_teach(self, arg: str) -> None:
        arg = arg.strip()
        if not arg:
            print("  ⚠️ 用法：/teach <事实>（或显式锚点：/teach K | Q | A）")
            return
        parts = [p.strip() for p in arg.split("|")]
        if len(parts) == 3 and all(parts):
            fact = {"K": parts[0], "Q": parts[1], "A": parts[2]}
            import hashlib
            fact["entity"] = hashlib.sha256(parts[0].encode("utf-8")).hexdigest()[:8]
            qa_note = "显式 K|Q|A 锚点"
        elif len(parts) == 1:
            try:
                fact = ivd.derive_qa(arg)
            except ValueError:
                print("  ⚠️ /teach 空文本——请输入一条事实")
                return
            qa_note = f"自动推导锚点（末句判对）: Q={fact['Q']!r} A={fact['A']!r}"
        else:
            print("  ⚠️ 格式错误：/teach <事实> 或 /teach K | Q | A（恰好三段）")
            return
        # 真实 KAL 对**事实陈述 K** 读 P(known)（教学信号=模型是否已知道该事实；
        # 不读 Q——自动推导的复述提示是元问题，读数无语义）。分档标签按 InquiryRouter
        # 默认阈值解释（≤0.4 完全空白区 / 0.4–0.7 可学习区 / ≥0.7 已掌握区）。
        real_cert = ivd.read_certainty(self.model, self.tok, fact["K"], self.dev)
        zone = ("完全空白区" if real_cert <= 0.4
                else ("可学习区" if real_cert < 0.7 else "已掌握区"))
        taught = ivd.teach_facts(self.model, self.tok, [fact], self.store, self.executor,
                                 self.router, self.dev, self.a_layers)
        t = taught[0]
        print(f"  [teach] {qa_note}")
        print(f"  真实 KAL certainty（对事实陈述 K）={real_cert:.3f}（KAL 判读：{zone}）；"
              f"路由（demo 占位 0.55 可学习区）→ {t['action']}")
        if not t["written"]:
            print("  ❌ 写入失败（CrossVerifier 未通过）——未收割、未入库")
            self.log("teach_failed", fact=fact, certainty=real_cert, action=t["action"])
            return
        self.taught.append(t)
        # 立即对照测试（baseline vs KV 注入各一次）
        g_base = ivd.answer_baseline(self.model, self.tok, fact, self.dev, max_new=8)
        g_kv = ivd.answer_with_kv_inject(self.model, self.tok, fact, t["kv"],
                                         self.a_layers, self.dev, max_new=8)
        ok_base = ivd.answer_correct(g_base, fact["A"])
        ok_kv = ivd.answer_correct(g_kv, fact["A"])
        print(f"  ✅ 写入 draft 块 + KV 收割完成（运行时零梯度，不动权重）")
        print(f"  即时对照：baseline {'✅' if ok_base else '❌'} {g_base[:50]!r}")
        print(f"            KV 注入 {'✅' if ok_kv else '❌'} {g_kv[:50]!r}"
              f"（A={fact['A']!r}，宽松判对）")
        self.log("teach", fact={k: fact[k] for k in ("K", "Q", "A", "entity")},
                 qa_note=qa_note, certainty=real_cert, certainty_on="K", certainty_zone=zone,
                 action=t["action"], written=True, baseline_ok=ok_base, kv_ok=ok_kv,
                 baseline_gen=g_base, kv_gen=g_kv)

    # ------------------------------------------------------------------
    # /quiz
    # ------------------------------------------------------------------
    def cmd_quiz(self) -> None:
        if not self.taught:
            print("  ⚠️ 本次会话尚未教过任何事实（先 /teach）")
            return
        print(f"  [quiz] 对 {len(self.taught)} 条已教事实逐条对照：")
        per = []
        for t in self.taught:
            fact, kv = t["fact"], t["kv"]
            g_base = ivd.answer_baseline(self.model, self.tok, fact, self.dev, max_new=8)
            g_kv = ivd.answer_with_kv_inject(self.model, self.tok, fact, kv,
                                             self.a_layers, self.dev, max_new=8)
            ok_base = ivd.answer_correct(g_base, fact["A"])
            ok_kv = ivd.answer_correct(g_kv, fact["A"])
            per.append({"entity": fact["entity"], "baseline_ok": ok_base, "kv_ok": ok_kv,
                        "baseline_gen": g_base, "kv_gen": g_kv})
            print(f"    [{fact['entity']}] baseline {'✅' if ok_base else '❌'} "
                  f"{g_base[:36]!r} | KV 注入 {'✅' if ok_kv else '❌'} {g_kv[:36]!r}")
        n = len(per)
        acc_base = sum(p["baseline_ok"] for p in per) / n
        acc_kv = sum(p["kv_ok"] for p in per) / n
        print(f"  [quiz 汇总] baseline={acc_base:.3f} vs KV 注入={acc_kv:.3f}"
              f"（n={n}；已知判据 n=16 召回 0.625，小样本看方向：注入>基线）")
        self.log("quiz", n=n, acc_baseline=acc_base, acc_kv_inject=acc_kv, per_fact=per)

    # ------------------------------------------------------------------
    # /probe
    # ------------------------------------------------------------------
    def cmd_probe(self, arg: str) -> None:
        arg = arg.strip()
        if not arg:
            print("  ⚠️ 用法：/probe <文本>")
            return
        try:
            sig = ivd.probe_hidden_signals(self.model, self.tok, arg, self.dev, self.manifold)
        except ValueError as e:
            print(f"  ⚠️ {e}")
            self.log("probe_rejected", text=arg, reason=str(e))
            return
        print(f"  续答: {sig['gen']!r}（轨迹 T={sig['n_tokens']} tokens）")
        print(f"  KAL certainty = {sig['certainty']:.3f}")
        print(f"  ThoughtManifold 3D 轨迹：路径长度 {sig['path_length']:.3f}，"
              f"位移范数 {sig['displacement']:.3f}，平均步长 {sig['mean_step']:.3f}")
        print(f"    随机游走基线位移 {sig['random_walk_displacement_mean']:.3f}"
              f"（有序性比值 {sig['orderliness']:.2f}）")
        print(f"  GridCodeProbe grid_score = {sig['grid_score']:.3f}"
              f"（阈值 {sig['grid_threshold']}，{'成立' if sig['grid_hit'] else '不成立——预期，未挂路径积分训练'}）")
        print(f"    口径：{sig['grid_positions_note']}；{sig['manifold_note']}")
        self.log("probe", text=arg, **{k: v for k, v in sig.items()
                                       if k not in ("manifold_note", "grid_positions_note")})

    # ------------------------------------------------------------------
    # /blocks
    # ------------------------------------------------------------------
    def cmd_blocks(self) -> None:
        rows = []
        for tier in ("L0", "L1", "L2"):
            od = self.store._store.get(tier)  # 只读遍历（不触发 usage/recency 副作用）
            if od is None:
                continue
            for bid, payload in od.items():
                if isinstance(payload, dict):
                    kind = payload.get("kind", "draft-text")
                    version = payload.get("version", "-")
                    draft = payload.get("draft", False)
                else:
                    kind, version, draft = type(payload).__name__, "-", "-"
                ns = str(bid).split("/")[0]
                rows.append({"block_id": bid, "tier": tier, "namespace": ns,
                             "kind": kind, "version": version, "draft": draft})
        if not rows:
            print("  BlockStore 为空（尚无块写入）")
            self.log("blocks", n=0)
            return
        print(f"  BlockStore 共 {len(rows)} 块：")
        print(f"    {'block_id':<34} {'tier':<4} {'ns':<10} {'载体':<12} {'ver':<4} draft")
        for r in rows:
            print(f"    {r['block_id']:<34} {r['tier']:<4} {r['namespace']:<10} "
                  f"{str(r['kind']):<12} {str(r['version']):<4} {r['draft']}")
        self.log("blocks", n=len(rows), blocks=rows)

    # ------------------------------------------------------------------
    # /sleep
    # ------------------------------------------------------------------
    def cmd_sleep(self) -> None:
        # v1.1 自适应 CA1 门：CrossVerifier 二次复核回调（边缘带 RE_VERIFY）+
        # 会话级信源可信度在线学习 tracker
        report, per_block = ivd.sleep_consolidate(
            self.store, self.model_embed,
            verify_fn=ivd.make_cross_verify_fn(self.executor),
            cred_tracker=self.cred_tracker)
        if report.n_practiced == 0:
            print("  ⚠️ 无 inquiry draft 块可固化（先 /teach 写入）")
            self.log("sleep", n_practiced=0)
            return
        print(f"  [sleep] 固化报告：分簇={report.n_clusters} 提取={report.n_practiced} "
              f"PROMOTE={report.n_promoted} QUARANTINE={report.n_quarantined} "
              f"REJECT={report.n_rejected}（边缘带补验证 {report.n_reverified} 块）")
        for b in per_block:
            print(f"    [{b['verdict']}] {b['block_id']}（{b['source']}）: {b['reason']}")
        cred = dict(self.cred_tracker.cred)
        print(f"  信源可信度（会话级 EMA）："
              + ", ".join(f"{k}={v:.2f}" for k, v in sorted(cred.items())))
        self.log("sleep", n_clusters=report.n_clusters, n_practiced=report.n_practiced,
                 n_promoted=report.n_promoted, n_quarantined=report.n_quarantined,
                 n_rejected=report.n_rejected, n_reverified=report.n_reverified,
                 promoted_ids=report.promoted_ids, per_block=per_block,
                 credibility=cred, reverify_log=report.reverify_log)


def main() -> None:
    ap = argparse.ArgumentParser(description="TAIS Obsidian 0.1B 交互式全链验证 REPL")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--log", default=DEFAULT_LOG)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sess = Session(args.ckpt, args.tokenizer, dev, args.log)
    print("=" * 70)
    print("TAIS Obsidian 0.1B 交互式全链验证 REPL（统一 checkpoint 已训强度）")
    print("  自由文本 = 对话（验证部件信号，0.1B 聊天质量差属已知）；/help 看命令")
    print("=" * 70)
    try:
        while True:
            try:
                line = input("你> ").strip()
            except EOFError:
                print("\n[bye] 输入结束（EOF）")
                break
            except KeyboardInterrupt:
                print("\n[bye] 中断退出")
                break
            if not line:
                continue
            sess.turn += 1
            try:
                if line in ("/quit", "/exit", "/q"):
                    print("[bye] 退出")
                    break
                elif line == "/help":
                    print(HELP_TEXT)
                    sess.log("help")
                elif line == "/quiz":
                    sess.cmd_quiz()
                elif line == "/blocks":
                    sess.cmd_blocks()
                elif line == "/sleep":
                    sess.cmd_sleep()
                elif line.startswith("/teach"):
                    sess.cmd_teach(line[len("/teach"):])
                elif line.startswith("/probe"):
                    sess.cmd_probe(line[len("/probe"):])
                elif line.startswith("/"):
                    print(f"  ⚠️ 未知命令 {line.split()[0]!r}——/help 看命令一览")
                    sess.log("unknown_command", line=line)
                else:
                    sess.chat(line)
            except Exception as e:  # REPL 健壮性：单轮异常不杀死会话
                print(f"  ⚠️ 本轮处理异常（会话继续）: {type(e).__name__}: {e}")
                sess.log("error", line=line, error=f"{type(e).__name__}: {e}")
    finally:
        sess.close()
        print(f"[save] 会话日志 → {sess.log_path}")


if __name__ == "__main__":
    main()
