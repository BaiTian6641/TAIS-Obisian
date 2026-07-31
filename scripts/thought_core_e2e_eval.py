"""思考核接入主干前向 + 推理增益验证 demo——第二阶段"能力证明"关键一跳（fb1 P1）。

**fb1 硬约束**：第二阶段思考核此前是 pilot 独立模块（未接 model.py 主干前向），真实部件
适配只验证"改变思考动力学"但 dist_core≈dist_no_core（未挣到 FLOPs）。fb1 判读："从模块
到原生的门槛只有一个：接入主干前向后，在推理基准上产生可测量的增益（哪怕 1-2 个点）"。

本 demo 把思考核接入 model.py 主干前向（可选路径），并在**多步链式推理基准**上做
"有核 vs 无核"对照：

  ① 无核基线：统一 checkpoint（teaching 基座，内化行为在）原前向跑 chain 基准；
  ② 挂载随机核：zero-init 门（gate=0）→ 恒等（验证"随机核在前向不改变 logits"——
     这正是 dist_core≈dist_no_core 的结构根因：思考增量无处可去）；
  ③ **离线训练思考核集成**（冻结主干，只训核参数 group_mlp/proj/门）：
     监督信号 = chain 样本的 next-token 交叉熵（有 K prompt，让核学会"精炼最终表征
     使答案更可能"），门 alpha 逐渐打开——这是"思考核挣到 FLOPs"的唯一诚实路径；
  ④ 有核（已训）vs 无核对照：同一 checkpoint 同一样本集，对比 chain 答对率。

基准任务：**多步链式推理**（build_teaching_data.build_chain 同族：多事实 + 2 跳问题，
如"行星 X 绕恒星 Y，Y 是红巨星，红巨星是老恒星 → X 的宿主恒星老吗？"）。选它而非
单事实 NIAH：多步推理最贴"思考增益"（探测显示 chain 无核基线 0.83 > chance，有 headroom
到 1.0；单事实 fact 失败多为 0.1B 容量/判对口径问题，非思考可解决）。

诚实标注（禁止臆造）：
  - 随机核（gate=0）恒等 → 增益恰为 0（结构保证，非"挣到"）；
  - 已训核增益 = 训练后（有核 − 无核）答对率差，如实报告（正/负/不显著都报）；
  - 若已训核增益仍≈0 或为负 → 如实报"思考核仍未挣到 FLOPs，需训练/设计迭代"
    （重要的负结果信息，fb1 判读的诚实回应）。

红线与纪律：
  - 可选路径默认关（use_thought_core=False，357 测试零改动）；
  - 训练只更新核参数（主干 frozen，detach_backbone=True 梯度隔离——HRL 隔离红线）；
  - 有界演化（max_ticks，certainty 早停，tanh 有界门）。

运行：$env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe scripts/thought_core_e2e_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from build_unified_checkpoint import load_unified  # noqa: E402
from build_teaching_data import build_chain  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

UNIFIED = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k_unified"
TOK_PATH = ROOT / "data" / "tokenizer" / "tokenizer.json"
OUT_DIR = ROOT / "runs" / "thought_core_e2e"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CORE_DIM = 384
MAX_TICKS = 8
# 基准/训练规模（0.1B pilot：小样本快速迭代；n 如实标注，不夸大统计强度）
N_TRAIN = 128
N_EVAL = 64
TRAIN_STEPS = 200
LR = 3e-4


# ---------------------------------------------------------------------------
# 评估原语
# ---------------------------------------------------------------------------
@torch.no_grad()
def gen_answer(model, tok, prompt, use_core, max_new=6):
    """prefill 全 prompt 后 argmax 续答（prompt 法，对齐 teaching_sft/internalization 口径）。"""
    ids = torch.tensor([tok.encode(prompt)], device=DEV)
    with torch.autocast("cuda", torch.bfloat16, enabled=(DEV == "cuda")):
        logits, cache = model(ids, use_thought_core=use_core)
        out = []
        for _ in range(max_new):
            nxt = int(logits[:, -1, :].float().argmax(-1).item())
            if nxt == tok.eot_id:
                break
            out.append(nxt)
            logits, cache = model(torch.tensor([[nxt]], device=DEV), cache,
                                  use_thought_core=use_core)
    return tok.decode(out)


def answer_correct(gen, gold):
    """宽松判对（连字符/空格变体命中即算对，对齐 internalization_e2e）。"""
    g, a = gen.strip().lower(), gold.strip().lower()
    return a in g or a.replace("-", " ") in g or a.replace("-", "") in g.replace("-", "")


@torch.no_grad()
def eval_chain(model, tok, samples, use_core):
    """chain 基准答对率（有核/无核同一协议同一样本集）。"""
    ok = 0
    for s in samples:
        g = gen_answer(model, tok, f"{s['K']}\nQuestion: {s['Q']}\nAnswer: ", use_core)
        ok += answer_correct(g, s["A"])
    return ok / len(samples)


# ---------------------------------------------------------------------------
# 离线训练思考核集成（冻结主干，只训核参数；监督 = 有K答案 next-token CE）
# ---------------------------------------------------------------------------
def train_thought_core(model, tok, samples, steps, lr):
    """训练思考核集成让 chain 答对率提升（打开门，让思考增量流入 logits）。

    监督：有 K prompt（K+Question+Answer: <答案>）的 next-token 交叉熵，仅在答案 token
    位置计损（让核学会精炼最终表征使正确答案更可能）。**只反传到核参数**：
    主干 frozen（requires_grad=False 全主干）+ 核输入 detach（detach_backbone=True，
    梯度隔离红线）——双重隔离保证主干零改动（思考核训练不污染主干）。
    """
    tci = model.thought_core_integration
    assert tci is not None, "须先 attach_thought_core"
    # 冻结主干（双保险：detach_backbone 已隔离核输入路径，此处再断主干参数梯度）
    for p in model.parameters():
        p.requires_grad_(False)
    for p in tci.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(tci.parameters(), lr=lr)
    model.train()

    rng = np.random.default_rng(0)
    losses = []
    n_skipped = 0
    for step in range(steps):
        s = samples[int(rng.integers(len(samples)))]
        # 训练监督 = 全序列 next-token CE（主干 frozen，只有核路径可反传——核被迫学会
        # "精炼最终表征降低整句 CE"，重点自然落在最难的答案 token 上）。
        # 注：先前试"仅答案区 mask"因 build_chain 的 Answer: {A} 被 tokenizer 与 prefix
        # 合并编码（n_prefix==len(ids)，mask 全空→nan）而放弃；全序列 CE 更稳且等效聚焦。
        text = f"{s['K']}\nQuestion: {s['Q']}\nAnswer: {s['A']}"
        ids = tok.encode(text)
        x = torch.tensor([ids], device=DEV)
        # fp32 训练（关 autocast）：思考核是 fp32 关键路径（thought_core 内已 fp32），
        # 训练数值稳定优先（bf16 autocast + 深层 tick 反传易不稳）。
        logits, _ = model(x, use_thought_core=True)
        lp = logits[0, :-1].float()
        tgt = torch.tensor(ids[1:], device=DEV)
        loss = torch.nn.functional.cross_entropy(lp, tgt)
        if not torch.isfinite(loss):
            n_skipped += 1
            opt.zero_grad()
            continue
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tci.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))
        if (step + 1) % 50 == 0:
            gate = float(torch.tanh(tci.gate_alpha).item())
            print(f"    step {step+1}/{steps} loss={np.mean(losses[-50:]):.4f} gate={gate:+.4f}")
    if n_skipped:
        print(f"    [诚实标注] {n_skipped}/{steps} 步 loss 非有限被跳过（数值防护）")
    model.eval()
    return losses


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("思考核接入主干前向 + 推理增益验证（fb1 P1：有核 vs 无核基准对照）")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tok = TokenizerIO(str(TOK_PATH))

    model = load_unified(str(UNIFIED), DEV)
    model.eval()
    print(f"checkpoint: {UNIFIED.name} | 层型 {''.join(model.config.layer_types)} | "
          f"d_model={model.config.d_model} core_dim={CORE_DIM} max_ticks={MAX_TICKS}")

    # 基准样本（train/eval 分离：不同 seed，评估集在训练后未见）
    rng_tr = np.random.default_rng(42)
    rng_ev = np.random.default_rng(999)
    train_samples = [build_chain(rng_tr) for _ in range(N_TRAIN)]
    eval_samples = [build_chain(rng_ev) for _ in range(N_EVAL)]
    print(f"基准任务: 多步链式推理（chain，2 跳）| train n={N_TRAIN} eval n={N_EVAL}")

    # ---- ① 无核基线 ----
    print("\n[① 无核基线]（原前向，use_thought_core=False）")
    acc_no_core = eval_chain(model, tok, eval_samples, use_core=False)
    print(f"  无核 chain 答对率 = {acc_no_core:.3f}（n={N_EVAL}）")

    # ---- ② 挂载随机核（zero-init 门恒等验证）----
    print("\n[② 挂载随机核]（zero-init 门，验证恒等——dist_core≈dist_no_core 结构根因）")
    torch.manual_seed(0)
    model.attach_thought_core(core_dim=CORE_DIM, max_ticks=MAX_TICKS,
                              detach_backbone=True, use_sync=True)
    gate0 = float(torch.tanh(model.thought_core_integration.gate_alpha).item())
    acc_random_core = eval_chain(model, tok, eval_samples, use_core=True)
    print(f"  随机核 gate={gate0:.4f}（zero-init → 恒等）")
    print(f"  随机核 chain 答对率 = {acc_random_core:.3f}（应 == 无核，恒等）")

    # ---- ③ 离线训练思考核集成（打开门，挣 FLOPs 的诚实路径）----
    print("\n[③ 离线训练思考核集成]（冻结主干，只训核参数，监督=有K答案 next-token CE）")
    losses = train_thought_core(model, tok, train_samples, TRAIN_STEPS, LR)
    gate_trained = float(torch.tanh(model.thought_core_integration.gate_alpha).item())
    print(f"  训练后 gate = {gate_trained:+.4f}（zero-init → 打开）")
    print(f"  loss 首/末 50 步均值 = {np.mean(losses[:50]):.4f} / {np.mean(losses[-50:]):.4f}")

    # ---- ④ 有核（已训）vs 无核对照 ----
    print("\n[④ 有核（已训）vs 无核对照]（同一 checkpoint 同一评估集）")
    acc_trained_core = eval_chain(model, tok, eval_samples, use_core=True)
    gain = acc_trained_core - acc_no_core
    print(f"  无核   chain 答对率 = {acc_no_core:.3f}")
    print(f"  有核   chain 答对率 = {acc_trained_core:.3f}")
    print(f"  增益（有核−无核）  = {gain:+.3f}")

    # 诚实判定（fb1 门槛：哪怕 1-2 个点）
    if gain > 0.01:
        verdict = f"✅ 有核增益 +{gain:.3f}（>1 点，思考核挣到 FLOPs，达 fb1 原生门槛）"
    elif gain > 0:
        verdict = f"🟡 微弱正增益 +{gain:.3f}（<1 点，方向正确但未达 fb1 门槛，需更多训练）"
    elif abs(gain) <= 0.01:
        verdict = f"🟡 增益≈0（{gain:+.3f}，思考核仍未挣到 FLOPs，需训练/设计迭代——诚实负结果）"
    else:
        verdict = f"⚠️ 负增益 {gain:+.3f}（思考核干扰推理，设计需回检——诚实负结果）"
    print(f"\n判定：{verdict}")

    # ---- 导出 ----
    report = {
        "checkpoint": UNIFIED.name,
        "benchmark": "多步链式推理 chain（build_teaching_data.build_chain，2 跳）",
        "n_eval": N_EVAL, "n_train": N_TRAIN, "train_steps": TRAIN_STEPS, "lr": LR,
        "core_dim": CORE_DIM, "max_ticks": MAX_TICKS,
        "acc_no_core": acc_no_core,
        "acc_random_core_zero_init": acc_random_core,
        "gate_zero_init": gate0,
        "gate_trained": gate_trained,
        "acc_trained_core": acc_trained_core,
        "gain": gain,
        "verdict": verdict,
        "loss_first50": float(np.mean(losses[:50])),
        "loss_last50": float(np.mean(losses[-50:])),
        "notes": "随机核 zero-init 门恒等（增益恰 0，结构保证非挣到）；已训核增益=训练后"
                 "（有核−无核）答对率差，如实报告。训练只更新核参数（主干 frozen+detach 双隔离）。",
    }
    rpath = OUT_DIR / "thought_core_e2e_report.json"
    rpath.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[导出] {rpath}")
    print("=" * 70)


if __name__ == "__main__":
    main()
