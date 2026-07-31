"""KAL 真值锚微调（GDN-2 10k checkpoint 适配版，第二阶段真实部件适配前置）。

背景（第二阶段真实部件适配发现，scripts/thinking_real_adapter_demo.py）：
checkpoints/pilot_0p1b_gdn2_10k/final 训练时 kernel_enabled=False，from_pretrained 后
model.kernel=None——须显式 attach_kernel() 挂载内核（内核头随机初始化未微调），
导致 KAL L1 P(IK) 的 known 概率对输入敏感漂移（0.001~0.99），**不可作元认知判据**。
本脚本把真值语义写回内生 kal_l1 头权重（随 checkpoint 存取），让 certainty 可靠。

与原脚本（kal_truth_finetune.py，第一阶段 pilot_0p1b_kal）的关键差异：
- 原脚本 `if model.kernel is None: sys.exit(1)`——10k checkpoint kernel=None 会退出；
  本脚本 `attach_kernel()` 挂载（幂等），注释标注"内核头随机初始化需真值锚微调"。
- 读点层：GDN-2 模型层型 G2G2G2A ×3（12 层），GDN 层 index 为 0,1,2,4,5,6,8,9,10
  （非 hybrid 的 GGGAGGGA）；KAL sense 读点是 GDN 层，先小范围扫描读点（默认 ℓ8/ℓ10）
  选 init AUROC 最高者再微调（中间~末段 GDN 层信号最强，SAPLMA/ITI 文献+0.1B 实证 ℓ8）。

红线与纪律（与原脚本一致）：
- detach 主干 hidden（no_grad 提取，监测/执行分置 + 探针冻结语义），梯度只进 kal_l1；
- 三态头用 [0,2] 二分类退化（"不确定"无标签来源）；diverse 数据源默认（OOD 鲁棒，规范 §7.2）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/kal_truth_finetune_gdn2.py \
      [--scan_layers 8 10] [--steps 500] [--data_source diverse]
产出：checkpoints/pilot_0p1b_gdn2_10k_kaltruth/（含校准内核权重）+ runs/kal_truth_gdn2/report.json。
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
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402
import kal_probe as kp  # noqa: E402

# 默认常量（10k checkpoint 路径 + 微调产出路径）
DEFAULT_CKPT = "checkpoints/pilot_0p1b_gdn2_10k/final"
DEFAULT_OUT = "checkpoints/pilot_0p1b_gdn2_10k_kaltruth"
DEFAULT_REPORT = "runs/kal_truth_gdn2/report.json"
# GDN-2 层型 G2G2G2A ×3：GDN 层 index（KAL sense 读点候选；注意力层 3,7,11 除外）
GDN_LAYERS = [0, 1, 2, 4, 5, 6, 8, 9, 10]


@torch.no_grad()
def collect_hidden(model, id_list, layer, device, batch_size, pooling):
    """复用 kal_probe.forward_collect 取 pooled hidden [N,d]（单流=内容流）。"""
    feats, _ = kp.forward_collect(model, id_list, [layer], device, batch_size, pooling)
    return feats[layer]


def make_batch(tok, val_shards, rng, n_each, T, layer, model, device, batch_size, pooling,
               data_source: str = "diverse"):
    """构造一批真值样本：n_each 条 known + n_each 条 unknown，编码→hidden→标签。

    data_source：
    - "diverse"（默认）：known/unknown 经 diverse_truth_data 多样化生成（多句式短句/
      疑问/否定/contrast-pair + 程序化虚构词 + 真实事实句），OOD 鲁棒（规范 §7.2）；
    - "template"：known=val 分布内文本，unknown=kal_probe 8 类长模板伪事实（v1 同分布）。

    返回 (hidden [2*n_each, d]（no_grad 提取）, labels [2*n_each]（known=0/unknown=2）)。
    """
    if data_source == "diverse":
        import diverse_truth_data as dt
        texts, labels_np = dt.build_diverse_truth_dataset(rng, n_each, n_each)
        ids = kp.encode_fixed(tok, texts, T)
        labels = labels_np
    else:  # template（v1）
        x, _ = val_shards.get_batch(n_each, T, "cpu", rng)
        known_ids = x.numpy().tolist()
        fake_ids = kp.encode_fixed(tok, kp.build_fake_fact_texts(rng, n_each), T)
        ids = known_ids + fake_ids
        labels = np.array([0] * n_each + [2] * n_each, dtype=np.int64)  # 0=知道, 2=空白
    hidden = collect_hidden(model, ids, layer, device, batch_size, pooling)
    return torch.from_numpy(hidden).to(device), torch.from_numpy(labels).to(device)


def main() -> None:
    ap = argparse.ArgumentParser(description="KAL 真值锚微调（GDN-2 10k 适配版）")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tokenizer", default=kp.DEFAULT_TOK)
    ap.add_argument("--shards", default=kp.DEFAULT_SHARDS)
    ap.add_argument("--out_dir", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT)
    ap.add_argument("--scan_layers", type=int, nargs="+", default=[8, 10],
                    help="读点扫描候选层（GDN 层）；选 init AUROC 最高者微调")
    ap.add_argument("--pooling", choices=["last", "mean"], default="last")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_each", type=int, default=64, help="每批 known/fake 各 n_each 条")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data_source", choices=["template", "diverse"], default="diverse",
                    help="diverse=v2 多样化真值（OOD 鲁棒，规范 §7.2，默认）；template=v1 模板伪事实")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tok = TokenizerIO(args.tokenizer)
    print(f"[kaltruth-gdn2] ckpt={args.ckpt} steps={args.steps} lr={args.lr} "
          f"scan_layers={args.scan_layers} data_source={args.data_source}")

    # ------------------------------------------------------------------
    # 加载 + attach_kernel（10k checkpoint kernel_enabled=False → kernel=None，
    # 须显式挂载；内核头随机初始化未微调，正是本脚本要校准的对象）
    # ------------------------------------------------------------------
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.eval()
    kernel_attached_fresh = False
    if model.kernel is None:
        print("[kaltruth-gdn2] model.kernel=None（10k kernel_enabled=False 训练）→ "
              "attach_kernel() 挂载（内核头随机初始化，需真值锚微调）")
        model.attach_kernel()
        kernel_attached_fresh = True
    else:
        print("[kaltruth-gdn2] model.kernel 已挂载（含内核权重，继续微调）")
    # 冻结全部，只解冻 kal_l1（红线：梯度只进 KAL 头，不污染主干/内核其余部件）
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.kernel.kal_l1.parameters():
        p.requires_grad_(True)
    head = model.kernel.kal_l1
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    val_shards = Shards(args.shards, "val")

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_auroc(layer, n_eval=200):
        """真值 AUROC：known(val) vs fake，score = logit[0]-logit[2]。"""
        ids, labels_np, subset = kp.build_l1_dataset(
            tok, args.shards, np.random.default_rng(args.seed + 999), n_eval, n_eval // 2, 0, args.seq_len)
        hidden = collect_hidden(model, ids, layer, args.device, args.batch_size, args.pooling)
        h = torch.from_numpy(hidden).to(args.device)
        logits = head(h).float()
        scores = (logits[:, 0] - logits[:, 2]).cpu().numpy()
        known_binary = (labels_np == 1).astype(np.int64)  # build_l1_dataset: known=1
        fake_mask = (subset == "known") | (subset == "fake")
        return (kp.auroc(scores, known_binary),
                kp.auroc(scores[fake_mask], known_binary[fake_mask]))

    # ------------------------------------------------------------------
    # 读点扫描：候选 GDN 层逐个测 init AUROC，选最高者作微调读点
    # ------------------------------------------------------------------
    scan: dict[int, tuple[float, float]] = {}
    for cand in args.scan_layers:
        if cand not in GDN_LAYERS:
            print(f"  ⚠️ 层 {cand} 非 GDN 层（GDN_LAYERS={GDN_LAYERS}），跳过")
            continue
        ov, fk = eval_auroc(cand)
        scan[cand] = (ov, fk)
        print(f"  [scan] ℓ{cand}: overall AUROC {ov:.3f} | fake {fk:.3f}")
    if not scan:
        print("❌ 无有效 GDN 读点候选")
        sys.exit(1)
    layer = max(scan, key=lambda l: scan[l][0])
    init_overall, init_fake = scan[layer]
    print(f"[kaltruth-gdn2] 读点选定 ℓ{layer}（init overall AUROC {init_overall:.3f} 最高）")

    # ------------------------------------------------------------------
    # 真值锚微调（detach 主干，梯度只进 kal_l1）
    # ------------------------------------------------------------------
    t0 = time.time()
    head.train()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        hidden, labels = make_batch(tok, val_shards, rng, args.n_each, args.seq_len,
                                    layer, model, args.device, args.batch_size, args.pooling,
                                    data_source=args.data_source)
        logits = head(hidden)  # [B,3]（hidden 已 detach 提取，梯度只进 head）
        ce = F.cross_entropy(logits.float(), labels)
        ce.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if (step + 1) % args.eval_every == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps} ce={ce.item():.4f} lr={lr:.2e}")

    final_overall, final_fake = eval_auroc(layer)
    dur = time.time() - t0
    print(f"[kaltruth-gdn2] 微调后：overall AUROC {init_overall:.3f}→{final_overall:.3f} | "
          f"fake {init_fake:.3f}→{final_fake:.3f} 用时 {dur:.0f}s")
    verdict = ("✅ 真值锚达标（overall≥0.8），certainty 可作元认知门控" if final_overall >= 0.8
               else "⚠️ 未达 0.8（真值锚有效但需更多步/数据，或读点/单头限制）")
    print(f"判定：{verdict}")

    # ------------------------------------------------------------------
    # certainty 可靠性抽查：known 文本 P(known) 高、fake 文本 P(known) 低
    # ------------------------------------------------------------------
    @torch.no_grad()
    def prob_known(texts, n_show=5):
        ids = kp.encode_fixed(tok, texts, args.seq_len)
        hidden = collect_hidden(model, ids, layer, args.device, args.batch_size, args.pooling)
        h = torch.from_numpy(hidden).to(args.device)
        probs = torch.softmax(head(h).float(), dim=-1)  # [N,3]
        return probs[:, 0].cpu().numpy()  # known 类（类0）概率

    import diverse_truth_data as dt
    probe_rng = np.random.default_rng(args.seed + 777)
    known_texts = dt.build_real_statements(probe_rng, 8)
    fake_texts = kp.build_fake_fact_texts(probe_rng, 8)
    pk_known = prob_known(known_texts)
    pk_fake = prob_known(fake_texts)
    print(f"[kaltruth-gdn2] certainty 抽查：known 文本 P(known) 均值 {pk_known.mean():.3f} "
          f"（应高）| fake 文本 P(known) 均值 {pk_fake.mean():.3f}（应低）")
    certainty_ok = bool(pk_known.mean() > 0.5 and pk_fake.mean() < 0.5)
    print(f"  certainty 方向：{'✅ 语义正确（known 高 / fake 低）' if certainty_ok else '⚠️ 方向异常'}")

    # ------------------------------------------------------------------
    # 保存微调后 checkpoint（含校准后的内核权重，供真实部件适配/推理循环用）
    # ------------------------------------------------------------------
    out_dir = Path(args.out_dir)
    model.save_pretrained(out_dir)
    report = {
        "ckpt": args.ckpt, "layer": layer, "steps": args.steps, "lr": args.lr,
        "n_each": args.n_each, "seq_len": args.seq_len, "seed": args.seed,
        "data_source": args.data_source, "scan": {str(k): list(v) for k, v in scan.items()},
        "kernel_attached_fresh": kernel_attached_fresh,
        "method": "真值锚（fake=unknown/real=known，detach 主干，梯度只进 kal_l1；GDN-2 10k 适配版）",
        "init": {"overall": init_overall, "fake": init_fake},
        "final": {"overall": final_overall, "fake": final_fake},
        "certainty_probe": {
            "known_p_known_mean": float(pk_known.mean()),
            "fake_p_known_mean": float(pk_fake.mean()),
            "direction_ok": certainty_ok,
        },
        "verdict": verdict, "duration_s": round(dur, 1), "out_dir": str(out_dir),
    }
    rp = Path(args.report)
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[kaltruth-gdn2] checkpoint → {out_dir}；report → {rp}")


if __name__ == "__main__":
    main()
