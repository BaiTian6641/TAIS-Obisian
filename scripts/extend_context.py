"""渐进扩窗微调（RoPE 缓存扩容 + 可选 YaRN scaling + 阶段课程）——fb1 P1，1M 上下文必经工程。

背景（docs/memories/niah-length-scan-gate-adaptive.md 实测）：max_seq=1024 是真实架构硬限——
三级栈滑窗分支的 RoPE 缓存 rope_cos/rope_sin 按 cfg.max_seq 构建（1024 行），cache 语义下
`_rope(k, 0)` 对全量 key 从 0 重算，序列超 1024 行即越界 RuntimeError。本脚本从既有
checkpoint 出发，按阶段（如 4K→16K→64K→256K）逐段微调解除硬限并适配长上下文行为。

方法选择（YaRN vs NTK，2026-07-31 决策记录）：
- **架构事实**：本架构 RoPE 只承载三级栈滑窗分支（L0，window=512）的绝对位置；CSA 选择检索
  （L1）与 HCA gist（L2）为 NoPE 内容寻址，GDN-2 为递归状态无位置编码。滑窗分支注意力分数
  经 RoPE 相对性 q·R(i)·R(j)ᵀ·k = q·R(i−j)·k 只依赖相对距离 i−j ≤ 512，全部落在训练所见
  相位域内——**纯缓存扩容（scaling="none"）在数学上即精确**，扩窗解除的是工程硬限而非外推误差。
- **选 YaRN 而非 NTK 作为 scaling 选项**：①设计文档 §3 明确"CSA 层 partial RoPE + 训练内
  YaRN"（OLMo 3 实证：只对注意力层应用 YaRN 效果最佳），1.5B CSA partial-RoPE 谱系对齐；
  ②YaRN 逐维 ramp（β_slow=1/β_fast=32）令高频维**精确不动**（γ=1），短上下文局部位置区分
  零扰动，满足"扩窗后短上下文尽量不变"红线；NTK-aware（θ' = θ·s^{D/(D-2)}）对所有频率整体
  压缩，短距离相位全体偏移、局部区分轻微模糊；③YaRN 插值单调无相位跳变，配合渐进微调
  （"训练内"使用）吸收二阶扰动。NTK 作为已知替代记录在案，不实现。
- YaRN logit 温度 mscale = 0.1·ln(s)+1（q/k 各乘 √mscale）：论文 §3.4 对插值致注意力分布
  变平坦的补偿。注意本架构滑窗相位基本未被插值（高频维不动），mscale 属可选二阶修正——
  消融可用 --rope_scaling none 对照（见下方实测）。

用法（0.1B 冒烟，PRO 4000 单卡视图）：
  CUDA_VISIBLE_DEVICES=1 python scripts/extend_context.py \
      --ckpt checkpoints/pilot_0p1b_gdn2_10k/final \
      --out_dir checkpoints/pilot_0p1b_gdn2_10k_ctx4k \
      --stages "4096:20" --rope_scaling yarn --micro_batch 1 --grad_accum 4

阶段语法：--stages "seq_len:steps[:lr], ..."（lr 缺省用 --lr）。每段 seq_len×步数×lr 独立，
WSD（warmup 10% + 末段 decay 20%）复用 train.py 的 lr_at/set_lr（不改其语义）。
每段末存 out_dir/latest.pt（含 stage_index，供断点定位）；全部完成后 save_pretrained → out_dir/final
（config.json 携带新 max_seq/rope_* 字段，from_pretrained 直接可加载）。
断点续训：加 --resume out_dir/latest.pt ——与 train.py 同语义（恢复权重/优化器/RNG/阶段内步数，
跳过已完成阶段；RoPE 缓存不进 state_dict，首个执行阶段自动完整重建）。

显存自适应：micro_batch 超 --token_budget（默认 16K tokens，=1024×16 基线）时自动降 micro
（micro = max(1, min(micro_batch, token_budget//seq_len))），grad_accum 按 --global_tokens
（默认 64K tokens/step）配平。OOM 时再自动减半一次（micro=1 兜底）。

长 seq 成本（实测见 scripts/bench_long_seq_cost.py + docs/上下文扩充256K_实施计划.md）：
滑窗分支 O(T·w) 线性；CSA indexer 打分 O(T·S·d_index)（S=T/4）与压缩注意力 logits
O(T·S·D) 随 T **平方**增长（常数 1/4）；HCA dense O(T·T/128·D) 同平方（常数 1/128）。
256K 时 CSA 打分 FLOPs 将超主干其余部分总和，须分块/fp8（设计 §3 的 V4 口径），见计划文档。

实测记录（2026-07-31，0.1B pilot_0p1b_gdn2_10k 底座，PRO 4000）：
  见文件末尾"实测记录"注释块（冒烟 loss / NIAH 复测 / 成本实测）——由本轮实验回填。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if hasattr(sys.stdout, "reconfigure"):  # ipykernel OutStream 无此方法（notebook import 兼容）
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tri_attention import TriRetrievalAttention  # noqa: E402
from tais_obsidian.train import (  # noqa: E402
    build_optimizer,
    chunked_ce,
    eval_val,
    lr_at,
    save_checkpoint,
    set_lr,
)


def parse_stages(s: str, default_lr: float) -> list[dict]:
    """解析 "4096:200,16384:100:5e-5" → [{seq_len, steps, lr}, ...]（lr 可缺省）。"""
    stages = []
    for part in s.split(","):
        fields = part.strip().split(":")
        assert 2 <= len(fields) <= 3, f"阶段格式 seq_len:steps[:lr]，收到 {part!r}"
        stages.append(
            {
                "seq_len": int(fields[0]),
                "steps": int(fields[1]),
                "lr": float(fields[2]) if len(fields) == 3 else default_lr,
            }
        )
    assert stages and all(st["seq_len"] > 0 and st["steps"] > 0 for st in stages)
    return stages


def main() -> None:
    ap = argparse.ArgumentParser(description="渐进扩窗微调（RoPE 扩容 + YaRN scaling + 阶段课程）")
    ap.add_argument("--ckpt", required=True, help="底座 checkpoint（save_pretrained 目录，只读）")
    ap.add_argument("--out_dir", required=True, help="扩窗产物目录（latest.pt + final/）")
    ap.add_argument("--stages", required=True, help='"4096:200,16384:100[:lr]" 阶段课程')
    ap.add_argument("--data_dir", default="data/shards")
    ap.add_argument("--rope_scaling", default="yarn", choices=["none", "yarn"],
                    help="none=纯缓存扩容（滑窗相对性数学精确）；yarn=逐维 ramp 插值（设计 §3）")
    ap.add_argument("--lr", type=float, default=2e-4, help="各阶段缺省 peak lr（微调量级）")
    ap.add_argument("--micro_batch", type=int, default=16, help="micro 上限（按 token_budget 自降）")
    ap.add_argument("--grad_accum", type=int, default=None,
                    help="固定 accum（缺省按 global_tokens/(micro×seq) 自动配平）")
    ap.add_argument("--token_budget", type=int, default=16384,
                    help="micro×seq 上限 tokens（默认 16K = 1024×16 基线显存口径）")
    ap.add_argument("--global_tokens", type=int, default=65536, help="每步全局 tokens 配平目标")
    ap.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"])
    ap.add_argument("--muon_lr", type=float, default=0.02)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--val_batches", type=int, default=8)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None,
                    help="out_dir/latest.pt 断点续训（与 train.py 同语义：恢复权重/优化器/RNG/"
                         "阶段索引与阶段内步数，跳过已完成阶段）")
    args = ap.parse_args()

    stages = parse_stages(args.stages, args.lr)
    max_stage = max(st["seq_len"] for st in stages)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, args.device)
    model.train()
    original_max = int(model.config.max_seq)  # 底座原始窗口（YaRN L_orig / scale 基准）
    print(f"[extend] 底座 {args.ckpt}（原 max_seq={original_max}）→ 阶段 {stages}，"
          f"scaling={args.rope_scaling}")

    # 优化器只建一次（矩跨阶段延续）；train.py 的 set_lr 按各阶段 WSD 逐组写入（语义不动）
    opt_cfg = {
        "optimizer": args.optimizer, "lr": args.lr, "muon_lr": args.muon_lr,
        "weight_decay": args.weight_decay,
    }
    opt = build_optimizer(model, opt_cfg)

    # 断点续训（与 train.py 同语义）：恢复权重/优化器/RNG + 阶段索引与阶段内步数。
    # RoPE 缓存 persistent=False 不进 state_dict，首个执行阶段仍走完整 extend_context 重建。
    resume_stage, resume_step = 0, 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        resume_step = ckpt["step"]
        resume_stage = ckpt["train_cfg"].get("stage_index", 0)
        # map_location=args.device 会把 CPU RNG ByteTensor 也映射上 GPU，set_rng_state 只收 CPU 张量
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if ckpt["rng"]["cuda"] is not None and args.device == "cuda":
            try:
                # 双卡 checkpoint 单卡 resume：切片到本机可见设备数；
                # map_location 会把 CPU ByteTensor 映射上 GPU（isinstance 校验失败），逐个搬回 CPU；
                # 异常降级为警告
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in ckpt["rng"]["cuda"][: torch.cuda.device_count()]]
                )
            except Exception as e:
                print(f"[resume] 警告：CUDA RNG 状态恢复失败（{e}），按当前种子继续")
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng"]["numpy"]
        print(f"[resume] 从 {args.resume} 恢复：阶段 {resume_stage} 步 {resume_step}")

    train_shards = Shards(args.data_dir, "train")
    val_shards = Shards(args.data_dir, "val")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"ckpt": args.ckpt, "original_max_seq": original_max,
                    "rope_scaling": args.rope_scaling, "stages": []}
    rope_extended = False  # 首个执行的阶段做完整 extend_context（fresh 的 si=0 或 resume 首个未跳阶段）
    for si, st in enumerate(stages):
        if si < resume_stage:
            print(f"[resume] 跳过已完成阶段 {si}（seq={st['seq_len']} steps={st['steps']}）")
            continue
        seq, steps, lr_peak = st["seq_len"], st["steps"], st["lr"]
        # 扩窗：max_seq 一次到顶（缓存行数按最终阶段构建，避免每段重分配大缓冲）；
        # YaRN scale 按本阶段窗口/原始窗口逐段调整（训练内 YaRN 课程口径），重建仅 fp32
        # 拷贝 67MB/层 级成本，ms 级完成。
        scale = max(1.0, seq / original_max) if args.rope_scaling == "yarn" else 1.0
        if not rope_extended:
            model.extend_context(max_seq=max_stage, rope_scaling=args.rope_scaling,
                                 rope_scale=scale, rope_original_max_seq=original_max)
            rope_extended = True
        else:
            model.config.rope_scale = scale
            for layer in model.layers:
                if isinstance(layer.mixer, TriRetrievalAttention):
                    layer.mixer.rebuild_rope_cache()
        # micro/accum 自适应：token_budget 内降 micro，global_tokens 配平 accum
        micro = max(1, min(args.micro_batch, args.token_budget // seq))
        accum = args.grad_accum or max(1, round(args.global_tokens / (micro * seq)))
        # 阶段级 WSD：warmup 10%（≥2 步）→ 恒定 → 末 20% 线性降 0（复用 train.lr_at）；
        # stage_index 入 cfg 供断点定位（lr_at 只读 warmup/max_steps/decay_frac/lr，多余键无害）
        stage_cfg = {"lr": lr_peak, "muon_lr": args.muon_lr,
                     "warmup": max(2, steps // 10), "max_steps": steps, "decay_frac": 0.2,
                     "stage_index": si}
        tokens_per_step = micro * accum * seq
        print(f"[stage {si}] seq={seq} steps={steps} lr={lr_peak:.1e} scale={scale:.1f} "
              f"micro={micro}×accum={accum} = {tokens_per_step/1024:.0f}k tok/step")

        ema = None
        t0 = time.time()
        n_tok = 0
        oom_halved = False
        step = resume_step if si == resume_stage else 0
        if step >= steps:
            # 阶段末保存后中断的场景：该阶段已训完，直接进入下一阶段
            print(f"[resume] 阶段 {si} 已训完（step={step}>={steps}），进入下一阶段")
            continue
        while step < steps:
            lr = lr_at(step, stage_cfg)
            set_lr(opt, lr, opt_cfg | {"lr": lr_peak})
            opt.zero_grad(set_to_none=True)
            loss_accum = 0.0
            try:
                for _ in range(accum):
                    x, y = train_shards.get_batch(micro, seq, args.device, rng)
                    with torch.autocast("cuda", torch.bfloat16, enabled=(args.device == "cuda")):
                        logits, _ = model(x)
                        loss = chunked_ce(logits, y)
                    (loss / accum).backward()
                    loss_accum += loss.item() / accum
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                if not oom_halved and micro > 1:
                    micro //= 2
                    accum *= 2
                    oom_halved = True
                    tokens_per_step = micro * accum * seq
                    torch.cuda.empty_cache()
                    print(f"[oom] micro 降至 {micro}，accum 升至 {accum}，重试本步")
                    continue
                raise
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            step += 1
            n_tok += tokens_per_step
            ema = loss_accum if ema is None else 0.98 * ema + 0.02 * loss_accum
            if step % args.log_every == 0 or step == 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0.0
                print(f"  [stage {si}] step {step:4d}/{steps} | loss {loss_accum:.4f} "
                      f"(ema {ema:.4f}) | lr {lr:.2e} | gnorm {gnorm.item():.2f} | "
                      f"mem {mem:.2f}GB", flush=True)

        dt = time.time() - t0
        vl = eval_val(model, val_shards,
                      {"micro_batch": micro, "seq_len": seq, "val_batches": args.val_batches},
                      args.device, rng)
        mem = torch.cuda.max_memory_allocated() / 1024**3 if args.device == "cuda" else 0.0
        # CSA/HCA 长 seq 成本记录（条目数 + 打分矩阵规模；实测计时见 bench_long_seq_cost.py）
        csa_entries, hca_entries = seq // model.config.tri_csa_stride, seq // model.config.tri_hca_stride
        stage_rec = {
            "seq_len": seq, "steps": steps, "lr": lr_peak, "rope_scale": scale,
            "micro": micro, "accum": accum, "train_loss_ema": round(ema, 4), "val_loss": round(vl, 4),
            "tok_per_s": round(n_tok / dt, 1), "peak_mem_gb": round(mem, 2), "sec": round(dt, 1),
            "csa_entries": csa_entries, "hca_entries": hca_entries,
            "csa_score_elems_per_layer": seq * csa_entries,  # indexer/logits 打分矩阵元素数（随 T²）
        }
        report["stages"].append(stage_rec)
        save_checkpoint(out_dir / "latest.pt", model, opt, step, stage_cfg, rng)
        print(f"[stage {si}] 完成：val {vl:.4f} | {n_tok/dt/1e3:.1f}k tok/s | 峰值 {mem:.2f}GB | "
              f"CSA {csa_entries} 条目 / HCA {hca_entries} 条目 | ckpt → {out_dir/'latest.pt'}",
              flush=True)

    model.save_pretrained(out_dir / "final")
    report["final_dir"] = str(out_dir / "final")
    (out_dir / "extension_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] 扩窗 final → {out_dir/'final'}；报告 → {out_dir/'extension_report.json'}")


if __name__ == "__main__":
    main()
