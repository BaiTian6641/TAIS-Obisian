"""训练循环：WSD 调度 + AdamW(decay 分组) + bf16 autocast + checkpoint/resume + tensorboard。

标准配方：参数 fp32，autocast(bf16) 前向，grad clip 1.0；
AdamW β=(0.9,0.95) eps=1e-8 wd=0.1（embedding/norm/1D 参数不衰减），fused 若可用；
WSD：warmup 线性升 → 稳定段恒定 → 最后 decay_frac 步数线性降到 0。

用法：
  python -m tais_obsidian.train --config configs/pilot_0p1b.json
  python -m tais_obsidian.train --config configs/pilot_0p1b.json --resume checkpoints/pilot_0p1b/latest.pt
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ipykernel OutStream 无 reconfigure 方法（notebook import 即炸），hasattr 守卫
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.config import ModelConfig
from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM

DEFAULTS = dict(
    run_name="pilot_0p1b",
    seed=42,
    out_dir="checkpoints/pilot_0p1b",
    data_dir="data/shards",
    seq_len=1024,
    max_steps=2000,
    micro_batch=16,
    grad_accum=4,
    lr=1e-3,
    warmup=200,
    decay_frac=0.15,
    weight_decay=0.1,
    grad_clip=1.0,
    val_every=250,
    val_batches=20,
    log_every=20,
    ckpt_every=500,
    # KAL 内生训练（T1 预训练后期；默认 0.0=关，既有训练零改动）：
    # kal_aux_weight>0 时启用 P(IK) 内生辅助损失（kal_pik_aux_loss），
    # 探针梯度只进 KAL 头（detach 主干，监测/执行分置红线）。
    kal_aux_weight=0.0,
    kal_sense_layers=None,  # 例 [4, 8]；None=全部 GDN 层
    # 优化器选择（设计 §14.3/§21：预训练与 W4 固化同优化器 Muon，arXiv:2605.06654）：
    # "adamw"（默认，pilot 向后兼容）| "muon"（2D 矩阵 Newton-Schulz 正交化 + 非 2D AdamW）。
    optimizer="adamw",
    muon_lr=0.02,          # Muon 组 lr（Muon 对 lr 不敏感，典型 0.02）
    muon_momentum=0.95,    # Muon SGD-momentum
    muon_ns_steps=5,       # Newton-Schulz 迭代步数
    muon_per_head=False,   # Per-Head Muon（K3 借鉴：Q/K/V 按头分块正交化）
)


def lr_at(step: int, cfg: dict) -> float:
    """WSD：warmup 线性升到 peak → 恒定 → 末段线性降到 0。"""
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    decay_start = int(cfg["max_steps"] * (1 - cfg["decay_frac"]))
    if step < decay_start:
        return cfg["lr"]
    return cfg["lr"] * (cfg["max_steps"] - step) / (cfg["max_steps"] - decay_start)


def set_lr(opt: torch.optim.Optimizer, lr: float, cfg: dict) -> None:
    """把 WSD 当前 lr 写入优化器各组。

    AdamW 组读 ``g["lr"]``；Muon 组读 ``g["muon_lr"]``/``g["adamw_lr"]``
    （见 optim/muon.py）。只设 ``g["lr"]`` 时 Muon 全程恒 lr（warmup/decay 失效），
    故对 Muon 组按 WSD 比例（lr / peak）同步缩放 muon_lr，adamw_lr 直接跟随 lr。
    """
    peak = cfg["lr"]
    scale = lr / peak if peak > 0 else 0.0
    muon_peak = cfg.get("muon_lr", 0.02)
    for g in opt.param_groups:
        g["lr"] = lr
        if "muon_lr" in g:
            g["muon_lr"] = muon_peak * scale
        if "adamw_lr" in g:
            g["adamw_lr"] = lr


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    """按 cfg["optimizer"] 构建优化器（默认 adamw 向后兼容；muon 可选）。

    decay 分组：≥2D 且非 embedding 的参数衰减；norm/1D/embedding 不衰减。
    Muon（设计 §14.3/§21：预训练与 W4 固化同优化器，arXiv:2605.06654）：
    2D 矩阵参数走 Newton-Schulz 正交化动量（lr=muon_lr），非 2D 走内部 AdamW（lr=lr）。
    红线：Muon 只影响优化器更新，前向/反向计算图不变（GDN/PM-stream/grad ckpt 兼容）。
    """
    if cfg.get("optimizer", "adamw") == "muon":
        from tais_obsidian.optim.muon import build_muon_optimizer

        return build_muon_optimizer(
            model,
            muon_lr=cfg.get("muon_lr", 0.02),
            adamw_lr=cfg["lr"],
            weight_decay=cfg["weight_decay"],
            per_head_qkv=cfg.get("muon_per_head", False),
            n_heads=getattr(model.config, "n_q_heads", None),
            head_dim=getattr(model.config, "head_dim", None),
            momentum=cfg.get("muon_momentum", 0.95),
            ns_steps=cfg.get("muon_ns_steps", 5),
        )
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name:
            decay.append(p)
        else:
            no_decay.append(p)
    groups = [
        {"params": decay, "weight_decay": cfg["weight_decay"]},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    fused = torch.cuda.is_available()
    print(f"[opt] AdamW fused={fused}, decay {sum(p.numel() for p in decay)/1e6:.1f}M, "
          f"no_decay {sum(p.numel() for p in no_decay)/1e6:.1f}M")
    return torch.optim.AdamW(
        groups, lr=cfg["lr"], betas=(0.9, 0.95), eps=1e-8, fused=fused
    )


def chunked_ce(logits: torch.Tensor, targets: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """分块 fp32 cross entropy，避免一次性 fp32 logits 副本撑爆显存。

    用 reshape 而非 view：允许非连续输入（如 cache/切片来源的张量）。
    """
    flat = logits.reshape(-1, logits.size(-1))
    tgt = targets.reshape(-1)
    total = None
    for i in range(0, flat.shape[0], chunk):
        part = F.cross_entropy(flat[i : i + chunk].float(), tgt[i : i + chunk], reduction="sum")
        total = part if total is None else total + part
    return total / flat.shape[0]


def kal_pik_aux_loss(model, x: torch.Tensor, y: torch.Tensor, sense_layers: list[int]) -> torch.Tensor | None:
    """KAL P(IK) 内生辅助损失（T1 预训练后期，Kadavath arXiv:2207.05221 范式的最小实现）。

    在线自标注（自包含）：用主干自己 next-token 正确性生成 P(IK) 伪标签——
    主干预测对的位置 → "知道"（类 0），预测错 → "未知"（类 2；类 1"不确定"暂欠）。
    这避免引入外部已知/未知数据集，P(IK) 信号直接锚在主干自身的知识边界上。

    红线（监测/执行分置 + 探针冻结语义，部件详细计划 Part B）：
    - **辅助损失只进 KAL 头**：对主干 hidden state（captures）**detach**，探针梯度
      绝不回传到主干残差流——否则模型会把"是否知道"重编码到不可读基底
      （NeurIPS 激活监控的安全警示 + "探针冻结"红线）。
    - 内核未挂载或层无 sense 输出时返回 None（不阻塞主干训练）。

    返回标量辅助损失（调用方乘 kal_aux_weight 后并入总损失）。
    """
    if model.kernel is None:
        return None
    with torch.no_grad():
        # 主干 next-token 伪标签（detach，不反传）
        logits, _, captures = model(x, capture_layers=sense_layers)
        pred = logits.argmax(dim=-1)
        known = (pred == y).float()  # [B,T]，1=知道
    # 二分类交叉熵：对每层 sense 输出的 P(IK) logits 取 [..., [0,2]] 两类
    total = None
    n = 0
    for i in sense_layers:
        if i not in captures or i not in model.kernel_sense_index():
            continue
        pm = captures[i]["pm"] if isinstance(captures[i], dict) else captures[i]
        sense = model.kernel.sense(pm.detach())  # detach 主干，探针梯度只进 KAL 头
        pik = sense.pik_logits  # [B,T,3]
        two = torch.stack([pik[..., 0], pik[..., 2]], dim=-1)  # [B,T,2] = 知道/未知
        ce = F.cross_entropy(two.reshape(-1, 2).float(), known.reshape(-1).long())
        total = ce if total is None else total + ce
        n += 1
    return (total / n) if n > 0 else None


@torch.no_grad()
def eval_val(model, val: Shards, cfg: dict, device: str, rng: np.random.Generator) -> float:
    model.eval()
    losses = []
    for _ in range(cfg["val_batches"]):
        x, y = val.get_batch(cfg["micro_batch"], cfg["seq_len"], device, rng)
        # enabled=(device=="cuda")：CPU 冒烟（extend_context --device cpu 等）下跳过 autocast；
        # cuda 路径行为与原硬编码逐行一致
        with torch.autocast("cuda", torch.bfloat16, enabled=(device == "cuda")):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(path: Path, model, opt, step: int, cfg: dict, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),  # fp32
            "opt": opt.state_dict(),
            "step": step,
            "train_cfg": cfg,
            "model_cfg": model.config.__dict__,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy": rng.bit_generator.state,
            },
        },
        path,
    )


def copy_tokenizer_to_final(final_dir: str | Path) -> None:
    """把 data/tokenizer/tokenizer.json 复制进 final 目录（随权重产物，下载即可推理）。

    路径相对仓库根（train.py 上两级）；tokenizer 不存在时只警告不炸（训练产物不受影响）。
    """
    src = Path(__file__).resolve().parents[2] / "data" / "tokenizer" / "tokenizer.json"
    dst = Path(final_dir) / "tokenizer.json"
    if not src.exists():
        print(f"[warn] tokenizer 不存在：{src}，final 目录未携带（generate 需显式 --tokenizer 或回退仓库内默认）")
        return
    shutil.copy(src, dst)
    print(f"[tokenizer] 已随权重产物复制：{src} → {dst}")


# resume 时校验的 model_cfg 关键字段（结构/词表/上下文语义错配即拒绝续训）
RESUME_CRITICAL_FIELDS = (
    "vocab_size", "d_model", "n_layer",
    "n_q_heads", "n_kv_heads", "head_dim", "n_v_heads", "n_qk_heads",
    "mlp_hidden", "max_seq", "rope_scaling", "rope_scale", "gdn_decay_g_min",
)


def validate_resume_model_cfg(ckpt_model_cfg: dict | None, cur: ModelConfig) -> None:
    """resume 时校验 checkpoint 的 model_cfg 与当前 config 关键字段一致。

    结构不一致续训轻则静默错训、重则 state_dict 形状崩溃；不一致即报错退出并打印
    差异字段表。极旧 checkpoint 无 model_cfg 字段时降级为警告（不阻塞）。
    """
    if not ckpt_model_cfg:
        print("[resume] 警告：checkpoint 无 model_cfg 字段（极旧格式），跳过结构一致性校验")
        return
    diffs = [(k, ckpt_model_cfg.get(k, "<缺失>"), getattr(cur, k))
             for k in RESUME_CRITICAL_FIELDS
             if ckpt_model_cfg.get(k, "<缺失>") != getattr(cur, k)]
    if diffs:
        table = "\n".join(f"  {k}: ckpt={old!r} ≠ 当前={new!r}" for k, old, new in diffs)
        raise SystemExit(f"[resume] model_cfg 关键字段不一致，拒绝续训：\n{table}")


def build_model_config(cfg: dict) -> ModelConfig:
    """从训练 cfg 字典构建 ModelConfig（train.main 与 scripts/train_dp.py 共用）。

    模型尺寸字段缺省 = ModelConfig 0.1B 默认值，既有 config 零改动；
    0.5B 等更大配置在 JSON 显式写 d_model/n_layer/mlp_hidden/头数 + "check_0p1b_params": false。
    """
    return ModelConfig(
        vocab_size=cfg.get("vocab_size", 32768),
        d_model=cfg.get("d_model", 768),
        n_layer=cfg.get("n_layer", 12),
        n_q_heads=cfg.get("n_q_heads", 12),
        n_kv_heads=cfg.get("n_kv_heads", 4),
        head_dim=cfg.get("head_dim", 64),
        n_v_heads=cfg.get("n_v_heads", 12),
        n_qk_heads=cfg.get("n_qk_heads", 6),
        mlp_hidden=cfg.get("mlp_hidden", 2048),
        max_seq=cfg.get("max_seq", 1024),
        # RoPE 上下文扩充（2026-07-31）：默认 none/1.0/None = 与旧版逐 bit 一致；
        # 扩窗 config JSON 显式 "rope_scaling": "yarn" + "rope_scale" + "max_seq"。
        rope_scaling=cfg.get("rope_scaling", "none"),
        rope_scale=cfg.get("rope_scale", 1.0),
        rope_original_max_seq=cfg.get("rope_original_max_seq", None),
        check_0p1b_params=cfg.get("check_0p1b_params", True),
        # block_pattern 缺省 = ModelConfig 默认 G2G2G2A（GDN-2，2026-07-27 切换）；
        # config JSON 可显式 "block_pattern": ["G","G","G","A"] 做 GDN-1 消融对照。
        block_pattern=cfg.get("block_pattern", ["G2", "G2", "G2", "A"]),
        pm_stream=cfg.get("pm_stream", 1),
        pm_constrain=cfg.get("pm_constrain", True),
        tri_window=cfg.get("tri_window", 512),
        tri_csa_stride=cfg.get("tri_csa_stride", 4),
        tri_csa_topk=cfg.get("tri_csa_topk", 128),
        tri_hca_stride=cfg.get("tri_hca_stride", 128),
        # tri_use_indexer 默认对齐 ModelConfig 扶正值 True（2000 步消融扶正的 V4 正式路径）；
        # config JSON 可显式 "tri_use_indexer": false 做 NSA 消融对照。
        tri_use_indexer=cfg.get("tri_use_indexer", True),
        tri_index_heads=cfg.get("tri_index_heads", 4),
        tri_index_dim=cfg.get("tri_index_dim", 32),
        # GDN decay 参数化（2026-07-27 有界化）：config JSON 显式 "gdn_decay_g_min" 控制——
        # -5.0=K3 式有界 sigmoid（新训练默认）；null=旧式无界（复现旧 checkpoint 语义）。
        # 缺省用 ModelConfig 默认（-5.0 有界）；无界对照 run 须显式 "gdn_decay_g_min": null。
        gdn_decay_g_min=cfg.get("gdn_decay_g_min", -5.0),
        kernel_enabled=cfg.get("kal_aux_weight", 0.0) > 0.0,
        kernel_sense_layers=cfg.get("kal_sense_layers") or [],
        # 以下字段此前不透传（config JSON 显式写也被静默忽略），2026-07-31 补齐；
        # 缺省一律回退 ModelConfig 默认值，既有 config JSON 行为零变化。
        pm_sk_t_max=cfg.get("pm_sk_t_max", 20),
        grad_checkpoint=cfg.get("grad_checkpoint", True),
        rope_theta=cfg.get("rope_theta", 10000.0),
        rms_eps=cfg.get("rms_eps", 1e-6),
        conv_kernel=cfg.get("conv_kernel", 4),
        kernel_dg_dim=cfg.get("kernel_dg_dim", 256),
        kernel_dg_topk=cfg.get("kernel_dg_topk", 32),
        manifold_dim=cfg.get("manifold_dim", 64),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="latest.pt 路径；断点续训")
    ap.add_argument("--max_steps", type=int, default=None, help="临时覆盖 max_steps（短跑验证用）")
    ap.add_argument("--micro_batch", type=int, default=None, help="临时覆盖 micro_batch（新机显存标定用）")
    ap.add_argument("--grad_accum", type=int, default=None, help="临时覆盖 grad_accum（配合 micro_batch 保持全局 batch）")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_dir", default=None, help="临时覆盖数据目录（中训练换数据用，与 --max_steps 同模式）")
    ap.add_argument("--init_from", default=None,
                    help="save_pretrained 目录：仅载权重（step=0、全新优化器），中训练/退火阶段初始化用；与 --resume 互斥")
    args = ap.parse_args()
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("max_steps", "micro_batch", "grad_accum", "run_name", "out_dir", "data_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(f"[cfg] {json.dumps(cfg, ensure_ascii=False)}")

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    device = "cuda"
    # TF32：GDN 核心算子内部 fp32 GEMM 走 tensor core（训练吞吐 ~1.4x；单元对拍不开启，仍全 fp32）
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # 效率加固：eager 路径 fp32 GEMM 统一走 TF32 tensor core；固定 shape 下 cudnn 算法自动调优。
    # 两者仅影响训练吞吐，不改变数值语义（仍 bf16 autocast + fp32 主权重）；单元测试不经过 main，不受影响。
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

    # pm_stream/pm_constrain 缺省 = 1/True（单流基线零改动）；PM 消融在 config JSON 中加 "pm_stream": 5
    # 模型尺寸字段（0.5B 等更大配置）：缺省 = ModelConfig 0.1B 默认值，既有 config 零改动；
    # 0.5B 配置 JSON 显式写 d_model/n_layer/mlp_hidden/头数 + "check_0p1b_params": false。
    model_cfg = build_model_config(cfg)
    model = TaisObsidianForCausalLM(model_cfg).to(device)
    opt = build_optimizer(model, cfg)
    train_shards = Shards(cfg["data_dir"], "train")
    val_shards = Shards(cfg["data_dir"], "val")
    print(f"[data] train {train_shards.total/1e6:.1f}M tokens, val {val_shards.total/1e6:.1f}M")

    from torch.utils.tensorboard import SummaryWriter

    out_dir = Path(cfg["out_dir"])
    writer = SummaryWriter(log_dir=f"runs/{cfg['run_name']}")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # 结构一致性校验：model_cfg 关键字段与当前 config 不符即报错退出（防错配静默续训）
        validate_resume_model_cfg(ckpt.get("model_cfg"), model_cfg)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        # map_location=device 会把 CPU RNG ByteTensor 也映射上 GPU，set_rng_state 只收 CPU 张量
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        if ckpt["rng"]["cuda"] is not None:
            try:
                # 双卡 checkpoint 单卡 resume：状态列表切片到本机可见设备数；
                # map_location=device 会把 CPU ByteTensor 映射上 GPU（isinstance 校验失败），逐个搬回 CPU；
                # 越界/损坏等异常降级为警告（按当前种子继续，不阻塞续训）
                torch.cuda.set_rng_state_all(
                    [s.cpu() for s in ckpt["rng"]["cuda"][: torch.cuda.device_count()]]
                )
            except Exception as e:
                print(f"[resume] 警告：CUDA RNG 状态恢复失败（{e}），按当前种子继续")
        rng = np.random.default_rng()
        rng.bit_generator.state = ckpt["rng"]["numpy"]
        print(f"[resume] 从 {args.resume} 恢复，step={start_step}")

    if args.init_from:
        # 中训练/退火初始化：仅载 save_pretrained 权重，step=0、全新优化器（OLMo Dolmino
        # 式独立退火 run 惯例——WSD 调度从本阶段 step 0 重新计，与 --resume 互斥）。
        if args.resume:
            raise SystemExit("[init_from] 与 --resume 互斥：中训练用 --init_from，断点续训用 --resume")
        src_model = TaisObsidianForCausalLM.from_pretrained(args.init_from, device="cpu")
        model.load_state_dict(src_model.state_dict(), strict=True)
        del src_model
        print(f"[init_from] 从 {args.init_from} 载入权重（step=0，全新优化器）")

    # micro_batch OOM 自动降级：16 → 8，grad_accum 翻倍保持 global batch
    oom_checked = False
    step = start_step
    tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
    print(f"[train] global batch {tokens_per_step/1024:.0f}k tokens/step, "
          f"micro {cfg['micro_batch']}×accum {cfg['grad_accum']}×seq {cfg['seq_len']}")

    ema_loss = None
    t_log = time.time()
    # try/finally：训练中途崩溃（OOM 不可恢复、键盘中断等）也尽量 writer.close() flush 已写标量
    try:
        while step < cfg["max_steps"]:
            lr = lr_at(step, cfg)
            set_lr(opt, lr, cfg)
            opt.zero_grad(set_to_none=True)
            loss_accum = 0.0
            try:
                for _ in range(cfg["grad_accum"]):
                    x, y = train_shards.get_batch(cfg["micro_batch"], cfg["seq_len"], device, rng)
                    with torch.autocast("cuda", torch.bfloat16):
                        logits, _ = model(x)
                        loss = chunked_ce(logits, y)
                        # KAL 内生辅助损失（T1 预训练后期；默认关）。
                        # 注意：kal_pik_aux_loss 内部对主干 hidden detach，探针梯度只进 KAL 头
                        # （监测/执行分置红线）；主干 CE 仍走完整计算图。
                        if cfg.get("kal_aux_weight", 0.0) > 0.0:
                            sense_layers = model.kernel_sense_index()
                            aux = kal_pik_aux_loss(model, x, y, sense_layers)
                            if aux is not None:
                                loss = loss + cfg["kal_aux_weight"] * aux
                    (loss / cfg["grad_accum"]).backward()
                    loss_accum += loss.item() / cfg["grad_accum"]
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                if not oom_checked and cfg["micro_batch"] > 1:
                    cfg["micro_batch"] //= 2
                    cfg["grad_accum"] *= 2
                    oom_checked = True
                    tokens_per_step = cfg["micro_batch"] * cfg["grad_accum"] * cfg["seq_len"]
                    try:
                        torch.cuda.empty_cache()
                    except torch.AcceleratorError:
                        # CUDA 上下文已被 OOM 污染（kernel 内 OOM 不可恢复），只能重启进程
                        raise RuntimeError(
                            "CUDA OOM 发生在 kernel 内，上下文不可恢复；请以更小 micro_batch 重启"
                        ) from e
                    # 不重建优化器：micro/accum 只影响 batch 切分，与优化器状态（动量/二阶矩）
                    # 无关；此前 build_optimizer 重建会静默丢失全部一阶/二阶矩（训练质量跳变）。
                    print(f"[oom] micro_batch 降至 {cfg['micro_batch']}，grad_accum 升至 {cfg['grad_accum']}，重试本步")
                    continue
                raise
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            step += 1

            ema_loss = loss_accum if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_accum
            if step % cfg["log_every"] == 0 or step == 1:
                dt = time.time() - t_log
                n_tok = tokens_per_step * (cfg["log_every"] if step > 1 else 1)
                tok_s = n_tok / dt if step > 1 else tokens_per_step / dt
                mem = torch.cuda.max_memory_allocated() / 1024**3
                print(f"step {step:5d} | loss {loss_accum:.4f} (ema {ema_loss:.4f}) | lr {lr:.2e} | "
                      f"gnorm {grad_norm.item():.2f} | {tok_s/1e3:.1f}k tok/s | mem {mem:.2f}GB")
                writer.add_scalar("train/loss", loss_accum, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                writer.add_scalar("train/tok_per_s", tok_s, step)
                t_log = time.time()
            if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
                vl = eval_val(model, val_shards, cfg, device, rng)
                print(f"step {step:5d} | val loss {vl:.4f}")
                writer.add_scalar("val/loss", vl, step)
            if step % cfg["ckpt_every"] == 0 or step == cfg["max_steps"]:
                save_checkpoint(out_dir / "latest.pt", model, opt, step, cfg, rng)
                print(f"[ckpt] step {step} → {out_dir/'latest.pt'}")

        # 结束：另存 final bf16 save_pretrained + 复制 tokenizer（随权重产物，下载即可推理）
        model.save_pretrained(out_dir / "final")
        copy_tokenizer_to_final(out_dir / "final")
        print(f"[done] final 模型（bf16）→ {out_dir/'final'}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
