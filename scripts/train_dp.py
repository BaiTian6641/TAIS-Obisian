"""双卡单进程手动数据并行（DP）训练脚本（Windows 无 NCCL，DDP/gloo 不可用）。

架构：同一进程内两份模型实例 + worker 独立线程（计算重叠）——
  - master  = cuda:1（RTX PRO 4000，24GB 计算卡），主线程持有 Muon/AdamW 优化器，
    负责梯度汇总、grad clip、optimizer step、checkpoint、tensorboard；
  - worker  = cuda:0（RTX 4070，8GB 副卡/显示卡），**独立线程**跑 forward+backward，
    与 master 的 micro-batch 循环真正并行（两卡不同 CUDA device，kernel 并发）。

每个 step（含 grad_accum 个 micro 步）的执行时序：
  1. 主线程发 "run" 命令后立即开始自己的 accum 循环；worker 线程同时：
     a. 等 master 上一步 opt.step 的 CUDA event（ev_after_step），拉取最新参数
        （master→worker 广播与 master 本步前若干个 micro-batch 计算重叠）；
     b. 各吃不同大小的 micro-batch 与 accum 数（时间均衡分片：慢卡小 micro × 多 accum，
        --bench 按实测速度比自动测定），bf16 autocast forward+backward，损失按
        `本卡 tokens / 全 batch tokens` 加权——两卡梯度之和 = 全 batch 平均梯度的无偏估计；
     c. backward 完成后，把梯度在专用 side stream 上 D2D 拷进 master 侧预分配缓冲
        （与 master 计算重叠；--bf16_transfer 可压 bf16 省一半 PCIe 带宽）；
  2. 主线程等 worker 梯度就绪（threading.Queue，异常一并传播），CUDA stream 等
     传输 event 后把缓冲累加进 master 梯度；
  3. master 上 grad clip 1.0 + optimizer step，记录 ev_after_step 供下一步广播等待。

线程安全约束：worker 线程只触碰 worker 模型参数/梯度、自己的 rng 流与 master 侧
梯度缓冲（经 CUDA event 定序）；优化器/checkpoint/eval 只在主线程。两卡写 master
参数的时间窗由 event 链严格隔开（opt.step 写 → ev_after_step → 广播读 → worker
fwd/bwd → 梯度就绪 → 下一次 opt.step 写）。

注意：**不要设 CUDA_VISIBLE_DEVICES**（此前单卡命令前缀 =1 只用 PRO 4000；
双卡模式需要两张卡同时可见，cuda:0=4070 / cuda:1=PRO 4000）。

用法：
  # bench 模式：两卡各跑 N 步测 tok/s，自动定 worker micro-batch 后进入正式训练
  python scripts/train_dp.py --config configs/pilot_0p5b_gdn2.json --bench 10
  # 手动指定 worker micro-batch（跳过 bench）
  python scripts/train_dp.py --config configs/pilot_0p5b_gdn2.json --worker_batch 2
  # 断点续训
  python scripts/train_dp.py --config configs/pilot_0p5b_gdn2.json --resume checkpoints/pilot_0p5b_gdn2/latest.pt

checkpoint 只存 master 模型，格式与 train.py 的 save_checkpoint 完全一致
（latest.pt 可被 train.py --resume 与 save_pretrained/from_pretrained 链路复用，
final/ 目录可被 `python -m tais_obsidian.generate --ckpt` 直接加载）。
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# 允许从仓库根目录直接 `python scripts/train_dp.py` 运行（src 布局包装进 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tais_obsidian.data.memmap import Shards
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.train import (
    DEFAULTS,
    build_model_config,
    build_optimizer,
    chunked_ce,
    eval_val,
    lr_at,
    save_checkpoint,
    set_lr,
)


def enable_tf32() -> None:
    """TF32 tensor core（对齐 train.py main 的效率加固，不改数值语义）。"""
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def sync_worker_from_master(master: torch.nn.Module, worker: torch.nn.Module) -> None:
    """把 master 参数/缓冲广播回 worker（初始化/续训时的阻塞式全量同步）。"""
    with torch.no_grad():
        for p_m, p_w in zip(master.parameters(), worker.parameters()):
            p_w.copy_(p_m.to(p_w.device), non_blocking=True)
        for b_m, b_w in zip(master.buffers(), worker.buffers()):
            b_w.copy_(b_m.to(b_w.device), non_blocking=True)
    torch.cuda.synchronize()


def weighted_micro_step(
    model: torch.nn.Module,
    shards: Shards,
    micro_batch: int,
    seq_len: int,
    device: str,
    rng: np.random.Generator,
    weight: float,
) -> torch.Tensor:
    """单卡一个 micro 步：forward + 按 token 占比加权的 backward，返回 detach 的 loss 张量。

    weight = 本卡本步 tokens / 全 step 总 tokens（含 grad_accum 归一）。
    两卡梯度之和即全 batch 平均梯度（与单卡全 batch 等价，差数值误差）。
    返回 GPU 上的 loss 张量而非 float：避免 .item() 强制同步把两卡串行化。
    """
    x, y = shards.get_batch(micro_batch, seq_len, device, rng)
    with torch.autocast("cuda", torch.bfloat16):
        logits, _ = model(x)
        loss = chunked_ce(logits, y)
    (loss * weight).backward()
    return loss.detach()


def zero_grads(*models: torch.nn.Module) -> None:
    for m in models:
        for p in m.parameters():
            p.grad = None


class WorkerNode:
    """worker 卡计算节点：独立线程跑 参数拉取 → accum fwd/bwd → 梯度回传 master 缓冲。

    与主线程的同步原语：
      - ``_cmd`` Queue：主→worker 命令（"run" / "stop"）；
      - ``_ready`` Queue：worker→主 结果（("ok", losses, ev_xfer) / ("err", traceback)）；
      - ``ev_after_step`` CUDA event：master 上一次 opt.step 完成标记（worker 拉参数前等待）；
      - ``ev_xfer`` CUDA event：worker 梯度拷贝完成标记（master 累加前等待）。
    异常不静默：worker 内任何异常经 ("err", tb) 传回主线程抛出。
    """

    def __init__(
        self,
        worker: torch.nn.Module,
        wg_buf: list[torch.Tensor],
        ev_after_step: torch.cuda.Event,
        dev_w: str,
        dev_m: str,
        shards: Shards,
        micro_batch: int,
        seq_len: int,
        grad_accum: int,
        weight: float,
        bf16_transfer: bool,
        rng: np.random.Generator,
    ):
        self.worker = worker
        self.wg_buf = wg_buf            # master 设备上的梯度缓冲（主线程预分配/清零）
        self.ev_after_step = ev_after_step
        self.dev_w, self.dev_m = dev_w, dev_m
        self.shards = shards
        self.micro_batch, self.seq_len = micro_batch, seq_len
        self.grad_accum, self.weight = grad_accum, weight
        self.bf16_transfer = bf16_transfer
        self.rng = rng                  # 仅 worker 线程使用（与主线程 rng_m 不同流）
        self._cmd: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="dp-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def run_async(self) -> None:
        self._cmd.put("run")

    def wait_ready(self) -> tuple[list[torch.Tensor], torch.cuda.Event]:
        """阻塞等 worker 完成本步；worker 异常在此重新抛出（不静默吞掉）。"""
        tag, *payload = self._ready.get()
        if tag == "err":
            raise RuntimeError(f"worker 线程异常：\n{payload[0]}")
        return payload[0], payload[1]

    def stop(self) -> None:
        self._cmd.put("stop")
        self._thread.join(timeout=30)

    def _loop(self) -> None:
        try:
            torch.cuda.set_device(self.dev_w)  # 线程首次用 CUDA 前设定设备
            s_pull = torch.cuda.Stream(self.dev_w)    # 参数拉取流（与 master 计算重叠）
            s_xfer = torch.cuda.Stream(self.dev_m)    # 梯度回传流（与 master 计算重叠）
            params = [p for p in self.worker.parameters() if p.requires_grad]
            while True:
                cmd = self._cmd.get()
                if cmd == "stop":
                    return
                # 1) 拉取 master 最新参数（等上次 opt.step 完成；与 master 本步计算重叠）
                with torch.cuda.stream(s_pull):
                    s_pull.wait_event(self.ev_after_step)
                    with torch.no_grad():
                        for p_w, p_m in zip(self.worker.parameters(), self.master_params):
                            p_w.copy_(p_m, non_blocking=True)
                ev_pull = torch.cuda.Event()
                ev_pull.record(s_pull)
                torch.cuda.current_stream(self.dev_w).wait_event(ev_pull)
                # 2) 清梯度 + accum 个 micro 步 fwd/bwd（损失已含 token 权重）
                for p in params:
                    p.grad = None
                losses: list[torch.Tensor] = []
                for _ in range(self.grad_accum):
                    losses.append(weighted_micro_step(
                        self.worker, self.shards, self.micro_batch, self.seq_len,
                        self.dev_w, self.rng, self.weight,
                    ))
                # 3) 梯度拷进 master 侧缓冲（side stream，等 backward 完成；与 master 计算重叠）
                ev_bwd = torch.cuda.Event()
                ev_bwd.record(torch.cuda.current_stream(self.dev_w))
                with torch.cuda.stream(s_xfer):
                    s_xfer.wait_event(ev_bwd)
                    with torch.no_grad():
                        for buf, p in zip(self.wg_buf, params):
                            if p.grad is None:
                                continue
                            g = p.grad.to(torch.bfloat16) if self.bf16_transfer else p.grad
                            buf.copy_(g, non_blocking=True)
                ev_xfer = torch.cuda.Event()
                ev_xfer.record(s_xfer)
                self._ready.put(("ok", losses, ev_xfer))
        except Exception:
            self._ready.put(("err", traceback.format_exc()))

    # master 参数列表由主线程注入（worker 只读，配合 ev_after_step 定序）
    master_params: list[torch.Tensor] = []


def bench_card(
    model: torch.nn.Module,
    shards: Shards,
    micro_batch: int,
    seq_len: int,
    device: str,
    rng: np.random.Generator,
    steps: int,
) -> float:
    """在指定卡上跑 steps 个 micro 步（fwd+bwd），返回 tok/s（含一步 warmup）。"""
    x, y = shards.get_batch(micro_batch, seq_len, device, rng)
    with torch.autocast("cuda", torch.bfloat16):
        logits, _ = model(x)
        loss = chunked_ce(logits, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    t0 = time.time()
    for _ in range(steps):
        x, y = shards.get_batch(micro_batch, seq_len, device, rng)
        with torch.autocast("cuda", torch.bfloat16):
            logits, _ = model(x)
            loss = chunked_ce(logits, y)
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    return micro_batch * seq_len * steps / dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="latest.pt 路径；断点续训（载入 master 后广播 worker）")
    ap.add_argument("--max_steps", type=int, default=None, help="临时覆盖 max_steps（短跑验证用）")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--data_dir", default=None, help="临时覆盖 data_dir（冒烟用 0.1B shards）")
    ap.add_argument("--bench", type=int, default=0, metavar="N",
                    help="bench 模式：两卡各跑 N 步测 tok/s，自动定 worker micro-batch 后进入训练")
    ap.add_argument("--worker_batch", type=int, default=None,
                    help="worker(4070) micro-batch；不给则需 --bench 自动测定，或配置 JSON 的 dp_worker_batch")
    ap.add_argument("--worker_accum", type=int, default=None,
                    help="worker 独立 accum 数（时间均衡：慢卡小 micro × 多 accum ≈ 快卡大 micro × 少 accum；"
                         "不给则 = grad_accum，或 --bench 自动按速度比测定）")
    ap.add_argument("--master_device", default="cuda:1", help="master 设备（PRO 4000）")
    ap.add_argument("--worker_device", default="cuda:0", help="worker 设备（4070）")
    ap.add_argument("--bf16_transfer", action="store_true",
                    help="梯度传输压缩为 bf16（省 PCIe 带宽；默认 fp32 保精度）")
    args = ap.parse_args()

    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(args.config).read_text(encoding="utf-8")))
    for key in ("max_steps", "run_name", "out_dir", "data_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    print(f"[cfg] {json.dumps(cfg, ensure_ascii=False)}")

    dev_m, dev_w = args.master_device, args.worker_device
    assert torch.cuda.device_count() >= 2, "双卡 DP 需要 ≥2 张可见 GPU（勿设 CUDA_VISIBLE_DEVICES=1）"

    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    np.random.seed(cfg["seed"])
    enable_tf32()

    model_cfg = build_model_config(cfg)
    master = TaisObsidianForCausalLM(model_cfg).to(dev_m)
    worker = TaisObsidianForCausalLM(model_cfg).to(dev_w)
    sync_worker_from_master(master, worker)  # 保证两卡初始化完全一致
    opt = build_optimizer(master, cfg)  # 优化器只在 master

    train_shards = Shards(cfg["data_dir"], "train")
    val_shards = Shards(cfg["data_dir"], "val")
    print(f"[data] train {train_shards.total/1e6:.1f}M tokens, val {val_shards.total/1e6:.1f}M")
    # 两卡各自独立随机流采样不同 micro-batch（同流会采到重复数据）；
    # rng_m 主线程用，rng_w 移交 worker 线程专用
    rng_m = np.random.default_rng(cfg["seed"])
    rng_w = np.random.default_rng(cfg["seed"] + 1000)

    from torch.utils.tensorboard import SummaryWriter

    out_dir = Path(cfg["out_dir"])
    writer = SummaryWriter(log_dir=f"runs/{cfg['run_name']}")

    micro_m = cfg["micro_batch"]
    tm = tw = None  # bench 实测速度（tok/s），仅 --bench 路径有值
    # worker micro-batch：CLI > 配置 JSON(dp_worker_batch) > --bench 自动测定
    micro_w = args.worker_batch or cfg.get("dp_worker_batch")
    if micro_w is None:
        if args.bench <= 0:
            raise SystemExit("需 --worker_batch / 配置 dp_worker_batch / --bench N 三选一确定 worker micro-batch")
        print(f"[bench] 两卡各跑 {args.bench} 步测 tok/s（master micro {micro_m}）…")
        tm = bench_card(master, train_shards, micro_m, cfg["seq_len"], dev_m, rng_m, args.bench)
        print(f"[bench] master({dev_m}) {tm/1e3:.1f}k tok/s")
        # worker(4070) 显存小：从 micro_m 起折半重试直到不 OOM（上限即该卡可装的 micro 上限）
        tw = None
        micro_w_fit = micro_m
        while micro_w_fit >= 1:
            try:
                tw = bench_card(worker, train_shards, micro_w_fit, cfg["seq_len"], dev_w, rng_w, args.bench)
                break
            except (torch.cuda.OutOfMemoryError, torch.AcceleratorError) as e:
                if "out of memory" not in str(e).lower():
                    raise
                micro_w_fit //= 2
                print(f"[bench] worker micro OOM → 折半重试 micro {micro_w_fit}")
                # OOM 后上下文通常可用（算子分配失败非 kernel 内 OOM）；清空缓存再继续
                try:
                    torch.cuda.empty_cache()
                except torch.AcceleratorError:
                    raise RuntimeError("worker CUDA 上下文被 OOM 污染，请减小 micro 重启") from e
        if tw is None:
            raise SystemExit("worker 即使 micro=1 也 OOM，无法双卡 DP")
        mem_w = torch.cuda.max_memory_allocated(dev_w) / 1024**3
        print(f"[bench] worker({dev_w}) {tw/1e3:.1f}k tok/s @ micro {micro_w_fit}（峰值 {mem_w:.2f}GB）")
        # 按速度比分片（时间均衡：micro_w/tw ≈ micro_m/tm），同时不超过 worker 显存上限
        micro_w = max(1, min(micro_w_fit, round(micro_m * tw / tm)))
        print(f"[bench] 速度比 master:worker = {tm/tw:.2f}:1 → worker micro-batch = {micro_w}")
    # worker 独立 accum（时间均衡：慢卡小 micro × 多 accum ≈ 快卡大 micro × 少 accum，
    # 让两卡每步耗时相近、token 贡献不再受 worker 显存上限压缩）：
    # CLI > 配置 dp_worker_accum > bench 按速度比自动 > 退回 = grad_accum
    accum_w = args.worker_accum or cfg.get("dp_worker_accum")
    if accum_w is None:
        if tm is not None and tw is not None:
            accum_w = max(1, round(cfg["grad_accum"] * micro_m * tw / (micro_w * tm)))
            print(f"[bench] 时间均衡 → worker accum = {accum_w}（master accum {cfg['grad_accum']}）")
        else:
            accum_w = cfg["grad_accum"]
    tokens_m = micro_m * cfg["grad_accum"] * cfg["seq_len"]
    tokens_w = micro_w * accum_w * cfg["seq_len"]
    tokens_per_step = tokens_m + tokens_w
    if tm is not None and tw is not None:
        # 重叠 DP 预估吞吐 ≈ tokens_per_step / max(两卡各自耗时)；低于单卡 master 吞吐则建议单卡
        t_m = tokens_m / tm
        t_w = tokens_w / tw
        est = tokens_per_step / max(t_m, t_w)
        print(f"[bench] 重叠 DP 预估 {est/1e3:.1f}k tok/s（vs 单卡 master {tm/1e3:.1f}k）")
        if est < tm:
            print("[bench] 注意：预估双卡无收益（worker 贡献不足以覆盖开销），建议改用单卡训练")
    print(f"[dp] master micro {micro_m}×accum {cfg['grad_accum']} + worker micro {micro_w}×accum {accum_w}，"
          f"global batch {tokens_per_step / 1024:.0f}k tokens/step（worker 贡献 {tokens_w/tokens_per_step:.0%}）")

    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=dev_m, weights_only=False)
        master.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        rng_m = np.random.default_rng()
        rng_m.bit_generator.state = ckpt["rng"]["numpy"]
        rng_w = np.random.default_rng(cfg["seed"] + 1000 + start_step)
        sync_worker_from_master(master, worker)
        print(f"[resume] 从 {args.resume} 恢复，step={start_step}（已广播 worker）")

    # 每 micro 步的梯度权重 = 本卡 tokens / 全 step tokens（含各自 accum 归一）
    w_m = micro_m * cfg["seq_len"] / tokens_per_step
    w_w = micro_w * cfg["seq_len"] / tokens_per_step

    # master 侧 worker 梯度缓冲（worker 线程经 side stream 写入；fp32 或 bf16 传输）
    buf_dtype = torch.bfloat16 if args.bf16_transfer else torch.float32
    wg_buf = [torch.zeros(p.shape, dtype=buf_dtype, device=dev_m)
              for p in master.parameters() if p.requires_grad]
    master_params = [p for p in master.parameters() if p.requires_grad]
    ev_after_step = torch.cuda.Event()
    ev_after_step.record(torch.cuda.current_stream(dev_m))  # step 0：初始同步即"已完成"
    node = WorkerNode(worker, wg_buf, ev_after_step, dev_w, dev_m, train_shards,
                      micro_w, cfg["seq_len"], accum_w, w_w,
                      args.bf16_transfer, rng_w)
    node.master_params = master_params
    node.start()

    ema_loss = None
    t_log = time.time()
    step = start_step
    try:
        while step < cfg["max_steps"]:
            lr = lr_at(step, cfg)
            set_lr(opt, lr, cfg)  # Muon 组同步缩放 muon_lr/adamw_lr（WSD 生效）
            zero_grads(master)
            for buf in wg_buf:
                buf.zero_()
            node.run_async()  # worker：拉参数（等 ev_after_step）→ fwd/bwd → 梯度回传
            losses_m: list[torch.Tensor] = []
            for _ in range(cfg["grad_accum"]):
                losses_m.append(weighted_micro_step(master, train_shards, micro_m, cfg["seq_len"], dev_m, rng_m, w_m))
            losses_w, ev_xfer = node.wait_ready()
            torch.cuda.current_stream(dev_m).wait_event(ev_xfer)
            with torch.no_grad():
                for p, buf in zip(master_params, wg_buf):
                    if p.grad is None:
                        continue  # 未参与本步前向的参数（两卡均无梯度贡献）
                    p.grad.add_(buf.to(p.grad.dtype) if buf.dtype != p.grad.dtype else buf)
            grad_norm = torch.nn.utils.clip_grad_norm_(master.parameters(), cfg["grad_clip"])
            opt.step()
            ev_after_step.record(torch.cuda.current_stream(dev_m))  # 供 worker 下一步拉参数等待
            step += 1

            # 加权平均 loss（按 tokens 占比；两卡之和即全 batch 平均 loss；
            # 分母 = 两卡各自 micro×accum 之和，seq 约去）
            loss_m = torch.stack(losses_m).sum().item()
            loss_w = torch.stack(losses_w).sum().item()
            loss_step = (loss_m * micro_m + loss_w * micro_w) / (
                micro_m * cfg["grad_accum"] + micro_w * accum_w)
            ema_loss = loss_step if ema_loss is None else 0.98 * ema_loss + 0.02 * loss_step
            if step % cfg["log_every"] == 0 or step == 1:
                dt = time.time() - t_log
                n_tok = tokens_per_step * (cfg["log_every"] if step > 1 else 1)
                tok_s = n_tok / dt
                mem_m = torch.cuda.max_memory_allocated(dev_m) / 1024**3
                mem_w = torch.cuda.max_memory_allocated(dev_w) / 1024**3
                print(f"step {step:5d} | loss {loss_step:.4f} (ema {ema_loss:.4f}) | lr {lr:.2e} | "
                      f"gnorm {grad_norm.item():.2f} | {tok_s/1e3:.1f}k tok/s | "
                      f"mem M {mem_m:.2f}GB / W {mem_w:.2f}GB")
                writer.add_scalar("train/loss", loss_step, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_norm.item(), step)
                writer.add_scalar("train/tok_per_s", tok_s, step)
                t_log = time.time()
            if step % cfg["val_every"] == 0 or step == cfg["max_steps"]:
                vl = eval_val(master, val_shards, cfg, dev_m, rng_m)
                print(f"step {step:5d} | val loss {vl:.4f}")
                writer.add_scalar("val/loss", vl, step)
            if step % cfg["ckpt_every"] == 0 or step == cfg["max_steps"]:
                save_checkpoint(out_dir / "latest.pt", master, opt, step, cfg, rng_m)
                print(f"[ckpt] step {step} → {out_dir/'latest.pt'}")

        master.save_pretrained(out_dir / "final")
        print(f"[done] final 模型（bf16，仅 master）→ {out_dir/'final'}")
    finally:
        node.stop()
    writer.close()


if __name__ == "__main__":
    main()
