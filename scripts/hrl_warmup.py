"""T2 HRL LightningIndexer KL warmup（DSA 稀疏训练阶段范式）。

目标：把 HRL 的 LightningIndexer 从"随机初始化结构通路"训练成"有语义的检索器"——
用 KL 散度对齐 indexer 分布到稠密教师分布（DeepSeek V3.2 DSA warmup 范式：
先冻结主干，短校准对齐 indexer 到稠密主注意力分布，再开 top-k 稀疏训练）。

数据源与教师（诚实标注，避免盲目猜测）：
- query/candidates = 真实 checkpoint 的某 GDN sense 层 hidden state（capture_layers 提取，
  PM-stream 关闭=单流时 captures[i] 即该层输出 [B,T,d]）；
- 稠密教师 teacher_scores [B,Tq,Tk] = 同层段上真实注意力 q·k 分数（用第一个 "A" 层的
  归一化 q/k 投影逐 token 点积，跨 kv 头均值）——这是模型自带的、无需外部标签的
  "什么样的检索分布是对的"的内生答案（对映 HRLIndexer.init_indexer_from_model 的
  q_proj 打分方向聚合初始化，同一信号源的分布级对齐）。

红线落实：
- 主干全程 frozen（requires_grad=False + no_grad 提 hidden/教师），梯度只进 indexer 权重；
- kl_warmup_loss 内部 teacher.detach()；HRLIndexer.forward detach_input=True 双保险；
- warmup 只做 token 域（压缩条目域同构，块域检索复用同一打分器，设计 §11.1）。

产出：
- runs/hrl_warmup/warmed_indexer.pt（LightningIndexer state_dict，可灌回内核）；
- runs/hrl_warmup/report.json（loss 曲线端点 / 收敛比 / 超参 / top-k 重叠率）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/hrl_warmup.py \
      [--ckpt checkpoints/pilot_0p1b_ws/final] [--steps 1000] [--seq_len 512]
"""
from __future__ import annotations

import argparse
import json
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
from tais_obsidian.model.tais_kernel import TAISKernel  # noqa: E402

DEVICE = "cuda"


@torch.no_grad()
def extract_hidden_and_teacher(
    model: TaisObsidianForCausalLM, x: torch.Tensor, sense_layer: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次前向提取 sense 层 hidden [B,T,d] 与第一个 A 层的稠密教师分数 [B,T,T]。

    教师 = 该 A 层归一化 q·k 逐 token 点积（跨 kv 头均值），detach。因果 mask 由
    kl_warmup_loss 的 softmax 天然处理（对齐分布即可；可选下三角 mask，见下）。
    """
    model.eval()
    _, _, captures = model(x, capture_layers=[sense_layer])
    hidden = captures[sense_layer]  # [B,T,d]（单流：直接是该层输出）
    # 第一个 "A" 层的 q/k 投影派生教师分数
    for layer in model.layers:
        if layer.type == "A":
            mixer = layer.mixer
            break
    else:
        raise RuntimeError("无 'A' 层，无法派生教师分数（fail-closed）")
    B, T, _ = hidden.shape
    # q_norm/k_norm 作用在 head_dim（64），须先 view 拆头再归一化（对齐 tri_attention 前向实现）
    q = mixer.q_norm(mixer.q_proj(hidden).view(B, T, mixer.n_q, mixer.head_dim))
    k = mixer.k_norm(mixer.k_proj(hidden).view(B, T, mixer.n_kv, mixer.head_dim))
    # 跨 kv 头分组（GQA：n_q//n_kv 个 q 头共享一个 kv 头），对组内均值 → [B,T,hd]
    group = mixer.n_q // mixer.n_kv
    q_grp = q.view(hidden.shape[0], hidden.shape[1], mixer.n_kv, group, mixer.head_dim).mean(dim=3)
    # 教师分数 [B,Tq,Tk]：q_grp·k 点积 / sqrt(hd)，跨 kv 头均值
    scale = mixer.head_dim ** -0.5
    teacher = torch.einsum("bihd,bjhd->bhij", q_grp, k).mean(dim=1) * scale  # [B,T,T]
    # 因果 mask：未来位置置 -inf（对齐自回归检索分布）
    T = teacher.shape[-1]
    mask = torch.triu(torch.ones(T, T, device=teacher.device, dtype=torch.bool), diagonal=1)
    teacher = teacher.masked_fill(mask, float("-inf"))
    return hidden.detach(), teacher.detach()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/pilot_0p1b_ws/final")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--sense_layer", type=int, default=8, help="GDN sense 读点层（0.1B ℓ8 探针最强）")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_index", type=int, default=32)
    ap.add_argument("--eval_topk", type=int, default=16, help="收敛评估的 top-k 重叠率 k")
    args = ap.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    rng = np.random.default_rng(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # 主干 frozen：compat 加载旧 checkpoint（旧 CSAAttention → 新 TriRetrievalAttention）
    model = TaisObsidianForCausalLM.from_pretrained(args.ckpt, strict=False).to(DEVICE)
    for p in model.parameters():
        p.requires_grad_(False)
    d_model = model.config.d_model
    print(f"[warmup] 主干 {args.ckpt} 已冻结，d_model={d_model}，sense_layer={args.sense_layer}")

    # HRL 内核 + LightningIndexer；q_proj 方向初始化（§11.1 近似）。
    # TAISKernel 内部 HRLIndexer(d_model) 默认 use_lightning=True, n_heads=4, d_index=32。
    # 自定义头/维时直接替换 kernel.hrl_indexer.lightning 的投影形状（此处用默认，args 仅记录）。
    kernel = TAISKernel(d_model=d_model).to(DEVICE)
    li = kernel.hrl_indexer.lightning
    assert li is not None, "LightningIndexer 未启用（fail-closed）"
    assert li.n_heads == args.n_heads and li.d_index == args.d_index, (
        f"heads/d_index 与内核默认不符（{li.n_heads}/{li.d_index} vs {args.n_heads}/{args.d_index}），"
        "请改 args 或扩展 TAISKernel 透传")
    init_layer = kernel.init_indexer_from_model(model)
    print(f"[warmup] indexer 初始化自 A 层 ℓ{init_layer} q_proj（方向聚合）")
    opt = torch.optim.AdamW(kernel.hrl_indexer.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)

    train_shards = Shards(ROOT / "data" / "shards", "train")
    val_shards = Shards(ROOT / "data" / "shards", "val")

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        # 线性衰减到 0.1×
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (1.0 - 0.9 * prog)

    @torch.no_grad()
    def eval_loss_and_overlap() -> tuple[float, float]:
        """验证集 KL + 教师/indexer top-k 重叠率（检索语义的直接指标）。"""
        losses, overlaps = [], []
        for _ in range(4):
            x, _ = val_shards.get_batch(args.batch, args.seq_len, DEVICE, rng)
            hidden, teacher = extract_hidden_and_teacher(model, x, args.sense_layer)
            loss = kernel.indexer_kl_warmup_loss(hidden, hidden, teacher)
            losses.append(loss.item())
            # top-k 重叠率：教师与 indexer 各自 top-k 的 Jaccard（忽略对角 trivial 自检索）
            idx_scores = kernel.route_candidates(hidden, hidden, k=None, detach_input=True)
            k_eff = min(args.eval_topk, hidden.shape[1] - 1)
            t_idx = teacher.topk(k_eff + 1, dim=-1).indices[..., 1:]  # 去掉 self
            s_idx = idx_scores.topk(k_eff + 1, dim=-1).indices[..., 1:]
            inter = (t_idx.unsqueeze(-1) == s_idx.unsqueeze(-2)).any(-1).float().mean()
            overlaps.append(inter.item())
        return float(np.mean(losses)), float(np.mean(overlaps))

    print(f"[warmup] steps={args.steps} lr={args.lr} heads={args.n_heads} d_index={args.d_index}")
    init_loss, init_overlap = eval_loss_and_overlap()
    print(f"[warmup] 初始：val_kl={init_loss:.4f} top-{args.eval_topk} 重叠率={init_overlap:.4f}")

    t0 = time.time()
    kernel.train()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        x, _ = train_shards.get_batch(args.batch, args.seq_len, DEVICE, rng)
        hidden, teacher = extract_hidden_and_teacher(model, x, args.sense_layer)
        loss = kernel.indexer_kl_warmup_loss(hidden, hidden, teacher)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kernel.hrl_indexer.parameters(), 1.0)
        opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step+1:4d}/{args.steps} kl={loss.item():.4f} lr={lr:.2e}")

    final_loss, final_overlap = eval_loss_and_overlap()
    dur = time.time() - t0
    conv_ratio = final_loss / max(init_loss, 1e-9)
    print(f"[warmup] 完成：val_kl {init_loss:.4f}→{final_loss:.4f}（×{conv_ratio:.3f}）"
          f" 重叠率 {init_overlap:.4f}→{final_overlap:.4f} 用时 {dur:.0f}s")

    # 判定：KL 显著下降（收敛比 <0.7）且重叠率上升（检索语义对齐）
    verdict = ("✅ warmup 达标（KL 收敛 + 重叠率升）"
               if (conv_ratio < 0.7 and final_overlap > init_overlap)
               else "⚠️ warmup 信号弱（短校准早期信号，T2 正式标定再判）")
    print(f"判定：{verdict}")

    out_dir = ROOT / "runs" / "hrl_warmup"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(kernel.hrl_indexer.state_dict(), out_dir / "warmed_indexer.pt")
    report = {
        "ckpt": args.ckpt, "sense_layer": args.sense_layer, "steps": args.steps,
        "n_heads": args.n_heads, "d_index": args.d_index, "lr": args.lr,
        "init_val_kl": init_loss, "final_val_kl": final_loss, "conv_ratio": conv_ratio,
        "init_topk_overlap": init_overlap, "final_topk_overlap": final_overlap,
        "eval_topk": args.eval_topk, "duration_s": round(dur, 1), "verdict": verdict,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[warmup] report → {out_dir/'report.json'}；indexer → {out_dir/'warmed_indexer.pt'}")


if __name__ == "__main__":
    main()
