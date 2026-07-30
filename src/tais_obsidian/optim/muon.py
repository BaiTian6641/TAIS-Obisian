"""Muon 优化器（MomentUm Orthogonalized by Newton-schulz，Keller Jordan 谱系 arXiv:2412.02684）。

设计依据（TAIS Obsidian）：
- 预训练与 W4 固化**同优化器**（设计 §14.3/§21 优化器一致性，arXiv:2605.06654 降遗忘）；
- Kimi K3 Per-Head Muon 借鉴（注意力 Q/K/V 投影按头分块各自正交化，头间均衡）。

核心原理：对 2D 矩阵参数（线性层权重）的动量做 Newton-Schulz 迭代正交化后更新
（隐式谱归一化，收敛快、对 lr 不敏感）；对非 2D 参数（embedding/norm/bias/1D）走内部 AdamW。

红线：
- Muon **只影响优化器更新**，前向/反向计算图完全不变（GDN 递归/chunked、PM-stream
  sinkhorn、grad checkpoint 等自定义路径天然兼容——它们只产生梯度，优化器不感知）。
- 正交化作用于**动量**（而非梯度/参数本身），与 Keller Jordan 参考实现一致。
"""
from __future__ import annotations

import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz 迭代求矩阵正交因子（USV^T 中 U V^T，即"零幂"最近正交矩阵）。

    Keller Jordan 参考实现的 quintic 变体（5 步迭代），系数 (3.4445, -4.7750, 2.0315)
    为经验调优值。作用于动量矩阵，使其谱范数归一（奇异值推向 1），实现隐式谱归一化。
    输入 G [..., m, n]（支持批量尾两维为矩阵），输出同形状正交化矩阵。
    """
    assert G.ndim >= 2, f"Newton-Schulz 输入须 ≥2D，实际 {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    # 归一化到谱范数 ≤1（Frobenius 范数是上界，保证迭代收敛域）
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT  # 使行数 ≤ 列数，减少迭代矩阵规模
        transposed = True
    # 谱范数粗估计：用 Frobenius 范数归一（≥谱范数，保证 X Xᵀ 特征值 ∈[0,1]）
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """Muon 优化器：2D 矩阵参数走 Newton-Schulz 正交化动量更新，非 2D 走内部 AdamW。

    参数分组（构建时按 ndim/名称自动划分，亦可外部传入自定义分组）：
      - **Muon 组**（ndim≥2 且非 embedding 的矩阵参数）：SGD-momentum + Newton-Schulz
        正交化 + 可选 weight decay。lr 用 `muon_lr`（Muon 对 lr 不敏感，典型 0.02）。
      - **AdamW 组**（embedding/norm/bias/1D 参数）：标准 AdamW，lr 用 `adamw_lr`。

    Per-Head Muon（K3 借鉴，可选 `per_head_dims`）：
      对注意力 Q/K/V 投影权重 [n_heads·head_dim, d]，按头分块 [head_dim, d] 各自
      Newton-Schulz 正交化（全矩阵正交化会让大头主导更新方向，逐头均衡）。
      通过 param 属性 `attrs["per_head"] = (n_heads, head_dim)` 标记，或构建时传入。
    """

    def __init__(
        self,
        params,
        muon_lr: float = 0.02,
        adamw_lr: float = 1e-3,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
    ):
        defaults = dict(
            muon_lr=muon_lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            use_muon = group.get("use_muon", False)
            if use_muon:
                self._muon_step(group)
            else:
                self._adamw_step(group)
        return loss

    def _muon_step(self, group) -> None:
        """2D 矩阵参数：SGD-momentum → Newton-Schulz 正交化 → 更新。"""
        lr = group["muon_lr"]
        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        wd = group["weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(g)
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(g)
            # nesterov：用 (g + momentum·buf) 作有效动量方向
            eff = g.add(buf, alpha=momentum) if nesterov else buf
            # Per-Head 分块正交化（K3 借鉴）或全矩阵正交化
            per_head = getattr(p, "per_head", None)  # (n_heads, head_dim) 或 None
            if per_head is not None:
                n_heads, head_dim = per_head
                o = torch.zeros_like(eff)
                for h in range(n_heads):
                    blk = eff[h * head_dim : (h + 1) * head_dim]
                    o[h * head_dim : (h + 1) * head_dim] = zeropower_via_newtonschulz5(blk, ns_steps)
            else:
                o = zeropower_via_newtonschulz5(eff, ns_steps)
            # 尺度修正：Keller Jordan 用 sqrt(max(1, rows/cols)) 保持更新幅度与 AdamW 可比
            scale = max(1.0, o.size(-2) / o.size(-1)) ** 0.5
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.add_(o, alpha=-lr * scale)

    def _adamw_step(self, group) -> None:
        """非 2D 参数（embedding/norm/bias/1D）：标准 AdamW。"""
        lr = group["adamw_lr"]
        b1, b2 = group["adamw_betas"]
        eps = group["adamw_eps"]
        wd = group["adamw_weight_decay"]
        for p in group["params"]:
            if p.grad is None:
                continue
            g = p.grad
            state = self.state[p]
            if "exp_avg" not in state:
                state["exp_avg"] = torch.zeros_like(g)
                state["exp_avg_sq"] = torch.zeros_like(g)
                state["step"] = 0
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
            state["step"] += 1
            t = state["step"]
            exp_avg.mul_(b1).add_(g, alpha=1 - b1)
            exp_avg_sq.mul_(b2).addcmul_(g, g, value=1 - b2)
            bias_c1 = 1 - b1**t
            bias_c2 = 1 - b2**t
            denom = (exp_avg_sq / bias_c2).sqrt().add_(eps)
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)
            p.addcdiv_(exp_avg, denom, value=-lr / bias_c1)


def build_muon_optimizer(
    model: torch.nn.Module,
    muon_lr: float,
    adamw_lr: float,
    weight_decay: float = 0.0,
    per_head_qkv: bool = False,
    n_heads: int | None = None,
    head_dim: int | None = None,
    momentum: float = 0.95,
    ns_steps: int = 5,
) -> Muon:
    """按参数维度/名称自动分组构建 Muon（对齐 train.py 的 decay 分组语义）。

    - **Muon 组**：ndim≥2 且非 embedding 的矩阵参数（线性层权重），weight_decay 生效；
    - **AdamW 组**：embedding/norm/bias/1D 参数，weight_decay=0（对齐 train.py no_decay）。
    per_head_qkv=True 时，对名称含 q_proj/k_proj/v_proj 的权重标记 per_head 属性
    （需提供 n_heads/head_dim），启用 Per-Head Muon（K3 借鉴）。
    """
    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "embed" not in name:
            if per_head_qkv and any(k in name for k in ("q_proj", "k_proj", "v_proj")):
                assert n_heads is not None and head_dim is not None, (
                    "per_head_qkv 需提供 n_heads/head_dim"
                )
                p.per_head = (n_heads, head_dim)
            muon_params.append(p)
        else:
            adamw_params.append(p)
    groups = [
        {"params": muon_params, "use_muon": True},
        {"params": adamw_params, "use_muon": False},
    ]
    n_muon = sum(p.numel() for p in muon_params)
    n_adamw = sum(p.numel() for p in adamw_params)
    print(f"[opt] Muon（矩阵 {n_muon/1e6:.1f}M，lr={muon_lr}）+ AdamW（非矩阵 {n_adamw/1e6:.1f}M，lr={adamw_lr}）"
          f"，per_head_qkv={per_head_qkv}，ns_steps={ns_steps}")
    return Muon(
        groups,
        muon_lr=muon_lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        momentum=momentum,
        ns_steps=ns_steps,
    )
