"""公共小部件：RMSNorm / 门控 RMSNorm（fp32 内部计算）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).type_as(x)


class RMSNormGated(nn.Module):
    """输出门控 RMSNorm：y = rmsnorm(x) * weight * sigmoid(g)（per-head 门控）。"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        y = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        y = y * self.weight.float() * torch.sigmoid(g.float())
        return y.type_as(x)
