"""TaisObsidianForCausalLM：tied embedding + block_pattern 堆叠的自研 causal LM。

每层 = {mixer(GDN 或 CSA Attn), SwiGLU MLP}，各带 pre-norm(RMSNorm) + residual；
final RMSNorm 后经 tied embedding 得 logits。不依赖 transformers 建模组件。
save_pretrained/from_pretrained：config.json + model.safetensors（bf16 存储）。
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .attention import CSAAttention
from .common import RMSNorm
from .gdn import GDNBlock


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    """一个 mixer(GDN/Attn) + 一个 MLP，各自 pre-norm + residual。"""

    def __init__(self, cfg: ModelConfig, layer_type: str):
        super().__init__()
        self.type = layer_type
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mixer = GDNBlock(cfg) if layer_type == "G" else CSAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)

    def forward(
        self, x: torch.Tensor, state: dict | None = None, offset: int = 0
    ) -> tuple[torch.Tensor, dict]:
        if self.type == "A":
            m, new_state = self.mixer(self.norm1(x), state, offset)
        else:
            m, new_state = self.mixer(self.norm1(x), state)
        x = x + m
        x = x + self.mlp(self.norm2(x))
        return x, new_state


class TaisObsidianForCausalLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList([Block(cfg, t) for t in cfg.layer_types])
        self.norm_f = RMSNorm(cfg.d_model, cfg.rms_eps)
        # lm_head 与 embed 共享权重（tied）
        self.apply(self._init_weights)
        # 残差分支输出投影缩小初始化（GPT-2 惯例）
        std_out = 0.02 / math.sqrt(2 * cfg.n_layer)
        for name, p in self.named_parameters():
            if name.endswith(("o_proj.weight", "mlp.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=std_out)
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] 参数量 {n_params/1e6:.2f}M（tied embedding；层型 {''.join(cfg.layer_types)}）")
        if cfg.check_0p1b_params:
            assert 90e6 <= n_params <= 130e6, f"参数量 {n_params/1e6:.1f}M 不在 90–130M 区间"

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, cache: dict | None = None
    ) -> tuple[torch.Tensor, dict]:
        """input_ids [B,T] → (logits [B,T,V], new_cache)。

        cache = {"pos": int, "layers": [各层 state]}，用于推理时增量生成。
        """
        offset = 0 if cache is None else cache["pos"]
        layer_states = [None] * len(self.layers) if cache is None else cache["layers"]
        x = self.embed(input_ids)
        new_states = []
        for layer, st in zip(self.layers, layer_states):
            if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                # 梯度检查点：激活不保留，反向时重算（训练省显存）
                x, nst = torch.utils.checkpoint.checkpoint(layer, x, st, offset, use_reentrant=False)
            else:
                x, nst = layer(x, st, offset)
            new_states.append(nst)
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)  # tied lm_head
        new_cache = {"pos": offset + input_ids.shape[1], "layers": new_states}
        return logits, new_cache

    def save_pretrained(self, dir: str | Path) -> None:
        """存 config.json + model.safetensors（bf16）。"""
        from safetensors.torch import save_file

        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        self.config.to_json(dir / "config.json")
        sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in self.state_dict().items()}
        save_file(sd, str(dir / "model.safetensors"))

    @classmethod
    def from_pretrained(cls, dir: str | Path, device: str | torch.device = "cpu") -> "TaisObsidianForCausalLM":
        from safetensors.torch import load_file

        dir = Path(dir)
        cfg = ModelConfig.from_json(dir / "config.json")
        model = cls(cfg)
        sd = load_file(str(dir / "model.safetensors"))
        model.load_state_dict(sd)  # bf16 自动 cast 到 fp32 参数
        return model.to(device)
