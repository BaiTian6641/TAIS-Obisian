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
from .pmstream import PMStreamMix


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    """一个 mixer(GDN/Attn) + 一个 MLP，各自 pre-norm + residual。

    pm_stream>1 时残差改为 mHC 多流（mixer/mlp 各为一个 mHC 层，见 pmstream.py）；
    pm_stream=1 时 forward 与既有单流版本逐行一致（默认，数值零改动）。
    """

    def __init__(self, cfg: ModelConfig, layer_type: str):
        super().__init__()
        self.type = layer_type
        self.pm_stream = cfg.pm_stream
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mixer = GDNBlock(cfg) if layer_type == "G" else CSAAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)
        if cfg.pm_stream > 1:
            # 每个子层一套 mHC 混合系数（arXiv:2512.24880 Fig.3 按 Attention/FFN 展开计层）
            self.mix_mixer = PMStreamMix(cfg.d_model, cfg.pm_stream, cfg.rms_eps, constrain=cfg.pm_constrain)
            self.mix_mlp = PMStreamMix(cfg.d_model, cfg.pm_stream, cfg.rms_eps, constrain=cfg.pm_constrain)

    def forward(
        self, x: torch.Tensor, state: dict | None = None, offset: int = 0
    ) -> tuple[torch.Tensor, dict]:
        if self.pm_stream > 1:
            return self._forward_pm(x, state, offset)
        if self.type == "A":
            m, new_state = self.mixer(self.norm1(x), state, offset)
        else:
            m, new_state = self.mixer(self.norm1(x), state)
        x = x + m
        x = x + self.mlp(self.norm2(x))
        return x, new_state

    def _forward_pm(
        self, S: torch.Tensor, state: dict | None = None, offset: int = 0
    ) -> tuple[torch.Tensor, dict]:
        """mHC 多流路径：S [B,T,n,d]；mixer 与 mlp 各按 Eq.3 做一次 读→F→写。"""
        h_pre, h_post, h_res = self.mix_mixer(S)
        u = self.mix_mixer.read(S, h_pre)
        if self.type == "A":
            # 注入写点（设计文档 §13.4）：HRL 载荷/人格向量在 CSA 残差前经 H_post
            # 写入 PM-stream（S[..., -1, :]）——E+-6 HRL 头簇在此接入（接口位，暂未启用）
            m, new_state = self.mixer(self.norm1(u), state, offset)
        else:
            m, new_state = self.mixer(self.norm1(u), state)
        S = self.mix_mixer.write(S, m, h_post, h_res)
        # 感知读点（设计文档 §13.4）：KAL 各头统一从 GDN-MemBlock 输出处的
        # PM-stream（S[..., -1, :]）读取——经 capture_layers 暴露（见 model.forward）
        h_pre, h_post, h_res = self.mix_mlp(S)
        u = self.mix_mlp.read(S, h_pre)
        S = self.mix_mlp.write(S, self.mlp(self.norm2(u)), h_post, h_res)
        return S, new_state


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
        self,
        input_ids: torch.Tensor,
        cache: dict | None = None,
        capture_layers: list[int] | None = None,
    ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, dict[int, torch.Tensor]]:
        """input_ids [B,T] → (logits [B,T,V], new_cache)；指定 capture_layers 时追加返回 captures。

        cache = {"pos": int, "layers": [各层 state]}，用于推理时增量生成。
        capture_layers：hidden-state 捕获挂点（KAL 探针/机制分析用，纯仪表件，只读不改前向数值）。
            传入层索引列表时返回三元组，captures = {layer_idx: 该 Block 输出处（mlp 残差之后）
            的残差流张量 [B,T,d_model]}（增量路径下 T=1）；为 None 时严格返回二元组（默认行为不变）。
            捕获张量与残差流共享存储、不 detach；grad checkpoint 路径下捕获的是 checkpoint
            返回的 Block 真实输出（重算仅发生在反向且数值相同），与普通路径一致。
            注意：捕获会延长激活/计算图存活时间，训练循环不要使用（避免显存驻留），
            探针捕获请在 eval + no_grad 下进行。
            pm_stream>1（mHC 多流）时捕获语义扩展为 dict：
            captures[i] = {"content": 内容流 0 [B,T,d]（与单流捕获同语义）,
                           "pm": PM-stream（末位流）[B,T,d]}（设计文档 §13.4 的 KAL 读点）。
        """
        offset = 0 if cache is None else cache["pos"]
        layer_states = [None] * len(self.layers) if cache is None else cache["layers"]
        capture_set = None if capture_layers is None else set(capture_layers)
        captures: dict[int, torch.Tensor] = {}
        x = self.embed(input_ids)
        new_states = []
        if self.config.pm_stream > 1:
            # mHC 多流路径（pmstream.py）：流初始化 = 嵌入复制 n 份（HC §2.1，恒等初始化）
            n = self.config.pm_stream
            S = x.unsqueeze(2).expand(x.shape[0], x.shape[1], n, x.shape[2])
            for i, (layer, st) in enumerate(zip(self.layers, layer_states)):
                if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                    # 梯度检查点：流状态不保留，反向时重算（与单流路径同一纪律）
                    S, nst = torch.utils.checkpoint.checkpoint(layer, S, st, offset, use_reentrant=False)
                else:
                    S, nst = layer(S, st, offset)
                new_states.append(nst)
                if capture_set is not None and i in capture_set:
                    captures[i] = {"content": S[:, :, 0, :], "pm": S[:, :, -1, :]}
            # 输出聚合：流均值（非 HC 的求和——RMSNorm eps 破坏尺度不变性，见 pmstream.py）
            x = S.mean(dim=2)
        else:
            for i, (layer, st) in enumerate(zip(self.layers, layer_states)):
                if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                    # 梯度检查点：激活不保留，反向时重算（训练省显存）
                    x, nst = torch.utils.checkpoint.checkpoint(layer, x, st, offset, use_reentrant=False)
                else:
                    x, nst = layer(x, st, offset)
                new_states.append(nst)
                if capture_set is not None and i in capture_set:
                    captures[i] = x
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)  # tied lm_head
        new_cache = {"pos": offset + input_ids.shape[1], "layers": new_states}
        if capture_set is None:
            return logits, new_cache
        return logits, new_cache, captures

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
