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
from .common import RMSNorm
from .gdn import GDNBlock
from .gdn2 import GDN2Block
from .pmstream import PMStreamMix
from .tais_kernel import TAISKernel
from .tri_attention import TriRetrievalAttention


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w13 = nn.Linear(d_model, 2 * hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.w13(x).chunk(2, dim=-1)
        return self.w2(F.silu(a) * b)


class Block(nn.Module):
    """一个 mixer(GDN-MemBlock 或 TriRetrievalAttention) + 一个 MLP，各自 pre-norm + residual。

    pm_stream>1 时残差改为 mHC 多流（mixer/mlp 各为一个 mHC 层，见 pmstream.py）；
    pm_stream=1 时 forward 与既有单流版本逐行一致（默认，数值零改动）。
    """

    def __init__(self, cfg: ModelConfig, layer_type: str):
        super().__init__()
        self.type = layer_type
        self.pm_stream = cfg.pm_stream
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_eps)
        if layer_type == "G":
            self.mixer = GDNBlock(cfg)
        elif layer_type == "G2":
            # GDN-2 层（erase/write 解耦，arXiv:2605.22791）：NVIDIA 论文实证优于 GDN-1
            # （RULER 检索大幅领先）；tied 退化=GDN-1（严格一般化）。2026-07-27 采纳切换。
            self.mixer = GDN2Block(cfg)
        else:
            # "A" 层统一为 TriRetrievalAttention（三级检索注意力，DeepSeek V4/NSA 谱系）；
            # 2026-07 起移除旧 RetrievalAttention 占位与 attn_only 对照组。
            self.mixer = TriRetrievalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mlp = SwiGLU(cfg.d_model, cfg.mlp_hidden)
        if cfg.pm_stream > 1:
            # 每个子层一套 mHC 混合系数（arXiv:2512.24880 Fig.3 按 Attention/FFN 展开计层）
            # pm_sk_t_max：Sinkhorn 迭代数（吞吐优化，默认 20=原文精确语义向后兼容，
            # 训练中可调小如 10 提速 ×1.7，谱范数红线不破）。
            sk_t_max = getattr(cfg, "pm_sk_t_max", 20)
            self.mix_mixer = PMStreamMix(cfg.d_model, cfg.pm_stream, cfg.rms_eps, t_max=sk_t_max, constrain=cfg.pm_constrain)
            self.mix_mlp = PMStreamMix(cfg.d_model, cfg.pm_stream, cfg.rms_eps, t_max=sk_t_max, constrain=cfg.pm_constrain)

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
        # 三级栈层号写入（inject_hca_entries 的 namespace 五元组校验用）
        for i, layer in enumerate(self.layers):
            if isinstance(layer.mixer, TriRetrievalAttention):
                layer.mixer.layer_idx = i
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
        # TAIS 内核挂点（M1–M8；默认关闭，forward 行为零改动）
        self.kernel: TAISKernel | None = None
        if cfg.kernel_enabled:
            self.attach_kernel()

    def attach_kernel(self) -> None:
        """挂载 TAIS 内核（聚合 KAL/HRL 内生头，随 state_dict 存取）。

        内核权重进 self.kernel（nn.Module 子模块），save_pretrained/from_pretrained
        自动随 state_dict 存取（无需特殊处理）。重复调用安全（幂等）。
        设备对齐：内核移到主干 embedding 权重所在设备/dtype（from_pretrained(..., device)
        后 attach 时内核默认在 CPU，须显式跟随主干，防 sense/inject 设备不匹配）。
        """
        cfg = self.config
        self.kernel = TAISKernel(cfg.d_model, dg_dim=cfg.kernel_dg_dim, dg_topk=cfg.kernel_dg_topk)
        p = self.embed.weight
        self.kernel = self.kernel.to(device=p.device, dtype=p.dtype)

    def kernel_sense_index(self) -> list[int]:
        """KAL sense 读点层索引（config.kernel_sense_layers；空=全部 GDN 层）。

        供 train.py 的 KAL 辅助损失定位 sense 输出层。
        """
        if self.config.kernel_sense_layers:
            return list(self.config.kernel_sense_layers)
        # GDN 系层（"G"=GDN-1，"G2"=GDN-2）均为 KAL sense 读点（递归状态 W-State）
        return [i for i, t in enumerate(self.config.layer_types) if t in ("G", "G2")]

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
        run_kernel: bool = False,
        injector=None,
        inject_payloads: dict[int, list] | None = None,
    ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, dict[int, torch.Tensor]]:
        """input_ids [B,T] → (logits [B,T,V], new_cache)；指定 capture_layers 时追加返回 captures。

        run_kernel=True（且 kernel 已挂载）时，在 forward 中调 TAIS 内核：
        - sense：在每个 sense 读点层（config.kernel_sense_layers；空=全部 GDN 层）后，
          对该层输出 PM-stream（pm_stream>1）或内容流（单流）调 kernel.sense()——**监测只读**；
        - inject：在 CSA 层前，对该层输入残差前 PM-stream 调 kernel.inject(payloads, injector)
          ——**执行写入**（监测/执行分置：sense 读 GDN 层，inject 写 CSA 层，不同层）。
          inject_payloads = {layer_idx: [BlockPayload, ...]}。
        返回的 kernel_signals 追加到 captures["__kernel__"]（dict per layer）。
        默认 run_kernel=False：forward 行为与现状逐行一致（94 项基线测试零改动）。
        """
        offset = 0 if cache is None else cache["pos"]
        layer_states = [None] * len(self.layers) if cache is None else cache["layers"]
        capture_set = None if capture_layers is None else set(capture_layers)
        captures: dict[int, torch.Tensor] = {}
        kernel_signals: dict[int, dict] = {}
        sense_layers = set(self.config.kernel_sense_layers) if self.config.kernel_sense_layers else None
        x = self.embed(input_ids)
        new_states = []
        use_kernel = run_kernel and self.kernel is not None
        if self.config.pm_stream > 1:
            # mHC 多流路径（pmstream.py）：流初始化 = 嵌入复制 n 份（HC §2.1，恒等初始化）
            n = self.config.pm_stream
            S = x.unsqueeze(2).expand(x.shape[0], x.shape[1], n, x.shape[2])
            for i, (layer, st) in enumerate(zip(self.layers, layer_states)):
                # 执行写入（CSA 残差前 PM-stream）：在该 CSA 层前注入
                if use_kernel and layer.type == "A" and inject_payloads and i in inject_payloads:
                    pm_pre = S[:, :, -1, :]
                    S = S.clone()
                    S[:, :, -1, :] = self.kernel.inject(
                        pm_pre, inject_payloads[i], injector=injector
                    )
                if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                    S, nst = torch.utils.checkpoint.checkpoint(layer, S, st, offset, use_reentrant=False)
                else:
                    S, nst = layer(S, st, offset)
                new_states.append(nst)
                if capture_set is not None and i in capture_set:
                    captures[i] = {"content": S[:, :, 0, :], "pm": S[:, :, -1, :]}
                # 监测只读（GDN 输出 PM-stream）：在该 GDN 层后感知（"G"/"G2" 均为 GDN 系）
                if use_kernel and (sense_layers is None or i in sense_layers) and layer.type in ("G", "G2"):
                    kernel_signals[i] = {"sense": self.kernel.sense(S[:, :, -1, :])}
            x = S.mean(dim=2)
        else:
            for i, (layer, st) in enumerate(zip(self.layers, layer_states)):
                if use_kernel and layer.type == "A" and inject_payloads and i in inject_payloads:
                    x = self.kernel.inject(x, inject_payloads[i], injector=injector)
                if self.config.grad_checkpoint and self.training and torch.is_grad_enabled():
                    x, nst = torch.utils.checkpoint.checkpoint(layer, x, st, offset, use_reentrant=False)
                else:
                    x, nst = layer(x, st, offset)
                new_states.append(nst)
                if capture_set is not None and i in capture_set:
                    captures[i] = x
                if use_kernel and (sense_layers is None or i in sense_layers) and layer.type in ("G", "G2"):
                    kernel_signals[i] = {"sense": self.kernel.sense(x)}
        x = self.norm_f(x)
        logits = F.linear(x, self.embed.weight)  # tied lm_head
        new_cache = {"pos": offset + input_ids.shape[1], "layers": new_states}
        if kernel_signals:
            captures["__kernel__"] = kernel_signals
        if capture_set is None and not kernel_signals:
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
    def from_pretrained(
        cls,
        dir: str | Path,
        device: str | torch.device = "cpu",
        strict: bool = True,
        skip_keys: tuple[str, ...] = (),
    ) -> "TaisObsidianForCausalLM":
        """加载 checkpoint。**strict=False 兼容模式**：允许缺失/多余键——
        旧 checkpoint（attn_impl=full 时代的 CSAAttention 权重）在新架构
        （TriRetrievalAttention 三级栈，含 csa_comp/hca_comp/gate_w/gate_b 新参数）
        下结构不同；strict=False 时旧主干权重（embedding/GDN/MLP/注意力 q/k/v/o_proj）
        仍载入，新三级栈参数随机初始化（供后续微调）。默认 strict=True（同架构往返）。

        ``skip_keys``（strict=True 时仍生效）：加载前剔除以这些前缀开头的键——
        用于**形状演进**的部件（如 side_heads.conflict 由 Linear(d,1) 升级 Linear(d,3)），
        旧形状权重无法 strict 载入，剔除后该部件随机初始化待微调，其余权重严格载入。
        """
        from safetensors.torch import load_file

        dir = Path(dir)
        cfg = ModelConfig.from_json(dir / "config.json")
        model = cls(cfg)
        sd = load_file(str(dir / "model.safetensors"))
        if skip_keys:
            skipped = [k for k in sd if any(k.startswith(p) for p in skip_keys)]
            for k in skipped:
                del sd[k]
            # 用当前模型的随机初始化值填补被剔除的键——strict=True 要求所有模型键有值，
            # 剔除的旧形状键由新形状的随机初始化顶替（待后续微调），保证 strict 载入通过。
            cur = model.state_dict()
            for k in cur:
                if any(k.startswith(p) for p in skip_keys) and k not in sd:
                    sd[k] = cur[k]
            if skipped:
                print(f"[from_pretrained] skip_keys 剔除 {len(skipped)} 键（形状演进部件随机初始化顶替）："
                      f"{[k for k in skipped[:3]]}{'...' if len(skipped) > 3 else ''}")
        # strict=False 时还需过滤形状不匹配的键（load_state_dict 对形状冲突仍报错）：
        # 仅保留与当前模型形状一致的键（缺失/多余/形状冲突键均跳过）。
        if not strict:
            cur = model.state_dict()
            sd = {k: v for k, v in sd.items() if k in cur and cur[k].shape == v.shape}
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if not strict and (missing or unexpected):
            print(f"[from_pretrained] 兼容模式：missing {len(missing)} 键（新参数随机初始化），"
                  f"unexpected {len(unexpected)} 键（忽略）")
        return model.to(device)
