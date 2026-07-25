"""CSA 原生块通路原型（设计文档 §11.1）：stride-4 学习压缩器 + 块 KV 收割/导出 + namespace 校验注入。

机制验证专用，不训练压缩器：
- ``CSACompressor``：把 CSA 层连续 stride 个 token 的 k/v 条目各压成 1 条；
- ``harvest_block_kv``：无 cache 前向后从返回 cache 收割各 "A" 层全量 k/v 并压缩导出；
- ``inject_block_kv``：namespace 五元组校验通过后，把压缩块 KV 前置拼入目标 "A" 层 state。

已知偏差（设计 §11.1 风险① 前缀偏差）：注入条目的 RoPE 相位来自源编码位置，
与重算不期望逐点等价；APE 式自适应缩放 / 关键 token 重算 / Block-Attention 式训练目标
均留给 E+ 训练阶段处理。本原型保证的是：形状、namespace 纪律与 offset 簿记正确。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig

# 压缩器结构/权重版本标识：结构与训练目标对齐设计 §11.1，权重训练留给 E+ 阶段。
COMPRESSOR_VERSION = "csa-comp-v0.1"


class NamespaceMismatchError(RuntimeError):
    """namespace 五元组校验失败（fail-closed）。

    运行时回退策略 = 重算 / 文本 RAG（设计 §11.1："失败一律 fail-closed 回退重算/文本 RAG"）；
    本原型只做 raise，由调用方捕获后走回退路径。
    """


class CSACompressor(nn.Module):
    """stride-4 学习压缩器原型：连续 stride 个 token 的 k/v 条目压成 1 条。

    per-head 共享一个 Linear(stride*head_dim → head_dim)，k、v 各自一个投影，无 bias。
    权重随机初始化即可——结构与训练目标对齐设计 §11.1，权重训练留给 E+ 阶段。

    尾部策略：**丢弃**不足 stride 的尾部 token（floor(T/stride)）。选择理由：压缩块
    要求定长窗口语义，短窗会产生变长块、破坏 namespace/注入长度的簿记一致性；被丢弃的
    尾部属于块边界截断，正式方案（边界标记/短窗保留）随 E+ 训练一并确定。
    """

    def __init__(self, head_dim: int, stride: int = 4):
        super().__init__()
        self.head_dim = head_dim
        self.stride = stride
        self.k_proj = nn.Linear(stride * head_dim, head_dim, bias=False)
        self.v_proj = nn.Linear(stride * head_dim, head_dim, bias=False)

    def _compress(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        # x: [B, n_kv, T, head_dim] → [B, n_kv, T//stride, head_dim]
        B, H, T, D = x.shape
        n = T // self.stride
        x = x[:, :, : n * self.stride, :]  # 丢弃不足 stride 的尾部
        x = x.reshape(B, H, n, self.stride * D)
        return proj(x)

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k/v: [B, n_kv, T, head_dim]（RoPE 后的 k）→ 压缩后 [B, n_kv, T//stride, head_dim]。"""
        return self._compress(k, self.k_proj), self._compress(v, self.v_proj)


def model_signature(cfg: ModelConfig) -> str:
    """模型签名：标识"同模型同层"绑定（设计 §11.1 风险② 载体绑定）。"""
    return (
        f"d{cfg.d_model}-L{cfg.n_layer}-h{cfg.head_dim}"
        f"-kv{cfg.n_kv_heads}-V{cfg.vocab_size}"
    )


def make_namespace(cfg: ModelConfig, layer_idx: int, dtype: torch.dtype) -> dict:
    """构造 namespace 五元组：(model_id, layer_idx, compressor_version, dtype, rope_theta)。"""
    return {
        "model_id": model_signature(cfg),
        "layer_idx": layer_idx,
        "compressor_version": COMPRESSOR_VERSION,
        "dtype": str(dtype),
        "rope_theta": float(cfg.rope_theta),
    }


def check_namespace(expected: dict, got: dict) -> None:
    """五元组逐字段比对；任一不匹配即 fail-closed 抛 NamespaceMismatchError。"""
    fields = ("model_id", "layer_idx", "compressor_version", "dtype", "rope_theta")
    bad = [f for f in fields if expected.get(f) != got.get(f)]
    if bad:
        raise NamespaceMismatchError(
            f"namespace 校验失败（fail-closed，回退重算/文本 RAG）："
            f"字段 {bad} 不匹配；expected={ {f: expected.get(f) for f in bad} }，"
            f"got={ {f: got.get(f) for f in bad} }"
        )


@torch.no_grad()
def harvest_block_kv(
    model: nn.Module,
    ids: torch.Tensor,
    compressor_by_layer: dict[int, CSACompressor],
    device: str | torch.device,
) -> dict:
    """无 cache 前向，收割各 "A" 层全量 k/v，过压缩器导出块 KV。

    返回 {"namespace": {layer_idx: 五元组}, "layers": {layer_idx: (k_comp, v_comp)}}；
    仅对 cfg.layer_types 为 "A" 且在 compressor_by_layer 中的层产出条目。
    GDN 层无 KV cache，不参与本通路（LoRA/steering 载荷才层无关）。
    """
    cfg = model.config
    ids = ids.to(device)
    _, cache = model(ids)
    layers: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    namespaces: dict[int, dict] = {}
    for i, lt in enumerate(cfg.layer_types):
        if lt != "A" or i not in compressor_by_layer:
            continue
        st = cache["layers"][i]
        comp = compressor_by_layer[i]
        k_comp, v_comp = comp(st["k"], st["v"])
        layers[i] = (k_comp, v_comp)
        namespaces[i] = make_namespace(cfg, i, k_comp.dtype)
    return {"namespace": namespaces, "layers": layers}


def inject_block_kv(cache: dict, block_kv: dict, cfg: ModelConfig) -> dict:
    """校验通过后，把压缩块 KV 前置拼入各目标 "A" 层 state 的 k/v（dim=2 前端）。

    - namespace 五元组校验：任一字段不匹配即 fail-closed 抛 NamespaceMismatchError
      （运行时回退策略 = 重算/文本 RAG，本原型只 raise）；
    - 目标层必须是 "A" 层，否则同样 fail-closed；
    - ``cache["pos"]`` 增加注入长度（每条压缩条目占 1 个位置槽），使后续新 token 的
      RoPE offset 与拼接后序列一致；各层注入长度必须一致；
    - "G" 层 state 完全不动（张量对象原样保留）。

    返回新 cache（不原地修改入参）。已知偏差：注入条目的 RoPE 相位来自源编码位置，
    与重算不期望逐点等价（设计 §11.1 风险① 前缀偏差，训练阶段才处理）；本函数保证
    形状、namespace 纪律与 offset 簿记正确。
    """
    new_layers = list(cache["layers"])
    total_inj: int | None = None
    for layer_idx, (k_comp, v_comp) in block_kv["layers"].items():
        if cfg.layer_types[layer_idx] != "A":
            raise NamespaceMismatchError(
                f"层 {layer_idx} 不是 CSA 层（type={cfg.layer_types[layer_idx]}），拒绝注入"
            )
        st = cache["layers"][layer_idx]
        expected = make_namespace(cfg, layer_idx, st["k"].dtype)
        check_namespace(expected, block_kv["namespace"][layer_idx])
        k_new = torch.cat([k_comp.to(device=st["k"].device, dtype=st["k"].dtype), st["k"]], dim=2)
        v_new = torch.cat([v_comp.to(device=st["v"].device, dtype=st["v"].dtype), st["v"]], dim=2)
        n_inj = k_comp.shape[2]
        assert k_comp.shape[:3] == v_comp.shape[:3], "k/v 压缩长度不一致"
        if total_inj is None:
            total_inj = n_inj
        elif total_inj != n_inj:
            raise NamespaceMismatchError(
                f"各层注入长度不一致：{total_inj} vs {n_inj}（层 {layer_idx}），拒绝注入"
            )
        new_layers[layer_idx] = {**st, "k": k_new, "v": v_new}
    return {"pos": cache["pos"] + (total_inj or 0), "layers": new_layers}
