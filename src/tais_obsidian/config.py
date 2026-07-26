"""模型配置：ModelConfig dataclass + JSON 读写。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """TAIS Obsidian 模型配置（D-0 级 0.1B 先导默认值）。

    block_pattern 循环重复至 n_layer；"G"=GDN 层，"A"=CSA 全注意力层。
    attn_only=True 时全部层替换为注意力层（对照孪生）。
    """

    vocab_size: int = 32768
    d_model: int = 768
    n_layer: int = 12
    block_pattern: list[str] = field(default_factory=lambda: ["G", "G", "G", "A"])
    # CSA 注意力
    n_q_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    rope_theta: float = 10000.0
    # GDN
    n_v_heads: int = 12
    n_qk_heads: int = 6
    conv_kernel: int = 4
    # MLP（SwiGLU）
    mlp_hidden: int = 2048
    rms_eps: float = 1e-6
    max_seq: int = 1024
    attn_only: bool = False
    # 训练时逐 block 梯度检查点（8GB 显存下 micro_batch=16 的必需项；重算换显存）
    grad_checkpoint: bool = True
    # 构建时是否断言参数量落在 0.1B 区间（tiny 冒烟配置关闭）
    check_0p1b_params: bool = True
    # PM-stream（mHC 多流残差，arXiv:2512.24880；实现见 model/pmstream.py）：
    # 1 = 现状单流残差（默认，数值路径与既有版本逐行一致）；5 = 4 内容流 + 1 感知-记忆流
    # （设计文档 §12.2/§13.4；>1 的其它整数值同理，末位流为 PM-stream）
    pm_stream: int = 1
    # Sinkhorn-Knopp 双随机约束开关：True = mHC 原文（默认）；False = 无约束 HC 消融对照
    pm_constrain: bool = True
    # 三级注意力栈（E+-7，设计文档 §17；实现见 model/tri_attention.py）：
    # "full" = CSA 全注意力（默认，既有数值路径零改动）；
    # "tri"  = 滑窗 + CSA 选择检索 + HCA 重压缩三级栈（DeepSeek V4/NSA 谱系）。
    # 纪律：attn_only=True（对照组）时始终全注意力，本开关不生效。
    attn_impl: str = "full"
    tri_window: int = 512      # 滑窗分支窗口（L0 工作记忆，NSA w=512）
    tri_csa_stride: int = 4    # CSA 压缩 stride（L1 情景记忆，V4 m=4）
    tri_csa_topk: int = 128    # CSA indexer top-k（仅因果压缩集合内）
    tri_hca_stride: int = 128  # HCA 重压缩比（L2 gist，V4 m'=128）

    @property
    def layer_types(self) -> list[str]:
        """展开后的逐层类型列表，长度为 n_layer。"""
        if self.attn_only:
            return ["A"] * self.n_layer
        types = [self.block_pattern[i % len(self.block_pattern)] for i in range(self.n_layer)]
        assert all(t in ("G", "A") for t in types), f"未知层类型: {types}"
        return types

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)
