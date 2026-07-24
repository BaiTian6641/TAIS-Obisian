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
