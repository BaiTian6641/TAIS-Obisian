"""模型配置：ModelConfig dataclass + JSON 读写。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """TAIS Obsidian 模型配置（D-0 级 0.1B 先导默认值）。

    block_pattern 循环重复至 n_layer；"G"=GDN 层，"A"=CSA 全注意力层。
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
    # 注意力层 = TriRetrievalAttention（三级检索注意力，DeepSeek V4/NSA 谱系；实现见
    # model/tri_attention.py）：滑窗 L0 + CSA 选择检索 L1 + HCA gist L2。"A" 层统一用此
    # （2026-07 起移除旧 RetrievalAttention 占位与 attn_only 对照组，全部走三级栈）。
    tri_window: int = 512      # 滑窗分支窗口（L0 工作记忆，NSA w=512）
    tri_csa_stride: int = 4    # CSA 压缩 stride（L1 情景记忆，V4 m=4）
    tri_csa_topk: int = 128    # CSA indexer top-k（仅因果压缩集合内）
    tri_hca_stride: int = 128  # HCA 重压缩比（L2 gist，V4 m'=128）
    # TriRetrievalAttention CSA 分支的选择机制（V4 最优组合）：
    # tri_use_indexer=True = V4 CSA 式独立 LightningIndexer 在压缩条目上打分选 top-k
    #   （DeepSeek V4 正式路径，与 HRL 的 LightningIndexer 同构共享——设计 §11.1
    #   "一个打分器两种检索对象"；**默认，2026-07-26 经 2000 步消融扶正**：NSA val 5.3543
    #   vs V4 val 5.3583，Δ+0.0041<0.02 不劣化、吞吐+1.4%、显存持平、参数+0.031M）；
    # False = NSA 式（复用压缩注意力分数 Softmax(q·K̃) 选 top-k，保留作消融对照）。
    tri_use_indexer: bool = True
    tri_index_heads: int = 4   # CSA indexer 头数（DSA lightning indexer 式，少头低维）
    tri_index_dim: int = 32    # CSA indexer 维度（低维，吞吐考虑）
    # TAIS 内核挂点（M1–M8；默认关闭，既有 checkpoint/train/generate 零改动）：
    # kernel_enabled=False = 不构建内核（forward 行为与现状逐行一致）；
    # True = 构建 TAISKernel 并允许 forward(run_kernel=True) 调 sense/inject。
    kernel_enabled: bool = False
    kernel_dg_dim: int = 256   # DG 投影维度
    kernel_dg_topk: int = 32   # DG 稀疏 key top-k
    kernel_sense_layers: list[int] = field(default_factory=list)  # sense 读点层（空=全部 GDN 层）

    @property
    def layer_types(self) -> list[str]:
        """展开后的逐层类型列表，长度为 n_layer（"G"=GDN-MemBlock，"A"=TriRetrievalAttention）。"""
        types = [self.block_pattern[i % len(self.block_pattern)] for i in range(self.n_layer)]
        assert all(t in ("G", "A") for t in types), f"未知层类型: {types}"
        return types

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        """从 config.json 读配置。**向后兼容**：忽略未知字段（如旧 checkpoint 的
        `attn_only`/`attn_impl`——2026-07 移除后旧 config.json 仍含这些键），
        使旧 checkpoint 可在新代码下加载。"""
        import dataclasses
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        valid = {f.name for f in dataclasses.fields(cls)}
        dropped = sorted(set(data) - valid)
        if dropped:
            print(f"[config] 忽略未知字段（向后兼容）: {dropped}")
        return cls(**{k: v for k, v in data.items() if k in valid})
