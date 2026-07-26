"""KAL 分层元认知头（E+-3 原型，设计文档 §8.3-1 P(IK)、§16.2 分层元认知）。

层级划分（设计 §16.2）：
- L1 知识感知头（P(IK)）：W[d, 3]，三态分类 = 知道 / 不确定 / 空白（Kadavath et al.,
  arXiv:2207.05221 的 P(IK) 辅助目标；SAPLMA 证据：内部状态含"是否知道"信号，中间层最强）；
- L2 语境情感感知头：W[d, 2]，valence / arousal 两维 logit（与 L1 共享读点与训练管线，
  成本≈0）；L3 语境一致性感知为远期，不在本文件。

结构纪律：
- 头本体即 nn.Linear（对齐设计 §8.4"KAL 三态头 = checkpoint 内权重"），state_dict
  可随主干存取；本原型（E+-3）仅以"探针"形式离线训练（scripts/kal_probe.py），
  合入主干 checkpoint 留后续任务。
- 读点：capture_layers 暴露的 Block 输出处残差流（model.forward docstring）。
  单流 checkpoint（pm_stream=1，本原型所用 pilot_0p1b_ws）captures[i] 即内容流张量；
  PM-stream 配置（pm_stream>1）下 captures[i] = {"content", "pm"}，设计 §13.4/§17.4
  规范 KAL 读点为 PM-stream（"pm" 键）——本原型读内容流，PM 读点切换留待
  PM 模型定稿后的配置变更（见 read_point 的 stream 参数）。
- 防自指红线（设计 §16.1）：L2 情感标签的 ground truth 不得来自模型自己的头，
  必须从外部信号 bootstrap（本原型用 dair-ai/emotion 外部标注，见 scripts/kal_probe.py）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

# 默认读点层位（0.1B pilot，12 层 GGGAGGGA）：ℓ4/ℓ8 均为 GDN 层 Block 输出处残差流。
# 依据：SAPLMA/ITI 文献显示"是否知道"信号在中间层最强；1.5B 设计读点 ℓ10/14/18（W[2048,3]）
# 按深度比例折算到 12 层约为 ℓ4/ℓ7/ℓ10，本原型取 ℓ4/ℓ8 两组对照。
DEFAULT_READ_LAYERS: tuple[int, int] = (4, 8)


class KALHead(nn.Module):
    """KAL 线性读出头：W[d, n_classes]，输入残差流 hidden，输出 logits。

    L1 知识感知头实例化 n_classes=3（知道/不确定/空白）；L2 情感头 n_classes=2
    （dim0 = valence logit，dim1 = arousal logit，二分类各维独立 BCE）。
    输入支持 [..., d]：逐位置 [B,T,d] 或池化后 [B,d] 均可（nn.Linear 逐位线性）。
    """

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h [..., d] → logits [..., n_classes]。"""
        return self.proj(h)

    @torch.no_grad()
    def predict_proba(self, h: torch.Tensor) -> torch.Tensor:
        """概率形式（二分类/多分类均按 softmax；L2 两维独立时调用方应改用 sigmoid）。"""
        return torch.softmax(self.proj(h).float(), dim=-1)


def make_l1_head(d_model: int) -> KALHead:
    """L1 知识感知头（P(IK)）：三态 知道/不确定/空白（W[d,3]，设计 §8.3-1）。

    注意：0.1B 预演数据协议只有 已知/未知 两类标签（"不确定"中间态无标签来源），
    探针实验退化为二分类头训练（scripts/kal_probe.py 用 KALHead(d, 2)）；
    三态规格在此保留，待正式 Phase 1 协议提供"不确定"标签后启用。
    """
    return KALHead(d_model, 3)


def make_l2_head(d_model: int) -> KALHead:
    """L2 语境情感感知头：valence/arousal 两维 logit（W[d,2]，设计 §16.2）。"""
    return KALHead(d_model, 2)


def read_point(caps: dict, layer: int, stream: str = "content") -> torch.Tensor:
    """从 capture_layers 输出取 KAL 读点张量 [B,T,d]。

    单流 checkpoint：caps[layer] 为张量，直接返回（本原型路径）。
    PM-stream 配置：caps[layer] = {"content": ..., "pm": ...}；stream="content"
    取内容流（本原型默认），stream="pm" 取 PM-stream——设计 §13.4 的 KAL 规范读点，
    待 PM 模型定稿后切换默认。
    """
    cap = caps[layer]
    if isinstance(cap, dict):
        return cap[stream]
    return cap
