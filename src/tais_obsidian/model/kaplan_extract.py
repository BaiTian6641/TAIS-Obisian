"""Kaplan 内词典提取（真实模型实现）：动态词表 concept_slot 的免微调向量提取。

设计依据（必须逐条对齐，禁止凭记忆扩展）：
- Kaplan《From Tokens to Words》（arXiv:2410.05864，ICLR 2025）：LLM 早期-中层天然
  detokenize，把多 token 碎片融合为词表示，对 OOV 专名同样成立（OOV 多 token 检出
  64% @ layers 5-7，Llama2-7B）。免微调提取 = 取概念多 token 序列**末 token 在最早
  成功层 ℓ 的 detokenized hidden state**，**一次前向即得**——末 token 已把前缀碎片
  "读进"自身表示（单向注意力下末 token 聚合全序列），无需任何微调。
- 设计 §28.2 正式口径 = ℓ10–14（28 层 36–50% 深度，与 KAL 挂点同带）；ℓ5–15 探索区间。
  本 pilot 为 0.1B **12 层**（d_model=768），同深度比例对应约 **ℓ4–6**；实扫描
  得 ℓ3 detokenize 最强（gap 0.196，见 ``DEFAULT_KAPLAN_LAYER`` 注释），故默认 **ℓ3**。
- 监测/执行分置红线：本提取 **no_grad 只读**（前向 hidden state 捕获，不回传梯度、
  不触碰权重），属"监测"侧；注入写入在内核 inject() 向量路径（另一侧）。
- 载体能力边界：提取出的 concept_slot 是**位置不变向量**（输入侧"单槽理解"），
  非事实查表（factual_recall=False）——它 steer 模型"把这段碎片当一个已学概念理解"，
  不携带可逐字召回的事实（token 寻址载体才可事实召回）。
- Over-Tokenized（ICML 2025）：第 0 级**输入侧** concept_slot 免费零风险；
  输出侧不升格 tied embedding（输出扩张对小模型有害）。

数据源：model.forward(capture_layers=[layer]) → captures[layer]。
  pm_stream=1（本 pilot）时 captures[layer] = 该层输出 hidden [B,T,d]；
  pm_stream>1 时 captures[layer] 为 {"content","pm"} dict，取内容流 content。
  取末 token（[: , -1, :]）即概念 detokenized 向量。
"""
from __future__ import annotations

import torch

from ..tokenizer_io import TokenizerIO

# 默认提取层：0.1B 12 层 pilot 的 detokenize 最强层。
# 设计口径 ℓ10–14@28层（36–50% 深度）≙ 12 层约 ℓ4–6；但对 pilot 0.1B 实扫描
# ℓ3–8 的同类/不同类余弦 gap（2026-07-29，GDN-2 10k checkpoint）得 ℓ3 最强
# （sim_mean 0.502 vs diff_mean 0.305，gap 0.196；ℓ4 0.167、ℓ5 0.163 递减）——
# 小模型 detokenize 峰值比设计探索区间略前移（Kaplan 原文 OOV 检出峰也在层 5-7/32，
# 同属偏早），故 pilot 默认取实测最强的 ℓ3，正式 1.5B 28 层回到 ℓ10–14。
# 可传 layer 覆盖以扫描选层。
DEFAULT_KAPLAN_LAYER = 3


def make_kaplan_extract_fn(
    model,
    layer: int | None = None,
    tokenizer=None,
    device: str | torch.device | None = None,
):
    """构造真实 Kaplan extract_fn：text → concept_slot 向量 [d_model]（no_grad 只读）。

    参数：
      model     ：TaisObsidianForCausalLM（已加载 checkpoint，eval）。
      layer     ：Kaplan 提取层 ℓ（None → DEFAULT_KAPLAN_LAYER=5）。
      tokenizer ：TokenizerIO 或 tokenizer.json 路径（None → data/tokenizer/tokenizer.json）。
      device    ：推理设备（None → 跟 model.embed.weight 所在设备）。

    返回 extract_fn(text) -> torch.Tensor [d_model]（float32，model 设备上）。
    每次调用 = 一次前向（编码 → capture 末 token hidden → 取向量），零梯度。
    """
    if layer is None:
        layer = DEFAULT_KAPLAN_LAYER
    if tokenizer is None:
        tokenizer = TokenizerIO("data/tokenizer/tokenizer.json")
    elif isinstance(tokenizer, (str,)):
        tokenizer = TokenizerIO(tokenizer)
    if device is None:
        device = next(model.parameters()).device
    n_layer = model.config.n_layer
    if not (0 <= layer < n_layer):
        raise ValueError(f"Kaplan 提取层 ℓ{layer} 越界（模型 {n_layer} 层）")

    pm_multi = getattr(model.config, "pm_stream", 1) > 1

    @torch.no_grad()
    def extract_fn(text: str) -> torch.Tensor:
        ids = tokenizer.encode(text)
        if not ids:  # 空文本 fail-closed（无法提取）
            raise RuntimeError(f"Kaplan 提取失败：text 编码为空（{text!r}）")
        x = torch.tensor([ids], dtype=torch.long, device=device)
        use_cuda = x.device.type == "cuda"
        with torch.autocast("cuda", torch.bfloat16, enabled=use_cuda):
            _logits, _cache, captures = model(x, capture_layers=[layer])
        cap = captures[layer]
        hidden = cap["content"] if pm_multi else cap  # 多流取内容流；单流即 hidden
        # 末 token = 概念碎片融合后的 detokenized 词表示（Kaplan 核心：单向注意力下
        # 末 token 聚合全序列碎片，OOV 多 token 融合为单一词向量）。
        vec = hidden[:, -1, :].float().squeeze(0)  # [d_model]
        if not torch.isfinite(vec).all():
            raise RuntimeError(f"Kaplan 提取得到非有限向量（{text!r} @ ℓ{layer}）")
        return vec

    return extract_fn
