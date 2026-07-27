"""动态词表（M7，第 0 级 concept_slot）：检测→提取→注册→注入（零梯度，运行时）。

设计依据（必须逐条对齐，禁止凭记忆扩展）：
- 部件实现详细计划 Part G1 / 设计 §28.2 第 0 级：KAL「词表摩擦」检测 → Kaplan 内词典
  提取 → concept_slot 注册（页表=动态词表 codebook）→ 输入侧注入（零风险）。
- Kaplan《From Tokens to Words》（arXiv:2410.05864，ICLR 2025）：LLM 早期-中层天然
  detokenize，把多 token 碎片融合为词表示，对 OOV 同样成立；免微调提取 = 取概念多 token
  序列末 token 在最早成功层 ℓ 的 detokenized hidden state，**一次前向即得**。
  设计 §28.2 正式口径 = ℓ10–14（28 层 36–50% 深度，与 KAL 挂点同带）；ℓ5–15 为探索区间。
- Over-Tokenized（ICML 2025）：输入词表↔loss log-linear、输入扩张**无条件正向**、
  输出扩张对小模型有害 → 第 0 级**输入侧免费、零风险**，输出侧暂不升格（tied embedding）。
- 载体能力边界：concept_slot 为**位置不变向量**（输入侧"单槽理解"），非事实查表
  （model/tais_kernel.py VECTOR_KINDS，factual_recall=False）。

纪律：
- 提取的真实模型前向（ℓ5-15 hidden state）由 ``extract_fn`` 回调注入（正式接
  TaisObsidianForCausalLM 的 capture_layers）；骨架阶段允许注入提取函数做离线对拍。
- 注册走 runtime PageTable（BlockSpec compiled_kind="concept_slot"，namespace 五元组）。
- 注入走内核 inject() 向量路径（单次加法，steer 行为/单槽理解）。
- 🔧 输出侧不升格（Over-Tokenized 输出有害；升格走第 1 级，CA1 门+自蒸馏 CPT，后续）。
"""
from __future__ import annotations

import torch

from ..runtime.pagetable import BlockSpec, PageTable


def vocab_friction_score(entropy: float, p_ik: float, repeat_cooccur: float) -> float:
    """KAL「词表摩擦」打分（BLT 熵 patching 信号的 token 级对应物，设计 §28.2-1）。

    高熵碎片段 + 反复共现 + P(IK) 异常低的专名区域 → 高摩擦（值得升格为概念槽）。
    简单线性组合（骨架；正式权重经路由器学习）。
    """
    return 0.5 * entropy + 0.3 * repeat_cooccur + 0.2 * (1.0 - p_ik)


class DynamicVocab:
    """动态词表第 0 级：concept_slot 全生命周期（零梯度）。

    - ``detect``：KAL 词表摩擦超阈 → 候选概念；
    - ``extract``：Kaplan 内词典提取（extract_fn 回调，正式接模型 capture_layers）；
    - ``register``：注册 concept_slot 到页表（codebook）；
    - ``inject_vector``：返回输入侧向量（内核 inject() 向量路径用）。
    """

    def __init__(self, pagetable: PageTable, namespace: tuple, extract_fn=None,
                 friction_thresh: float = 0.6, blockstore=None):
        self.pagetable = pagetable
        self.namespace = namespace
        self.extract_fn = extract_fn  # fn(text) -> Tensor [d]（Kaplan ℓ5-15 hidden state）
        self.friction_thresh = friction_thresh
        # 块载荷存储（M4 BlockStore）：promote 注册后把 concept 向量存为 BlockPayload，
        # 使概念可被 Pager fail-closed 检索、经内核 inject 向量路径注入（注册→检索→注入闭环）。
        # 不给时仅注册元数据（向后兼容，向量由调用方自存）。
        self.blockstore = blockstore

    def detect(self, entropy: float, p_ik: float, repeat_cooccur: float) -> bool:
        """词表摩擦超阈即候选。"""
        return vocab_friction_score(entropy, p_ik, repeat_cooccur) >= self.friction_thresh

    def extract(self, text: str) -> torch.Tensor:
        """Kaplan 内词典提取（一次前向，零梯度）。须注入 extract_fn。"""
        if self.extract_fn is None:
            raise RuntimeError("未注入 extract_fn（Kaplan 提取需模型 capture_layers 前向）")
        with torch.no_grad():
            return self.extract_fn(text)

    def register(self, text: str, vec: torch.Tensor) -> BlockSpec:
        """注册 concept_slot 到页表（compiled_kind=concept_slot，输入侧向量+markdown 源）。"""
        spec = BlockSpec(
            block_id=f"concept/{text}",
            route_key=text,
            namespace=self.namespace,
            compiled_kind="concept_slot",
            factual_recall=False,  # 位置不变向量，非事实查表（载体能力边界）
        )
        ok = self.pagetable.register(spec)
        if not ok:
            raise ValueError(f"concept_slot 注册被页表拒绝: {text!r}")
        # 载荷（向量）由调用方经 BlockStore/内核单独存放；页表只存元数据
        return spec

    def promote(self, text: str) -> BlockSpec:
        """检测→提取→注册→（可选）载荷入 BlockStore 一步到位（零梯度）。

        挂 blockstore 时，把 concept 向量存为 BlockPayload（concept_slot，位置不变向量），
        使概念可被 Pager 检索、经内核 inject 向量路径注入——打通 注册→检索→注入 闭环。
        """
        vec = self.extract(text)
        spec = self.register(text, vec)
        if self.blockstore is not None:
            from .tais_kernel import BlockPayload
            self.blockstore.put(
                spec.block_id,
                BlockPayload(block_id=spec.block_id, compiled_kind="concept_slot",
                             vector=vec, layer_ns=self.namespace),
            )
        return spec


def make_dynamic_vocab(pagetable: PageTable, namespace: tuple, extract_fn=None, **kw) -> DynamicVocab:
    """工厂函数。"""
    return DynamicVocab(pagetable, namespace, extract_fn=extract_fn, **kw)
