"""三级注意力栈原型（E+-7，设计文档 §17）：滑窗 + CSA 选择检索 + HCA 重压缩 + 学习门控融合。

DeepSeek V4 混合压缩注意力 / NSA（arXiv:2502.11089）谱系的项目化实现，经 config
（2026-07 起为唯一注意力层实现——旧 TriRetrievalAttention 占位与 attn_only 对照组已移除，全部走三级栈）。
只替换 "A" 层；"G" 层（GDN 无 KV cache）不动。

三分支（与 L0/L1/L2 存储层级一一对应，设计 §17.2）：

- **滑窗分支（L0 精确）**：最近 ``tri_window`` 个 token 的精确注意力（RoPE + GQA，
  与 TriRetrievalAttention 同纪律的 masked SDPA）。NSA §3.3.3：独立滑窗分支防止局部模式
  短路压缩/选择分支的学习。
- **CSA 分支（L1 情景）**：stride-``tri_csa_stride`` 学习压缩器把全量 k/v 压成 T/stride
  条目；importance = 压缩注意力分数（NSA Eq.8：p=Softmax(q·K̃)，**非独立 indexer 模块**；
  GQA 组内头求和共享选择，NSA Eq.10），仅因果集合内取 top-``tri_csa_topk``，
  query 对选中压缩条目做注意力（V4 CSA 式：选择对象为压缩条目本身，而非 NSA 的
  细粒度原文块——设计 §17.1 的命名收敛即此含义）。
- **HCA 分支（L2 gist）**：``tri_hca_stride``:1 重压缩（V4 m'=128，无重叠），对全部
  因果内条目做 dense 注意力（V4 §2.3.2：HCA 不做稀疏选择）。HCA 区是知识块注入的
  原生落点（设计 §17.3）：``inject_hca_entries`` 把外部条目前置拼入 HCA 区。

融合：NSA Eq.5 门控形式 o = Σ_c g^c·Attn^c，g = sigmoid(线性(q))，per-head per-branch，
独立 sigmoid（不强制和为 1）；**零初始化权重 + bias=-ln2 → init 精确均等 1/3**（记录：
plan 要求 init 均等；NSA 原文只说 sigmoid MLP，未给初始化——推断实现）。

原文核对结论（2026-07-25 联网核实，引用落到注释）：

1. 压缩块构造：NSA Eq.7（MLP+块内位置编码，l=32/d=16 重叠）；V4 §2.3.1 Eq.9–12
   （softmax 门控池化 + 学习位置偏置 B∈R^{m×c}，CSA 重叠 2m 窗口，HCA 不重叠 §2.3.2）。
   **本实现采用 V4 式 softmax 门控池化、CSA/HCA 均不重叠**（简化：CSA 重叠减半信息碎裂
   但参数翻倍，0.1B/seq-1024 尺度影响有限，留作后续消融；不重叠与 blockpath.BlockCompressor
   的定长块语义一致）。
2. 重要性分数来源：NSA Eq.8–10，复用压缩注意力分数（分数参与压缩注意力训练，
   选择本身离散无梯度——本实现照此：top-k 索引无梯度，选中条目的注意力 logits
   正常回传到 q/压缩器/k）。V4 另有独立 lightning indexer（Eq.13–16），原型不引入。
3. 门控形式：NSA Eq.5，sigmoid(MLP(输入特征))，g∈[0,1] 逐分支独立。本实现自 q 产生
   （plan §4；q 即输入特征的投影，内容路由与 NSA 一致）。
4. top-k 梯度路径：NSA 无 straight-through/辅助 loss——选择只影响前向的值聚合，
   重要性分数即压缩注意力分数、随压缩分支训练。照抄，禁止自创梯度路径（plan 红线）。
5. 位置编码：V4 压缩条目只带学习位置偏置（Eq.11），无 RoPE；NSA 压缩器含块内位置
   编码。原文均未明示压缩分支的 RoPE 处理 → **推断实现**：压缩/HCA 分支 q/k 均不施加
   RoPE（NoPE 内容寻址；对多块混合向量施加单点旋转无位置语义），块内位置由压缩器的
   学习位置偏置承担；滑窗分支保持 RoPE（全局 NoPE + 局部 RoPE 是稀疏注意力的常见
   混合模式）。V4 §2.3.3 的 Q/KV entry normalization 以 q_norm/k_norm 落实（压缩
   k 条目过 k_norm）。
6. 因果性（V4 §2.3.3 明示）：query 只能看到**严格先于本块**的压缩条目——块尾 j 的
   条目只对 >j 的 query 可见；看不到自己所在块内的其它 token（由滑窗分支补足局部）。

KV cache（生成路径，原型级）：state 存**全量** k/v（[B,T,n_kv,hd] 布局，k 为 k_norm 后、
**RoPE 前**——与 TriRetrievalAttention 存 RoPE 后不同，注意勿混用），每步由全量 cache 现算三分支
（O(L)/token，0.1B/seq 1024 可接受）。**生产路径应增量维护压缩/HCA 条目 cache**
（V4 的 1M 下 10% KV 正由此来；HCA 区 = 滑窗原始 KV + 压缩条目，V4 §3.5.1）；原型的
现算方式不改变前向数值语义（RoPE 只依赖绝对位置，压缩只依赖完整块内容）。
HCA 注入区随 state 携带（"hca_inj_k"/"hca_inj_v"），注入条目对所有 query 恒可见、
不占 token 位置槽（无 RoPE 相位，cache["pos"] 不变——与 blockpath 的 token 流注入
簿记不同，正是设计 §17.3"前缀偏差从结构上消失"的体现）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ModelConfig
from .blockpath import NamespaceMismatchError, check_namespace, make_namespace
from .common import RMSNorm


def _yarn_scaled_inv_freq(
    inv_freq: torch.Tensor,
    scale: float,
    original_max_seq: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """YaRN 逐维 ramp 插值（arXiv:2309.00071 §3.3；设计文档 §3"训练内 YaRN"）。

    对每个频率维 i：波长 λ_i = 2π/inv_freq_i，r_i = L_orig/λ_i，
    ramp γ_i = clamp((r_i − β_slow)/(β_fast − β_slow), 0, 1)（β_slow=1, β_fast=32 为原文默认），
    插值 inv'_i = inv_i·[(1−γ_i)/scale + γ_i]：
    - 高频维（λ_i ≤ L_orig/32，γ=1）：**精确不动**——局部（短距离）位置区分逐 bit 保持；
    - 低频维（λ_i ≥ L_orig，γ=0）：全插值 1/scale——把 scale 倍位置范围压回训练所见相位域；
    - 中间维按比例混合（保单调、无相位跳变）。

    返回 (inv', mscale)：mscale = 0.1·ln(scale)+1 为 YaRN logit 温度修正（论文 §3.4
    插值致注意力分布变平坦的补偿；q/k 各乘 √mscale → logits × mscale）。
    """
    if scale <= 1.0:
        return inv_freq, 1.0
    wavelengths = 2.0 * math.pi / inv_freq  # λ_i
    r = original_max_seq / wavelengths  # r_i = L_orig/λ_i
    gamma = ((r - beta_slow) / (beta_fast - beta_slow)).clamp(0.0, 1.0)
    inv_scaled = inv_freq * ((1.0 - gamma) / scale + gamma)
    mscale = 0.1 * math.log(scale) + 1.0
    return inv_scaled, mscale


def _build_rope_cache(cfg: ModelConfig, head_dim: int) -> tuple[torch.Tensor, torch.Tensor, float]:
    """按 cfg 构建滑窗分支 RoPE 缓存 → (cos [max_seq, D/2], sin, mscale)。

    - scaling="none" 且 max_seq ≤ 8192：旧式 fp32 构造（与 2026-07-31 前版本逐 bit 一致，
      既有 checkpoint/测试数值零漂移）；
    - scaling="yarn" 或 max_seq > 8192：fp64 计算 cos/sin 后落 fp32——大位置（如 256K）
      下 fp32 直接算 t·inv_freq 的辐角误差达 ~3e-2 rad（最高频维），fp64 构造把它压到
      1e-7 量级（仅缓存构建期一次性成本）。
    - rope_original_max_seq 缺省回填 max_seq/scale（如 262144/256 → 1024）。
    """
    inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    mscale = 1.0
    scaling = getattr(cfg, "rope_scaling", "none")
    scale = float(getattr(cfg, "rope_scale", 1.0))
    if scaling == "yarn" and scale > 1.0:
        orig = getattr(cfg, "rope_original_max_seq", None) or max(1, int(round(cfg.max_seq / scale)))
        inv_freq, mscale = _yarn_scaled_inv_freq(inv_freq, scale, orig)
    elif scaling not in ("none", "yarn"):
        raise ValueError(f"未知 rope_scaling={scaling!r}（支持 none/yarn）")
    if scaling == "none" and cfg.max_seq <= 8192:
        t = torch.arange(cfg.max_seq).float()
        freqs = torch.outer(t, inv_freq)
        return freqs.cos(), freqs.sin(), mscale
    t = torch.arange(cfg.max_seq, dtype=torch.float64)
    freqs = torch.outer(t, inv_freq.to(torch.float64))
    return freqs.cos().float(), freqs.sin().float(), mscale


class ChunkCompressor(nn.Module):
    """V4 式 softmax 门控池化压缩器（无重叠，stride:1）：连续 stride 个 token 的 k/v 各压成 1 条目。

    参照 DeepSeek V4 技术报告 §2.3.1–2.3.2（Eq.9–12；HCA 不重叠，本原型 CSA 亦不重叠——
    见模块 docstring 核对结论 1）：z = W_z(x)；s = softmax_块内位置(z + B_pos)；
    entry = Σ_j s_j ⊙ x_j。B_pos ∈ R^{stride×head_dim} 为学习块内位置偏置（V4 Eq.11，
    零初始化 = 初始无位置先验）；k、v 各一套投影/偏置，kv 头间共享（V4 为单序列
    MQA 式条目；本实现按 kv 头分别压缩、结构对齐）。
    尾部策略：丢弃不足 stride 的尾部 token（floor(T/stride)，与 blockpath.BlockCompressor
    同纪律——定长块语义；被丢弃的尾部由滑窗分支以原始 k/v 覆盖）。
    """

    def __init__(self, head_dim: int, stride: int):
        super().__init__()
        self.stride = stride
        self.k_z = nn.Linear(head_dim, head_dim, bias=False)
        self.v_z = nn.Linear(head_dim, head_dim, bias=False)
        self.k_pos = nn.Parameter(torch.zeros(stride, head_dim))
        self.v_pos = nn.Parameter(torch.zeros(stride, head_dim))

    def _comp(self, x: torch.Tensor, z_proj: nn.Linear, pos: torch.Tensor) -> torch.Tensor:
        # x: [B, n_kv, T, head_dim] → [B, n_kv, T//stride, head_dim]
        B, H, T, D = x.shape
        n = T // self.stride
        x = x[:, :, : n * self.stride, :].reshape(B, H, n, self.stride, D)
        z = z_proj(x) + pos  # 学习块内位置偏置（broadcast [stride, D]）
        s = torch.softmax(z.float(), dim=3).type_as(x)  # 块内 stride 个位置归一（V4 Eq.11）
        return (s * x).sum(dim=3)  # 通道级加权和（V4 Eq.12 的 Hadamard 形式）

    def forward(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """k/v: [B, n_kv, T, head_dim] → 压缩条目 [B, n_kv, T//stride, head_dim]。"""
        return self._comp(k, self.k_z, self.k_pos), self._comp(v, self.v_z, self.v_pos)


class TriRetrievalAttention(nn.Module):
    """三级注意力层：滑窗 + CSA 选择检索 + HCA 重压缩，学习门控融合（详见模块 docstring）。

    forward 签名与 TriRetrievalAttention 一致（x, state, offset）→ (out, new_state)，供 Block 直接替换。
    state = {"k","v"}（[B,T,n_kv,hd]，k 为 k_norm 后 RoPE **前**）+ 可选 HCA 注入区
    （"hca_inj_k"/"hca_inj_v"，[B,n_kv,N,hd]）。
    aux：测试/仪表挂点——传入 dict 时填入分支输出/门控/选择掩码/参考用张量（默认 None 零开销）。
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.n_q = cfg.n_q_heads
        self.n_kv = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        assert self.n_q % self.n_kv == 0
        self.window = cfg.tri_window
        self.topk = cfg.tri_csa_topk
        self.q_proj = nn.Linear(d, self.n_q * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d, self.n_kv * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_q * self.head_dim, d, bias=False)
        # QK-Norm：按 head_dim 归一（V4 §2.3.3 Query/KV entry normalization）
        self.q_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(self.head_dim, cfg.rms_eps)
        # 三分支共享同一组 k/v 投影（plan §4：滑窗 = 全量 k/v 尾部窗口，cache 存全量 k/v）——
        # 偏离 NSA §3.3.3 的分支独立 k/v（plan 显式指定共享：参数更省、单一 KV cache）
        self.csa_comp = ChunkCompressor(self.head_dim, cfg.tri_csa_stride)
        self.hca_comp = ChunkCompressor(self.head_dim, cfg.tri_hca_stride)
        # CSA 分支选择机制（V4 最优组合，tri_use_indexer=True 时启用）：
        # 独立 LightningIndexer 在压缩条目上打分选 top-k（DeepSeek V4 CSA 正式路径，
        # 与 HRL 的 model/hrl_indexer.py LightningIndexer 同构共享——设计 §11.1
        # "一个打分器两种检索对象"）。默认 False=NSA 式复用压缩注意力分数（数值兼容）。
        self.use_indexer = cfg.tri_use_indexer
        if self.use_indexer:
            from .hrl_indexer import LightningIndexer
            # indexer 作用在 q_nope（[B,n_q,T,D]）× 压缩条目 kc（[B,n_kv,S,D]）：
            # 按 kv 头共享一个 indexer（V4 单序列 MQA 式条目；query 侧多头、key 侧共享）。
            self.csa_indexer = LightningIndexer(
                d_model=self.head_dim, n_heads=cfg.tri_index_heads, d_index=cfg.tri_index_dim
            )
        # 门控（NSA Eq.5 sigmoid 形式，自 q 产生，per-head per-branch；裸 Parameter 实现，
        # 避免被 model._init_weights 的 nn.Linear 规则覆盖 → init 精确均等 1/3，记录见 docstring）
        self.gate_w = nn.Parameter(torch.zeros(3, self.head_dim))
        self.gate_b = nn.Parameter(torch.full((3,), -math.log(2.0)))  # sigmoid(-ln2) = 1/3
        # RoPE 缓存 [max_seq, head_dim/2]（half-split NeoX 风格；唯一构造点——
        # tri_attention_gated/decoupled/fully_decoupled 变体均 attach 到本类实例、
        # 复用本缓冲与 _rope，故全部经此一处覆盖）。max_seq 可扩至 256K：
        # 每行 32 float32 ×2（cos/sin）≈ 256B，256K 行 ≈ 67MB/层（0.1B 3 个 A 层 ≈ 201MB）。
        # scaling/YaRN 见 _build_rope_cache；默认 none 与旧版逐 bit 一致。
        cos, sin, self.rope_mscale = _build_rope_cache(cfg, self.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # namespace 校验用层号（由 model.__init__ 按层序写入；standalone 使用需手动设置）
        self.layer_idx: int = -1

    def rebuild_rope_cache(self) -> None:
        """按当前 self.cfg 原地重建 RoPE 缓存（扩窗/scaling 变更后调用，keep 设备）。

        供 scripts/extend_context.py 与 model.extend_context() 在加载旧 checkpoint 后
        扩 max_seq / 切 scaling 用；权重不动（缓存 persistent=False 本就不进 state_dict）。
        """
        cos, sin, self.rope_mscale = _build_rope_cache(self.cfg, self.head_dim)
        dev = self.rope_cos.device
        self.register_buffer("rope_cos", cos.to(dev), persistent=False)
        self.register_buffer("rope_sin", sin.to(dev), persistent=False)

    def _rope(self, x: torch.Tensor, offset: int) -> torch.Tensor:
        # x: [B, T, H, D]，half-split（NeoX 风格）旋转（与 TriRetrievalAttention._rope 同一实现）
        T = x.shape[1]
        cos = self.rope_cos[offset : offset + T]  # [T, D/2]
        sin = self.rope_sin[offset : offset + T]
        cos = torch.cat([cos, cos], dim=-1)[None, :, None, :]
        sin = torch.cat([sin, sin], dim=-1)[None, :, None, :]
        x1, x2 = x[..., : self.head_dim // 2], x[..., self.head_dim // 2 :]
        rot = torch.cat([-x2, x1], dim=-1)
        out = x * cos + rot * sin
        if self.rope_mscale != 1.0:
            # YaRN logit 温度修正：q/k 各乘 √mscale → 滑窗 logits × mscale（仅 yarn 生效）
            out = out * math.sqrt(self.rope_mscale)
        return out

    def inject_hca_entries(
        self,
        state: dict,
        entries: tuple[torch.Tensor, torch.Tensor],
        namespace: dict,
    ) -> dict:
        """把外部 HCA 条目前置拼入 state 的 HCA 注入区（知识块注入原生落点，设计 §17.3）。

        - entries = (k_inj, v_inj)，[B, n_kv, N, head_dim]；namespace = blockpath 五元组；
        - namespace 校验 fail-closed（复用 blockpath check_namespace/NamespaceMismatchError），
          任一字段不匹配即抛错，由调用方走重算/文本 RAG 回退；
        - 注入条目在 HCA 区排在压缩条目**之前**，对所有 query 恒可见（gist 前缀）；
          不占 token 位置槽（无 RoPE 相位，``cache["pos"]`` 不变）；
        - 返回新 state（不原地修改入参）。

        本任务只做结构与校验，注入条目的训练留给后续阶段（同 blockpath 原型纪律）。
        """
        k_inj, v_inj = entries
        if self.layer_idx < 0:
            raise NamespaceMismatchError("layer_idx 未设置（应由 model 按层序写入），拒绝注入")
        if "k" not in state or "v" not in state:
            raise NamespaceMismatchError("state 缺少 k/v（需先 prefill），拒绝注入")
        if k_inj.shape != v_inj.shape or k_inj.dim() != 4:
            raise NamespaceMismatchError(f"注入条目 k/v 形状不一致或非 4 维：{k_inj.shape} vs {v_inj.shape}")
        B, H, _, D = k_inj.shape
        if H != self.n_kv or D != self.head_dim or B != state["k"].shape[0]:
            raise NamespaceMismatchError(
                f"注入条目形状 {[B, H, int(k_inj.shape[2]), D]} 与层结构 "
                f"(B={state['k'].shape[0]}, n_kv={self.n_kv}, hd={self.head_dim}) 不匹配，拒绝注入"
            )
        expected = make_namespace(self.cfg, self.layer_idx, state["k"].dtype)
        check_namespace(expected, namespace)
        k_inj = k_inj.to(device=state["k"].device, dtype=state["k"].dtype)
        v_inj = v_inj.to(device=state["v"].device, dtype=state["v"].dtype)
        new_state = dict(state)
        if "hca_inj_k" in state:
            # 再次注入：新条目排最前（前置拼入）
            new_state["hca_inj_k"] = torch.cat([k_inj, state["hca_inj_k"]], dim=2)
            new_state["hca_inj_v"] = torch.cat([v_inj, state["hca_inj_v"]], dim=2)
        else:
            new_state["hca_inj_k"] = k_inj
            new_state["hca_inj_v"] = v_inj
        return new_state

    @staticmethod
    def _masked_softmax(logits: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        """fp32 masked softmax；整行无可选条目时返回 0 行（分支无贡献，由门控×0 吸收）。

        logits/keep: [..., S]；keep=False 处视为 -inf。softmax 内层 fp32 保稳定性。
        """
        neg_inf = torch.finfo(logits.dtype).min
        logits = logits.masked_fill(~keep, neg_inf)
        p = torch.softmax(logits.float(), dim=-1)
        # 全不可见行：softmax(全 -inf) = nan → 置 0
        p = torch.where(keep.any(dim=-1, keepdim=True), p, torch.zeros((), dtype=p.dtype, device=p.device))
        return p.type_as(logits)

    def forward(
        self,
        x: torch.Tensor,
        state: dict | None = None,
        offset: int = 0,
        aux: dict | None = None,
    ) -> tuple[torch.Tensor, dict]:
        B, T, _ = x.shape
        D = self.head_dim
        # q/k/v 投影 + QK-Norm（k 保持 RoPE 前入 cache；压缩分支用 NoPE 内容寻址）
        q = self.q_norm(self.q_proj(x).view(B, T, self.n_q, D))
        k = self.k_norm(self.k_proj(x).view(B, T, self.n_kv, D))
        v = self.v_proj(x).view(B, T, self.n_kv, D)
        if state is not None:
            k = torch.cat([state["k"], k], dim=1)
            v = torch.cat([state["v"], v], dim=1)
        Tk = k.shape[1]
        inj_k = state.get("hca_inj_k") if state else None
        inj_v = state.get("hca_inj_v") if state else None
        new_state = {"k": k, "v": v}
        if inj_k is not None:
            new_state["hca_inj_k"], new_state["hca_inj_v"] = inj_k, inj_v
        # RoPE：q 在 offset（cache 尾部对齐）；k 全量从 0（state k 覆盖绝对位置 0..Tk-1）
        q_rope = self._rope(q, offset).transpose(1, 2)  # [B, n_q, T, D]
        q_nope = q.transpose(1, 2)
        k_rope = self._rope(k, 0).transpose(1, 2)  # [B, n_kv, Tk, D]
        k_nope = k.transpose(1, 2)
        v = v.transpose(1, 2)
        rep = self.n_q // self.n_kv
        # 绝对位置：query 右对齐（Tk-T..Tk-1），key 0..Tk-1（与 TriRetrievalAttention cache 语义一致）
        i_abs = torch.arange(Tk - T, Tk, device=x.device)
        j_abs = torch.arange(Tk, device=x.device)

        # ── 滑窗分支（L0 精确，RoPE + GQA + masked SDPA）────────────────────────
        win = (j_abs[None, :] <= i_abs[:, None]) & (j_abs[None, :] > i_abs[:, None] - self.window)
        k_e = k_rope.repeat_interleave(rep, dim=1)
        v_e = v.repeat_interleave(rep, dim=1)
        o_win = F.scaled_dot_product_attention(q_rope, k_e, v_e, attn_mask=win[None, None])

        # ── CSA 分支（L1 情景：压缩 + 因果内 top-k 选择检索）────────────────────
        m = self.csa_comp.stride
        S = Tk // m
        if S > 0:
            kc, vc = self.csa_comp(k_nope, v)  # [B, n_kv, S, D]
            kc = self.k_norm(kc)  # KV entry normalization（V4 §2.3.3）
            # 条目因果性：块尾 j=m(s+1)-1 只对 >j 的 query 可见（V4 §2.3.3：看不到本块）
            tail = m * (torch.arange(S, device=x.device) + 1) - 1
            vis = tail[None, :] < i_abs[:, None]  # [T, S]
            kc_e = kc.repeat_interleave(rep, dim=1)
            vc_e = vc.repeat_interleave(rep, dim=1)
            logits = (q_nope @ kc_e.transpose(-1, -2)) / math.sqrt(D)  # [B, n_q, T, S]
            if self.use_indexer:
                # V4 CSA 式：独立 LightningIndexer 在压缩条目上打分选 top-k（正式路径）。
                # indexer 打分 query=q_nope([B,n_q,T,D]→按 kv 头取代表)× 压缩条目 kc([B,n_kv,S,D])，
                # 得 [B,n_q,T,S]；GQA 组内头求和共享选择（与 NSA Eq.10 同纪律），照 DSA/V4 原文。
                # indexer 输入按 kv 头分组：把 q_nope 的 rep 个 q 头求和到 kv 头（共享选择）。
                q_g = q_nope.view(B, self.n_kv, rep, T, self.head_dim).sum(dim=2)  # [B,n_kv,T,D]
                idx_scores = []
                for h in range(self.n_kv):
                    # LightningIndexer(x_q [B,T,D], x_k [B,S,D]) → [B,T,S]
                    s_h = self.csa_indexer(q_g[:, h], kc[:, h])  # [B,T,S]
                    idx_scores.append(s_h)
                imp = torch.stack(idx_scores, dim=1)  # [B,n_kv,T,S]
            else:
                # NSA 式（默认）：重要性分数 = 压缩注意力分数（NSA Eq.8），GQA 组内头求和共享选择
                p = self._masked_softmax(logits, vis[None, None].expand(B, self.n_q, T, S))
                imp = p.view(B, self.n_kv, rep, T, S).sum(dim=2)  # [B, n_kv, T, S]
            # top-k 仅因果集合内；选择离散无梯度（NSA 式：分数随压缩注意力训练，照抄不自创）
            k_eff = min(self.topk, S)
            topv, topi = imp.masked_fill(~vis[None, None], -torch.finfo(imp.dtype).max).topk(k_eff, dim=-1)
            keep = torch.zeros_like(vis[None, None].expand(B, self.n_kv, T, S))
            keep.scatter_(-1, topi, topv > -torch.finfo(imp.dtype).max)
            keep_e = keep.repeat_interleave(rep, dim=1)
            attn = self._masked_softmax(logits, keep_e)
            o_csa = attn @ vc_e
        else:
            keep = None
            o_csa = torch.zeros_like(o_win)

        # ── HCA 分支（L2 gist：128:1 重压缩，因果内 dense 恒可见）────────────────
        m2 = self.hca_comp.stride
        S2 = Tk // m2
        parts_k, parts_v = [], []
        if inj_k is not None:
            parts_k.append(inj_k)
            parts_v.append(inj_v)
        if S2 > 0:
            kh, vh = self.hca_comp(k_nope, v)
            parts_k.append(self.k_norm(kh))
            parts_v.append(vh)
        if parts_k:
            k_h = torch.cat(parts_k, dim=2)  # [B, n_kv, N_inj+S2, D]（注入区在前）
            v_h = torch.cat(parts_v, dim=2)
            n_inj = inj_k.shape[2] if inj_k is not None else 0
            vis_h = [torch.ones(T, n_inj, dtype=torch.bool, device=x.device)] if n_inj else []
            if S2 > 0:
                tail2 = m2 * (torch.arange(S2, device=x.device) + 1) - 1
                vis_h.append(tail2[None, :] < i_abs[:, None])
            vis_h = torch.cat(vis_h, dim=1)  # [T, N_inj+S2]
            k_he = k_h.repeat_interleave(rep, dim=1)
            v_he = v_h.repeat_interleave(rep, dim=1)
            logits_h = (q_nope @ k_he.transpose(-1, -2)) / math.sqrt(D)
            attn_h = self._masked_softmax(logits_h, vis_h[None, None].expand(B, self.n_q, T, k_h.shape[2]))
            o_hca = attn_h @ v_he
        else:
            n_inj = 0
            o_hca = torch.zeros_like(o_win)

        # ── 学习门控融合（NSA Eq.5；顺序 [win, csa, hca]，init 精确 1/3）─────────
        g = torch.sigmoid(q_nope.transpose(1, 2) @ self.gate_w.T + self.gate_b)  # [B, T, n_q, 3]
        o = (
            g[..., 0:1] * o_win.transpose(1, 2)
            + g[..., 1:2] * o_csa.transpose(1, 2)
            + g[..., 2:3] * o_hca.transpose(1, 2)
        )
        o = o.reshape(B, T, self.n_q * D)

        if aux is not None:
            aux.update(
                o_win=o_win, o_csa=o_csa, o_hca=o_hca,  # [B, n_q, T, D] 分支输出（门控前）
                gates=g,  # [B, T, n_q, 3]（win/csa/hca 顺序）
                q_rope=q_rope, k_rope=k_rope, v=v,  # 滑窗参考实现用（kv 未展开）
                i_abs=i_abs, sel_keep=keep,  # 选择合法性检查用（keep: [B, n_kv, T, S] 或 None）
                n_csa=S, n_hca=S2, n_hca_inj=n_inj,
            )
        return self.o_proj(o), new_state
