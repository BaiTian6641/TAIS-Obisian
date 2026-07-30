"""HCA 门控扩容（GatedFusionMLP）——突破 585 线性门控容量瓶颈的可选升级（不改原 tri_attention.py）。

背景（runs/retrieval_recall/report.json + memories/repo/retrieval-recall-training.md）：
原门控 `g = sigmoid(q @ gate_w.T + gate_b)`（NSA Eq.5 线性标量门控，gate_w[3,head_dim]+gate_b[3]，
**每层 195 参数、3 个 A 层共 585**）——KV 注入召回训到 0.188 即触顶（<< in-context 上界 0.70）。
诊断：线性标量门控学不会"对注入条目开权重"的复杂内容路由（容量瓶颈，非通路问题）。

本模块把门控扩为**小 MLP**（仅增强门控表达力，三分支融合语义不变）：
    g = sigmoid( MLP(q_nope) )，MLP = Linear(head_dim→hidden) + GELU + Linear(hidden→3)

关键设计（红线）：
- **恒等初始化**：输出层 W2=0、b2=-ln2（第一层 W1/b1=0）→ 初始 g=sigmoid(-ln2)=1/3，
  与原门控零初始化精确一致——挂上 GatedFusionMLP 后初始前向行为与原线性门控**逐位相同**，
  不破坏既有 checkpoint 行为（扩容是可选升级，默认行为不变）。
- **注入式（不改原类）**：`attach_gated_fusion(mixer)` 给 TriRetrievalAttention 实例挂上
  `mixer.gate_mlp` 并**预绑定 forward**（types.MethodType）替换门控计算；原 `gate_w/gate_b`
  Parameter 保留在 state_dict（向后兼容：旧 checkpoint 加载不报错、键不缺失）。
  `detach_gated_fusion` 恢复原 forward（unload 兼容路径）。
- **融合语义不变**：仍是 win/csa/hca 三分支 sigmoid 独立门控加权和（仅门控产生方式由线性换 MLP）。
- **主干 frozen 纪律**：召回头训练只训 GatedFusionMLP（~8.4k/层），不动 q/k/v/o 投影与 gate_w/b。

向后兼容处理：旧 checkpoint 的 gate_w/gate_b 保留在模块 state_dict（加载不报错）；其值可通过
`from_linear_gate` 融入 MLP 第一层（W1[:, :3]=gate_w.T）使 MLP 初始即等价旧线性门控——默认
`fuse_linear=False` 用纯恒等初始化（g=1/3，与原零初始化门控一致；旧已训 gate 值默认舍弃，注释说明）。
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

# 恒等初始化偏置：sigmoid(-ln2) = 1/3（与原门控 gate_b 零初始化+ bias=-ln2 精确一致）
_INIT_BIAS = -math.log(2.0)


class GatedFusionMLP(nn.Module):
    """门控小 MLP：q_nope → 3 分支门控 logit（Linear+GELU+Linear，恒等初始化 g=1/3）。

    结构：head_dim → hidden（GELU）→ 3（win/csa/hca）。参数量 = head_dim*hidden + hidden*3
    （bias 另计 hidden+3）；head_dim=64、hidden=128 时 64*128+128*3 + 131 = 8707/层。

    恒等初始化：W1/b1/W2 全 0、b2=-ln2 → 任意输入 q 初始输出 logit=-ln2 → g=1/3，
    与原线性门控（gate_w=0, gate_b=-ln2）的初始行为精确一致（不破坏既有 checkpoint）。
    """

    def __init__(self, head_dim: int, hidden: int = 128, fuse_gate_w: torch.Tensor | None = None):
        super().__init__()
        self.head_dim = head_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(head_dim, hidden)
        self.fc2 = nn.Linear(hidden, 3)
        # 恒等初始化保 g=1/3：fc2 权重 0 + bias=-ln2 → 任意输入初始 logit=-ln2 → g=1/3，
        # 与原线性门控零初始化精确一致（不破坏既有 checkpoint 行为）。
        # fc1 用小随机初始化破对称（std=0.02）：fc2=0 时 fc1 不影响初始输出（仍 g=1/3），
        # 但提供随机特征基供 fc2 学习——若 fc1 也 0，所有隐藏单元梯度相同退化为线性门控，
        # 且 fc2 单点强梯度易发散（实测 lr 5e-3/5e-4 均 CE 爆炸）；随机 fc1 破对称后 MLP
        # 表达力真正可用、训练稳定。
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, _INIT_BIAS)
        if fuse_gate_w is not None:
            self.from_linear_gate(fuse_gate_w)

    def from_linear_gate(self, gate_w: torch.Tensor) -> None:
        """把旧线性门控 gate_w [3, head_dim] 融入第一层（W1[:, :3]=gate_w.T → fc1 前 3 个隐藏单元
        初始输出 q@gate_w.T，经 GELU≈线性段近似传递）——可选兼容路径（默认不用，纯恒等初始化）。

        注：GELU 非严格线性，此为**近似**融入；精确等价需 bypass。本项目默认纯恒等初始化
        （g=1/3 与原零初始化一致），旧已训 gate 值舍弃重训（召回头本就只训门控）。
        """
        with torch.no_grad():
            h = min(self.hidden, gate_w.shape[0])
            self.fc1.weight[:h, :] = gate_w[:h, :].to(self.fc1.weight.dtype)

    def forward(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope: [B, T, n_q, head_dim] → 门控 logit [B, T, n_q, 3]（外层再 sigmoid）。"""
        return self.fc2(F.gelu(self.fc1(q_nope)))

    def gate(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope: [B, T, n_q, head_dim] → g = sigmoid(logit) [B, T, n_q, 3]（win/csa/hca 顺序）。"""
        return torch.sigmoid(self.forward(q_nope))


# ---------------------------------------------------------------------------
# 注入式挂载（不改原 TriRetrievalAttention 类；预绑定 forward 替换门控计算）
# ---------------------------------------------------------------------------
def _gated_forward(self, x, state=None, offset: int = 0, aux: dict | None = None):
    """替换 TriRetrievalAttention.forward：仅门控由 MLP 产生，其余三分支逻辑与原一致。

    与原 forward 唯一差异在门控行：`g = self.gate_mlp.gate(q_nope_bt)`（MLP）替代
    `g = sigmoid(q_nope @ gate_w.T + gate_b)`（线性）。分支计算/注入/融合语义不变。
    """
    B, T, _ = x.shape
    D = self.head_dim
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
    q_rope = self._rope(q, offset).transpose(1, 2)      # [B, n_q, T, D]
    q_nope = q.transpose(1, 2)                          # [B, n_q, T, D]
    k_rope = self._rope(k, 0).transpose(1, 2)           # [B, n_kv, Tk, D]
    k_nope = k.transpose(1, 2)
    v = v.transpose(1, 2)                               # [B, n_kv, Tk, D]
    rep = self.n_q // self.n_kv
    i_abs = torch.arange(Tk - T, Tk, device=x.device)
    j_abs = torch.arange(Tk, device=x.device)

    # ── 滑窗分支（RoPE + GQA + masked SDPA，与原一致）──────────────────────────
    win = (j_abs[None, :] <= i_abs[:, None]) & (j_abs[None, :] > i_abs[:, None] - self.window)
    k_e = k_rope.repeat_interleave(rep, dim=1)
    v_e = v.repeat_interleave(rep, dim=1)
    o_win = F.scaled_dot_product_attention(q_rope, k_e, v_e, attn_mask=win[None, None])

    # ── CSA 分支（压缩 + 因果内 top-k 选择检索，与原一致）──────────────────────
    m = self.csa_comp.stride
    S = Tk // m
    if S > 0:
        kc, vc = self.csa_comp(k_nope, v)
        kc = self.k_norm(kc)
        tail = m * (torch.arange(S, device=x.device) + 1) - 1
        vis = tail[None, :] < i_abs[:, None]
        kc_e = kc.repeat_interleave(rep, dim=1)
        vc_e = vc.repeat_interleave(rep, dim=1)
        logits = (q_nope @ kc_e.transpose(-1, -2)) / math.sqrt(D)
        if self.use_indexer:
            q_g = q_nope.view(B, self.n_kv, rep, T, self.head_dim).sum(dim=2)
            idx_scores = [self.csa_indexer(q_g[:, h], kc[:, h]) for h in range(self.n_kv)]
            imp = torch.stack(idx_scores, dim=1)
        else:
            p = self._masked_softmax(logits, vis[None, None].expand(B, self.n_q, T, S))
            imp = p.view(B, self.n_kv, rep, T, S).sum(dim=2)
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

    # ── HCA 分支（128:1 重压缩 + 注入区，因果内 dense 恒可见，与原一致）─────────
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
        k_h = torch.cat(parts_k, dim=2)
        v_h = torch.cat(parts_v, dim=2)
        n_inj = inj_k.shape[2] if inj_k is not None else 0
        vis_h = [torch.ones(T, n_inj, dtype=torch.bool, device=x.device)] if n_inj else []
        if S2 > 0:
            tail2 = m2 * (torch.arange(S2, device=x.device) + 1) - 1
            vis_h.append(tail2[None, :] < i_abs[:, None])
        vis_h = torch.cat(vis_h, dim=1)
        k_he = k_h.repeat_interleave(rep, dim=1)
        v_he = v_h.repeat_interleave(rep, dim=1)
        logits_h = (q_nope @ k_he.transpose(-1, -2)) / math.sqrt(D)
        attn_h = self._masked_softmax(logits_h, vis_h[None, None].expand(B, self.n_q, T, k_h.shape[2]))
        o_hca = attn_h @ v_he
    else:
        n_inj = 0
        o_hca = torch.zeros_like(o_win)

    # ── 门控融合：唯一差异——MLP 门控替代线性门控（win/csa/hca 顺序不变）─────────
    g = self.gate_mlp.gate(q_nope.transpose(1, 2))      # [B, T, n_q, 3]
    o = (
        g[..., 0:1] * o_win.transpose(1, 2)
        + g[..., 1:2] * o_csa.transpose(1, 2)
        + g[..., 2:3] * o_hca.transpose(1, 2)
    )
    o = o.reshape(B, T, self.n_q * D)

    if aux is not None:
        aux.update(
            o_win=o_win, o_csa=o_csa, o_hca=o_hca,
            gates=g,
            q_rope=q_rope, k_rope=k_rope, v=v,
            i_abs=i_abs, sel_keep=keep,
            n_csa=S, n_hca=S2, n_hca_inj=n_inj,
        )
    return self.o_proj(o), new_state


def attach_gated_fusion(mixer, hidden: int = 128, fuse_linear: bool = False) -> GatedFusionMLP:
    """给 TriRetrievalAttention 实例挂载 GatedFusionMLP 并预绑定 forward（注入式，不改原类）。

    - mixer：TriRetrievalAttention 实例；hidden：MLP 隐藏维（默认 128）；
    - fuse_linear：True 时把旧 gate_w 融入 MLP 第一层（近似等价旧线性门控，见 from_linear_gate）；
      默认 False = 纯恒等初始化（g=1/3，与原零初始化门控一致，旧已训 gate 值舍弃重训）。
    - 原 gate_w/gate_b 保留在 state_dict（向后兼容：旧 checkpoint 加载不报错、键不缺失）；
      gate_mlp 是 nn.Module 子模块（随 mixer.state_dict 存取，键 "gate_mlp.*"）。
    - 记录原 forward 到 mixer._orig_forward（detach_gated_fusion 恢复用）。
    返回挂载的 GatedFusionMLP（训练目标参数来源）。
    """
    mlp = GatedFusionMLP(
        mixer.head_dim, hidden,
        fuse_gate_w=(mixer.gate_w.detach() if fuse_linear else None),
    ).to(device=mixer.gate_w.device, dtype=mixer.gate_w.dtype)
    mixer.gate_mlp = mlp  # nn.Module 子模块：随 state_dict 存取（"gate_mlp.*"）
    if not hasattr(mixer, "_orig_forward"):
        mixer._orig_forward = mixer.forward  # 记录原 forward（类方法绑定，detach 恢复用）
    mixer.forward = types.MethodType(_gated_forward, mixer)  # 预绑定替换（实例级）
    return mlp


def detach_gated_fusion(mixer) -> None:
    """恢复原线性门控 forward 并移除 gate_mlp（unload/消融对照用）。"""
    if hasattr(mixer, "_orig_forward"):
        mixer.forward = mixer._orig_forward
        del mixer._orig_forward
    if hasattr(mixer, "gate_mlp"):
        del mixer.gate_mlp
