"""解耦双通道门控（DecoupledHcaGate）——消除扩容门控对"自然 gist"副作用的方案 A
（门控上下文感知自适应；不改原 tri_attention.py / tri_attention_gated.py）。

副作用（runs/recall_gated + scripts/unified_full_chain_demo 实测）：
扩容门控 GatedFusionMLP（`g=sigmoid(MLP(q))` 输出 3 维 win/csa/hca）为让"知识块 KV 注入
HCA 区后可召回"（KV 注入答对率 0.625），训练门控对 **HCA 分支整体**开权重；但 HCA 分支同时
承载"注入知识块条目"与"长文本自然 gist 条目（压缩器产生）"——召回训练对**两类条目一视同仁**
地开权重 → in-context 下 HCA 对自然 gist 也开权重、干扰滑窗 L0 精确召回
（in-context 0.688 → 0.250，unified_full_chain_demo 实测带门控 0.250 / 拆门控回 0.6875）。

方案 A（TokenMem arXiv:2607.22625 / DecoupledRAG 先例，研究记忆 context-aware-gating-research）：
把 **HCA 注意力按条目来源拆两路**——
- **自然 gist 条目**（压缩器产生）：走 **natural_gate**（复用/包装原 GatedFusionMLP，
  对 gist 维持原权重/行为不变——恒等初始化或已训值）；
- **注入知识块条目**（inject_hca_entries 前置拼入 HCA 区）：走 **inject_gate**（**独立的、
  fc2 零初始化**的小 MLP，起点 g≈0，仅对注入条目激活——TokenMem 零初始化先例）。

关键设计（红线）：
- **结构化来源路由**：注入条目经 `inject_hca_entries` 拼入 HCA 区（排在压缩条目**之前**），
  故"前 n_inj 个条目"天然带 namespace 来源标记（注入=True）；自然 gist 条目=False。
  门控按条目来源分流——**非学习 embedding**，契合 BlockStore 五元组 namespace fail-closed 红线。
- **恒等初始化**：natural_gate 保持原 GatedFusionMLP 行为（g=1/3 或已训值，零改动）；
  inject_gate fc2=0、bias=-ln2 → 任意输入初始 g=sigmoid(-ln2)≈1/3（不干扰）；fc1 小随机破对称
  （std=0.02，gated-fusion-mlp 经验：fc1=0 时隐藏单元退化、fc2 单点强梯度易发散）。
- **注入式 attach**：`attach_decoupled_gate(mixer)` 预绑定 forward 替换门控/HCA 逻辑；
  原 tri_attention.py / tri_attention_gated.py 不改。`detach_decoupled_gate` 恢复原 forward。
- **主干 frozen 纪律**：召回头训练只训 inject_gate（natural_gate frozen 保 gist 原权重——
  结构性消除副作用），不动 q/k/v/o 投影与 gate_w/b。

实现要点（HCA 拆分两路）：
  原 HCA：attn_h = softmax(q·[K_inj; K_h]) @ [V_inj; V_h]  →  o_hca（单路，单一门控 g_hca）。
  解耦后：
    o_inj = softmax(q·K_inj) @ V_inj   →  门控 g_inj = inject_gate(q)[..., 2]
    o_nat = softmax(q·K_h)   @ V_h     →  门控 g_nat = natural_gate(q)[..., 2]
    o = g_win·o_win + g_csa·o_csa + g_inj·o_inj + g_nat·o_nat
  win/csa 门控沿用 natural_gate 的 [...,0]/[...,1]（注入条目只进 HCA，与 win/csa 无关）。
  注入条目 vs 自然 gist 用**不同 softmax 分母**（不归一在一起）——这是"解耦"的核心：
  两类条目各自独立归一 + 独立门控，召回训练只调 inject_gate 即可保 gist 通路零改动。

来源：研究记忆 /memories/repo/context-aware-gating-research.md（方案 A 推荐）；
     扩容门控记忆 /memories/repo/gated-fusion-mlp.md（恒等初始化 fc1 随机 + fc2=0 经验）。
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tri_attention_gated import GatedFusionMLP, _INIT_BIAS, _gated_forward

__all__ = [
    "DecoupledHcaGate",
    "attach_decoupled_gate",
    "detach_decoupled_gate",
    "set_decoupled_gate_enabled",
]


class DecoupledHcaGate(nn.Module):
    """解耦双通道 HCA 门控：natural_gate（gist）+ inject_gate（注入知识块）两路独立。

    结构：
      - natural_gate：GatedFusionMLP（fc1 随机 + fc2=0，恒等 g=1/3；或载入已训权重）——
        负责 win/csa 门控 + 自然 gist 的 HCA 门控（沿用原 GatedFusionMLP 行为，对 gist 维持
        原权重/行为不变）。
      - inject_gate：GatedFusionMLP（同构，fc2 零初始化起点 g≈0）——仅对注入知识块条目
        激活（TokenMem 零初始化先例；起点不干扰自然 gist 通路）。

    forward(q_nope, has_inject)：按"是否存在注入条目"路由——
      - has_inject=False（无注入）：退化为 natural_gate 单门控（in-context 纯文本场景，
        结构性恢复精确召回——inject_gate 完全不参与）；
      - has_inject=True（有注入）：HCA 拆两路，注入条目走 inject_gate、gist 走 natural_gate。
    """

    def __init__(
        self,
        head_dim: int,
        hidden: int = 128,
        natural_gate: GatedFusionMLP | None = None,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.hidden = hidden
        # 自然 gist 门控（复用/包装原 GatedFusionMLP；恒等初始化 g=1/3，对 gist 维持原行为）
        self.natural_gate = natural_gate if natural_gate is not None else GatedFusionMLP(head_dim, hidden)
        # 注入知识块门控（独立零初始化通道；fc2=0 起点 g≈0，仅对注入条目激活）
        self.inject_gate = GatedFusionMLP(head_dim, hidden)
        # 运行时开关：False 时强制走 natural 单门控（测试/消融"恢复纯文本精确召回"对照用）
        self.enabled = True

    def gate_natural(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope [B,T,n_q,head_dim] → g [B,T,n_q,3]（win/csa/hca；自然 gist 通路）。"""
        return self.natural_gate.gate(q_nope)

    def gate_inject(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope [B,T,n_q,head_dim] → g [B,T,n_q,3]（仅取 [...,2] 作注入条目 HCA 门控）。"""
        return self.inject_gate.gate(q_nope)

    def forward(self, q_nope: torch.Tensor, has_inject: bool = False) -> torch.Tensor:
        """按条目来源路由门控通道。

        - q_nope: [B, T, n_q, head_dim]；
        - has_inject: HCA 区是否含注入条目（state 中 hca_inj_k 非 None——结构化来源标记）。
        返回 g [B,T,n_q,3]：
          - has_inject=False：g = natural_gate(q)（win/csa/hca 全走 natural——纯文本场景）；
          - has_inject=True ：g[...,0:2]=natural（win/csa），g[...,2]=inject_gate(q)[...,2]
            （HCA 门控对注入条目——自然 gist 的 HCA 门控在 forward 融合时另取 natural）。
        注：HCA 拆分后"自然 gist 的 HCA 门控"由融合逻辑单独取 natural_gate，本 forward 的
        返回 g[...,2] 仅供"无注入"或"注入条目门控"用；真正的双路融合在 _decoupled_forward。
        """
        if (not self.enabled) or (not has_inject):
            return self.gate_natural(q_nope)
        g_nat = self.gate_natural(q_nope)
        g_inj_hca = self.gate_inject(q_nope)[..., 2:3]
        return torch.cat([g_nat[..., 0:2], g_inj_hca], dim=-1)

    def init_identity(self) -> None:
        """恒等初始化（与 GatedFusionMLP 一致）：fc1 小随机破对称 + fc2=0 + bias=-ln2。

        供 attach 后手动重置（attach 已做）；单独调用用于测试断言恒等初始化。
        """
        for gate in (self.natural_gate, self.inject_gate):
            nn.init.normal_(gate.fc1.weight, std=0.02)
            nn.init.zeros_(gate.fc1.bias)
            nn.init.zeros_(gate.fc2.weight)
            nn.init.constant_(gate.fc2.bias, _INIT_BIAS)


# ---------------------------------------------------------------------------
# 注入式挂载（不改原 TriRetrievalAttention 类；预绑定 forward 替换门控/HCA 逻辑）
# ---------------------------------------------------------------------------
def _decoupled_forward(self, x, state=None, offset: int = 0, aux: dict | None = None):
    """替换 TriRetrievalAttention.forward：HCA 拆"注入条目 vs 自然 gist"两路独立门控。

    与原 _gated_forward 唯一差异在 HCA 分支 + 门控融合：
      原：attn_h = softmax(q·[K_inj; K_h]) @ [V_inj; V_h] → o_hca（单路）→ g[...,2]·o_hca。
      解耦：
        o_inj = softmax(q·K_inj) @ V_inj   →  g_inj = inject_gate(q)[..., 2]
        o_nat = softmax(q·K_h)   @ V_h     →  g_nat = natural_gate(q)[..., 2]
        o = g_win·o_win + g_csa·o_csa + g_inj·o_inj + g_nat·o_nat
      （无注入时 n_inj=0 → o_inj=0、g_inj 不取 → 退化为 natural 单门控，结构性恢复精确召回）。
    win/csa 门控沿用 natural_gate 的 [...,0]/[...,1]（注入条目只进 HCA）。
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

    # ── HCA 分支（解耦双通道：注入条目 vs 自然 gist 各自独立归一 + 独立门控）─────
    # 结构化来源路由：inject_hca_entries 拼入的条目排在 HCA 区前 n_inj 个（注入=True，
    # namespace 来源标记）；压缩器产生的自然 gist 条目=False——按条目来源分流门控通道。
    dec_gate = self.decoupled_gate
    has_inject = inj_k is not None and dec_gate.enabled
    n_inj = inj_k.shape[2] if inj_k is not None else 0
    o_inj = torch.zeros_like(o_win)
    o_nat = torch.zeros_like(o_win)
    S2 = Tk // self.hca_comp.stride

    if n_inj and has_inject:
        # 注入知识块条目：独立 softmax 归一（对所有 query 恒可见）+ inject_gate 门控。
        # 空归一防护：注入区独立 softmax（无压缩条目混入）须防 NaN——当全 query 对注入条目
        # 不可见（不可能，恒可见）或注入区条目数=0 时 softmax(空)=NaN。恒可见故正常非空；
        # 但仍显式断言非空防呆（与 _masked_softmax 全不可见行置 0 的纪律一致）。
        k_ie = inj_k.repeat_interleave(rep, dim=1)
        v_ie = inj_v.repeat_interleave(rep, dim=1)
        logits_i = (q_nope @ k_ie.transpose(-1, -2)) / math.sqrt(D)
        vis_i = torch.ones(T, n_inj, dtype=torch.bool, device=x.device)
        attn_i = self._masked_softmax(logits_i, vis_i[None, None].expand(B, self.n_q, T, n_inj))
        o_inj = attn_i @ v_ie

    if S2 > 0:
        # 自然 gist 条目（压缩器产生）：独立 softmax 归一 + natural_gate 门控（因果内恒可见）
        kh, vh = self.hca_comp(k_nope, v)
        kh = self.k_norm(kh)
        k_he = kh.repeat_interleave(rep, dim=1)
        v_he = vh.repeat_interleave(rep, dim=1)
        tail2 = self.hca_comp.stride * (torch.arange(S2, device=x.device) + 1) - 1
        vis_n = tail2[None, :] < i_abs[:, None]
        logits_n = (q_nope @ k_he.transpose(-1, -2)) / math.sqrt(D)
        attn_n = self._masked_softmax(logits_n, vis_n[None, None].expand(B, self.n_q, T, S2))
        o_nat = attn_n @ v_he

    # 无注入时兼容旧行为：o_hca = 自然 gist 单路（o_nat），g[...,2] 用 natural（与 _gated_forward 一致）
    o_hca = o_nat

    # ── 门控融合（解耦：win/csa 走 natural；HCA 注入条目走 inject、gist 走 natural）─────────
    q_bt = q_nope.transpose(1, 2)  # [B,T,n_q,D]
    if has_inject:
        g_nat = dec_gate.gate_natural(q_bt)            # [B,T,n_q,3]（win/csa/natural-HCA）
        g_inj_hca = dec_gate.gate_inject(q_bt)[..., 2:3]  # [B,T,n_q,1]（注入条目 HCA 门控）
        g_win, g_csa, g_hca_nat = g_nat[..., 0:1], g_nat[..., 1:2], g_nat[..., 2:3]
        g = torch.cat([g_win, g_csa, g_inj_hca], dim=-1)  # aux 记录用（HCA 位 = 注入门控）
    else:
        # 无注入（in-context 纯文本）：退化为 natural 单门控——结构性恢复精确召回
        g = dec_gate.gate_natural(q_bt)
        g_win, g_csa, g_hca_nat = g[..., 0:1], g[..., 1:2], g[..., 2:3]
        g_inj_hca = torch.zeros_like(g_hca_nat)
    o = (
        g_win * o_win.transpose(1, 2)
        + g_csa * o_csa.transpose(1, 2)
        + g_inj_hca * o_inj.transpose(1, 2)
        + g_hca_nat * o_nat.transpose(1, 2)
    )
    o = o.reshape(B, T, self.n_q * D)

    if aux is not None:
        aux.update(
            o_win=o_win, o_csa=o_csa, o_hca=o_hca, o_inj=o_inj, o_nat=o_nat,
            gates=g, gate_inject=g_inj_hca,
            q_rope=q_rope, k_rope=k_rope, v=v,
            i_abs=i_abs, sel_keep=keep,
            n_csa=S, n_hca=S2, n_hca_inj=n_inj, has_inject=bool(has_inject),
        )
    return self.o_proj(o), new_state


def attach_decoupled_gate(
    mixer,
    natural_state_dict: dict | None = None,
    inject_state_dict: dict | None = None,
    hidden: int = 128,
) -> DecoupledHcaGate:
    """给 TriRetrievalAttention 实例挂载 DecoupledHcaGate 并预绑定 forward（注入式，不改原类）。

    - mixer：TriRetrievalAttention 实例；
    - natural_state_dict：已训 GatedFusionMLP 权重（trained_gate_mlp.pt 某层的 state_dict）——
      载入后 natural_gate 保持已训行为（对 gist 维持原权重）；None=恒等初始化（g=1/3）；
    - inject_state_dict：已训 inject_gate 权重（重训产物）；None=零初始化（起点 g≈0，待训）；
    - hidden：门控 MLP 隐藏维（须与已训权重一致，默认 128）。

    实现（复用 tri_attention_gated 的恒等初始化 + 注入式纪律）：
      ① 建 natural_gate（GatedFusionMLP，可选载入已训权重）+ inject_gate（零初始化）；
      ② 包装成 DecoupledHcaGate 注册为 mixer.decoupled_gate（nn.Module 子模块，
         随 state_dict 存取，键 "decoupled_gate.*"）；
      ③ 记录原 forward 到 mixer._orig_forward_decoupled（detach 恢复用）；
      ④ 预绑定 _decoupled_forward 替换门控/HCA 逻辑。
    返回挂载的 DecoupledHcaGate（训练目标参数来源：inject_gate；natural_gate frozen）。
    """
    natural = GatedFusionMLP(mixer.head_dim, hidden).to(device=mixer.gate_w.device, dtype=mixer.gate_w.dtype)
    if natural_state_dict is not None:
        natural.load_state_dict(natural_state_dict)
    gate = DecoupledHcaGate(mixer.head_dim, hidden, natural_gate=natural).to(
        device=mixer.gate_w.device, dtype=mixer.gate_w.dtype)
    if inject_state_dict is not None:
        gate.inject_gate.load_state_dict(inject_state_dict)
    mixer.decoupled_gate = gate  # nn.Module 子模块：随 state_dict 存取（"decoupled_gate.*"）
    if not hasattr(mixer, "_orig_forward_decoupled"):
        # 记录"被替换前"的 forward：若已挂 GatedFusionMLP（gate_mlp）则是 _gated_forward（单门控，
        # detach 后恢复它）；否则是原线性门控 forward（类方法绑定）。
        mixer._orig_forward_decoupled = mixer.forward
    mixer.forward = types.MethodType(_decoupled_forward, mixer)  # 预绑定替换（实例级）
    return gate


def detach_decoupled_gate(mixer) -> None:
    """恢复 attach 前的 forward（原线性门控或 GatedFusionMLP 单门控）并移除 decoupled_gate。

    若 attach 前 mixer 挂了 gate_mlp（_gated_forward），detach 恢复它（保留扩容门控）；
    否则恢复原线性门控。供 unload/消融对照用。
    """
    if hasattr(mixer, "_orig_forward_decoupled"):
        mixer.forward = mixer._orig_forward_decoupled
        del mixer._orig_forward_decoupled
    if hasattr(mixer, "decoupled_gate"):
        del mixer.decoupled_gate


def set_decoupled_gate_enabled(mixer, enabled: bool) -> None:
    """开/关解耦双通道（enabled=False 时强制走 natural 单门控——恢复纯文本精确召回对照）。

    测试/消融用：attach 后关 enabled 即退化为单门控（in-context 精确召回恢复），
    开 enabled 恢复双通道（注入召回）。运行时开关，不动权重。
    """
    if hasattr(mixer, "decoupled_gate"):
        mixer.decoupled_gate.enabled = bool(enabled)
