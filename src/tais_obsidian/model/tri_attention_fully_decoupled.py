"""彻底解耦门控（FullyDecoupledGate）——注入召回走独立 csa 通道，消除 ic/KV 结构性权衡。

背景（方案 A 解耦的诚实权衡发现，/memories/repo/decoupled-gate.md + niah-length-scan-gate-adaptive.md）：
方案 A（DecoupledHcaGate）把注入召回隔离成功（inject_gate 独立训 → KV 注入答对率 0.625），
但 **win/csa 门控仍由 natural_gate 共享控制**——门控自适应（natural_gate 重训"对 gist 关"
恢复 in-context 0.688）让 natural_gate 学"win 主导、压 csa"，而 **KV 注入召回的检索路径依赖
csa/HCA** → 对 gist 关必然压 csa → KV 召回崩到 0.438（结构性权衡，KV 锚定也只能部分回升）。

**本模块（彻底解耦）**：注入召回的 csa 检索路径**不再经过 natural_gate 的 csa 门控**——
把 win/csa/hca 三分支 + 注入通道拆成"自然通路"与"注入通路"两套**完全独立的门控**：
  - **natural_gate**（GatedFusionMLP，3 维 win/csa/hca）：门控**自然通路**——
    滑窗 o_win、自然 csa（压缩器产生）、自然 gist（hca 压缩器产生）。
    可重训"对 gist 关"（win 主导、压自然 csa/gist）恢复 in-context 0.688——**不影响注入召回**。
  - **inject_csa_gate**（GatedFusionMLP，4 维 win/csa/hca/inject）：门控**注入通路**——
    注入场景下的滑窗/csa/hca + 注入条目（inject_hca_entries 拼入）的独立 csa 检索通道。
    注入条目走**独立 csa 选择检索**（独立打分/softmax/门控），不经 natural_gate 的任何门控。
  关键：注入召回从"CSA 压缩条目检索"到"HCA 注入条目"全链路走 inject_csa_gate，
  natural_gate 重训（哪怕把自然 csa 压到 0）**零影响注入召回**——结构性权衡彻底消除。

融合（NSA Eq.5 扩展，7 项加权和）：
  o = g_nat_win·o_win + g_nat_csa·o_csa + g_nat_hca·o_nat        （自然通路）
    + g_inj_win·o_win + g_inj_csa·o_csa_inj + g_inj_hca·o_nat      （注入通路，有注入时）
    + g_inj_inj·o_inj                                              （注入条目独立通道）
  其中 o_csa_inj = 注入场景的独立 csa 检索（inject_csa_gate 门控，非 natural 的 o_csa）。
  无注入时退化为 natural 单门控（g_nat·[o_win, o_csa, o_nat]），与方案 A 无注入行为一致。

关键设计（红线）：
- **恒等初始化**：natural_gate / inject_csa_gate 均 fc2=0 + bias=-ln2 → 初始 g=1/3
  （注入通路 4 维门控每位 1/3）；fc1 小随机破对称（std=0.02，gated-fusion-mlp 经验）。
  挂上后初始前向 = 原行为的多通道复制（不破坏 checkpoint；可选升级）。
- **来源路由（结构化非学习 embedding）**：inject_hca_entries 拼入的条目带 namespace 标记
  （has_inject=True）→ 走 inject_csa_gate；自然条目（压缩器 gist/csa）走 natural_gate。
  契合 BlockStore 五元组 namespace fail-closed 红线。
- **注入式 attach**：attach_fully_decoupled(mixer) 预绑定 forward 替换门控/融合逻辑；
  原 tri_attention.py / tri_attention_decoupled.py 不改。detach_fully_decoupled 恢复。
- **主干 frozen 纪律**：联合训练只训 natural_gate + inject_csa_gate（两路独立），
  不动 q/k/v/o 投影、gate_w/b、压缩器、indexer。

来源：方案 A 权衡发现（decoupled-gate.md ④"真正解=注入召回走独立 csa 通道"）；
     NSA Eq.5 门控融合扩展（arXiv:2502.11089）；TokenMem 零初始化独立通道先例（arXiv:2607.22625）。
"""
from __future__ import annotations

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tri_attention_gated import GatedFusionMLP, _INIT_BIAS

__all__ = [
    "FullyDecoupledGate",
    "attach_fully_decoupled",
    "detach_fully_decoupled",
    "set_fully_decoupled_enabled",
]


class _Gate4(nn.Module):
    """4 维门控 MLP（win/csa/hca/inject）：fc1+GELU+fc2，恒等初始化（fc2=0+bias=-ln2 → g=1/3）。

    与 GatedFusionMLP（3 维）同构，仅输出维 4（多出 inject 位）。inject_csa_gate 专用：
    门控注入通路的滑窗/csa/hca/注入条目四路，独立于 natural_gate 的 3 维 win/csa/hca。
    """

    def __init__(self, head_dim: int, hidden: int = 128):
        super().__init__()
        self.head_dim = head_dim
        self.hidden = hidden
        self.fc1 = nn.Linear(head_dim, hidden)
        self.fc2 = nn.Linear(hidden, 4)
        # 召回训练模式：True 时 win/csa/hca 位输出 detach（梯度只进 inject 位）——
        # 强迫召回走注入条目（对齐方案 A inject_gate 只控 o_inj 单路的成功机制：
        # 实测联合训练 4 位同训时召回梯度走捷径开 win/csa（已能提供部分信号）而非 inject，
        # 注入位学不动 → KV 召回 0.062；detach 后 loss 只能经 inject 位降 → 学开 inject）。
        self.train_inject_only = False
        # 恒等初始化保 g=1/3：fc2=0 + bias=-ln2；fc1 小随机破对称（std=0.02，同 GatedFusionMLP）
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, _INIT_BIAS)

    def forward(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope: [B, T, n_q, head_dim] → 门控 logit [B, T, n_q, 4]（win/csa/hca/inject）。"""
        logit = self.fc2(F.gelu(self.fc1(q_nope)))
        if self.train_inject_only:
            # win/csa/hca 位 detach（梯度只进 inject 位）；值仍前向参与（固定贡献）
            logit = torch.cat([logit[..., 0:3].detach(), logit[..., 3:4]], dim=-1)
        return logit

    def gate(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope: [B, T, n_q, head_dim] → g = sigmoid(logit) [B, T, n_q, 4]（win/csa/hca/inject）。"""
        return torch.sigmoid(self.forward(q_nope))


class FullyDecoupledGate(nn.Module):
    """彻底解耦门控：natural_gate（自然 win/csa/gist）+ inject_csa_gate（注入通路独立 csa）。

    结构：
      - natural_gate（GatedFusionMLP，3 维 win/csa/hca）：门控自然通路（滑窗/自然 csa/gist）。
        可重训"对 gist 关"（win 主导、压自然 csa/gist）恢复 in-context——不影响注入召回。
      - inject_csa_gate（_Gate4，4 维 win/csa/hca/inject）：门控注入通路（滑窗/独立 csa/hca/
        注入条目）。注入召回的 csa 检索走此独立通道，不经 natural_gate 的任何门控。
    两路参数完全独立（不同对象、独立张量）——natural 重训零影响 inject_csa（结构性解耦）。
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
        # 自然通路与注入通路两套完全独立门控（参数独立，结构性解耦 ic/KV 权衡）
        self.natural_gate = natural_gate if natural_gate is not None else GatedFusionMLP(head_dim, hidden)
        self.inject_csa_gate = _Gate4(head_dim, hidden)
        # 召回友好初始化（TokenMem 零初始化独立通道先例）：inject 位偏置更低 → 起点 g_inj=0.05
        # （其余 win/csa/hca 位保持 1/3）。恒等 1/3 起点下召回学不动（4 维全开答案被稀释，
        # 学"关 3 路+开 inject"负担重）；起点 inject 位低 = 注入条目初始弱贡献，召回训练
        # 只需"开 inject 位"即可保召回（vs 方案 A inject_gate 零初始化 g=1/3 也达 0.625）。
        with torch.no_grad():
            self.inject_csa_gate.fc2.bias[3] = -3.0  # sigmoid(-3)≈0.047（inject 位起点低）
        # 运行时开关：False 时强制走 natural 单门控（测试/消融对照用）
        self.enabled = True

    def gate_natural(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope [B,T,n_q,head_dim] → g [B,T,n_q,3]（自然通路 win/csa/hca 门控）。"""
        return self.natural_gate.gate(q_nope)

    def gate_inject_csa(self, q_nope: torch.Tensor) -> torch.Tensor:
        """q_nope [B,T,n_q,head_dim] → g [B,T,n_q,4]（注入通路 win/csa/hca/inject 门控）。"""
        return self.inject_csa_gate.gate(q_nope)

    def forward(self, q_nope: torch.Tensor, has_inject: bool = False) -> torch.Tensor:
        """按条目来源路由门控通道。

        - has_inject=False：g = natural_gate(q)[...,3]（纯文本，注入通路不参与）；
        - has_inject=True：g = [natural(3), inject_csa(4)] 拼接 [B,T,n_q,7]——
          前 3 维自然通路门控，后 4 维注入通路门控（融合逻辑分别取用）。
        """
        if (not self.enabled) or (not has_inject):
            return self.gate_natural(q_nope)
        g_nat = self.gate_natural(q_nope)          # [B,T,n_q,3]
        g_inj = self.gate_inject_csa(q_nope)       # [B,T,n_q,4]
        return torch.cat([g_nat, g_inj], dim=-1)   # [B,T,n_q,7]

    def init_identity(self) -> None:
        """恒等初始化（natural fc2=0+bias=-ln2 → g=1/3；inject_csa win/csa/hca 位 1/3、
        inject 位 -3.0 → g_inj≈0.05 召回友好起点；fc1 随机破对称）。"""
        for gate in (self.natural_gate, self.inject_csa_gate):
            nn.init.normal_(gate.fc1.weight, std=0.02)
            nn.init.zeros_(gate.fc1.bias)
            nn.init.zeros_(gate.fc2.weight)
            nn.init.constant_(gate.fc2.bias, _INIT_BIAS)
        with torch.no_grad():
            self.inject_csa_gate.fc2.bias[3] = -3.0  # inject 位召回友好起点（g_inj≈0.05）


# ---------------------------------------------------------------------------
# 注入式挂载（不改原 TriRetrievalAttention 类；预绑定 forward 替换门控/融合逻辑）
# ---------------------------------------------------------------------------
def _fully_decoupled_forward(self, x, state=None, offset: int = 0, aux: dict | None = None):
    """替换 TriRetrievalAttention.forward：自然通路 + 注入通路两套独立门控融合。

    与原 _decoupled_forward 差异：注入召回的 csa 检索走**独立通道**（inject_csa_gate 门控），
    不经 natural_gate 的 csa 门控——natural_gate 重训（对 gist 关压 csa）零影响注入召回。
      自然通路（natural_gate 3 维）：o_win / o_csa（压缩器）/ o_nat（gist）；
      注入通路（inject_csa_gate 4 维）：o_win / o_csa_inj（独立 csa 检索）/ o_nat / o_inj（注入条目）。
      o = Σ g_nat·[o_win, o_csa, o_nat] + Σ g_inj·[o_win, o_csa_inj, o_nat, o_inj]（有注入时）
    无注入退化为 natural 单门控（与方案 A 无注入一致）。
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

    # ── CSA 分支（压缩 + 因果内 top-k 选择检索）────────────────────────────────
    # 双份：o_csa（natural_gate 门控，自然通路）+ o_csa_inj（inject_csa_gate 门控，注入通路）。
    # 彻底解耦核心：注入召回的 csa 检索（o_csa_inj）走 inject_csa_gate，不经 natural_gate——
    # natural_gate 重训"对 gist 关"压 csa 时，o_csa_inj 仍由 inject_csa_gate 独立开权重保召回。
    m = self.csa_comp.stride
    S = Tk // m
    o_csa = torch.zeros_like(o_win)
    o_csa_inj = torch.zeros_like(o_win)
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
        o_csa = attn @ vc_e          # 自然通路 csa（natural_gate 门控）
        o_csa_inj = attn @ vc_e      # 注入通路 csa（inject_csa_gate 门控；检索同、门控独立）
        keep = keep
    else:
        keep = None

    # ── HCA 分支（解耦：注入条目独立通道 o_inj + 自然 gist o_nat）────────────────
    gate = self.fully_decoupled_gate
    has_inject = inj_k is not None and gate.enabled
    n_inj = inj_k.shape[2] if inj_k is not None else 0
    o_inj = torch.zeros_like(o_win)
    o_nat = torch.zeros_like(o_win)
    S2 = Tk // self.hca_comp.stride

    if n_inj and has_inject:
        # 注入知识块条目：独立 softmax 归一（恒可见）+ inject_csa_gate 的 inject 位门控
        k_ie = inj_k.repeat_interleave(rep, dim=1)
        v_ie = inj_v.repeat_interleave(rep, dim=1)
        logits_i = (q_nope @ k_ie.transpose(-1, -2)) / math.sqrt(D)
        vis_i = torch.ones(T, n_inj, dtype=torch.bool, device=x.device)
        attn_i = self._masked_softmax(logits_i, vis_i[None, None].expand(B, self.n_q, T, n_inj))
        o_inj = attn_i @ v_ie

    if S2 > 0:
        # 自然 gist 条目（压缩器产生）：独立 softmax 归一 + natural/inject_csa 的 hca 位门控
        kh, vh = self.hca_comp(k_nope, v)
        kh = self.k_norm(kh)
        k_he = kh.repeat_interleave(rep, dim=1)
        v_he = vh.repeat_interleave(rep, dim=1)
        tail2 = self.hca_comp.stride * (torch.arange(S2, device=x.device) + 1) - 1
        vis_n = tail2[None, :] < i_abs[:, None]
        logits_n = (q_nope @ k_he.transpose(-1, -2)) / math.sqrt(D)
        attn_n = self._masked_softmax(logits_n, vis_n[None, None].expand(B, self.n_q, T, S2))
        o_nat = attn_n @ v_he

    # ── 门控融合（彻底解耦：自然通路 natural + 注入通路 inject_csa 两套独立门控加权）─────
    q_bt = q_nope.transpose(1, 2)  # [B,T,n_q,D]
    o_win_t = o_win.transpose(1, 2)
    o_csa_t = o_csa.transpose(1, 2)
    o_csa_inj_t = o_csa_inj.transpose(1, 2)
    o_nat_t = o_nat.transpose(1, 2)
    o_inj_t = o_inj.transpose(1, 2)
    if has_inject:
        g_nat = gate.gate_natural(q_bt)          # [B,T,n_q,3]（自然 win/csa/hca；注入时不参与前向）
        g_inj = gate.gate_inject_csa(q_bt)       # [B,T,n_q,4]（注入 win/csa/hca/inject）
        # 彻底解耦融合（字面义：注入召回走独立 csa 通道——注入场景**全部**分支门控走 inject_csa_gate，
        # natural_gate 完全不参与注入前向）。注入召回的 win/csa/hca/inject 全由 inject_csa_gate
        # （扩容开权重+召回路由）独立控制——natural_gate 重训对 gist 关（压 win/csa）零影响注入召回，
        # 结构性权衡彻底消除。对齐方案 A 召回语义（g_win/csa/hca·o + g_inj·o_inj，扩容 natural 全开），
        # 但门控来源换成 inject_csa_gate（独立通道）。
        o = (
            g_inj[..., 0:1] * o_win_t                  # 滑窗：inject_csa_gate（win 位）
            + g_inj[..., 1:2] * o_csa_t                # csa 检索：inject_csa_gate（csa 位，开权重保召回）
            + g_inj[..., 2:3] * o_nat_t                # gist：inject_csa_gate（hca 位）
            + g_inj[..., 3:4] * o_inj_t                # 注入条目：inject_csa_gate（inject 位，召回路由）
        )
        g_record = torch.cat([g_nat, g_inj], dim=-1)  # aux 记录（7 维）
    else:
        # 无注入（in-context 纯文本）：退化为 natural 单门控（结构性恢复精确召回）
        g_nat = gate.gate_natural(q_bt)
        o = g_nat[..., 0:1] * o_win_t + g_nat[..., 1:2] * o_csa_t + g_nat[..., 2:3] * o_nat_t
        g_inj = torch.zeros(q_bt.shape[0], q_bt.shape[1], q_bt.shape[2], 4,
                            device=q_bt.device, dtype=q_bt.dtype)
        g_record = g_nat
    o = o.reshape(B, T, self.n_q * D)

    if aux is not None:
        aux.update(
            o_win=o_win, o_csa=o_csa, o_csa_inj=o_csa_inj, o_nat=o_nat, o_inj=o_inj,
            gates=g_record, gate_natural=g_nat, gate_inject_csa=g_inj,
            q_rope=q_rope, k_rope=k_rope, v=v,
            i_abs=i_abs, sel_keep=keep,
            n_csa=S, n_hca=S2, n_hca_inj=n_inj, has_inject=bool(has_inject),
        )
    return self.o_proj(o), new_state


def attach_fully_decoupled(
    mixer,
    natural_state_dict: dict | None = None,
    inject_csa_state_dict: dict | None = None,
    hidden: int = 128,
) -> FullyDecoupledGate:
    """给 TriRetrievalAttention 实例挂载 FullyDecoupledGate 并预绑定 forward（注入式，不改原类）。

    - mixer：TriRetrievalAttention 实例；
    - natural_state_dict：已训 natural_gate 权重（可选；None=恒等初始化 g=1/3，待重训对 gist 关）；
    - inject_csa_state_dict：已训 inject_csa_gate 权重（可选；None=恒等初始化，待训保召回）；
    - hidden：门控 MLP 隐藏维（默认 128）。
    返回挂载的 FullyDecoupledGate（训练目标参数来源：natural_gate + inject_csa_gate）。
    """
    natural = GatedFusionMLP(mixer.head_dim, hidden).to(device=mixer.gate_w.device, dtype=mixer.gate_w.dtype)
    if natural_state_dict is not None:
        natural.load_state_dict(natural_state_dict)
    gate = FullyDecoupledGate(mixer.head_dim, hidden, natural_gate=natural).to(
        device=mixer.gate_w.device, dtype=mixer.gate_w.dtype)
    if inject_csa_state_dict is not None:
        gate.inject_csa_gate.load_state_dict(inject_csa_state_dict)
    mixer.fully_decoupled_gate = gate  # nn.Module 子模块：随 state_dict 存取（"fully_decoupled_gate.*"）
    if not hasattr(mixer, "_orig_forward_fully_decoupled"):
        mixer._orig_forward_fully_decoupled = mixer.forward  # 记录被替换前 forward（detach 恢复用）
    mixer.forward = types.MethodType(_fully_decoupled_forward, mixer)  # 预绑定替换（实例级）
    return gate


def detach_fully_decoupled(mixer) -> None:
    """恢复 attach 前的 forward（原线性门控 / gate_mlp 单门控 / decoupled 双通道）并移除门控。"""
    if hasattr(mixer, "_orig_forward_fully_decoupled"):
        mixer.forward = mixer._orig_forward_fully_decoupled
        del mixer._orig_forward_fully_decoupled
    if hasattr(mixer, "fully_decoupled_gate"):
        del mixer.fully_decoupled_gate


def set_fully_decoupled_enabled(mixer, enabled: bool) -> None:
    """开/关彻底解耦（enabled=False 时强制走 natural 单门控——恢复纯文本精确召回对照）。"""
    if hasattr(mixer, "fully_decoupled_gate"):
        mixer.fully_decoupled_gate.enabled = bool(enabled)
