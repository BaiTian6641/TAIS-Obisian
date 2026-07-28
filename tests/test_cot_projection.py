"""CoT 投影层（CoT Projection Layer）测试：第二阶段（思维能力强化）迭代⑤ pilot 模块。

覆盖判据（对齐任务规范 §实现要求）：
  a) 解码形状：tick_state→思考段表征 [B,T,d_model]；trajectory→段序列
     （recall tick 标 `<|recall|>`）；
  b) grounded 监督损失：对齐段 loss < 随机段；可反传（decoder 有梯度）；
  c) 说-做一致性：构造"解码段与位移一致"vs"不一致"，一致者得分高；
  d) CMI 审计：输出诊断 dict 四键齐全，值域合理；
  e) recall 保留：recall_triggered tick 解码出 `<|recall|>`（对齐红线）；
  f) 忠实性判据：faithfulness_rate 在合成一致数据上高于不一致数据。
用法：python -m pytest tests/test_cot_projection.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.cot_projection import (
    CotFaithfulnessAudit,
    CotProjectionLayer,
    ThoughtSegmentDecoder,
    grounded_supervision_loss,
)
from tais_obsidian.model.reasoning_loop import RECALL_TOKEN, ReasoningTickState

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MANIFOLD_DIM = 64
D_MODEL = 384
B, T = 2, 10


def make_tick_state(
    tick_index: int,
    seed: int,
    recall: bool = False,
    coord: torch.Tensor | None = None,
    disp: torch.Tensor | None = None,
) -> ReasoningTickState:
    """构造一个 ReasoningTickState（pilot 测试用，不依赖完整 ReasoningLoop）。"""
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    if coord is None:
        coord = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
    if disp is None:
        disp = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
    return ReasoningTickState(
        tick_index=tick_index,
        current_coord=coord.to(DEVICE),
        disp=disp.to(DEVICE),
        certainty=0.2 if recall else 0.5,
        hrl_topk_idx=None,
        early_stop=False,
        recall_triggered=recall,
    )


def make_layer(seed: int = 42) -> CotProjectionLayer:
    torch.manual_seed(seed)
    return CotProjectionLayer(MANIFOLD_DIM, D_MODEL, use_mlp=True).to(DEVICE)


def fit_auditor(auditor: CotFaithfulnessAudit, layer: CotProjectionLayer, seed: int = 7) -> None:
    """用 (解码段, 对应流形坐标) 数据对拟合审计器的 Hidden→manifold 伪逆。

    说-做一致性须在流形空间度量（设计纪律），故须先 fit_back_projection。
    用真实 decoder 生成的 (段, 坐标) 对拟合，保证伪逆对齐本 decoder 的映射。
    """
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    coords = torch.randn(64, MANIFOLD_DIM, device=DEVICE, generator=g)
    with torch.no_grad():
        segs = layer.decoder(coords)
    auditor.fit_back_projection(segs, coords)


# ---------- a) 解码形状 ----------

def test_a_decode_tick_shape() -> None:
    """a) tick_state→思考段表征 [B,T,d_model]。"""
    layer = make_layer()
    ts = make_tick_state(0, seed=0)
    decoded, recall_tok = layer.decode_tick(ts)
    assert decoded.shape == (B, T, D_MODEL), (
        f"解码段形状 {tuple(decoded.shape)} ≠ {(B, T, D_MODEL)}"
    )
    assert recall_tok is None  # 本 tick 未触发 recall
    print(f"[a] decode_tick 形状 {tuple(decoded.shape)} OK")


def test_a_decode_trajectory_shape() -> None:
    """a) trajectory→段序列（长度=轨迹长度，recall tick 标 `<|recall|>`）。"""
    layer = make_layer()
    traj = [make_tick_state(i, seed=i, recall=(i == 1)) for i in range(3)]
    decoded_segs, tokens = layer.decode_trajectory(traj)
    assert len(decoded_segs) == 3, f"段序列长度 {len(decoded_segs)} ≠ 3"
    for seg in decoded_segs:
        assert seg.shape == (B, T, D_MODEL), f"段形状错误 {tuple(seg.shape)}"
    assert len(tokens) == 3
    assert tokens[1] == RECALL_TOKEN, f"recall tick 应标 <|recall|>，实得 {tokens[1]}"
    assert tokens[0] != RECALL_TOKEN and tokens[2] != RECALL_TOKEN
    print(f"[a] decode_trajectory 3 段 + tokens={tokens} OK")


# ---------- b) grounded 监督损失 ----------

def test_b_grounded_supervision_loss() -> None:
    """b) grounded 监督损失：对齐段 loss < 随机段；可反传（decoder 有梯度）。"""
    torch.manual_seed(0)
    layer = make_layer()
    layer.decoder.train()
    g = torch.Generator(device=DEVICE).manual_seed(1)
    coords = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
    target = torch.randn(B, T, D_MODEL, device=DEVICE, generator=g)  # ground truth 段表征

    # 对齐段 loss：直接以 target 自身为目标（上界应≈0，远小于随机段）
    loss_aligned = grounded_supervision_loss(target, target)
    assert loss_aligned.item() < 1e-6, f"对齐段 loss 应≈0，实得 {loss_aligned.item()}"

    # 随机段 loss：decoder 随机输出 vs target（应显著大于对齐段）
    decoded = layer.decoder(coords)
    loss_random = grounded_supervision_loss(decoded, target)
    assert loss_random.item() > loss_aligned.item(), (
        f"随机段 loss({loss_random.item():.4f}) 应 > 对齐段({loss_aligned.item():.6f})"
    )

    # 可反传：decoder 参数有梯度
    layer.decoder.zero_grad()
    loss_random.backward()
    grad_norm = layer.decoder.back_proj.proj.weight.grad
    assert grad_norm is not None and grad_norm.abs().sum().item() > 0, (
        "decoder.back_proj 应有梯度（grounded 监督可反传）"
    )
    print(
        f"[b] aligned={loss_aligned.item():.2e} < random={loss_random.item():.4f}，"
        f"decoder 梯度范数 {grad_norm.abs().sum().item():.4f} OK"
    )


# ---------- c) 说-做一致性 ----------

def test_c_speak_do_consistency() -> None:
    """c) 说-做一致性：构造"解码段与位移一致"vs"不一致"，一致者得分高。"""
    layer = make_layer()
    auditor = layer.auditor
    fit_auditor(auditor, layer)

    # 一致情形：disp 与坐标同向——构造 coord=0、disp=v，使思考状态=coord+disp=v
    # 解码段 = decoder(v)，反投影回流形应≈v（伪逆对齐本 decoder）→ 与 disp=v 余弦高。
    g = torch.Generator(device=DEVICE).manual_seed(11)
    v = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
    coord_zero = torch.zeros_like(v)
    # 思考状态 = coord + disp = v；disp = v（做的方向 = v）
    with torch.no_grad():
        seg_consistent = layer.decoder(coord_zero + v)  # 说 = decoder(思考结果 v)
    cos_consistent, _ = auditor.speak_do_consistency(seg_consistent, v)

    # 不一致情形：disp 取反（做的方向与说相反）→ 余弦应显著更低。
    cos_inconsistent, _ = auditor.speak_do_consistency(seg_consistent, -v)

    mean_c = float(cos_consistent.mean().item())
    mean_i = float(cos_inconsistent.mean().item())
    assert mean_c > mean_i, (
        f"一致者得分({mean_c:.4f}) 应高于不一致者({mean_i:.4f})"
    )
    # 一致情形余弦应明显为正（伪逆对齐本 decoder，同向位移应高余弦）
    assert mean_c > 0.0, f"一致情形余弦应为正，实得 {mean_c:.4f}"
    print(f"[c] 一致 cos={mean_c:.4f} > 不一致 cos={mean_i:.4f} OK")


# ---------- d) CMI 审计诊断 dict ----------

def test_d_audit_diag_dict() -> None:
    """d) CMI 审计：输出诊断 dict 四键齐全，值域合理。"""
    layer = make_layer()
    fit_auditor(layer.auditor, layer)
    traj = [make_tick_state(i, seed=100 + i) for i in range(3)]
    decoded_segs, _ = layer.decode_trajectory(traj)
    g = torch.Generator(device=DEVICE).manual_seed(200)
    output_repr = torch.randn(B, T, D_MODEL, device=DEVICE, generator=g)
    context_repr = torch.randn(B, T, D_MODEL, device=DEVICE, generator=g)

    diag = layer.audit(
        traj, decoded_segs, output_repr=output_repr, context_repr=context_repr
    )
    # 四键齐全
    for key in ("speak_do_consistency", "cmi_approx", "divergence_penalty", "faithfulness_rate"):
        assert key in diag, f"诊断 dict 缺键 {key}"
    # 值域合理
    assert -1.0 <= diag["speak_do_consistency"] <= 1.0, (
        f"speak_do_consistency 须 ∈ [-1,1]，实得 {diag['speak_do_consistency']}"
    )
    assert 0.0 <= diag["faithfulness_rate"] <= 1.0, (
        f"faithfulness_rate 须 ∈ [0,1]，实得 {diag['faithfulness_rate']}"
    )
    assert diag["divergence_penalty"] >= 0.0, (
        f"divergence_penalty 须 ≥ 0，实得 {diag['divergence_penalty']}"
    )
    assert isinstance(diag["cmi_approx"], float), "cmi_approx 须为 float"
    print(
        f"[d] 诊断四键齐全：speak_do={diag['speak_do_consistency']:.4f}, "
        f"cmi≈{diag['cmi_approx']:.4f}, pen={diag['divergence_penalty']:.4f}, "
        f"faith={diag['faithfulness_rate']:.4f} OK"
    )


# ---------- e) recall 保留（红线） ----------

def test_e_recall_token_preserved() -> None:
    """e) recall 保留：recall_triggered tick 解码出 `<|recall|>`（对齐红线）。"""
    layer = make_layer()
    # 触发 recall 的 tick（certainty < 阈值 → recall_triggered=True）
    ts_recall = make_tick_state(0, seed=5, recall=True)
    _, recall_tok = layer.decode_tick(ts_recall)
    assert recall_tok == RECALL_TOKEN, (
        f"recall_triggered tick 应解码出 <|recall|>，实得 {recall_tok}"
    )
    # 未触发 recall 的 tick → None
    ts_normal = make_tick_state(1, seed=6, recall=False)
    _, tok_normal = layer.decode_tick(ts_normal)
    assert tok_normal is None, f"正常 tick 不应标 recall，实得 {tok_normal}"

    # 轨迹级：recall tick 在 tokens 序列中标 <|recall|>
    traj = [ts_normal, ts_recall, make_tick_state(2, seed=8, recall=False)]
    _, tokens = layer.decode_trajectory(traj)
    assert tokens[1] == RECALL_TOKEN and tokens[0] != RECALL_TOKEN and tokens[2] != RECALL_TOKEN
    print(f"[e] recall tick 解码出 {RECALL_TOKEN}（红线）OK，tokens={tokens}")


# ---------- f) 忠实性判据 ----------

def test_f_faithfulness_rate_consistent_vs_inconsistent() -> None:
    """f) 忠实性判据：faithfulness_rate 在合成一致数据上高于不一致数据。"""
    layer = make_layer()
    fit_auditor(layer.auditor, layer)
    g = torch.Generator(device=DEVICE).manual_seed(31)

    def build_traj(consistent: bool) -> tuple[list[ReasoningTickState], list[torch.Tensor]]:
        """构造 3 tick 合成轨迹，控制"说"（解码的思考状态）与"做"（disp）是否同向。

        说 = decoder(思考状态 thought)，做 = disp。思考状态固定为 v（正向），
        说-做一致性由 disp 相对 v 的方向控制：
          consistent=True  → 说=v 且 做=+v（同向，一致，cos 高）；
          consistent=False → 说=v 但 做=−v（说做脱钩/反向，不一致，cos 低）。
        关键：思考状态始终解码 v（说 v），仅 disp（做）变号——这样"说"与"做"
        才真正脱钩；若二者同取反（说−v/做−v）反投影后仍同向，说-做仍一致。
        """
        traj: list[ReasoningTickState] = []
        for i in range(3):
            v = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)
            thought = v  # 说：思考状态（解码输入）恒为 v
            # 做：disp 相对说的方向——一致则 +v，不一致则 −v（说做反向）
            disp = v if consistent else -v
            # current_coord = thought − disp（使 current_coord+disp=thought=v，说的内容恒为 v）
            coord = thought - disp
            ts = ReasoningTickState(
                tick_index=i,
                current_coord=coord.to(DEVICE),
                disp=disp.to(DEVICE),
                certainty=0.5,
                hrl_topk_idx=None,
                early_stop=False,
                recall_triggered=False,
            )
            traj.append(ts)
        decoded_segs, _ = layer.decode_trajectory(traj)
        return traj, decoded_segs

    # 一致数据：说=v 且 做=+v（同向）→ 解码段反投影与 disp 高余弦 → faithfulness 高
    traj_c, segs_c = build_traj(consistent=True)
    diag_c = layer.audit(traj_c, segs_c)
    # 不一致数据：说=v 但 做=−v（说做反向）→ 余弦低 → faithfulness 低
    traj_i, segs_i = build_traj(consistent=False)
    diag_i = layer.audit(traj_i, segs_i)

    assert diag_c["faithfulness_rate"] > diag_i["faithfulness_rate"], (
        f"一致数据 faithfulness({diag_c['faithfulness_rate']:.4f}) 应高于"
        f"不一致数据({diag_i['faithfulness_rate']:.4f})"
    )
    print(
        f"[f] faithfulness：一致={diag_c['faithfulness_rate']:.4f} > "
        f"不一致={diag_i['faithfulness_rate']:.4f}（忠实性判据）OK"
    )
