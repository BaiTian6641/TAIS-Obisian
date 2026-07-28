"""CoT 投影层（CoT Projection Layer）——第二阶段（思维能力强化）迭代⑤ pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §4.2 + §6 迭代⑤。

核心理念（§4.2）：**CoT 保持为投影层而非计算层**——
    计算在思考流形上进行（潜在思考），但**每个 tick 强制解码出显式思考段
    （CoT 文本）作为 grounded 监督**。文本是思考的压缩投影与审计接口，
    **不是思考本身**。

本模块把迭代④ ReasoningLoop 产生的 tick 轨迹（ReasoningTickState）里的流形
思考状态，投影成显式思考段表征（CoT 投影），并提供忠实性审计。pilot 阶段
产出**思考段表征向量**（d_model 维，供后续接语言解码成文本）；文本解码留接口。

必须接住的潜在推理三条批评（§4.1）与对应解法：
1. **信心膨胀（Coda-Forno）**：连续思考的额外 latent token 放大信心但不增加
   算法结构 → 解法：**CMI 审计 + 说-做分歧惩罚**（忠实性纪律，CotFaithfulnessAudit）。
2. **P2 grounding/防漂移**：潜在方法缺 CoT 式逐步监督，训练易漂移 → 解法：
   **每 tick 强制解码显式思考段作 grounded 监督**（grounded_supervision_loss）。
3. **探索抑制（Zou 2026）**：高确定性抑制探索 → 解法：KAL 早停与 ε 探索联合
   调参（本模块**仅记录探索标记**，不深入——early_stop 由迭代④ ReasoningLoop 负责）。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] ITI steering 反投影思路（manifold_bridge.ManifoldToHidden，独立 Linear
  manifold_dim→d_model，读写解耦）；共形等距几何（manifold.py，相邻段位移∝语义
  步长）——说-做一致性在流形空间度量有了几何依据。
- [推测/独创] 把"解码思考段反投影回流形空间与真实位移求余弦相似度"作为
  说-做一致率、把"CMI 近似 = 段-输出相关 − 上下文-输出相关"作为信心膨胀审计——
  文献无直接先例（TAIS 独创外推，须经 pilot 验证 CoT 忠实性判据）。

红线与纪律：
- **CoT 是投影层非计算层**：本模块产出的思考段表征是思考的**压缩投影与审计
  接口**，pilot 运行时**只读投影**——不回流改变思考动力学（不回写 PM-stream、
  不修改 ReasoningLoop/主干任何前向）。解码器的训练走独立的 grounded 监督
  （离线），与思考动力学解耦。
- **`<|recall|>` 显形化红线不可破坏**：空白 tick（recall_triggered）解码出的
  思考段显式标 `<|recall|>`（复用 reasoning_loop.RECALL_TOKEN），对齐
  "`<|recall|>` 必须显式出现在 CoT 中"审计接口。
- **CMI 是 pilot 近似**：本模块的 cmi_approx **非精确互信息**，仅是
  "段-输出相关 − 上下文-输出相关"的趋势观测量，用于信心膨胀的相对审计，
  不能当作真实互信息的绝对值。
- **梯度边界**：decoder 独立训练（grounded 监督可反传）；audit 全程 no_grad
  只读（监测/执行分置，审计信号不建梯度路径）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .manifold_bridge import ManifoldToHidden
from .reasoning_loop import RECALL_TOKEN, ReasoningTickState, trajectory_to_recall_tokens


class ThoughtSegmentDecoder(nn.Module):
    """思考段解码器：把流形思考状态投影成显式思考段表征（CoT 投影）。

    输入：流形坐标 [B,T,manifold_dim]（ReasoningTickState 的 current_coord 或
    current_coord+disp——思考状态）。输出：思考段表征 [B,T,d_model]（供后续接
    语言解码成文本；pilot 阶段产出表征向量，文本解码留接口）。

    结构：`Linear(manifold_dim→d_model)` 反投影 + 可选小 MLP。

    **与 ManifoldToHidden 的关系（设计决策）**：本解码器**复用/对齐**
    manifold_bridge.ManifoldToHidden 的反投影思路——独立 `Linear(manifold_dim→d_model)`
    （读写解耦，读投影含不可逆 LayerNorm，伪逆只能是近似，故用独立可学习 Linear）。
    区别：ManifoldToHidden 是**写侧**（把流形位移反投影回 d_model 作 steering 写
    PM-stream，计算层）；本解码器是**读侧投影**（把流形思考状态解码成显式思考段
    表征作审计/监督，投影层）。二者同构（manifold_dim→d_model 反投影）但语义不同
    ——一个改变思考（写），一个观测思考（读投影）。

    **梯度边界（红线）**：pilot 独立训练（grounded_supervision_loss 可反传进本
    解码器）；**运行时只读投影**——本解码器不回写 PM-stream、不改变思考动力学，
    解码出的表征仅作审计/监督信号。
    """

    def __init__(self, manifold_dim: int, d_model: int, use_mlp: bool = True):
        """use_mlp：True 时反投影后接一个轻量 MLP（Linear→GELU→Linear，增强表达）。"""
        super().__init__()
        self.manifold_dim = manifold_dim
        self.d_model = d_model
        self.use_mlp = use_mlp
        # 反投影主体：与 ManifoldToHidden 同构的独立 Linear（manifold_dim→d_model）
        self.back_proj = ManifoldToHidden(manifold_dim, d_model)
        # 可选小 MLP：增强思考段表征的表达力（pilot 默认开，可用 use_mlp=False 消融）
        self.mlp = (
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            if use_mlp
            else nn.Identity()
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [..., manifold_dim] → 思考段表征 [..., d_model]。"""
        return self.mlp(self.back_proj(coords))


def grounded_supervision_loss(
    decoded_seg: torch.Tensor,
    target_seg: torch.Tensor,
) -> torch.Tensor:
    """grounded 监督损失（接住 P2 批评的关键：显式思考段须锚定真实推理，防漂移）。

    解码思考段 decoded_seg 与目标思考段 target_seg 的对齐损失（MSE）。
    target_seg = ground truth CoT 段表征 / 真实推理后续 hidden（监督锚点）。

    参数：
        decoded_seg: [..., d_model] 解码出的思考段表征（ThoughtSegmentDecoder 输出）。
        target_seg: [..., d_model] 目标思考段表征（与 decoded_seg 同形）。
    返回：标量 MSE 损失（可反传，梯度进 ThoughtSegmentDecoder）。

    设计决策（MSE 而非对比损失）：pilot 阶段 target_seg 是**逐位对齐**的监督锚点
    （同一 tick 的真实推理 hidden），MSE 直接拉近表征；对比损失（InfoNCE 谱系）适合
    "多候选中挑正例"的检索式监督，留作后续扩展。MSE 与 manifold.py 的共形等距
    损失同族（都是回归式对齐）。
    """
    if decoded_seg.shape != target_seg.shape:
        raise ValueError(
            f"形状不一致：decoded {tuple(decoded_seg.shape)} vs target {tuple(target_seg.shape)}"
        )
    return F.mse_loss(decoded_seg.float(), target_seg.float())


class CotFaithfulnessAudit:
    """CoT 忠实性审计（CMI + 说-做分歧）——接住 Coda-Forno 信心膨胀批评。

    度量"解码出的思考段（说）"与"流形实际位移/检索行为（做）"的一致性，并
    用 CMI 近似审计"段是否带来超出上下文的额外信息"（信心膨胀探针）。

    无自有可训练参数（纯诊断/审计，全程 no_grad——监测/执行分置红线）。

    持有：一个 Hidden→manifold 的**线性最小二乘伪逆投影**（audit_back），用于把
    d_model 思考段表征反投影回流形空间做说-做一致性度量。**设计决策**：说-做
    一致性须在流形空间度量（流形是思考的几何坐标系，disp 位移在流形空间），但
    解码段在 d_model 空间——故用一个从训练数据拟合的**固定线性伪逆**把解码段
    映回流形，再与真实 disp 求余弦。该伪逆**不是可学习部件**（最小二乘闭式解，
    fit 后固定），避免审计器自身引入可学习混淆。若未 fit，退化为用解码段与
    disp 各自投影到共有子空间的近似（见 speak_do_consistency 注释）。
    """

    def __init__(self, manifold_dim: int, d_model: int):
        self.manifold_dim = manifold_dim
        self.d_model = d_model
        # Hidden(d_model)→manifold 的最小二乘伪逆权重 [d_model, manifold_dim]，
        # 由 fit_back_projection 用 (解码段, 流形坐标) 数据对闭式解拟合；None=未拟合。
        self.audit_back: torch.Tensor | None = None

    # ------------------------------------------------------------------
    def fit_back_projection(
        self,
        decoded_segs: torch.Tensor,
        coords: torch.Tensor,
        ridge: float = 1e-4,
    ) -> None:
        """用 (解码段, 对应流形坐标) 数据对拟合 Hidden→manifold 线性伪逆（闭式解）。

        最小二乘：min_W ||decoded @ W − coords||² + ridge||W||²，
        解 W = (XᵀX + ridge·I)⁻¹ XᵀY（岭回归闭式解，X=decoded [N,d_model]，
        Y=coords [N,manifold_dim]）。拟合后固定（不参与训练，非可学习部件）。
        """
        X = decoded_segs.float().reshape(-1, self.d_model)
        Y = coords.float().reshape(-1, self.manifold_dim)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("decoded_segs 与 coords 样本数须一致")
        XtX = X.T @ X  # [d_model, d_model]
        reg = ridge * torch.eye(self.d_model, device=X.device, dtype=X.dtype)
        W = torch.linalg.solve(XtX + reg, X.T @ Y)  # [d_model, manifold_dim]
        self.audit_back = W.detach()

    # ------------------------------------------------------------------
    def _project_decoded_to_manifold(self, decoded_seg: torch.Tensor) -> torch.Tensor:
        """把 d_model 解码段表征反投影回流形空间 [..., manifold_dim]。

        已 fit：用最小二乘伪逆 audit_back。未 fit（pilot 兜底）：把解码段与 disp
        都放在 d_model 空间度量会失真（disp 在流形空间）——故未 fit 时**抛错提示须
        先 fit**，保证说-做一致性在正确的几何空间度量（设计纪律：流形是思考坐标系）。
        """
        if self.audit_back is None:
            raise RuntimeError(
                "CotFaithfulnessAudit 须先 fit_back_projection 拟合 Hidden→manifold "
                "伪逆（说-做一致性在流形空间度量，需把解码段映回流形）"
            )
        return decoded_seg.float() @ self.audit_back.to(decoded_seg.device)

    # ------------------------------------------------------------------
    def speak_do_consistency(
        self,
        decoded_seg: torch.Tensor,
        disp: torch.Tensor,
        threshold: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """说-做一致率：解码段（说）与流形实际位移（做）的一致性。

        简化实现：解码段反投影回流形空间，与真实位移 disp 求**余弦相似度** →
        一致性得分 ∈ [-1,1]；> threshold 判"说做一致"。

        参数：
            decoded_seg: [..., d_model] 解码出的思考段表征（说）。
            disp: [..., manifold_dim] 流形实际位移（做，ReasoningTickState.disp）。
            threshold: 一致判定阈值（余弦 > 此值 = 一致）。
        返回：(cos_sim [...] 逐位余弦相似度 ∈ [-1,1],
               consistent [...] bool 掩码，cos_sim > threshold)。

        几何依据 [推测/独创]：流形是思考的坐标系（manifold.py 共形等距），"说"的内容
        若忠实于"做"，则解码段映回流形后应指向真实位移方向（余弦高）。文献无先例。
        """
        seg_m = self._project_decoded_to_manifold(decoded_seg)  # [..., manifold_dim]
        d = disp.float()
        if seg_m.shape != d.shape:
            raise ValueError(
                f"反投影后形状须与 disp 一致：{tuple(seg_m.shape)} vs {tuple(d.shape)}"
            )
        cos = F.cosine_similarity(seg_m, d, dim=-1)  # [...] ∈ [-1,1]
        consistent = cos > threshold
        return cos, consistent

    # ------------------------------------------------------------------
    def cmi_approx(
        self,
        decoded_seg: torch.Tensor,
        output_repr: torch.Tensor,
        context_repr: torch.Tensor,
    ) -> float:
        """CMI（条件互信息）审计 [pilot 近似，非精确互信息，仅趋势观测]。

        估计"给定上下文后，解码段对最终输出的额外信息量"——
        **近似 = 段-输出相关 − 上下文-输出相关**：
            cmi_approx ≈ corr(decoded_seg, output) − corr(context, output)
        正值 = 解码段带来超出上下文的额外信息（健康）；≈0 或负 = 段未增信息
        （信心膨胀信号——Coda-Forno：latent token 放大信心但不增加算法结构）。

        相关用**逐维均值池化后的标量 Pearson 相关**（把 [..., d_model] 各展平成
        标量序列求相关）。**注释标注近似**：这是相关的差，不是真正的条件互信息
        I(seg; output | context)，仅作趋势观测，不能当绝对值。

        参数：
            decoded_seg: [..., d_model] 解码思考段表征。
            output_repr: [..., d_model] 最终输出表征（推理结果 hidden）。
            context_repr: [..., d_model] 上下文表征（不含解码段的输入侧）。
        返回：cmi_approx 标量（float，可正可负，趋势观测量）。
        """

        def _flat_corr(a: torch.Tensor, b: torch.Tensor) -> float:
            """两表征展平成标量序列后的 Pearson 相关（pilot 级近似）。"""
            x = a.float().reshape(-1)
            y = b.float().reshape(-1)
            n = min(x.numel(), y.numel())
            if n < 2:
                return 0.0
            x, y = x[:n], y[:n]
            xc, yc = x - x.mean(), y - y.mean()
            denom = xc.norm() * yc.norm()
            if denom < 1e-8:
                return 0.0
            return float((xc @ yc / denom).item())

        seg_out = _flat_corr(decoded_seg, output_repr)
        ctx_out = _flat_corr(context_repr, output_repr)
        return seg_out - ctx_out

    # ------------------------------------------------------------------
    def divergence_penalty(
        self,
        consistency_cos: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """说-做分歧惩罚：一致率低于阈值时输出惩罚信号（供训练）。

        简化实现：对每位 cos，惩罚 = relu(threshold − cos)——cos 越低于阈值惩罚
        越大（不一致 → 惩罚），cos ≥ 阈值惩罚为 0。返回逐位惩罚张量（调用侧取
        均值作训练惩罚项，权重由训练配方定）。

        参数：
            consistency_cos: [...] speak_do_consistency 返回的逐位余弦相似度。
            threshold: 一致判定阈值（与 speak_do_consistency 一致）。
        返回：[...] 逐位惩罚 ≥ 0（可反传进解码器的训练惩罚路径——注意：本审计
        默认 no_grad，此惩罚若要反传须由调用侧在梯度上下文中重算 cos）。
        """
        return F.relu(threshold - consistency_cos.float())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def audit(
        self,
        trajectory: list[ReasoningTickState],
        decoded_segs: list[torch.Tensor],
        output_repr: torch.Tensor | None = None,
        context_repr: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> dict:
        """整段轨迹的忠实性审计，返回诊断 dict（四键齐全）。

        对每个 tick：用 tick.disp（真实位移，做）与对应 decoded_seg（解码段，说）
        求说-做一致性；聚合得 faithfulness_rate（说做一致的 tick 占比）；CMI 近似
        （output/context 给定时）跨全轨迹池化。

        参数：
            trajectory: list[ReasoningTickState]（ReasoningLoop.run 返回的轨迹）。
            decoded_segs: list[Tensor [B,T,d_model]]（每 tick 的解码段表征，与
                trajectory 等长——见 CotProjectionLayer.decode_trajectory）。
            output_repr/context_repr: 可选 [...,d_model]，CMI 近似用；缺省时
                cmi_approx 记 0.0（接口位，趋势观测须显式提供两端表征）。
            threshold: 说-做一致判定阈值。
        返回诊断 dict：
            speak_do_consistency: 全轨迹逐位余弦均值 ∈ [-1,1]；
            cmi_approx: CMI 近似标量（pilot 近似，非精确互信息）；
            divergence_penalty: 全轨迹逐位分歧惩罚均值 ≥ 0；
            faithfulness_rate: 说做一致的位占比 ∈ [0,1]（迭代⑤验证判据）。
        """
        if len(trajectory) != len(decoded_segs):
            raise ValueError(
                f"trajectory({len(trajectory)}) 与 decoded_segs({len(decoded_segs)}) 须等长"
            )
        all_cos: list[torch.Tensor] = []
        all_pen: list[torch.Tensor] = []
        for ts, seg in zip(trajectory, decoded_segs):
            cos, _ = self.speak_do_consistency(seg, ts.disp, threshold=threshold)
            pen = self.divergence_penalty(cos, threshold=threshold)
            all_cos.append(cos)
            all_pen.append(pen)
        cos_cat = torch.cat([c.reshape(-1) for c in all_cos])
        pen_cat = torch.cat([p.reshape(-1) for p in all_pen])
        speak_do = float(cos_cat.mean().item()) if cos_cat.numel() > 0 else 0.0
        faith_rate = (
            float((cos_cat > threshold).float().mean().item()) if cos_cat.numel() > 0 else 0.0
        )
        div_pen = float(pen_cat.mean().item()) if pen_cat.numel() > 0 else 0.0
        # CMI 近似：全轨迹段拼接后与 output/context 求相关差（缺省 0.0 接口位）
        if output_repr is not None and context_repr is not None:
            seg_all = torch.cat([s.reshape(-1, s.shape[-1]) for s in decoded_segs], dim=0)
            cmi = self.cmi_approx(seg_all, output_repr, context_repr)
        else:
            cmi = 0.0
        return {
            "speak_do_consistency": speak_do,
            "cmi_approx": cmi,
            "divergence_penalty": div_pen,
            "faithfulness_rate": faith_rate,
        }


class CotProjectionLayer(nn.Module):
    """CoT 投影层封装：持有 decoder + 单 tick/轨迹解码 + 忠实性审计。

    **CoT 是投影层非计算层（红线）**：本层把流形思考状态投影成显式思考段表征
    （审计/监督接口），**不回流改变思考动力学**——不修改 ReasoningLoop、不回写
    PM-stream、运行时只读投影。decoder 的训练走独立 grounded 监督（离线）。

    持有：
      decoder: ThoughtSegmentDecoder（流形→思考段表征，CoT 投影）；
      audit_helper: CotFaithfulnessAudit（忠实性审计，无可训练参数）。
    """

    def __init__(
        self,
        manifold_dim: int,
        d_model: int,
        use_mlp: bool = True,
        consistency_threshold: float = 0.5,
    ):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.d_model = d_model
        self.consistency_threshold = consistency_threshold
        self.decoder = ThoughtSegmentDecoder(manifold_dim, d_model, use_mlp=use_mlp)
        self.auditor = CotFaithfulnessAudit(manifold_dim, d_model)

    # ------------------------------------------------------------------
    def _tick_coord(self, tick_state: ReasoningTickState) -> torch.Tensor:
        """取 tick 的流形思考状态（current_coord + disp = 位移后坐标，思考结果）。"""
        return (tick_state.current_coord + tick_state.disp).float()

    # ------------------------------------------------------------------
    def decode_tick(
        self, tick_state: ReasoningTickState
    ) -> tuple[torch.Tensor, str | None]:
        """单 tick 解码：流形思考状态 → 思考段表征 + recall 标记。

        参数：tick_state: ReasoningTickState（ReasoningLoop.reasoning_tick 返回）。
        返回：(decoded_seg [B,T,d_model] 思考段表征,
               recall_token: RECALL_TOKEN 若本 tick recall_triggered 否则 None)。

        红线：`<|recall|>` 显形化——空白 tick 解码出 `<|recall|>` 标记（审计接口）。
        """
        coord = self._tick_coord(tick_state)  # [B,T,manifold_dim]
        decoded = self.decoder(coord)  # [B,T,d_model]
        recall_tok = RECALL_TOKEN if tick_state.recall_triggered else None
        return decoded, recall_tok

    # ------------------------------------------------------------------
    def decode_trajectory(
        self, trajectory: list[ReasoningTickState]
    ) -> tuple[list[torch.Tensor], list[str]]:
        """整段轨迹解码成思考段序列（复用 reasoning_loop 的 RECALL_TOKEN 标记）。

        返回：(decoded_segs list[[B,T,d_model]]（每 tick 一段，与 trajectory 等长）,
               tokens list[str]（recall tick = `<|recall|>`，其余 = `<|tick_i|>`，
               复用 trajectory_to_recall_tokens——对齐"`<|recall|>` 显式出现在 CoT"红线）)。
        """
        decoded_segs: list[torch.Tensor] = []
        for ts in trajectory:
            seg, _ = self.decode_tick(ts)
            decoded_segs.append(seg)
        tokens = trajectory_to_recall_tokens(trajectory)  # 复用迭代④，recall 显形化
        return decoded_segs, tokens

    # ------------------------------------------------------------------
    def audit(
        self,
        trajectory: list[ReasoningTickState],
        decoded_segs: list[torch.Tensor],
        output_repr: torch.Tensor | None = None,
        context_repr: torch.Tensor | None = None,
    ) -> dict:
        """忠实性审计（委托 CotFaithfulnessAudit.audit，返回四键诊断 dict）。"""
        return self.auditor.audit(
            trajectory,
            decoded_segs,
            output_repr=output_repr,
            context_repr=context_repr,
            threshold=self.consistency_threshold,
        )


__all__ = [
    "CotFaithfulnessAudit",
    "CotProjectionLayer",
    "ThoughtSegmentDecoder",
    "grounded_supervision_loss",
]
