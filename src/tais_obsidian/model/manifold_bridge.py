"""思考流形 ↔ PM-stream 桥接模块——第二阶段（思维能力强化）迭代① × 前置工程③交汇点。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §1.3（推理循环）：
    每个思考 tick：
      GDN 状态 → CSA/TriRetrieval 注意力 glimpse → HRL 提议 → KAL 门控
        → 流形上位移一步 → 写 PM-stream（感知-记忆专用道）

**设计交汇语义**：PM-stream 末位流（PMStreamMix.pm_index(n)=n-1，capture 暴露的
"pm" 槽）是思考段的**载体/寄存器**；思考流形（manifold.py，manifold_dim=64）是
思考段的**几何坐标系**。本模块把两者接通：
  - 读：从 PM-stream 末位流读出思考段表征 → 经共享 projector 投影进流形坐标；
  - 写：流形位移经反投影回 d_model → 以 steering 式有界加法写回 PM-stream 末位流。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] ITI steering（Li et al. 2306.03341）：`h ← h + α·direction` 加法干预残差
  流可有效调制行为，α 须按残差 norm 比例有界（Braun 2505.22637：α 过大致崩溃，
  小模型更敏感，±0.1–0.2×norm 为安全区）；本项目 0.1B 实证 ITI steer 有效
  （iti_head.py docstring，α≈0.2×norm 翻转探针读数）。
- [推测/独创] 把流形位移"写回 PM-stream 末位流"作为思考 tick 的寄存器更新——
  文献无先例（TAIS 独创外推，须经 pilot 验证"写后坐标朝 target 移动且不崩人效"）。

写纪律红线（对齐项目红线总表"读写不对称"）：
- **运行时写 PM-stream 是 steering 式有界加法（W1–W2 零梯度快写），绝不触碰权重**——
  同 ITI steer（iti_head.py）与向量块 ICV-steering 的写纪律；W3+（梯度更新）仅离线
  睡眠期执行。
- 幅度 clamp：写回增量范数 ≤ alpha × pm_state norm（对齐 ITI max_alpha_frac 纪律，
  α 有界防 steering 崩溃）。
- **梯度边界**：write 的 displacement 计算路径经 `detach()`（steering 是推理期干预，
  非训练梯度路径——对齐 ITIHead 非可学习/方向 detach 的纪律）。桥内唯一可训练参数
  to_hidden 的梯度须由离线睡眠期的显式训练目标提供（如重建/对比损失），不经 tick
  的 steering 路径回流。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .manifold import ThoughtManifoldProjector


class ManifoldToHidden(nn.Module):
    """流形坐标/位移 → d_model 隐藏空间的反投影（与 projector 配对）。

    **设计决策：独立 Linear（非 projector 伪逆）**。
    - 独立 Linear(manifold_dim → d_model)：读写两侧解耦，可独立训练；写方向不必受
      读投影的几何约束（读投影含无仿射 LayerNorm，伪逆会把归一化几何强行带回隐藏
      空间，且 LayerNorm 不可逆，伪逆本就只能是近似）。
    - 初始化：默认 PyTorch 初始化；离线睡眠期可用显式目标训练（如 to_hidden(project(x))
      ≈ x 的重建损失），运行时只做有界 steering 快写，不更新本权重。
    """

    def __init__(self, manifold_dim: int, d_model: int):
        super().__init__()
        self.manifold_dim = manifold_dim
        self.d_model = d_model
        self.proj = nn.Linear(manifold_dim, d_model)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """coords [..., manifold_dim] → 隐藏空间表征 [..., d_model]。"""
        return self.proj(coords)


class ThoughtSegmentExtractor:
    """从 PM-stream 读出思考段表征并投影进流形（读侧，无自有参数）。

    复用传入的共享 ThoughtManifoldProjector 实例（不重复造投影器——迭代①已训的
    同一实例，保证"知识块 / PM 思考段 / W0 轨迹"三类对象在同一坐标空间）。
    """

    def extract(
        self,
        pm_stream_state: torch.Tensor,
        projector: ThoughtManifoldProjector,
    ) -> torch.Tensor:
        """逐 token 读出：pm_stream_state [B,T,d] → 流形坐标 [B,T,manifold_dim]。"""
        if pm_stream_state.dim() != 3:
            raise ValueError(
                f"pm_stream_state 须为 [B,T,d]，实得 shape={tuple(pm_stream_state.shape)}"
            )
        return projector.project(pm_stream_state)

    def extract_segments(
        self,
        pm_stream_state: torch.Tensor,
        segment_boundaries,
        projector: ThoughtManifoldProjector,
    ) -> torch.Tensor:
        """按思考段聚合读出：段内 token 均值池化 → [B, n_segments, manifold_dim]。

        段 = 思考 tick 的对应物（对齐设计"思考段"概念）：若干连续 token 的区间。
        参数：
            pm_stream_state: [B, T, d]（PM-stream 末位流）。
            segment_boundaries: 段边界索引（1D 升序整数序列/张量），为各段的**起始
                索引**，隐含终点 = T。例：T=10、boundaries=[0,4,7] ⇒ 3 段
                [0,4) [4,7) [7,10)。首元素须为 0，末段自动延伸至 T。
            projector: 共享 ThoughtManifoldProjector 实例。
        返回：[B, len(boundaries), manifold_dim]——先投影到流形再段内均值池化
        （均值池化在流形坐标上做，几何位移语义直接在段坐标间成立）。
        """
        if pm_stream_state.dim() != 3:
            raise ValueError(
                f"pm_stream_state 须为 [B,T,d]，实得 shape={tuple(pm_stream_state.shape)}"
            )
        B, T, _ = pm_stream_state.shape
        b = torch.as_tensor(segment_boundaries, dtype=torch.long).flatten().tolist()
        if len(b) == 0 or b[0] != 0:
            raise ValueError(f"segment_boundaries 首元素须为 0，实得 {b}")
        if any(b[i] >= b[i + 1] for i in range(len(b) - 1)) or b[-1] >= T:
            raise ValueError(f"segment_boundaries 须严格升序且 < T={T}，实得 {b}")
        coords = projector.project(pm_stream_state)  # [B,T,manifold_dim]
        ends = b[1:] + [T]
        segs = [coords[:, s:e, :].mean(dim=1) for s, e in zip(b, ends)]  # 各 [B,manifold_dim]
        return torch.stack(segs, dim=1)  # [B, n_segments, manifold_dim]


class ThoughtDisplacementWriter:
    """把流形位移写回 PM-stream（写侧，steering 式有界加法，W1–W2 零梯度快写）。

    红线（读写不对称）：运行时只读 + 只写 W0 日志 + W1–W2 零梯度快写；本写回是
    **steering 式加法（非权重更新）**且幅度有界——对齐 ITI steer（iti_head.py，
    α ≤ max_alpha_frac×norm）与向量块 ICV-steering 的写纪律。
    """

    def __init__(self, max_alpha_frac: float = 0.2):
        """max_alpha_frac：alpha 上限（相对 pm_state norm 的分数，对齐 ITI 安全区 0.2）。"""
        if not 0.0 < max_alpha_frac <= 1.0:
            raise ValueError("max_alpha_frac 须在 (0,1]")
        self.max_alpha_frac = max_alpha_frac

    def write(
        self,
        pm_stream_state: torch.Tensor,
        displacement_vec: torch.Tensor,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """有界写回：pm ← pm + α_eff · displacement，增量范数 ≤ alpha×norm（clamp）。

        参数：
            pm_stream_state: [B,T,d]（PM-stream 末位流）。
            displacement_vec: [B,T,d]（流形位移经反投影回 d_model 得到的方向×步长）。
            alpha: 期望写入强度（相对 pm_state norm 的分数），clamp 到 [0, max_alpha_frac]。
        返回：写后的 pm_stream_state（新张量，不改原输入）。
        幅度 clamp：增量逐 token 归一到范数 = α_eff × pm_state 平均 token 范数——
        与 ITI steer 的"α = alpha_frac × 残差 norm"同纪律，防 steering 过强崩溃。
        """
        if pm_stream_state.shape != displacement_vec.shape:
            raise ValueError(
                f"形状不一致：pm {tuple(pm_stream_state.shape)} vs disp {tuple(displacement_vec.shape)}"
            )
        alpha_eff = max(0.0, min(float(alpha), self.max_alpha_frac))  # α 有界钳制
        if alpha_eff == 0.0:
            return pm_stream_state.clone()
        pm = pm_stream_state.float()
        disp = displacement_vec.float()
        # 梯度边界（W1–W2 零梯度快写红线）：steering 是推理期干预，增量整体 detach——
        # scale 取自 pm 的范数但不建梯度路径（写后 pm_w 对 pm 的梯度为恒等，增量不回流）。
        scale = pm.detach().norm(dim=-1).mean()  # pm_state 平均 token 范数（对齐 ITI 标度）
        target_norm = alpha_eff * scale
        # 逐 token 归一化 displacement 到 target_norm（幅度 clamp 核心）；displacement
        # 本应由调用方 detach（tick 内已做），此处再 detach 兜底，双保险梯度边界。
        disp = disp.detach()
        disp_n = disp.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        increment = disp / disp_n * target_norm
        return (pm + increment).to(pm_stream_state.dtype)


class ThoughtManifoldBridge(nn.Module):
    """思考流形 ↔ PM-stream 桥接封装：读（extract）+ 写（write）+ 单 tick 闭环。

    持有：
      projector: 共享 ThoughtManifoldProjector 实例（复用迭代①训练好的；可传入，
                 缺省则新建）；本桥**不拥有**其训练权（读侧共享坐标映射）。
      to_hidden: ManifoldToHidden 反投影（桥内唯一可训练参数，离线睡眠期训练）。
      extractor/writer: 读/写两侧（无自有可训练参数）。

    写纪律：tick 内位移计算路径 detach（steering 是推理期干预，非梯度路径）；
    to_hidden 的梯度由离线显式目标提供，不经 tick 的 steering 路径回流。
    """

    def __init__(
        self,
        d_model: int,
        manifold_dim: int = 64,
        projector: ThoughtManifoldProjector | None = None,
        max_alpha_frac: float = 0.2,
    ):
        super().__init__()
        if projector is not None:
            if projector.d_model != d_model or projector.manifold_dim != manifold_dim:
                raise ValueError(
                    f"传入 projector 维度 ({projector.d_model},{projector.manifold_dim}) "
                    f"与桥 ({d_model},{manifold_dim}) 不一致"
                )
            self.projector = projector  # 共享实例（复用迭代①训练好的投影器）
        else:
            self.projector = ThoughtManifoldProjector(d_model, manifold_dim)
        self.to_hidden = ManifoldToHidden(manifold_dim, d_model)
        self.extractor = ThoughtSegmentExtractor()
        self.writer = ThoughtDisplacementWriter(max_alpha_frac=max_alpha_frac)
        self.d_model = d_model
        self.manifold_dim = manifold_dim

    def extract(self, pm_stream_state: torch.Tensor) -> torch.Tensor:
        """读：PM-stream 末位流 → 流形坐标 [B,T,manifold_dim]。"""
        return self.extractor.extract(pm_stream_state, self.projector)

    def extract_segments(self, pm_stream_state: torch.Tensor, segment_boundaries) -> torch.Tensor:
        """读（段聚合）：→ [B, n_segments, manifold_dim]。"""
        return self.extractor.extract_segments(pm_stream_state, segment_boundaries, self.projector)

    def write(
        self,
        pm_stream_state: torch.Tensor,
        displacement_vec: torch.Tensor,
        alpha: float = 0.1,
    ) -> torch.Tensor:
        """写：流形位移（已回 d_model）有界 steering 加法回 PM-stream。"""
        return self.writer.write(pm_stream_state, displacement_vec, alpha=alpha)

    def tick(
        self,
        pm_stream_state: torch.Tensor,
        target_coord: torch.Tensor,
        alpha: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """一个思考 tick 的桥接闭环（对齐 §1.3"流形上位移一步 → 写 PM-stream"）。

        流程：读出当前流形坐标 → 计算到 target_coord 的位移 → 反投影回 d_model
        → 有界写回 PM-stream。
        参数：
            pm_stream_state: [B,T,d]（PM-stream 末位流）。
            target_coord: [B,T,manifold_dim] 或 [B,manifold_dim]（广播到 T）目标流形坐标。
            alpha: 写入强度（clamp 到 writer 上限）。
        返回：(写后 pm_state, 当前坐标 current_coord [B,T,manifold_dim],
               流形位移 disp_manifold [B,T,manifold_dim])。
        梯度边界：位移经 to_hidden 时 detach——steering 是推理期干预（W1–W2 零梯度
        快写），to_hidden 的训练走离线显式目标，不经本 steering 路径回流。
        """
        current_coord = self.extract(pm_stream_state)  # [B,T,manifold_dim]
        if target_coord.dim() == 2:  # [B,manifold_dim] → 广播到 T
            target_coord = target_coord.unsqueeze(1).expand_as(current_coord)
        if target_coord.shape != current_coord.shape:
            raise ValueError(
                f"target_coord 须为 [B,T,{self.manifold_dim}] 或 [B,{self.manifold_dim}]，"
                f"实得 {tuple(target_coord.shape)}"
            )
        disp_manifold = target_coord - current_coord  # 流形位移（朝 target 方向）
        # 反投影回 d_model（detach：steering 干预非梯度路径，对齐 ITI 写纪律）
        disp_hidden = self.to_hidden(disp_manifold.detach())
        pm_written = self.write(pm_stream_state, disp_hidden, alpha=alpha)
        return pm_written, current_coord, disp_manifold
