"""思考轨迹可视化前端（Thought Visualizer）——第二阶段（思维能力强化）迭代⑦ pilot 模块。

设计依据：docs/TAIS_Obsidian_思维能力强化_架构设计文档.md §6 迭代⑦ + §1.1（维度修正）。

**迭代⑦**：思考轨迹 3D 投影实时渲染（CTM demo 式），归因监测头 + provenance 审计
可视化。**验证判据：坏路径可视监测**。

**§1.1 维度修正（关键红线）**：3D 投影**仅作为给人类看的可解释性视图**（归因监测/
审计前端）——表征本身保持高维（manifold_dim=64），3D 是压缩视图，**不参与任何
训练/推理计算**。本模块纯只读（no_grad，监测/执行分置），不回流改变思考动力学。

**用途**：像 CTM demo 一样实时渲染模型的思考路径（3D 投影）。它不是锦上添花——
是归因监测与 provenance 审计的天然可视化前端，也是对"坏路径"（偏离目标/信心
膨胀/漂移）最直观的监测手段。

**渲染边界**：本模块**不依赖 matplotlib/GUI**——只负责"数据→可视化就绪结构"，
导出 JSON + ASCII 终端渲染（pilot 级），外部渲染工具/前端消费 JSON 做 3D 实时渲染。

证据分级（写作纪律：区分已确立与独创外推）：
- [已确立] CTM demo（arXiv:2505.05522）实时渲染思考路径；KAL P(IK) 探针
  （SAPLMA/量化态探针 0.904–1.000 AUROC，kal.py）。
- [推测/独创] 把"思考轨迹 3D 投影 + 坏路径四类检测（信心膨胀/漂移/早停失败/
  recall 风暴）"作为归因监测/审计可视化前端——文献无先例（TAIS 独创外推，
  须经 pilot 验证"坏路径可视监测"判据）。

红线与纪律：
- **3D 仅人类视图**：可视化数据不参与任何训练/推理计算（注释强调）；本模块纯只读
  （no_grad，监测/执行分置），不回流改变思考动力学。
- **坏路径阈值定为模块常量**（可配），注释标注"pilot 经验值，待 T1 标定"。
- **监测/执行分置**：本模块只读 ReasoningTickState 轨迹 + project_3d（无梯度
  固定投影），不写任何状态。
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .manifold import ThoughtManifoldProjector
from .reasoning_loop import ReasoningTickState

# ---------------------------------------------------------------------------
# 坏路径检测阈值（pilot 经验值，待 T1 标定——0.1B pilot 阶段先给经验值，
# 正式 1.5B T1 阶段须按真实分布标定）。均为模块常量，可在构造时覆盖。
# ---------------------------------------------------------------------------
#: 信心膨胀阈值：certainty > 此值 且 speak_do_consistency < consistency_low 判信心膨胀
CONFIDENCE_INFLATION_CERTAINTY: float = 0.7
#: 说-做一致性低阈值：speak_do_consistency < 此值（配合高 certainty 判信心膨胀）
SPEAK_DO_CONSISTENCY_LOW: float = 0.3
#: 漂移阈值：相邻 tick 位移范数 > 此值 × 轨迹平均位移范数 判漂移（异常大位移）
DRIFT_DISP_RATIO: float = 3.0
#: 早停失败阈值：certainty 始终低于此值却跑满 max_ticks 判早停失败（未触发有效早停/求知）
EARLY_STOP_FAIL_CERTAINTY: float = 0.5
#: recall 风暴阈值：连续 recall_triggered tick 数 ≥ 此值判 recall 风暴（反复空白未解决）
RECALL_STORM_CONSECUTIVE: int = 3


@dataclass
class ThoughtTrajectoryPoint:
    """单个思考轨迹点的可视化就绪数据（3D 投影 + 审计信号）。

    字段：
        tick_index: 思考 tick 序号（0 起）。
        xyz: [3] 3D 投影坐标（经 ThoughtManifoldProjector.project_3d，固定 view3d，
            仅人类可视化视图，不参与训练/推理计算）。
        certainty: KAL P(IK) 读出标量 ∈ [0,1]。
        recall_triggered: 本 tick 是否触发 <|recall|>（空白检测）。
        early_stop: 本 tick 是否触发早停（certainty > stop_threshold）。
        is_bad_path: 本 tick 是否标记为坏路径（见 ThoughtVisualizer.detect_bad_path）。
        bad_reason: 坏路径原因 str（可组合，";" 分隔；is_bad_path=False 时为 ""）。
        speak_do_consistency: 可选 float | None，忠实性审计的说-做一致性（逐 tick
            余弦均值；无忠实性诊断时为 None）。
    """

    tick_index: int
    xyz: list[float]
    certainty: float
    recall_triggered: bool
    early_stop: bool
    is_bad_path: bool
    bad_reason: str
    speak_do_consistency: float | None = None


@dataclass
class ThoughtTrajectory:
    """一条完整思考轨迹的可视化结构（轨迹点列表 + 元数据）。

    持有：
        points: list[ThoughtTrajectoryPoint]（按 tick_index 顺序）。
        n_ticks: 轨迹长度（tick 总数）。
        stop_tick: 停止 tick 序号（early_stop 触发的 tick；未早停时为最后一个 tick）。
        recall_triggered_any: 是否有任一 tick 触发 recall。
        avg_certainty: 全轨迹平均 certainty。
        n_bad_points: 坏路径点数（is_bad_path=True 的 tick 数）。
    """

    points: list[ThoughtTrajectoryPoint] = field(default_factory=list)
    n_ticks: int = 0
    stop_tick: int = 0
    recall_triggered_any: bool = False
    avg_certainty: float = 0.0
    n_bad_points: int = 0

    def to_dict(self) -> dict:
        """导出为 dict（JSON 可序列化，外部渲染工具/前端消费）。

        结构：
            {
              "meta": {n_ticks, stop_tick, recall_triggered_any, avg_certainty, n_bad_points},
              "points": [ {tick_index, xyz, certainty, recall_triggered, early_stop,
                           is_bad_path, bad_reason, speak_do_consistency}, ... ]
            }
        """
        return {
            "meta": {
                "n_ticks": self.n_ticks,
                "stop_tick": self.stop_tick,
                "recall_triggered_any": self.recall_triggered_any,
                "avg_certainty": self.avg_certainty,
                "n_bad_points": self.n_bad_points,
            },
            "points": [asdict(p) for p in self.points],
        }

    def to_json(self, path: str | Path) -> None:
        """导出为 JSON 文件（外部渲染工具/前端消费）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """文本摘要（轨迹统计 + 坏路径告警）。"""
        lines = [
            f"思考轨迹摘要：{self.n_ticks} ticks，停止于 tick {self.stop_tick}",
            f"  平均 certainty = {self.avg_certainty:.3f}",
            f"  recall 触发：{'是' if self.recall_triggered_any else '否'}",
            f"  坏路径点数：{self.n_bad_points}",
        ]
        if self.n_bad_points > 0:
            bad_pts = [p for p in self.points if p.is_bad_path]
            lines.append("  ⚠️ 坏路径告警：")
            for p in bad_pts:
                lines.append(
                    f"    tick {p.tick_index}: {p.bad_reason} (certainty={p.certainty:.3f})"
                )
        return "\n".join(lines)


class ThoughtVisualizer:
    """思考轨迹构建器（pilot 可解释性前端）。

    从 ReasoningTickState 轨迹 + ThoughtManifoldProjector（用其 project_3d）构建
    ThoughtTrajectory。纯只读（no_grad，监测/执行分置），不回流改变思考动力学。

    持有：坏路径检测阈值（模块常量默认值，可在构造时覆盖——pilot 经验值待 T1 标定）。
    """

    def __init__(
        self,
        confidence_inflation_certainty: float = CONFIDENCE_INFLATION_CERTAINTY,
        speak_do_consistency_low: float = SPEAK_DO_CONSISTENCY_LOW,
        drift_disp_ratio: float = DRIFT_DISP_RATIO,
        early_stop_fail_certainty: float = EARLY_STOP_FAIL_CERTAINTY,
        recall_storm_consecutive: int = RECALL_STORM_CONSECUTIVE,
    ):
        self.confidence_inflation_certainty = confidence_inflation_certainty
        self.speak_do_consistency_low = speak_do_consistency_low
        self.drift_disp_ratio = drift_disp_ratio
        self.early_stop_fail_certainty = early_stop_fail_certainty
        self.recall_storm_consecutive = recall_storm_consecutive

    # ------------------------------------------------------------------
    @torch.no_grad()
    def build(
        self,
        trajectory: list[ReasoningTickState],
        projector: ThoughtManifoldProjector,
        faithfulness_diag: dict | None = None,
    ) -> ThoughtTrajectory:
        """从 ReasoningTickState 轨迹构建可视化就绪的 ThoughtTrajectory。

        参数：
            trajectory: list[ReasoningTickState]（ReasoningLoop.run 返回的轨迹）。
            projector: ThoughtManifoldProjector（用其 project_3d，固定 view3d 无梯度）。
            faithfulness_diag: 可选忠实性诊断 dict（CotFaithfulnessAudit.audit 返回，
                含 speak_do_consistency/cmi_approx/faithfulness_rate）；用于坏路径
                检测①信心膨胀（certainty 高但 speak_do_consistency 低）。
        返回：ThoughtTrajectory（轨迹点 + 元数据）。
        """
        if not trajectory:
            return ThoughtTrajectory(points=[], n_ticks=0)

        # 逐 tick 构建可视化点：current_coord → project_3d → xyz [3]
        # 取 current_coord 的 [B,T,manifold_dim] 均值池化成 [manifold_dim] 标量坐标
        # （pilot 级：每 tick 一个代表点；B/T 维是 batch/seq 维，可视化时聚合成单点）。
        points: list[ThoughtTrajectoryPoint] = []
        prev_coord: torch.Tensor | None = None
        consec_recall = 0  # 连续 recall 计数（recall 风暴检测）
        all_certainty: list[float] = []

        for i, ts in enumerate(trajectory):
            # current_coord [B,T,manifold_dim] → 均值池化 [manifold_dim]（pilot 聚合）
            coord = ts.current_coord.float().mean(dim=(0, 1))  # [manifold_dim]
            xyz = projector.project_3d(coord)  # [3]（固定 view3d，无梯度）
            xyz_list = xyz.tolist()

            # 坏路径检测（返回 is_bad, bad_reason）
            is_bad, bad_reason = self.detect_bad_path(
                ts, prev_coord, faithfulness_diag,
                consec_recall=consec_recall,
                is_last=(i == len(trajectory) - 1),
                n_ticks=len(trajectory),
            )
            if is_bad:
                pass  # is_bad 已由 detect_bad_path 返回
            # 连续 recall 计数更新（recall 风暴检测用）
            consec_recall = consec_recall + 1 if ts.recall_triggered else 0
            all_certainty.append(ts.certainty)

            # speak_do_consistency：faithfulness_diag 给定时取轨迹级均值（逐 tick
            # 一致性须逐 tick 调用 audit，pilot 简化取全轨迹均值）。
            speak_do = (
                float(faithfulness_diag.get("speak_do_consistency"))
                if faithfulness_diag is not None
                and "speak_do_consistency" in faithfulness_diag
                else None
            )

            points.append(
                ThoughtTrajectoryPoint(
                    tick_index=ts.tick_index,
                    xyz=xyz_list,
                    certainty=ts.certainty,
                    recall_triggered=ts.recall_triggered,
                    early_stop=ts.early_stop,
                    is_bad_path=is_bad,
                    bad_reason=bad_reason,
                    speak_do_consistency=speak_do,
                )
            )
            prev_coord = coord

        # 元数据聚合
        n_ticks = len(points)
        stop_tick = next((p.tick_index for p in points if p.early_stop), n_ticks - 1)
        recall_any = any(p.recall_triggered for p in points)
        avg_cert = sum(all_certainty) / len(all_certainty) if all_certainty else 0.0
        n_bad = sum(1 for p in points if p.is_bad_path)

        return ThoughtTrajectory(
            points=points,
            n_ticks=n_ticks,
            stop_tick=stop_tick,
            recall_triggered_any=recall_any,
            avg_certainty=avg_cert,
            n_bad_points=n_bad,
        )

    # ------------------------------------------------------------------
    def detect_bad_path(
        self,
        tick_state: ReasoningTickState,
        prev_coord: torch.Tensor | None,
        faithfulness_diag: dict | None,
        *,
        consec_recall: int = 0,
        is_last: bool = False,
        n_ticks: int = 0,
    ) -> tuple[bool, str]:
        """坏路径监测：标记以下情况（可组合），返回 (is_bad, bad_reason)。

        ① 信心膨胀（Coda-Forno）：certainty 高但 speak_do_consistency 低（说-做不一致）。
           判据：certainty > confidence_inflation_certainty 且
                 speak_do_consistency < speak_do_consistency_low（须 faithfulness_diag 给定）。
        ② 漂移：当前坐标远离上一坐标（位移异常大，超阈值——可能跑飞）。
           判据：||current_coord − prev_coord|| > drift_disp_ratio × 轨迹平均位移范数
                 （pilot 简化：prev_coord 给定时直接用 ||disp|| 与阈值比较）。
        ③ 早停失败：certainty 始终低于阈值却跑满 max_ticks（未触发有效早停/求知）。
           判据：is_last 且 certainty < early_stop_fail_certainty 且非 early_stop。
        ④ recall 风暴：连续多 tick recall_triggered（反复空白未解决）。
           判据：recall_triggered 且 consec_recall + 1 ≥ recall_storm_consecutive。

        参数：
            tick_state: 当前 tick 状态。
            prev_coord: 上一 tick 的流形坐标 [manifold_dim]（None=首 tick，跳过漂移检测）。
            faithfulness_diag: 可选忠实性诊断 dict（含 speak_do_consistency）。
            consec_recall: 截至上一 tick 的连续 recall 计数（recall 风暴检测）。
            is_last: 是否最后一个 tick（早停失败检测）。
            n_ticks: 轨迹总长度（早停失败检测参考）。
        返回：(is_bad, bad_reason)——is_bad 任一条满足即 True；bad_reason 为
            满足的条件组合（";" 分隔），is_bad=False 时为 ""。
        """
        reasons: list[str] = []

        # ① 信心膨胀：certainty 高但 speak_do_consistency 低（Coda-Forno 信号）
        if (
            tick_state.certainty > self.confidence_inflation_certainty
            and faithfulness_diag is not None
            and "speak_do_consistency" in faithfulness_diag
            and float(faithfulness_diag["speak_do_consistency"]) < self.speak_do_consistency_low
        ):
            reasons.append(
                f"信心膨胀(certainty={tick_state.certainty:.2f}>"
                f"{self.confidence_inflation_certainty},speak_do="
                f"{float(faithfulness_diag['speak_do_consistency']):.2f}<"
                f"{self.speak_do_consistency_low})"
            )

        # ② 漂移：当前坐标远离上一坐标（位移异常大）
        if prev_coord is not None:
            coord = tick_state.current_coord.float().mean(dim=(0, 1))  # [manifold_dim]
            disp_norm = float((coord - prev_coord).norm().item())
            # pilot 简化：位移范数 > drift_disp_ratio × 1.0（绝对阈值经验值，待 T1 标定）
            # 正式应按轨迹平均位移范数自适应；pilot 阶段用固定阈值。
            if disp_norm > self.drift_disp_ratio:
                reasons.append(
                    f"漂移(disp_norm={disp_norm:.2f}>{self.drift_disp_ratio})"
                )

        # ③ 早停失败：certainty 始终低于阈值却跑满 max_ticks
        if (
            is_last
            and not tick_state.early_stop
            and tick_state.certainty < self.early_stop_fail_certainty
        ):
            reasons.append(
                f"早停失败(certainty={tick_state.certainty:.2f}<"
                f"{self.early_stop_fail_certainty}且跑满{n_ticks} ticks未早停)"
            )

        # ④ recall 风暴：连续多 tick recall_triggered
        if tick_state.recall_triggered and (consec_recall + 1) >= self.recall_storm_consecutive:
            reasons.append(
                f"recall风暴(连续{consec_recall + 1} ticks触发recall≥"
                f"{self.recall_storm_consecutive})"
            )

        is_bad = len(reasons) > 0
        bad_reason = ";".join(reasons) if reasons else ""
        return is_bad, bad_reason


# ---------------------------------------------------------------------------
def render_ascii(trajectory: ThoughtTrajectory, width: int = 60, height: int = 20) -> str:
    """终端 ASCII 简易渲染（pilot 级，无需 GUI）——把 3D 轨迹投影到 2D 字符画。

    把 xyz 的 xy 平面投影到 width×height 字符网格，标出：
      - tick 序号（0-9 循环，>9 用字母 a-z）；
      - recall 点（R，recall_triggered=True）；
      - 坏路径点（X，is_bad_path=True，覆盖 tick 序号/recall 标记）。

    供快速命令行检视（无需 GUI/matplotlib）。

    参数：
        trajectory: ThoughtTrajectory（须含 points）。
        width/height: 字符画布尺寸（默认 60×20）。
    返回：ASCII 字符串（多行）。
    """
    if not trajectory.points:
        return "(空轨迹)"

    # 收集 xy 坐标（xyz 的前两维）
    xs = [p.xyz[0] for p in trajectory.points]
    ys = [p.xyz[1] for p in trajectory.points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    # 防退化（单点/共线时范围为零）
    x_range = max(x_max - x_min, 1e-6)
    y_range = max(y_max - y_min, 1e-6)

    # 初始化画布（空格填充）
    canvas = [[" "] * width for _ in range(height)]

    def _to_col(x: float) -> int:
        return int((x - x_min) / x_range * (width - 1))

    def _to_row(y: float) -> int:
        # y 轴翻转（屏幕坐标：大 y 在上）
        return height - 1 - int((y - y_min) / y_range * (height - 1))

    # 逐点绘制（坏路径 X > recall R > tick 序号）
    tick_symbols = "0123456789abcdefghijklmnopqrstuvwxyz"
    for p in trajectory.points:
        col = _to_col(p.xyz[0])
        row = _to_row(p.xyz[1])
        if p.is_bad_path:
            ch = "X"
        elif p.recall_triggered:
            ch = "R"
        else:
            ch = tick_symbols[p.tick_index % len(tick_symbols)]
        canvas[row][col] = ch

    lines = ["".join(row) for row in canvas]
    header = (
        f"思考轨迹 ASCII 渲染（{trajectory.n_ticks} ticks，xy 平面投影；"
        f"X=坏路径 R=recall 数字/字母=tick 序号）："
    )
    return header + "\n" + "\n".join(lines)
