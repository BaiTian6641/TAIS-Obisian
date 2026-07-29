"""第二阶段真实部件适配 demo：真实 0.1B 模型部件接入推理循环 + 有核/无核消融。

把 thinking_e2e_demo.py 的全 mock（glimpse=mean-pool、certainty=sigmoid(norm)、
state=随机张量）替换为**真实 0.1B 模型部件**（checkpoints/pilot_0p1b_gdn2_10k/final）：

  - 真实 GDN 状态：model.forward(capture_layers=[GDN 层]) 的层输出 [B,T,768]
    经适配层投影 → [B,T,384] 作思考核 state（GDN 持续状态读出）。
  - 真实 certainty：attach_kernel 后 model.forward(run_kernel=True) 的
    captures["__kernel__"][GDN层]["sense"].pik_logits → softmax → known 类概率。
    **诚实标注：10k checkpoint 的 KAL 头未微调（kernel_enabled=False 训练，
    attach_kernel 后随机初始化）——pik_logits 未校准，certainty 仅演示通路，
    非可靠元认知。**
  - 真实 glimpse：CSA 注意力层（type "A"）的层输出 [B,T,768] 经适配层 → [B,T,384]
    作观察（pilot 简化：用层输出残差流近似注意力观察，非真正"往哪看"选择注意）。

红线与纪律：
  - 监测/执行分置：真实模型 forward 与 sense 全程 no_grad（只读）；bridge.tick 写
    PM-stream detach（W1–W2 零梯度快写）；适配层随机初始化未训练（离线才允许训练）。
  - 共享 projector 单实例 + 复用 core.bridge（集成对齐纪律保持）。
  - 消融是 pilot 级：验证"思考核确实改变思考动力学"（轨迹/位移差异），非基准准确率。

运行：$env:CUDA_VISIBLE_DEVICES="1"; .venv/Scripts/python.exe scripts/thinking_real_adapter_demo.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 把 src 加入 import 路径（脚本直接运行用）
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.model.manifold import ThoughtManifoldProjector
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.reasoning_loop import ReasoningLoop
from tais_obsidian.model.thought_core import ThoughtCore

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints" / "pilot_0p1b_gdn2_10k" / "final"
OUT_DIR = ROOT / "runs" / "thinking_real"
D_MODEL = 768          # 真实 0.1B 模型 d_model
CORE_DIM = 384         # ThoughtCore 维度（思考核规格 §1.2 [256,512]）
N_GROUPS = 8
HISTORY = 4
MAX_TICKS = 8
MANIFOLD_DIM = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# 层型 G2G2G2A ×3：GDN 读点取最后一个 GDN 层（index 10），glimpse 取第一个 CSA 层（index 3）
GDN_LAYER = 10
ATTN_LAYER = 3


# ---------------------------------------------------------------------------
# 真实部件适配器
# ---------------------------------------------------------------------------
class RealThoughtAdapter(nn.Module):
    """真实 0.1B 模型部件 → 推理循环接口的适配层（768↔384 维度桥接 + 真实读出封装）。

    持有：
      model: TaisObsidianForCausalLM（真实 0.1B checkpoint，attach_kernel 挂载内核）；
      down_proj: 768→384 Linear（真实 GDN/CSA 层输出投到思考核维度）；
      up_proj: 384→768 Linear（思考核状态反投影，写回/对照用；pilot 未训练）。

    纪律：model 全程 no_grad 只读（监测/执行分置）；down/up_proj 随机初始化
    未训练（适配层训练走离线，本 demo 仅演示通路）。
    """

    def __init__(self, model: TaisObsidianForCausalLM, core_dim: int = CORE_DIM):
        super().__init__()
        self.model = model
        d_model = model.config.d_model
        self.d_model = d_model
        self.core_dim = core_dim
        self.down_proj = nn.Linear(d_model, core_dim)
        self.up_proj = nn.Linear(core_dim, d_model)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def read_gdn_state(
        self, input_ids: torch.Tensor, layer: int = GDN_LAYER
    ) -> torch.Tensor:
        """真实 GDN 状态读出：capture GDN 层输出 [B,T,768] → down_proj → [B,T,384]。

        思考核的 state 来源（GDN 持续状态/工作记忆寄存器角色）。
        """
        _, _, captures = self.model(input_ids, capture_layers=[layer])
        h = captures[layer]  # [B,T,768]（pm_stream=1 单流为层输出残差流）
        return self.down_proj(h.float())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def glimpse(
        self, input_ids: torch.Tensor, layer: int = ATTN_LAYER
    ) -> torch.Tensor:
        """真实 glimpse：capture CSA 注意力层（type "A"）输出 [B,T,768] → [B,T,384]。

        [pilot 简化] 用注意力层输出残差流作观察近似（非真正 CTM 式"往哪看"
        选择注意——正式应接 CSA 检索的 top-k 块观察；接口位，注释标注）。
        """
        _, _, captures = self.model(input_ids, capture_layers=[layer])
        h = captures[layer]
        return self.down_proj(h.float())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def certainty(
        self, input_ids: torch.Tensor, layer: int = GDN_LAYER
    ) -> float:
        """真实 certainty：run_kernel=True → sense.pik_logits → softmax → known 类概率。

        **KAL 未校准诚实标注**：10k checkpoint 的 KAL 头随机初始化未微调
        （kernel_enabled=False 训练，attach_kernel 后未载入内核权重）——
        pik_logits 未校准，返回的 known 概率仅演示"真实 sense 通路"可用，
        **非可靠元认知**（不做早停/空白判断依据）。
        """
        _, _, captures = self.model(input_ids, run_kernel=True)
        sense = captures["__kernel__"][layer]["sense"]
        probs = torch.softmax(sense.pik_logits[:, -1, :].float(), dim=-1)  # [B,3]
        return float(probs[:, 0].mean().item())  # known 类（类 0）概率均值


# ---------------------------------------------------------------------------
# 加载真实模型
# ---------------------------------------------------------------------------
def load_real_model(device: str = DEVICE) -> TaisObsidianForCausalLM:
    """加载 10k checkpoint 并挂载内核（attach_kernel 幂等；内核头随机初始化）。"""
    model = TaisObsidianForCausalLM.from_pretrained(CKPT_DIR, device=device)
    model.eval()
    if model.kernel is None:
        model.attach_kernel()  # 内核头随机初始化（KAL 未校准，见 certainty 标注）
    return model


# ---------------------------------------------------------------------------
# 构建真实部件接入的推理循环
# ---------------------------------------------------------------------------
def build_real_loop(
    adapter: RealThoughtAdapter,
    input_ids: torch.Tensor,
    seed: int = 42,
    mock_certainty: bool = True,
) -> dict:
    """真实部件接入 ReasoningLoop（替换 glimpse/certainty/state 来源）。

    集成对齐：共享 projector 单实例 + 复用 core.bridge（与 e2e demo 同一纪律）。
    certainty 缺省用 mock（低值跑满 tick）：真实 KAL 未校准仅演示通路（见上标注），
    不作早停/空白依据；mock_certainty=False 时可切真实通路演示（不推荐作判据）。
    """
    torch.manual_seed(seed)
    shared_projector = ThoughtManifoldProjector(
        d_model=CORE_DIM, manifold_dim=MANIFOLD_DIM
    ).to(DEVICE)
    thought_core = ThoughtCore(
        core_dim=CORE_DIM, n_groups=N_GROUPS, history=HISTORY,
        max_ticks=MAX_TICKS, manifold_dim=MANIFOLD_DIM,
        projector=shared_projector, use_sync=True,
    ).to(DEVICE)
    reasoning_loop = ReasoningLoop(
        thought_core=thought_core, bridge=thought_core.bridge, kernel=None,
    ).to(DEVICE)

    # 真实 GDN 状态作思考核初始 state（维度桥接 768→384）
    with torch.no_grad():
        real_state = adapter.read_gdn_state(input_ids)  # [B,T,384]

    # 替换 certainty 来源（真实 KAL 通路演示 or mock 跑满）
    if not mock_certainty:
        reasoning_loop.kal_certainty = lambda s: adapter.certainty(input_ids)
    else:
        reasoning_loop.kal_certainty = lambda s: 0.2  # mock 低 certainty → 跑满 tick

    # 共享 projector 一致性断言（集成红线）
    assert thought_core.bridge.projector is shared_projector
    assert reasoning_loop.bridge.projector is shared_projector
    return {
        "shared_projector": shared_projector,
        "thought_core": thought_core,
        "reasoning_loop": reasoning_loop,
        "real_state": real_state,
    }


# ---------------------------------------------------------------------------
# 有核 vs 无核消融
# ---------------------------------------------------------------------------
def run_ablation(
    adapter: RealThoughtAdapter,
    input_ids: torch.Tensor,
    seed: int = 42,
) -> dict:
    """有思考核 vs 无思考核消融（同一真实输入，pilot 级——验证核改变动力学）。

    (a) 有核：真实 GDN 状态 → ThoughtCore 多 tick 演化（+bridge 位移写 PM）；
    (b) 无核：真实 GDN 状态直接单步（一次 down_proj 读出，不经思考核演化）。
    对比：轨迹长度/累计流形位移/末坐标到 target 距离/certainty（真实通路演示）。
    """
    torch.manual_seed(seed)
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    built = build_real_loop(adapter, input_ids, seed=seed, mock_certainty=True)
    rl = built["reasoning_loop"]
    real_state = built["real_state"]
    B, T, _ = real_state.shape
    target_coord = torch.randn(B, T, MANIFOLD_DIM, device=DEVICE, generator=g)

    # (a) 有核：多 tick 演化
    final_core, trajectory, stop_tick = rl.run(
        real_state, target_coord=target_coord, max_ticks=MAX_TICKS,
        stop_threshold=0.9, recall_threshold=0.3, bridge_alpha=0.1,
    )
    # 累计流形位移（核改变动力学的直接证据）
    total_disp = sum(float(ts.disp.float().norm().item()) for ts in trajectory)
    # 有核末态投影坐标→target 距离（核演化后真实到达的流形位置；非 current+disp——
    # disp 定义为 target−current，二者相加恒等 target 无意义）
    sp = built["shared_projector"]
    core_final_coord = sp.project(final_core.float())  # [B,T,manifold]
    dist_core = float((core_final_coord - target_coord.float()).norm().item())

    # (b) 无核：真实 GDN 状态单步（同一真实初始状态，一次投影取坐标，不演化）
    with torch.no_grad():
        no_core_coord = sp.project(real_state.float())  # [B,T,manifold]（时间维对齐）
        dist_no_core = float((no_core_coord - target_coord.float()).norm().item())

    # 真实 certainty 通路演示（KAL 未校准，仅通路验证，非判据）
    real_cert = adapter.certainty(input_ids)
    # 真实 glimpse 通路演示（CSA 层输出，pilot 简化近似）
    real_glimpse = adapter.glimpse(input_ids)

    return {
        "n_ticks": len(trajectory),
        "stop_tick": stop_tick,
        "total_disp": total_disp,
        "dist_core": dist_core,
        "dist_no_core": dist_no_core,
        "real_certainty": real_cert,
        "real_glimpse_shape": list(real_glimpse.shape),
        "trajectory": trajectory,
        "final_core": final_core,
        "real_state_shape": list(real_state.shape),
    }


# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("第二阶段真实部件适配 demo：真实 0.1B 模型部件 + 有核/无核消融")
    print("=" * 70)
    print(f"设备: {DEVICE} | checkpoint: {CKPT_DIR.name} | d_model={D_MODEL} "
          f"core_dim={CORE_DIM} manifold_dim={MANIFOLD_DIM} max_ticks={MAX_TICKS}")

    model = load_real_model()
    adapter = RealThoughtAdapter(model).to(DEVICE)

    # 真实输入：vocab 内随机 token 序列（真实前向通路）
    g = torch.Generator(device=DEVICE).manual_seed(42)
    B, T = 2, 32
    input_ids = torch.randint(
        0, model.config.vocab_size, (B, T), device=DEVICE, generator=g
    )

    # 真实通路读出
    real_state = adapter.read_gdn_state(input_ids)
    real_glimpse = adapter.glimpse(input_ids)
    real_cert = adapter.certainty(input_ids)
    print("\n[① 真实部件读出]")
    print(f"  真实 GDN 状态（层{GDN_LAYER}→384）: {list(real_state.shape)}")
    print(f"  真实 glimpse（CSA 层{ATTN_LAYER}→384）: {list(real_glimpse.shape)}")
    print(f"  真实 certainty（KAL known 概率）: {real_cert:.4f} "
          f"（未校准，仅通路演示）")

    # 消融
    ab = run_ablation(adapter, input_ids, seed=42)
    print("\n[② 有核 vs 无核消融]")
    print(f"  有核 tick 数: {ab['n_ticks']}（stop_tick={ab['stop_tick']}）")
    print(f"  有核累计流形位移: {ab['total_disp']:.4f}")
    print(f"  有核末坐标→target 距离: {ab['dist_core']:.4f}")
    print(f"  无核单步坐标→target 距离: {ab['dist_no_core']:.4f}")
    print(f"  核改变动力学: {'是' if ab['n_ticks'] > 1 else '否'}"
          f"（多 tick 演化产生 {ab['n_ticks']} 步轨迹）")

    # 导出
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "checkpoint": str(CKPT_DIR),
        "d_model": D_MODEL, "core_dim": CORE_DIM, "manifold_dim": MANIFOLD_DIM,
        "gdn_layer": GDN_LAYER, "attn_layer": ATTN_LAYER,
        "real_certainty": ab["real_certainty"],
        "real_state_shape": ab["real_state_shape"],
        "real_glimpse_shape": ab["real_glimpse_shape"],
        "ablation": {
            "n_ticks": ab["n_ticks"], "stop_tick": ab["stop_tick"],
            "total_disp": ab["total_disp"],
            "dist_core": ab["dist_core"], "dist_no_core": ab["dist_no_core"],
        },
        "notes": "KAL 头未校准（10k 未微调），certainty 仅演示通路；消融 pilot 级"
                 "（验证核改变动力学，非基准准确率）；适配层随机初始化未训练。",
    }
    json_path = OUT_DIR / "real_adapter_ablation.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[③ 导出] {json_path}")
    print("\n" + "=" * 70)
    print("真实部件适配 demo 完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
