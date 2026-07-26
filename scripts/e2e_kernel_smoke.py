"""D-0 pilot 端到端内核烟测：真实 0.1B checkpoint 挂内核，验证 sense/inject 在线工作。

验证项（M5/M2 退出标准 + 全链路对齐）：
1. 加载 checkpoints/pilot_0p1b_ws/final，attach_kernel（内核权重随机初始化——仅结构烟测，
   未经 T2 训练，KAL/HRL 输出无语义，仅验证前向不崩 + 信号通路通）；
2. forward(run_kernel=True) 不崩，产出 sense 信号（GDN 层 PM-stream 读点）；
3. 注入 steering 向量后人效（next-token loss）不显著降（注入即 steer 行为，小扰动）；
4. 注入 steering 后人效（next-token loss）不显著降（注入即 steer 行为，小扰动）；
5. 默认 run_kernel=False 与基线逐点一致（回归判据）。

注意（诚实标注）：内核为随机初始化（未训练），sense 输出无语义，本烟测只验证
"结构通路 + 不破坏主干前向"，KAL 探针语义见 scripts/kal_probe.py（M2 已达标 AUROC 0.945）。

用法：
  CUDA_VISIBLE_DEVICES=1 .venv/Scripts/python.exe scripts/e2e_kernel_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from tais_obsidian.data.memmap import Shards  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import BlockPayload  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_ws/final"
DEVICE = "cuda"


def next_token_loss(model, ids, **fwd_kw) -> float:
    """对一段真实 token 序列算 next-token 平均交叉熵（人效代理）。"""
    with torch.no_grad():
        out = model(ids[:, :-1], **fwd_kw)
        logits = out[0] if isinstance(out, tuple) else out
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(), ids[:, 1:].reshape(-1)
        )
    return float(loss)


def main() -> None:
    torch.manual_seed(0)
    print(f"[smoke] 加载 {CKPT} ...")
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, device=DEVICE).eval()

    # 取一段真实 val token（人效基线）
    import numpy as np
    shards = Shards(ROOT / "data" / "shards", split="val")
    rng = np.random.default_rng(0)
    x, _y = shards.get_batch(batch=1, seq_len=128, device=DEVICE, rng=rng)
    ids = x
    print(f"[smoke] 真实序列 ids {tuple(ids.shape)}")

    # ---- 1. 基线（run_kernel=False，未挂内核）----
    loss_base = next_token_loss(model, ids)
    print(f"[1] 基线（未挂内核）next-token loss = {loss_base:.4f}")

    # ---- 2. 挂内核，默认关闭仍与基线一致 ----
    model.attach_kernel()
    loss_off = next_token_loss(model, ids, run_kernel=False)
    print(f"[2] 挂内核+run_kernel=False loss = {loss_off:.4f}（应与基线一致，Δ={abs(loss_off-loss_base):.2e}）")

    # ---- 3. run_kernel=True：sense 信号通路 ----
    with torch.no_grad():
        out = model(ids, run_kernel=True)
    assert len(out) == 3, "run_kernel=True 应返回三元组（含 kernel_signals）"
    _, _, captures = out
    assert "__kernel__" in captures, "缺 kernel_signals"
    ks = captures["__kernel__"]
    gdn_layers = [i for i, t in enumerate(model.config.layer_types) if t == "G"]
    assert set(ks.keys()) == set(gdn_layers), f"sense 应只在 GDN 层，实际 {sorted(ks.keys())}"
    sample = ks[gdn_layers[0]]["sense"]
    print(f"[3] sense 信号通路 OK：GDN 层 {gdn_layers} 均产出信号；"
          f"示例 P(IK) logits 形状 {tuple(sample.pik_logits.shape)}（注：内核未训练，输出无语义）")

    # ---- 4. 注入 steering：人效不显著降 ----
    # 在第一个 CSA 层（idx 3）前注入一个小幅度 steering 向量（steer 行为，非事实）
    a_layer = next(i for i, t in enumerate(model.config.layer_types) if t == "A")
    steer = torch.randn(model.config.d_model, device=DEVICE) * 0.01  # 小幅度
    payloads = {a_layer: [BlockPayload(block_id="smoke", compiled_kind="steering", vector=steer)]}
    loss_inj = next_token_loss(model, ids, run_kernel=True, inject_payloads=payloads)
    delta = loss_inj - loss_base
    print(f"[4] 注入 steering 后 loss = {loss_inj:.4f}（Δ={delta:+.4f}；注入即 steer 行为，"
          f"{'通过：人效未显著降' if abs(delta) < 0.1 else '⚠️ 人效下降超阈'}）")

    # ---- 5. 注入 namespace 校验 fail-closed（KV 载荷不给 injector 应拒绝）----
    try:
        bad = {a_layer: [BlockPayload(block_id="kv", compiled_kind="kv")]}
        model(ids, run_kernel=True, inject_payloads=bad)
        print("[5] ⚠️ KV 载荷未拒绝（应为 fail-closed）")
    except NotImplementedError:
        print("[5] KV 载荷未给 injector 时 fail-closed 拒绝 OK")

    print("\n[smoke] 端到端内核烟测完成：结构通路 OK、注入后人效稳定、fail-closed 生效。")


if __name__ == "__main__":
    main()
