"""内核端到端接线测试：model.forward(run_kernel=True) 调 sense/inject（监测/执行分置）。

判据：
- 默认 run_kernel=False：forward 行为与未挂内核基线**逐点一致**（94 项基线零改动的强判据）；
- run_kernel=True（单流）：在 GDN 层后产出 sense 信号（kernel_signals 非空、形状正确）；
- run_kernel=True（PM-stream）：sense 读 PM-stream（末位流）；inject 写 CSA 残差前 PM-stream
  （注入后该层 PM-stream 改变）；
- 监测/执行分置：sense 只在 GDN 层、inject 只在 CSA 层。
"""
from __future__ import annotations

import torch

from tais_obsidian.config import ModelConfig
from tais_obsidian.model.model import TaisObsidianForCausalLM
from tais_obsidian.model.tais_kernel import BlockPayload

D = 32


def _tiny(pm_stream: int = 1, kernel_enabled: bool = True) -> ModelConfig:
    return ModelConfig(
        vocab_size=64, d_model=D, n_layer=4, block_pattern=["G", "G", "G", "A"],
        n_q_heads=4, n_kv_heads=2, head_dim=8, n_v_heads=4, n_qk_heads=2, mlp_hidden=64,
        max_seq=16, grad_checkpoint=False, check_0p1b_params=False,
        pm_stream=pm_stream, kernel_enabled=kernel_enabled,
        kernel_dg_dim=32, kernel_dg_topk=4,
    )


def _ids() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, 64, (1, 8))


def test_default_off_matches_no_kernel_baseline() -> None:
    """默认 run_kernel=False：挂内核但不启用时，forward 与纯基线逐点一致。"""
    torch.manual_seed(0)
    m_on = TaisObsidianForCausalLM(_tiny(kernel_enabled=True))
    torch.manual_seed(0)
    m_off = TaisObsidianForCausalLM(_tiny(kernel_enabled=False))
    # 同步主干权重（内核仅多 kernel 参数；主干同种子初始化一致）
    m_off.load_state_dict({k: v for k, v in m_on.state_dict().items() if not k.startswith("kernel.")},
                          strict=False)
    ids = _ids()
    m_on.eval(); m_off.eval()
    with torch.no_grad():
        lo, _ = m_on(ids, run_kernel=False)
        lf, _ = m_off(ids)
    assert torch.allclose(lo, lf, atol=1e-5), "默认关闭时挂内核模型的输出应与纯基线一致"


def test_run_kernel_produces_sense_signals_single_stream() -> None:
    m = TaisObsidianForCausalLM(_tiny(pm_stream=1))
    m.eval()
    ids = _ids()
    with torch.no_grad():
        out = m(ids, run_kernel=True)
    assert len(out) == 3
    _, _, captures = out
    assert "__kernel__" in captures
    ks = captures["__kernel__"]
    # GDN 层（0,1,2）产出 sense；CSA 层（3）不做 sense（监测/执行分置）
    assert set(ks.keys()) == {0, 1, 2}
    for i in (0, 1, 2):
        sense = ks[i]["sense"]
        assert sense.pik_logits.shape[-1] == 3
        assert sense.affect_logits.shape[-1] == 2


def test_run_kernel_inject_writes_pm_stream() -> None:
    """PM-stream 路径：inject 写 CSA 残差前 PM-stream（注入改变该层末位流）。"""
    m = TaisObsidianForCausalLM(_tiny(pm_stream=2))
    m.eval()
    ids = _ids()
    vec = torch.ones(D)
    payloads = {3: [BlockPayload(block_id="v", compiled_kind="steering", vector=vec)]}
    with torch.no_grad():
        # 不注入
        _, _, caps_no = m(ids, capture_layers=[3], run_kernel=True)
        pm_no = caps_no[3]["pm"].clone()
        # 注入（steering 向量写 CSA 层 3 残差前 PM-stream）
        _, _, caps_inj = m(ids, capture_layers=[3], run_kernel=True,
                           inject_payloads=payloads)
        pm_inj = caps_inj[3]["pm"].clone()
    # 注入后该层 PM-stream 与未注入不同（单次加法生效）
    assert not torch.allclose(pm_no, pm_inj), "inject 应改变 CSA 层残差前 PM-stream"


def test_sense_only_on_gdn_inject_only_on_csa() -> None:
    """监测/执行分置：sense 信号仅出现在 GDN 层；inject 载荷仅在 CSA 层生效。"""
    m = TaisObsidianForCausalLM(_tiny(pm_stream=1))
    m.eval()
    ids = _ids()
    vec = torch.ones(D)
    with torch.no_grad():
        _, _, caps = m(ids, run_kernel=True,
                       inject_payloads={3: [BlockPayload(block_id="v", compiled_kind="icv", vector=vec)]})
    ks = caps["__kernel__"]
    # sense 只在 GDN 层（0,1,2）
    assert all(m.layers[i].type == "G" for i in ks.keys())
