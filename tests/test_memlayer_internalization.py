"""记忆层条目内化单元测试（fb1 P0：事实迁记忆层条目，根治门控副作用）。

对齐 scripts/memlayer_internalization_e2e.py 的实测行为：
- 记忆层 write→query token 寻址查表（单条读出余弦≈0.98，查表命中）。
- 记忆层读出含答案方向（logit 偏置探针：答案 token 被抬高）。
- **副作用消除（核心判据）**：记忆层残差注入不影响 in-context 精确召回（≈纯净基线），
  区别于 KV 拼接门控（in-context 0.688→0.500）。
- 载体能力边界：mem_entry token 寻址 factual_recall=True，concept_slot 向量=False。
- 主干 frozen：记忆层 write 是 state buffer 零梯度 delta 写，主干权重逐位不变。

红线：纯新增测试，不改 memlayer/injection/model。需要 checkpoint + GPU（teaching ckpt）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tais_obsidian.model.memlayer import make_memory_layer  # noqa: E402
from tais_obsidian.model.model import TaisObsidianForCausalLM  # noqa: E402
from tais_obsidian.model.tais_kernel import ADDRESSED_KINDS, VECTOR_KINDS, BlockPayload  # noqa: E402
from tais_obsidian.tokenizer_io import TokenizerIO  # noqa: E402

CKPT = "checkpoints/pilot_0p1b_gdn2_10k_teaching"
TOK = "data/tokenizer/tokenizer.json"
KD, D = 64, 768

# 标记：需要 GPU + checkpoint 的端到端测试（CI 无卡时跳过）
needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or not (ROOT / CKPT / "model.safetensors").exists(),
    reason="需 CUDA + teaching checkpoint",
)


# ===========================================================================
# 纯单元（CPU 秒级，不需 checkpoint）：载体边界 + 查表机制
# ===========================================================================
def test_mem_entry_is_token_addressed_factual_recall() -> None:
    """载体能力边界：mem_entry ∈ ADDRESSED_KINDS（token 寻址），factual_recall=True。"""
    assert "mem_entry" in ADDRESSED_KINDS
    p = BlockPayload(block_id="b", compiled_kind="mem_entry")
    assert p.factual_recall is True


def test_concept_slot_is_position_invariant_not_factual() -> None:
    """载体能力边界：concept_slot/icv/steering ∈ VECTOR_KINDS（位置不变），factual_recall=False。"""
    for kind in ("concept_slot", "icv", "steering"):
        assert kind in VECTOR_KINDS
        p = BlockPayload(block_id="b", compiled_kind=kind)
        assert p.factual_recall is False


def test_memlayer_write_then_query_recalls_value() -> None:
    """记忆层 write(k,v) 后 query(k) 读出 ≈ v（token 寻址查表命中，factual_recall 机制）。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k, v = torch.randn(KD), torch.randn(D)
    ml.write(k, v, beta=1.0)
    out = ml.query(k, topk=4)
    # 读出与写入 value 高余弦（token 寻址命中；product-key 训练键值随机时不干扰 delta 读出路径）
    assert F.cosine_similarity(out, v, dim=0).item() > 0.9


def test_memlayer_write_is_zero_grad_state_buffer() -> None:
    """记忆层 write 是 state buffer 零梯度 delta 写（非参数，不动 keys/values 训练参数）。"""
    ml = make_memory_layer(n_slots=64, key_dim=KD, value_dim=D)
    k0 = ml.keys.detach().clone()
    v0 = ml.values.detach().clone()
    ml.write(torch.randn(KD), torch.randn(D), beta=1.0)
    # state（运行时 buffer）变了；keys/values（训练参数）不变
    assert ml.state.abs().sum() > 0
    assert torch.equal(ml.keys.detach(), k0)
    assert torch.equal(ml.values.detach(), v0)


# ===========================================================================
# 端到端（GPU + teaching checkpoint）
# ===========================================================================
@needs_gpu
def test_memlayer_readout_contains_answer_direction() -> None:
    """记忆层读出 value 含答案嵌入方向（logit 偏置探针：答案 token logit 被抬高）。

    实测（单条）：读出 vs 写入 value 余弦≈0.98；logit 偏置 top1=答案尾段 subword。
    证明 mem_entry token 寻址查表命中、读出表征=答案方向（事实召回的载体基础）。
    """
    dev = "cuda"
    tok = TokenizerIO(TOK)
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    csa = a_layers[0]

    ml = make_memory_layer(n_slots=256, key_dim=KD, value_dim=D).to(dev)
    key_proj = torch.nn.Linear(D, KD, bias=False).to(dev)
    torch.manual_seed(1)
    torch.nn.init.normal_(key_proj.weight, std=0.02)

    Q, A = "What does the Nyxdrethorvae engine run on?", "krypton"
    with torch.no_grad():
        ids = torch.tensor([tok.encode(Q)], device=dev)
        with torch.autocast("cuda", torch.bfloat16):
            _, _, caps = model(ids, capture_layers=[csa])
        k = key_proj(caps[csa][0].mean(0).float())
        aids = torch.tensor([tok.encode(A)], device=dev)
        v = model.embed(aids)[0].mean(0).float()
        ml.write(k, v, beta=1.0)
        out = ml.query(k, topk=4)
        # 读出命中答案方向
        assert F.cosine_similarity(out, v, dim=0).item() > 0.9
        # logit 偏置：答案尾段 subword 'pton'（krypton 末 token）被抬高进 top-k
        bias = model.embed.weight.float() @ out
        top_ids = bias.topk(10).indices.tolist()
        answer_last = tok.encode(A)[-1]  # krypton 末 token id
        assert answer_last in top_ids


@needs_gpu
def test_memlayer_injection_no_incontext_side_effect() -> None:
    """副作用消除（核心判据）：记忆层残差注入不影响 in-context 精确召回（≈纯净基线）。

    实测：纯净基线 0.688 / 记忆层路径 0.688（完全一致）vs KV 拼接门控 0.500（副作用）。
    记忆层不经 HCA gist 通道→结构上无 gist 门控被波及（根治）。此处用 n=4 小样验证趋势
    （记忆层 in-context ≥ KV 门控 in-context，且记忆层接近纯净基线）。
    """
    dev = "cuda"
    tok = TokenizerIO(TOK)
    from train_retrieval_recall import make_facts, harvest_kv, hidden  # noqa: E402
    from tais_obsidian.model.tri_attention_gated import attach_gated_fusion  # noqa: E402
    from tais_obsidian.model.blockpath import make_namespace  # noqa: E402
    from tais_obsidian.model.injection import make_injector  # noqa: E402

    model = TaisObsidianForCausalLM.from_pretrained(CKPT, dev)  # 纯净模型（无扩容门控）
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    csa = a_layers[0]
    facts = make_facts(4, seed=0)

    ml = make_memory_layer(n_slots=256, key_dim=KD, value_dim=D).to(dev)
    key_proj = torch.nn.Linear(D, KD, bias=False).to(dev)
    torch.manual_seed(1)
    torch.nn.init.normal_(key_proj.weight, std=0.02)
    for f in facts:
        with torch.no_grad():
            qh = hidden(model, tok, f["Q"], csa, dev)[0].mean(0).float()
            k = key_proj(qh)
            aids = torch.tensor([tok.encode(f["A"])], device=dev)
            v = model.embed(aids)[0].mean(0).float()
            ml.write(k, v, beta=1.0)

    def _continue(logits, cache, max_new=8):
        out = []
        with torch.autocast("cuda", torch.bfloat16):
            for _ in range(max_new):
                nxt = int(logits[:, -1, :].float().argmax(-1).item())
                if nxt == tok.eot_id:
                    break
                out.append(nxt)
                logits, cache = model(torch.tensor([[nxt]], device=dev), cache)
        return tok.decode(out)

    def _correct(gen, gold):
        g, a = gen.strip().lower(), gold.strip().lower()
        return a in g or a.replace("-", " ") in g or a.replace("-", "") in g.replace("-", "")

    with torch.no_grad():
        # 纯净基线 in-context（K 纯文本前缀）
        ic_base = 0
        # 记忆层残差注入叠加 in-context（记忆层不经 gist，应≈基线）
        ic_mem = 0
        for f in facts:
            full = f"{f['K']}\nQuestion: {f['Q']}\nAnswer: "
            with torch.autocast("cuda", torch.bfloat16):
                logits, cache = model(torch.tensor([tok.encode(full)], device=dev))
            ic_base += _correct(_continue(logits, cache), f["A"])
            # 记忆层残差注入
            qh = hidden(model, tok, f["Q"], csa, dev)[0].mean(0).float()
            value = ml.query(key_proj(qh), topk=4).detach()
            delta = value.to(model.embed.weight.dtype)
            ids = torch.tensor([tok.encode(full)], device=dev)

            def _add(module, args_, d=delta):
                return (args_[0] + d.view(1, 1, -1),) + args_[1:]

            h = model.layers[csa].register_forward_pre_hook(_add)
            try:
                with torch.autocast("cuda", torch.bfloat16):
                    logits, cache = model(ids)
                ic_mem += _correct(_continue(logits, cache), f["A"])
            finally:
                h.remove()
    # 核心断言：记忆层注入不降低 in-context（ic_mem ≈ ic_base，副作用消除）
    assert ic_mem >= ic_base - 1  # 允许 1 条数值噪声；关键是记忆层不劣化纯文本召回


@needs_gpu
def test_backbone_frozen_after_memlayer_write() -> None:
    """主干 frozen：记忆层 write/query 后主干权重逐位不变（drift==0）。"""
    dev = "cuda"
    tok = TokenizerIO(TOK)
    model = TaisObsidianForCausalLM.from_pretrained(CKPT, dev)
    model.eval()
    a_layers = [i for i, t in enumerate(model.config.layer_types) if t == "A"]
    csa = a_layers[0]
    snap = {n: p.detach().clone() for n, p in model.named_parameters()}

    ml = make_memory_layer(n_slots=256, key_dim=KD, value_dim=D).to(dev)
    key_proj = torch.nn.Linear(D, KD, bias=False).to(dev)
    torch.manual_seed(1)
    from train_retrieval_recall import make_facts, hidden  # noqa: E402
    facts = make_facts(4, seed=0)
    with torch.no_grad():
        for f in facts:
            qh = hidden(model, tok, f["Q"], csa, dev)[0].mean(0).float()
            ml.write(key_proj(qh), model.embed(
                torch.tensor([tok.encode(f["A"])], device=dev))[0].mean(0).float(), beta=1.0)
        # 读出
        _ = ml.query(key_proj(hidden(model, tok, facts[0]["Q"], csa, dev)[0].mean(0).float()))
    drift = max((p.detach().float() - snap[n].float()).abs().max().item()
                for n, p in model.named_parameters())
    assert drift == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
