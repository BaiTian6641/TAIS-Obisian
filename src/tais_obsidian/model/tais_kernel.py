"""TAIS 内核（TAIS Kernel）骨架（M1）：聚合 KAL 与 HRL 内生头，统一 PM-stream 读写接口。

设计依据（必须逐条对齐，禁止凭记忆扩展）：
- 接口与实现计划 v1.0 §0/§2：TAISKernel = checkpoint 内生部件（前向可微、随 state_dict
  存取），聚合 KAL 各头 + HRL Indexer + DG 投影 + 侧信道头簇；sense/route/inject 三方法。
- 监测/执行分置红线（子系统架构规格 Part B；PMC9053853 监测/控制分离）：
  sense() 只读 **GDN-MemBlock 输出处 PM-stream**（S[..., -1, :]），零副作用；
  inject() 只写 **CSA-AttnBlock 残差前 PM-stream**——读写不同层，避免探针读到
  自己刚写的干预而自激（"探针读到自己的干预"为已知失败模式，设计 §8/§29）。
- 载体能力边界（接口计划 §6，已核实）：token 寻址载体（kv/mem_entry/gist/concept_slot）
  能事实召回；位置不变向量（icv/steering）不能，只能 steer 行为。BlockPayload 标
  factual_recall 字段。
- HRL checkpoint 边界（用户选方案 B）：学习型头（Indexer/DG/侧信道）内生本文件；
  数据/算法（页表/BlockStore/CA3 PPR/CA1 门）走 runtime/ 服务（M4，本骨架仅留接口位）。

纪律：
- 本骨架只做结构与对拍单测（前向不崩、PM 读写通）；注入训练、RL、真实块库接 runtime
  均在后续 milestone，禁止在本骨架引入未核实机制。
- 读点默认内容流（stream="content"），与 kal.py read_point 一致；PM 读点
  （stream="pm"）为设计 §13.4 规范读点，待 PM 模型定稿后切换默认。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .kal import KALHead, make_l1_head, make_l2_head

# 块载体类型（接口计划 §6；token 寻址 vs 位置不变的分类即载体能力边界）
BlockKind = str  # Literal["kv","mem_entry","icv","steering","concept_slot","lora","gist","route"]

# 位置不变向量载体（只能 steer 行为/单槽理解，不能事实召回——已核实边界）。
# concept_slot 属此类：输入侧"单槽理解"的向量注入（概念压缩成单向量），非事实查表。
VECTOR_KINDS: frozenset = frozenset({"icv", "steering", "concept_slot"})
# token 寻址载体（能事实召回：KV 前缀 / 记忆层条目 / gist / LoRA / 路径块）
ADDRESSED_KINDS: frozenset = frozenset({"kv", "mem_entry", "gist", "lora", "route"})


@dataclass
class BlockPayload:
    """注入载荷（接口计划 §5.1 的 BlockPayload 骨架）。

    factual_recall：载体能力边界标注——由 compiled_kind 推导（向量载体=False），
    不可由调用方伪造（__post_init__ 强校验，防"向量当事实用"红线）。
    """

    block_id: str
    compiled_kind: BlockKind
    vector: torch.Tensor | None = None       # icv/steering/concept_slot 载荷 [d]
    entries: tuple | None = None             # kv/gist/mem_entry 载荷 (k,v)（M5 接 tri/blockpath）
    layer_ns: tuple = ()                     # namespace 五元组（M5 接 blockpath 校验）
    signature: bytes = b""
    factual_recall: bool = field(init=False)

    def __post_init__(self) -> None:
        self.factual_recall = self.compiled_kind in ADDRESSED_KINDS
        if self.compiled_kind not in (VECTOR_KINDS | ADDRESSED_KINDS):
            raise ValueError(f"未知块载体类型: {self.compiled_kind!r}")


@dataclass
class SenseOut:
    """sense() 输出（零副作用，只读信号）。"""

    pik_logits: torch.Tensor          # [...,3] L1 三态（知道/不确定/空白）
    affect_logits: torch.Tensor       # [...,2] L2 情感（valence/arousal）
    write_salience: torch.Tensor      # [...,1] 写显著性头（惊讶度→W0 加标）
    conflict_logit: torch.Tensor      # [...,1] L3 冲突检测（远期占位）


@dataclass
class RouteOut:
    """route() 输出：内生 Indexer 分数 + DG 稀疏 key（喂 runtime Pager，M4 起用）。"""

    sparse_key: torch.Tensor          # [..., dg_dim] DG 稀疏 key
    score: torch.Tensor               # [...,1] Indexer 打分（块域）


class HRLIndexer(nn.Module):
    """HRL 统一 Indexer 打分头（方案 B 内生 checkpoint；接口计划 §4.1 / C2）。

    块域/token 域同构打分头；正式实现用 **CSA indexer 权重初始化**再做块域 KL 对齐（T2）。
    红线（MoE-RL 教训，§27.3）：**辅助损失梯度只进 Indexer，禁止污染主干**——调用方须
    对主干输入 detach（见 forward 的 detach_input 参数，默认 True=隔离，隔离即默认开）。
    """

    def __init__(self, d_model: int, use_lightning: bool = True, n_index_heads: int = 4, d_index: int = 32):
        super().__init__()
        self.score = nn.Linear(d_model, 1)
        # 真正的 CSA Indexer（DSA lightning indexer 式，model/hrl_indexer.py）：
        # 独立多头低维打分器，对候选块集合打分（区别于上面的单层骨架 nn.Linear(d,1)）。
        # use_lightning=True 时启用（默认）；False 退回骨架（消融对照）。
        self.use_lightning = use_lightning
        if use_lightning:
            from .hrl_indexer import LightningIndexer
            self.lightning = LightningIndexer(d_model, n_index_heads, d_index)
        else:
            self.lightning = None

    def forward(self, query: torch.Tensor, detach_input: bool = True) -> torch.Tensor:
        """query [...,d] → score [...,1]。正式 top-k 分块归并在 runtime（M4）。

        detach_input=True（默认）：对主干 query 输入 detach，保证辅助损失梯度只回传到
        Indexer 权重，不污染主干残差流（红线）。T3 统一 RL 时若需端到端梯度，可显式关。
        """
        if detach_input:
            query = query.detach()
        return self.score(query)

    def score_candidates(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        detach_input: bool = True,
    ) -> torch.Tensor:
        """对候选块集合打分（真正的 CSA Indexer 路径，DSA lightning indexer 式）。

        query [B,Tq,d]（当前思考段），candidates [B,Tk,d]（候选块表示）→ 分数 [B,Tq,Tk]。
        供 runtime Pager 做 top-k 块检索（M4 起用）；token 域/块域同构（一个打分器两种对象）。
        detach_input 隔离主干（红线）。未启用 lightning 时 fail-closed（RuntimeError）。
        """
        if self.lightning is None:
            raise RuntimeError("LightningIndexer 未启用（use_lightning=False），无法用 score_candidates")
        if detach_input:
            query = query.detach()
            candidates = candidates.detach()
        return self.lightning(query, candidates)

    def topk_candidates(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        k: int,
        detach_input: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """打分 + top-k 候选（离散，无梯度）。返回 (top_scores [B,Tq,k], top_idx [B,Tq,k])。"""
        if self.lightning is None:
            raise RuntimeError("LightningIndexer 未启用，无法用 topk_candidates")
        if detach_input:
            query = query.detach()
            candidates = candidates.detach()
        return self.lightning.topk_indices(query, candidates, k)

    def kl_warmup_loss(self, query: torch.Tensor, candidates: torch.Tensor, teacher_scores: torch.Tensor) -> torch.Tensor:
        """warmup：KL 散度对齐 indexer 分布到稠密教师（DSA warmup 范式，T2）。

        teacher_scores [B,Tq,Tk]：稠密教师（全块枚举打分），detach 后作目标。
        """
        if self.lightning is None:
            raise RuntimeError("LightningIndexer 未启用，无法 KL warmup")
        return self.lightning.kl_warmup_loss(query, candidates, teacher_scores)

    def load_from_csa_indexer(self, csa_indexer_weight: torch.Tensor) -> None:
        """用 CSA indexer 权重初始化（设计 §11.1：HRL 块索引器与 CSA token 索引器同构）。

        csa_indexer_weight [1, d] 或 [d]：CSA 压缩注意力 indexer 的打分向量；
        形状须与 self.score.weight 一致（d_model,），fail-closed 校验。
        """
        w = csa_indexer_weight.detach().reshape(self.score.weight.shape).to(self.score.weight.dtype)
        if w.shape != self.score.weight.shape:
            raise ValueError(f"CSA indexer 权重形状 {w.shape} 与 Indexer {self.score.weight.shape} 不符")
        with torch.no_grad():
            self.score.weight.copy_(w)

    def init_from_attention_qproj(self, q_proj_weight: torch.Tensor, d_model: int, n_q_heads: int, head_dim: int) -> None:
        """从注意力 q_proj 派生 indexer 初始化（设计 §11.1 的可提取近似）。

        依据：本仓库两个注意力实现（tri=TriRetrievalAttention（已移除旧全注意力占位））的"检索打分"都是
        query 对 key 的点积（tri 的 CSA 分支选择分数 = 压缩注意力分数 Softmax(q·K̃)，
        **非独立 indexer 模块**，见 tri_attention.py docstring）。故"CSA indexer 向量"在
        当前实现中无独立实体，最贴近的可提取来源是 **q_proj 的打分方向聚合**——
        q_proj: d_model → (n_q_heads·head_dim)，按 query 头聚合回 d_model 维"检索方向"，
        作为 HRL Indexer 的 warm-start。

        聚合方式：W_q [n_q*hd, d] → 按头分块取均值方向（对 d_model 各维取 |均值| 归一），
        得 [1, d_model]。这是**近似初始化**（诚实标注：非 §11.1 设想的独立 indexer，
        而是 query 打分方向的聚合；T2 仍须经块域 KL 对齐正式训练）。
        """
        W = q_proj_weight.detach().float()  # [n_q*hd, d]
        assert W.shape == (n_q_heads * head_dim, d_model), \
            f"q_proj 形状 {tuple(W.shape)} ≠ ({n_q_heads*head_dim}, {d_model})"
        # 按 query 头聚合：对 n_q 个头在 d_model 各维求均值方向，再归一
        per_head = W.view(n_q_heads, head_dim, d_model).mean(dim=1)  # [n_q, d]
        direction = per_head.mean(dim=0, keepdim=True)  # [1, d]
        direction = direction / (direction.norm() + 1e-6)
        with torch.no_grad():
            self.score.weight.copy_(direction.to(self.score.weight.dtype))


class DGProjection(nn.Module):
    """DG 模式分离投影（方案 B 内生；接口计划 §4.1 / C1）。

    route_key 稀疏化去相关防碰撞（潜空间几何各向异性的必要去相关，§15.2）。
    """

    def __init__(self, d_model: int, dg_dim: int, topk: int):
        super().__init__()
        self.dg_dim = dg_dim
        self.topk = topk
        self.proj = nn.Linear(d_model, dg_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [...,d] → 稀疏 key [...,dg_dim]（仅保留 top-k 激活，其余置零）。"""
        h = self.proj(x)
        if self.topk >= self.dg_dim:
            return h
        vals, idx = h.abs().topk(self.topk, dim=-1)
        mask = torch.zeros_like(h).scatter(-1, idx, 1.0)
        return h * mask


class SideChannelHeads(nn.Module):
    """侧信道头簇（方案 B 内生；接口计划 §4.2 / C6）。骨架含写显著性头；其余四头 M3 接。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.write_salience = nn.Linear(d_model, 1)   # 惊讶度 KL 阈值→W0 加标
        self.conflict = nn.Linear(d_model, 1)          # 块-上下文矛盾（远期占位）
        self.prefetch = nn.Linear(d_model, 1)          # 预取（远期占位）
        self.attribution = nn.Linear(d_model, 2)       # 注入质量/usage（远期占位）
        self.assoc = nn.Linear(d_model, 1)             # 联想触发（远期占位）

    def forward(self, pm_out: torch.Tensor) -> dict:
        return {
            "write_salience": self.write_salience(pm_out),
            "conflict": self.conflict(pm_out),
        }


class TAISKernel(nn.Module):
    """聚合 KAL + HRL 内生头（方案 B）。挂在主干 PM-stream 上，随 state_dict 存取。

    sense() / route() / inject() 三方法对应感知/路由/注入三通道；监测/执行分置由
    调用方（model.forward）保证读写不同层——sense 只读 GDN 输出层 PM-stream，
    inject 只写 CSA 残差前 PM-stream。
    """

    def __init__(self, d_model: int, dg_dim: int = 256, dg_topk: int = 32):
        super().__init__()
        self.kal_l1 = make_l1_head(d_model)      # L1 三态（P(IK)）
        self.kal_l2 = make_l2_head(d_model)      # L2 情感（valence/arousal）
        self.hrl_indexer = HRLIndexer(d_model)   # 统一打分头
        self.dg_proj = DGProjection(d_model, dg_dim, dg_topk)  # DG 模式分离
        self.side_heads = SideChannelHeads(d_model)  # 侧信道头簇

    def sense(self, pm_out: torch.Tensor) -> SenseOut:
        """读 GDN-MemBlock 输出处 PM-stream [B,T,d]，返只读信号（零副作用）。

        pm_out 应为 capture_layers 暴露的 PM-stream（S[..., -1, :]）或内容流
        （单流 checkpoint），由调用方按 stream 参数选择（kal.read_point 同语义）。
        """
        sc = self.side_heads(pm_out)
        return SenseOut(
            pik_logits=self.kal_l1(pm_out),
            affect_logits=self.kal_l2(pm_out),
            write_salience=sc["write_salience"],
            conflict_logit=sc["conflict"],
        )

    def route(self, query: torch.Tensor, detach_input: bool = True) -> RouteOut:
        """query [B,T,d] → RouteOut（DG 稀疏 key + Indexer 分数）。

        喂 runtime Pager 取候选块（M4 起用）；正式 top-k 分块归并在 runtime。
        detach_input 透传 Indexer 的梯度隔离开关（默认隔离主干）。
        """
        return RouteOut(
            sparse_key=self.dg_proj(query),
            score=self.hrl_indexer(query, detach_input=detach_input),
        )

    def route_candidates(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        k: int | None = None,
        detach_input: bool = True,
    ):
        """对候选块集合检索（真正的 CSA Indexer 路径，DSA lightning indexer 式）。

        query [B,Tq,d]（当前思考段），candidates [B,Tk,d]（候选块表示）；
        k=None 返回全部分数 [B,Tq,Tk]，否则返回 top-k (scores, idx)。
        token 域（压缩条目）/块域（知识块）同构——一个打分器两种检索对象（设计 §11.1）。
        detach_input 隔离主干（MoE-RL 红线）。
        """
        if k is None:
            return self.hrl_indexer.score_candidates(query, candidates, detach_input=detach_input)
        return self.hrl_indexer.topk_candidates(query, candidates, k, detach_input=detach_input)

    def indexer_kl_warmup_loss(self, query, candidates, teacher_scores):
        """HRL Indexer 的 KL warmup（T2，DSA warmup 范式，对齐稠密教师）。"""
        return self.hrl_indexer.kl_warmup_loss(query, candidates, teacher_scores)

    def load_indexer_from_csa(self, csa_indexer_weight: torch.Tensor) -> None:
        """用 CSA indexer 权重初始化 HRL Indexer（设计 §11.1 同构初始化）。"""
        self.hrl_indexer.load_from_csa_indexer(csa_indexer_weight)

    def init_indexer_from_model(self, model) -> int:
        """从主干的注意力层 q_proj 派生 HRL Indexer 初始化（设计 §11.1 的可提取近似）。

        取第一个注意力层（type "A"）的 q_proj，按头聚合出检索方向初始化 Indexer。
        返回初始化的层索引；无注意力层返回 -1（fail-closed，不初始化）。
        """
        for i, layer in enumerate(model.layers):
            if layer.type == "A":
                mixer = layer.mixer
                self.hrl_indexer.init_from_attention_qproj(
                    mixer.q_proj.weight, model.config.d_model,
                    mixer.n_q, mixer.head_dim,
                )
                return i
        return -1

    def inject(
        self,
        pm_pre: torch.Tensor,
        payloads: list[BlockPayload],
        alphas: list[float] | None = None,
        injector=None,
    ) -> torch.Tensor:
        """写 CSA-AttnBlock 残差前 PM-stream [B,T,d]：注入载荷。

        - 位置不变向量载体（icv/steering/concept_slot）：PM-stream 单次加法（steer 行为）。
        - token 寻址载体（kv/mem_entry/gist 等）：委托给 ``injector``（M5 的
          injection.Injector）——KV/gist 走 blockpath namespace 校验 + tri_attention
          HCA 拼接；mem_entry 走 memlayer 查询。**不给 injector 时 fail-closed 拒绝**
          （M5 起应传入以接通闭环）。
        注入即"紧邻检索层立刻参与注意力计算"（设计 §13.4 写点）。
        """
        if alphas is None:
            alphas = [1.0] * len(payloads)
        out = pm_pre
        for p, a in zip(payloads, alphas):
            if p.compiled_kind in VECTOR_KINDS:
                if p.vector is None:
                    raise ValueError(f"向量载体 {p.compiled_kind} 缺 vector 载荷")
                # 单次加法（位置不变偏移 = steer 行为，不做事实召回——载体能力边界）
                out = out + a * p.vector.to(out.dtype).unsqueeze(0).unsqueeze(0)
            elif p.compiled_kind in ADDRESSED_KINDS:
                if injector is None:
                    # 未接通 M5 注入器时 fail-closed（不静默注入 token 寻址载体）
                    raise NotImplementedError(
                        f"载体 {p.compiled_kind} 为 token 寻址注入，需传入 injector "
                        f"（M5 的 injection.Injector）以接通 KV 拼接/记忆层路径"
                    )
                # 委托 M5 注入器路由（KV/gist→blockpath 校验，mem_entry→memlayer）。
                # 向量型返回忽略（mem 查询/写入不走 PM-stream）；KV 条目由调用方拼入 HCA。
                injector.inject(p, namespace=p.layer_ns if p.layer_ns else None)
            else:
                raise ValueError(f"未知块载体类型: {p.compiled_kind!r}")
        return out


def make_kernel(d_model: int, **kw) -> TAISKernel:
    """工厂函数（M1 骨架）。"""
    return TAISKernel(d_model, **kw)
