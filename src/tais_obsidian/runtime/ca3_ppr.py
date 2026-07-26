"""CA3 PPR 联想检索（🟡 运行时图算法）：块图上 Personalized PageRank 扩散。

设计依据：
- 部件实现详细计划 Part C4 / 接口与实现计划 v1.0 §4：⭐ HippoRAG 式 Personalized
  PageRank 在块图上扩散（ε≈0.1），由内生 Indexer 分数作种子；多跳/类比联想
  （HippoRAG +20%）。🧠 CA3 自动联想。
- 纯 python 实现（无 numpy），骨架版用幂迭代。

种子语义：``seed_scores`` 为 Indexer 打分的 {block_id: score}，PPR 把它按
(1-alpha) 沿图边扩散、alpha 回流种子，得到全可达块的扩展分数。
"""
from __future__ import annotations


def ca3_ppr(
    seed_scores: dict[str, float],
    graph: dict[str, list[str]],
    alpha: float = 0.1,
    iters: int = 20,
) -> dict[str, float]:
    """Personalized PageRank（HippoRAG 式），种子为 Indexer 分数。

    :param seed_scores: 种子分数 {block_id: score}（内部 Indexer 输出，自动归一化）。
    :param graph: 邻接表 {block_id: [后继 block_id, ...]}（无出边节点视为悬挂，回流种子）。
    :param alpha: 回流（重启）概率 ε≈0.1。
    :param iters: 幂迭代轮数。
    :return: 扩展分数 {block_id: score}，覆盖所有可达块（含种子与邻居）。
    """
    nodes = set(graph.keys())
    for nbrs in graph.values():
        nodes.update(nbrs)
    nodes.update(seed_scores.keys())
    if not nodes:
        return {}

    # 种子归一化为个性化分布
    total = sum(max(v, 0.0) for v in seed_scores.values())
    if total <= 0:
        pers = {n: 1.0 / len(nodes) for n in nodes}
    else:
        pers = {n: max(seed_scores.get(n, 0.0), 0.0) / total for n in nodes}

    rank = dict(pers)
    for _ in range(iters):
        nxt = {n: alpha * pers[n] for n in nodes}
        for u in nodes:
            nbrs = graph.get(u, [])
            share = rank[u] * (1.0 - alpha)
            if nbrs:
                w = share / len(nbrs)
                for v in nbrs:
                    nxt[v] += w
            else:
                # 悬挂节点：质量均摊回全体（等价回流）
                w = share / len(nodes)
                for v in nodes:
                    nxt[v] += w
        rank = nxt
    return rank
