"""页表 Block Spec（🟡 运行时数据，非权重）：块注册表 + 元数据（Part C3）。

设计依据：
- 部件实现详细计划 Part C3：BlockSpec 字段规范 + 内容寻址 + 双形态
  （markdown 源=ground truth，编译产物=可失效缓存）；⭐ Zep 双时态
  ``valid_at/ingested_at``。
- 接口与实现计划 v1.0 §4：页表走 SQLite；查询经 SQLite + 向量库。
- 🧠 海马索引（Teyler-DiScenna）：内容寻址 + 双时态。

纪律（fail-closed）：
- ``register()`` 拒绝未知 ``compiled_kind`` 的 spec（接口计划 §6 载体能力边界；
  未知载体一律拒收，不静默落库）。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field

# 已知块载体类型（接口计划 §6 载体能力边界；与 model/tais_kernel.py 的
# VECTOR_KINDS | ADDRESSED_KINDS 并集一致，本包为运行时数据侧，独立维护以防循环依赖）
KNOWN_KINDS: frozenset = frozenset(
    {"kv", "mem_entry", "icv", "steering", "concept_slot", "lora", "gist", "route"}
)


@dataclass
class BlockSpec:
    """页表 Block Spec（Part C3；⭐ Zep 双时态）。

    运行时元数据（非权重）。markdown 源为 ground truth（审计/回滚依据），
    编译产物可失效重建。``factual_recall`` 为载体能力边界标注（token 寻址可事实召回，
    位置不变向量只能 steer 行为），须与 compiled_kind 一致。
    """

    block_id: str
    route_key: str
    affect: dict = field(default_factory=dict)      # {valence, arousal, saliency}
    temporal_ctx: tuple = ()                        # 时间上下文（占位，序列化为 JSON）
    spatial_coord: tuple | None = None              # 空间坐标（可选）
    namespace: tuple = ()                           # namespace 五元组
    version: int = 1
    signature: bytes = b""
    ttl: float = float("inf")
    usage_count: int = 0
    compiled_kind: str = "kv"
    factual_recall: bool = True
    merged_flag: bool = False
    valid_at: float = field(default_factory=time.time)      # ⭐ Zep 双时态：有效时间
    ingested_at: float = field(default_factory=time.time)   # ⭐ Zep 双时态：入库时间


class PageTable:
    """页表（SQLite 后端）。默认内存库 ``:memory:``；给路径则文件后端（持久化）。

    仅做元数据 CRUD + 内容寻址查询；不存块载荷（载荷走 BlockStore）。
    fail-closed：未知 compiled_kind 的 spec 一律拒收（返回 False，不抛给调用方）。
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS blocks (
        block_id TEXT PRIMARY KEY,
        route_key TEXT,
        affect TEXT,
        temporal_ctx TEXT,
        spatial_coord TEXT,
        namespace TEXT,
        version INTEGER,
        signature BLOB,
        ttl REAL,
        usage_count INTEGER,
        compiled_kind TEXT,
        factual_recall INTEGER,
        merged_flag INTEGER,
        valid_at REAL,
        ingested_at REAL
    )
    """

    def __init__(self, path: str | None = None):
        self._conn = sqlite3.connect(path or ":memory:")
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def register(self, spec: BlockSpec) -> bool:
        """注册块元数据。fail-closed：未知 compiled_kind → 返回 False，不落库。"""
        if spec.compiled_kind not in KNOWN_KINDS:
            return False
        self._conn.execute(
            "INSERT OR REPLACE INTO blocks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                spec.block_id,
                spec.route_key,
                json.dumps(spec.affect, ensure_ascii=False),
                json.dumps(list(spec.temporal_ctx)),
                json.dumps(list(spec.spatial_coord) if spec.spatial_coord is not None else None),
                json.dumps(list(spec.namespace)),
                spec.version,
                spec.signature,
                spec.ttl,
                spec.usage_count,
                spec.compiled_kind,
                int(spec.factual_recall),
                int(spec.merged_flag),
                spec.valid_at,
                spec.ingested_at,
            ),
        )
        self._conn.commit()
        return True

    def get(self, block_id: str) -> BlockSpec | None:
        """按 block_id 取元数据；不存在返回 None（fail-closed，不抛）。"""
        row = self._conn.execute(
            "SELECT * FROM blocks WHERE block_id = ?", (block_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_spec(row)

    def update_usage(self, block_id: str, delta: int = 1) -> None:
        """usage_count 自增（归因/淘汰 hint 用）。"""
        self._conn.execute(
            "UPDATE blocks SET usage_count = usage_count + ? WHERE block_id = ?",
            (delta, block_id),
        )
        self._conn.commit()

    def query_by_route_key(self, substr: str) -> list[BlockSpec]:
        """内容寻址：route_key 子串匹配（骨架版；正式向量检索在 M5+）。"""
        rows = self._conn.execute(
            "SELECT * FROM blocks WHERE route_key LIKE ?", (f"%{substr}%",)
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    def list_pending_promotion(self, min_usage: int) -> list[BlockSpec]:
        """列出 usage_count >= min_usage 且未 merged 的块（CA1 升格候选）。"""
        rows = self._conn.execute(
            "SELECT * FROM blocks WHERE usage_count >= ? AND merged_flag = 0",
            (min_usage,),
        ).fetchall()
        return [self._row_to_spec(r) for r in rows]

    @staticmethod
    def _row_to_spec(row) -> BlockSpec:
        # schema 列序：0 block_id,1 route_key,2 affect,3 temporal_ctx,4 spatial_coord,
        # 5 namespace,6 version,7 signature,8 ttl,9 usage_count,10 compiled_kind,
        # 11 factual_recall,12 merged_flag,13 valid_at,14 ingested_at
        sc = json.loads(row[4])
        return BlockSpec(
            block_id=row[0],
            route_key=row[1],
            affect=json.loads(row[2]),
            temporal_ctx=tuple(json.loads(row[3])),
            spatial_coord=tuple(sc) if sc is not None else None,
            namespace=tuple(json.loads(row[5])),
            version=row[6],
            signature=row[7] or b"",
            ttl=row[8],
            usage_count=row[9],
            compiled_kind=row[10],
            factual_recall=bool(row[11]),
            merged_flag=bool(row[12]),
            valid_at=row[13],
            ingested_at=row[14],
        )
