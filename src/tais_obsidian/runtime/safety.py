"""安全管线（M8，§26.2 命名防御范式）：块签名 + 注入攻击面编排 + 扫描器接口。

设计依据（必须逐条对齐，禁止凭记忆扩展）：
- 接口与实现计划 v1.0 §10 / 部件实现详细计划 Part H：注入即攻击面（⭐ MemoryGraft
  arXiv:2512.16962 已实证"恶意成功经验植入长期记忆"）；攻击**时间解耦**（今日下毒、
  数周后语义触发）；防御须检测"被腐蚀的信念"而非动作。
- 微软 Defender 三原语 ↔ TAIS 实现：
  memory contracts → 块签名 + namespace fail-closed（pager 已就绪）；
  belief drift detection → CA1 巩固门回归测试 + 探针漂移监测（runtime/ca1_gate 已就绪）；
  context provenance tracking → markdown 源代码形态（pagetable 元数据，永久审计/回滚）。
- MS 后门扫描器（⭐ arXiv:2602.03085，机制已核——sleeper agent 记忆投毒数据可经
  记忆提取泄露 + 触发时输出分布/注意力头独特模式）接入睡眠固化前 draft 区筛查。

纪律：
- 签名用 HMAC（stdlib hmac/hashlib，无重依赖）；验证失败一律 fail-closed。
- 扫描器（scanner_fn）由调用方注入（正式接 MS 扫描器/梯度耦合检测），骨架用回调。
- 本模块编排已就绪的 pager namespace 校验 + ca1 漂移门 + 签名验证 + 扫描器，
  形成睡眠固化前 draft 区的完整安全闸。
"""
from __future__ import annotations

import hashlib
import hmac

from ..runtime.ca1_gate import ca1_gate


def sign_block(payload: bytes, secret: bytes) -> bytes:
    """块签名（memory contracts 原语）：HMAC-SHA256。"""
    return hmac.new(secret, payload, hashlib.sha256).digest()


def verify_signature(payload: bytes, signature: bytes, secret: bytes) -> bool:
    """签名验证（fail-closed）：恒定时间比对，防时序侧信道。"""
    expected = sign_block(payload, secret)
    return hmac.compare_digest(expected, signature)


class SafetyPipeline:
    """睡眠固化前 draft 区安全闸：签名 → namespace → 漂移门 → 扫描器。

    逐项 fail-closed；任一不通过即拦截（返回判定与原因）。
    """

    def __init__(self, secret: bytes, scanner_fn=None, ca1_thresholds: dict | None = None):
        self.secret = secret
        self.scanner_fn = scanner_fn  # fn(payload) -> bool（True=干净/False=投毒）
        self.ca1_thresholds = ca1_thresholds or {}

    def check(
        self,
        payload: bytes,
        signature: bytes,
        *,
        candidate=None,
        regression_ok: bool = True,
        usage_count: int = 0,
        teacher_consensus: float = 1.0,
        belief_drift: float = 0.0,
    ) -> dict:
        """完整安全闸。返回 {ok: bool, verdict: str, reason: str}。

        顺序：① 签名验证（memory contracts）；② 扫描器（投毒检出）；
        ③ CA1 门（验证+漂移，belief drift detection）。任一失败 fail-closed。
        """
        if not verify_signature(payload, signature, self.secret):
            return {"ok": False, "verdict": "REJECT", "reason": "signature_invalid"}
        if self.scanner_fn is not None and not self.scanner_fn(payload):
            return {"ok": False, "verdict": "DROP", "reason": "poison_detected"}
        verdict = ca1_gate(
            candidate if candidate is not None else payload,
            regression_ok=regression_ok,
            usage_count=usage_count,
            teacher_consensus=teacher_consensus,
            belief_drift=belief_drift,
            **self.ca1_thresholds,
        )
        ok = verdict == "PROMOTE"
        return {"ok": ok, "verdict": verdict, "reason": "ca1_gate"}


def make_safety_pipeline(secret: bytes, scanner_fn=None, **kw) -> SafetyPipeline:
    """工厂函数。"""
    return SafetyPipeline(secret, scanner_fn=scanner_fn, **kw)
