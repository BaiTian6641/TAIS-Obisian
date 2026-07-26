"""M8 安全管线单元测试：签名、扫描器、CA1 门编排、fail-closed。

判据（接口与实现计划 v1.0 §10 / 部件实现详细计划 Part H）：
- sign_block/verify_signature HMAC 往返；篡改 fail-closed；
- SafetyPipeline：签名无效 REJECT、投毒 DROP、CA1 门 QUARANTINE/PROMOTE；
- 扫描器回调接入；恒定时间比对（防时序侧信道）。
"""
from __future__ import annotations

import pytest

from tais_obsidian.runtime import (
    SafetyPipeline,
    make_safety_pipeline,
    sign_block,
    verify_signature,
)

SECRET = b"tais-secret-key"
PAYLOAD = b"block payload bytes"


def test_sign_and_verify_roundtrip() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    assert verify_signature(PAYLOAD, sig, SECRET) is True


def test_verify_tampered_fail_closed() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    assert verify_signature(PAYLOAD + b"x", sig, SECRET) is False   # 载荷篡改
    assert verify_signature(PAYLOAD, sig, b"wrong-secret") is False  # 密钥错误
    assert verify_signature(PAYLOAD, sig[:-1] + b"0", SECRET) is False  # 签名篡改


def test_pipeline_signature_invalid_reject() -> None:
    sp = make_safety_pipeline(SECRET)
    out = sp.check(PAYLOAD, b"bad-signature", usage_count=99)
    assert out["ok"] is False and out["reason"] == "signature_invalid"


def test_pipeline_poison_detected_drop() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    sp = make_safety_pipeline(SECRET, scanner_fn=lambda p: False)  # 扫描器报警
    out = sp.check(PAYLOAD, sig, usage_count=99)
    assert out["ok"] is False and out["verdict"] == "DROP" and out["reason"] == "poison_detected"


def test_pipeline_belief_drift_quarantine() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    sp = make_safety_pipeline(SECRET, scanner_fn=lambda p: True)  # 扫描器干净
    out = sp.check(PAYLOAD, sig, usage_count=99, belief_drift=0.9)  # 信念漂移超阈
    assert out["ok"] is False and out["verdict"] == "QUARANTINE"


def test_pipeline_clean_promote() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    sp = make_safety_pipeline(SECRET, scanner_fn=lambda p: True)
    out = sp.check(PAYLOAD, sig, usage_count=99, teacher_consensus=0.9, belief_drift=0.1)
    assert out["ok"] is True and out["verdict"] == "PROMOTE"


def test_pipeline_no_scanner_still_gated_by_ca1() -> None:
    sig = sign_block(PAYLOAD, SECRET)
    sp = SafetyPipeline(SECRET)  # 无扫描器
    out = sp.check(PAYLOAD, sig, usage_count=1)  # usage 不足 → CA1 REJECT
    assert out["ok"] is False and out["verdict"] == "REJECT"
