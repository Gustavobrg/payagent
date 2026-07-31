"""Tests for the simulated merchant: state, idempotency, and driveable failure modes.

The most valuable case in this file is `TIMEOUT_AFTER_COMMIT` followed by a retry on the same
key: the charge landed but the response was lost, which is the real-world failure that produces
double charges. Exactly one charge must exist afterwards.

Nothing here sleeps. What is under test is the *handling* of a failure — the right exception,
no charge recorded, state unchanged — none of which depends on elapsed wall time.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from payagent.mcp_server.merchant_sim import (
    MerchantBehavior,
    MerchantDeclined,
    MerchantIdempotencyConflict,
    MerchantSim,
    MerchantTimeout,
    MerchantUnavailable,
)
from payagent.policy import EffectGrant, PolicyAction

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

MERCHANT_SIM_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "payagent" / "mcp_server" / "merchant_sim.py"
)


def _counter_id_factory():
    counter = {"n": 0}

    def factory(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:016d}"

    return factory


def _merchant(**kwargs) -> MerchantSim:
    kwargs.setdefault("clock", lambda: FIXED_NOW)
    kwargs.setdefault("id_factory", _counter_id_factory())
    return MerchantSim(**kwargs)


def _grant(action: PolicyAction = PolicyAction.EXECUTE_SETTLEMENT, amount_cents: int = 4999):
    return EffectGrant(
        action=action,
        amount_cents=amount_cents,
        currency="BRL",
        merchant_id="MERCH-02",
        decided_at=FIXED_NOW,
    )


def _settle(merchant: MerchantSim, *, key: str = "idem-key-settle-0001", amount_cents: int = 4999):
    return merchant.settle(
        _grant(amount_cents=amount_cents),
        idempotency_key=key,
        payment_mandate_id="PM-0000000000000002",
        amount_cents=amount_cents,
        currency="BRL",
        merchant_id="MERCH-02",
    )


def test_accept_records_exactly_one_charge():
    merchant = _merchant()

    charge = _settle(merchant)

    assert charge.amount_cents == 4999
    assert charge.currency == "BRL"
    assert len(merchant.charges) == 1
    assert merchant.charges[charge.charge_id] == charge


def test_same_key_twice_yields_one_charge_and_the_same_id():
    merchant = _merchant()

    first = _settle(merchant)
    second = _settle(merchant)

    assert second.charge_id == first.charge_id
    assert len(merchant.charges) == 1


def test_same_key_with_a_different_amount_is_a_conflict_and_records_nothing_new():
    """The merchant is a second, independent dedup layer — a dispatch-layer bug still can't double-charge."""
    merchant = _merchant()
    _settle(merchant, amount_cents=4999)

    with pytest.raises(MerchantIdempotencyConflict):
        _settle(merchant, amount_cents=500000)

    assert len(merchant.charges) == 1


def test_different_keys_with_the_same_arguments_produce_two_charges():
    merchant = _merchant()

    _settle(merchant, key="idem-key-settle-0001")
    _settle(merchant, key="idem-key-settle-0002")

    assert len(merchant.charges) == 2


@pytest.mark.parametrize(
    ("behavior", "expected"),
    [
        (MerchantBehavior.DECLINE, MerchantDeclined),
        (MerchantBehavior.TIMEOUT, MerchantTimeout),
        (MerchantBehavior.UNAVAILABLE, MerchantUnavailable),
    ],
)
def test_failure_behaviors_raise_and_record_no_charge(behavior, expected):
    merchant = _merchant(behavior=behavior)

    with pytest.raises(expected):
        _settle(merchant)

    assert merchant.charges == {}


def test_timeout_after_commit_records_the_charge_then_raises():
    merchant = _merchant(behavior=MerchantBehavior.TIMEOUT_AFTER_COMMIT)

    with pytest.raises(MerchantTimeout):
        _settle(merchant)

    assert len(merchant.charges) == 1


def test_retry_after_a_lost_response_charges_exactly_once():
    """The nastiest real case: the charge landed, the response didn't. A retry must not double it."""
    merchant = _merchant(
        scripted=[MerchantBehavior.TIMEOUT_AFTER_COMMIT, MerchantBehavior.ACCEPT]
    )

    with pytest.raises(MerchantTimeout):
        _settle(merchant, key="idem-key-settle-0001")

    charge = _settle(merchant, key="idem-key-settle-0001")

    assert len(merchant.charges) == 1
    assert charge.amount_cents == 4999


def test_behavior_by_merchant_isolates_merchants():
    """The POL-005 allowlist scenario: one merchant declines while another accepts."""
    merchant = _merchant(
        behavior_by_merchant={"MERCH-99": MerchantBehavior.DECLINE},
    )

    charge = _settle(merchant, key="idem-key-settle-0001")
    assert charge.merchant_id == "MERCH-02"

    with pytest.raises(MerchantDeclined):
        merchant.settle(
            _grant(),
            idempotency_key="idem-key-settle-0002",
            payment_mandate_id="PM-0000000000000009",
            amount_cents=4999,
            currency="BRL",
            merchant_id="MERCH-99",
        )


def test_scripted_behaviors_are_consumed_in_order_then_fall_back_to_the_default():
    merchant = _merchant(
        behavior=MerchantBehavior.ACCEPT,
        scripted=[MerchantBehavior.DECLINE],
    )

    with pytest.raises(MerchantDeclined):
        _settle(merchant, key="idem-key-settle-0001")

    charge = _settle(merchant, key="idem-key-settle-0002")
    assert charge.amount_cents == 4999


def test_refund_reverses_the_full_charge():
    merchant = _merchant()
    charge = _settle(merchant)

    refund = merchant.refund(
        _grant(action=PolicyAction.REFUND),
        idempotency_key="idem-key-refund-0001",
        charge_id=charge.charge_id,
        reason_code="defective",
    )

    assert refund.amount_cents == charge.amount_cents
    assert refund.currency == charge.currency
    assert merchant.is_refunded(charge.charge_id) is True


def test_a_second_refund_on_the_same_charge_is_rejected():
    """Full-refund-only means a repeat is a state error, never a summed partial."""
    merchant = _merchant()
    charge = _settle(merchant)
    merchant.refund(
        _grant(action=PolicyAction.REFUND),
        idempotency_key="idem-key-refund-0001",
        charge_id=charge.charge_id,
        reason_code="defective",
    )

    with pytest.raises(ValueError, match="already refunded"):
        merchant.refund(
            _grant(action=PolicyAction.REFUND),
            idempotency_key="idem-key-refund-0002",
            charge_id=charge.charge_id,
            reason_code="duplicate_charge",
        )


def test_refund_replayed_with_the_same_key_returns_the_same_refund():
    merchant = _merchant()
    charge = _settle(merchant)

    first = merchant.refund(
        _grant(action=PolicyAction.REFUND),
        idempotency_key="idem-key-refund-0001",
        charge_id=charge.charge_id,
        reason_code="defective",
    )
    second = merchant.refund(
        _grant(action=PolicyAction.REFUND),
        idempotency_key="idem-key-refund-0001",
        charge_id=charge.charge_id,
        reason_code="defective",
    )

    assert second.refund_id == first.refund_id


def test_refund_of_an_unknown_charge_is_rejected():
    merchant = _merchant()

    with pytest.raises(KeyError):
        merchant.refund(
            _grant(action=PolicyAction.REFUND),
            idempotency_key="idem-key-refund-0001",
            charge_id="CH-doesnotexist",
            reason_code="defective",
        )


def test_charges_view_is_read_only():
    """A test (or a handler) must not be able to fabricate merchant state through the view."""
    merchant = _merchant()
    _settle(merchant)

    with pytest.raises(TypeError):
        merchant.charges["CH-forged"] = None  # type: ignore[index]


def test_calls_log_records_every_attempt_including_failures():
    merchant = _merchant(behavior=MerchantBehavior.DECLINE)

    with pytest.raises(MerchantDeclined):
        _settle(merchant)

    assert len(merchant.calls) == 1
    assert merchant.calls[0]["outcome"] == "declined"


def test_merchant_sim_never_sleeps():
    """Simulated latency would test the clock, not the code — and would make CI flaky."""
    tree = ast.parse(MERCHANT_SIM_PATH.read_text(encoding="utf-8"))
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }

    assert "time" not in imported
    assert "asyncio" not in imported
    assert "anyio" not in imported
