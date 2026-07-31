"""Tests for mandate records, the authority seam, and the store.

Two properties carry this module. First, **P3**: no mandate projection that can leave the
server has a field capable of holding cryptographic material — asserted against the field
names themselves, so adding one breaks the suite. Second, **I3**: the only authority shipped
in `src/` cannot verify anything, so no code path in `src/` can settle. That is checked
exhaustively rather than by a single happy case, because "accidentally becomes fail-open" is
the exact failure mode a stub authority has.
"""

from __future__ import annotations

from dataclasses import fields
from datetime import UTC, datetime, timedelta

import pytest

from payagent.mandates import (
    InMemoryMandateStore,
    IntentMandate,
    MandateSummary,
    MandateVerification,
    PaymentMandate,
    UnsignedMandateAuthority,
    summarize,
)

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

# Substrings that would indicate a field able to carry signing material or key state.
CRYPTO_FIELD_MARKERS = ("signature", "jws", "jwt", "key", "secret", "private", "sign", "payload_b64")


def _intent(**overrides) -> IntentMandate:
    base = {
        "intent_mandate_id": "IM-0000000000000001",
        "purpose": "Buy a bluetooth speaker",
        "max_amount_cents": 50000,
        "currency": "BRL",
        "allowed_categories": ("electronics",),
        "allowed_merchant_ids": ("MERCH-02",),
        "issued_at": FIXED_NOW,
        "expires_at": FIXED_NOW + timedelta(hours=1),
    }
    return IntentMandate(**{**base, **overrides})


def _payment(**overrides) -> PaymentMandate:
    base = {
        "payment_mandate_id": "PM-0000000000000002",
        "intent_mandate_id": "IM-0000000000000001",
        "quote_id": "QT-00000000000000000000000000000003",
        "amount_cents": 4999,
        "currency": "BRL",
        "merchant_id": "MERCH-02",
        "sku": "SKU-SPEAKER",
        "quantity": 1,
        "issued_at": FIXED_NOW,
        "expires_at": FIXED_NOW + timedelta(minutes=30),
    }
    return PaymentMandate(**{**base, **overrides})


@pytest.mark.parametrize("model", [IntentMandate, PaymentMandate, MandateSummary])
def test_no_mandate_model_has_a_field_that_could_hold_crypto_material(model):
    """P3 expressed in the type system: there is no field for signing material to leak through."""
    for field_def in fields(model):
        lowered = field_def.name.lower()
        for marker in CRYPTO_FIELD_MARKERS:
            assert marker not in lowered, f"{model.__name__}.{field_def.name} looks like crypto material"


def test_payment_mandate_requires_a_parent_intent_mandate():
    """POL-009 §3: without a valid parent reference the mandate must be rejected."""
    with pytest.raises(ValueError, match="intent_mandate_id"):
        _payment(intent_mandate_id="")


def test_payment_mandate_rejects_a_float_amount():
    with pytest.raises(TypeError, match="int"):
        _payment(amount_cents=49.99)


def test_intent_mandate_rejects_a_float_ceiling():
    with pytest.raises(TypeError, match="int"):
        _intent(max_amount_cents=500.0)


def test_mandate_expiry_must_be_after_issuance():
    with pytest.raises(ValueError, match="expires_at"):
        _payment(expires_at=FIXED_NOW - timedelta(seconds=1))


def test_summarize_payment_mandate_is_human_readable_and_carries_no_ids_beyond_its_own():
    summary = summarize(_payment())

    assert isinstance(summary, MandateSummary)
    assert summary.kind == "payment"
    assert summary.mandate_id == "PM-0000000000000002"
    assert summary.amount_cents == 4999
    assert "49.99" in summary.scope
    assert "SKU-SPEAKER" in summary.scope


def test_summarize_intent_mandate_states_the_ceiling_and_scope():
    summary = summarize(_intent())

    assert summary.kind == "intent"
    assert summary.amount_cents is None  # an intent authorizes a ceiling, not an amount
    assert "500.00" in summary.scope
    assert "electronics" in summary.scope


@pytest.mark.parametrize(
    "now",
    [
        FIXED_NOW,  # well within validity
        FIXED_NOW + timedelta(minutes=1),
        FIXED_NOW + timedelta(hours=99),  # long expired
    ],
    ids=["valid", "still-valid", "expired"],
)
def test_unsigned_authority_never_verifies_any_mandate(now: datetime):
    """I3 by refusal: a well-formed, unexpired, amount-matching mandate still fails verification.

    Parametrized on purpose — a single case could pass because of an incidental detail, and the
    property being asserted is that *no* input produces `verified=True`.
    """
    verification = UnsignedMandateAuthority().verify_for_settlement(_payment(), now=now)

    assert isinstance(verification, MandateVerification)
    assert verification.verified is False
    assert verification.code == "MANDATE_SIGNATURE_UNVERIFIED"


def test_unsigned_authority_issues_the_record_it_was_given():
    authority = UnsignedMandateAuthority()
    intent = _intent()

    assert authority.issue_intent(intent) == intent
    assert authority.issue_payment(_payment()) == _payment()


def test_store_round_trips_both_mandate_kinds():
    store = InMemoryMandateStore()
    intent, payment = _intent(), _payment()

    store.put_intent(intent)
    store.put_payment(payment)

    assert store.get_intent(intent.intent_mandate_id) == intent
    assert store.get_payment(payment.payment_mandate_id) == payment


def test_store_returns_none_for_an_unknown_id_rather_than_raising():
    """Callers turn a miss into a structured tool error; an exception here would be echoed instead."""
    store = InMemoryMandateStore()

    assert store.get_intent("IM-doesnotexist") is None
    assert store.get_payment("PM-doesnotexist") is None
