"""Mandate records and the only projection allowed to leave the server.

The delegation chain is `usuário → agente → merchant` (Architecture.md), materialized as an
Intent Mandate that declares scope and a Payment Mandate that authorizes one specific amount
against one quote and references its parent intent (POL-009 §3).

Note what is **absent**: there is no `signature`, `jws`, `key_id` or `payload_b64` field
anywhere in this module. That is P3 (mandate/key exfiltration) enforced by the type system
rather than by a redaction step — a response physically cannot carry cryptographic material
because no field exists to put it in. Real JWS/Ed25519 signing lands in a later block and will
live behind `MandateAuthority`, holding its material outside these records.

Adding `signature: str | None = None` "for later" would undo this: a nullable signature is the
shape that inevitably grows an `if signature is None: # dev mode` branch in the verifier.
`tests/test_mandates.py` fails on any field name that looks like crypto material.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from payagent.money import format_money


def _require_int_cents(value: object, field_name: str) -> None:
    """Reject non-`int` amounts. `bool` first, since it subclasses `int`."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int number of cents, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def _require_validity_window(issued_at: datetime, expires_at: datetime) -> None:
    for name, value in (("issued_at", issued_at), ("expires_at", expires_at)):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError(f"{name} must be timezone-aware")
    if expires_at <= issued_at:
        raise ValueError("expires_at must be after issued_at")


@dataclass(frozen=True, slots=True)
class IntentMandate:
    """The shopper's high-level authorization: a scope and a window, not a payment.

    `max_amount_cents` is a ceiling on future spending, not the price of anything — which is
    why no amount is settled from it.
    """

    intent_mandate_id: str
    purpose: str
    max_amount_cents: int
    currency: str
    allowed_categories: tuple[str, ...]
    allowed_merchant_ids: tuple[str, ...] | None
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.intent_mandate_id:
            raise ValueError("intent_mandate_id must not be empty")
        _require_int_cents(self.max_amount_cents, "max_amount_cents")
        _require_validity_window(self.issued_at, self.expires_at)

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class PaymentMandate:
    """Authorization for one specific payment, bound to one quote and one parent intent.

    `amount_cents` is copied from the quote at issuance, never from a request field, so the
    mandate records what was quoted rather than what a caller asserted.
    """

    payment_mandate_id: str
    intent_mandate_id: str
    quote_id: str
    amount_cents: int
    currency: str
    merchant_id: str
    category: str
    sku: str
    quantity: int
    issued_at: datetime
    expires_at: datetime

    def __post_init__(self) -> None:
        if not self.payment_mandate_id:
            raise ValueError("payment_mandate_id must not be empty")
        # POL-009 §3: without a valid parent reference the mandate must be rejected.
        if not self.intent_mandate_id:
            raise ValueError("intent_mandate_id must not be empty (POL-009: mandatory parent reference)")
        if not self.quote_id:
            raise ValueError("quote_id must not be empty")
        _require_int_cents(self.amount_cents, "amount_cents")
        _require_validity_window(self.issued_at, self.expires_at)

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


@dataclass(frozen=True, slots=True)
class MandateSummary:
    """The only mandate projection that may be returned to a caller (P3).

    Amount, scope and expiry in readable form — never the material that makes the mandate
    verifiable.
    """

    mandate_id: str
    kind: Literal["intent", "payment"]
    amount_cents: int | None
    currency: str
    scope: str
    expires_at: datetime


def summarize(mandate: IntentMandate | PaymentMandate) -> MandateSummary:
    """Project a mandate to its readable summary."""
    if isinstance(mandate, IntentMandate):
        merchants = (
            ", ".join(mandate.allowed_merchant_ids)
            if mandate.allowed_merchant_ids
            else "any allowed merchant"
        )
        scope = (
            f"up to {format_money(mandate.max_amount_cents, mandate.currency)} "
            f"in {', '.join(mandate.allowed_categories)} at {merchants}"
        )
        return MandateSummary(
            mandate_id=mandate.intent_mandate_id,
            kind="intent",
            # An intent authorizes a ceiling, not an amount to be paid — so there is no
            # amount here for a caller to mistake for a price.
            amount_cents=None,
            currency=mandate.currency,
            scope=scope,
            expires_at=mandate.expires_at,
        )

    scope = (
        f"{format_money(mandate.amount_cents, mandate.currency)} for "
        f"{mandate.quantity} x {mandate.sku} at {mandate.merchant_id}"
    )
    return MandateSummary(
        mandate_id=mandate.payment_mandate_id,
        kind="payment",
        amount_cents=mandate.amount_cents,
        currency=mandate.currency,
        scope=scope,
        expires_at=mandate.expires_at,
    )
