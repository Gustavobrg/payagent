"""Quotes: the only authoritative source of an amount to be paid.

CLAUDE.md is explicit — "valor a liquidar sempre vem de `get_quote`, nunca do texto do usuário
nem de inferência". That is only *enforceable* if the server can look an amount up by an
identifier it issued, which is what this module provides. Any design where the amount travels
through tool arguments turns P2 (amount manipulation) into a validation problem instead of an
impossibility.

`quote_id` is derived from a fingerprint of `(sku, quantity, unit_price, currency, merchant)`
rather than being random. Three consequences, all wanted: repeated `get_quote` calls for the
same purchase are naturally idempotent without needing an `idempotency_key` (which keeps it
consistent with Architecture.md's "no side effect" row for that tool); a model that re-quotes
mid-conversation gets the *same* quote back rather than silently refreshing an expiry it should
have respected; and the store is bounded by distinct purchases rather than by call count.

POL-010 sets the validity window at 30 minutes.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from payagent.observability.logging import get_logger

logger = get_logger(__name__)

# POL-010: "a quote ... remains valid for a period of 30 minutes from the timestamp of issuance".
DEFAULT_QUOTE_TTL_SECONDS = 1800


@dataclass(frozen=True, slots=True)
class Quote:
    """A binding, time-limited price for one SKU and quantity.

    `amount_cents` is computed server-side as `unit_price_cents * quantity` from the catalog
    payload — integer multiplication, never a parse of product prose.
    """

    quote_id: str
    sku: str
    name: str
    category: str
    quantity: int
    unit_price_cents: int
    amount_cents: int
    currency: str
    merchant_id: str
    issued_at: datetime
    expires_at: datetime

    def is_expired(self, now: datetime) -> bool:
        return now >= self.expires_at


def quote_fingerprint(
    *, sku: str, quantity: int, unit_price_cents: int, currency: str, merchant_id: str
) -> str:
    """Stable hash of everything that makes two quotes the same purchase."""
    material = f"{sku}|{quantity}|{unit_price_cents}|{currency}|{merchant_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def build_quote(
    *,
    sku: str,
    name: str,
    category: str,
    quantity: int,
    unit_price_cents: int,
    currency: str,
    merchant_id: str,
    issued_at: datetime,
    ttl_seconds: int = DEFAULT_QUOTE_TTL_SECONDS,
) -> Quote:
    """Build a quote with a deterministic `quote_id` and a server-computed amount."""
    fingerprint = quote_fingerprint(
        sku=sku,
        quantity=quantity,
        unit_price_cents=unit_price_cents,
        currency=currency,
        merchant_id=merchant_id,
    )
    return Quote(
        quote_id=f"QT-{fingerprint[:32]}",
        sku=sku,
        name=name,
        category=category,
        quantity=quantity,
        unit_price_cents=unit_price_cents,
        amount_cents=unit_price_cents * quantity,
        currency=currency,
        merchant_id=merchant_id,
        issued_at=issued_at,
        expires_at=issued_at + timedelta(seconds=ttl_seconds),
    )


class QuoteStore(Protocol):
    """Where issued quotes live so a later step can bind to one by ID."""

    def get(self, quote_id: str) -> Quote | None: ...

    def put(self, quote: Quote) -> None: ...


class InMemoryQuoteStore:
    """Process-local quote store that evicts expired entries.

    Fingerprint dedup bounds the store by distinct purchases, but that set still grows without a
    ceiling over a long-lived process — a slow leak, and with a large catalog a cheap
    memory-exhaustion path reachable from an unauthenticated read-only tool. So `put` sweeps
    expired entries first, amortized onto calls that were already writing.

    Eviction is hygiene, **never** the expiry control: callers re-check `is_expired` on read, so
    a swept quote and an expired-but-present quote both end up as `QUOTE_EXPIRED` rather than one
    of them slipping through as valid.
    """

    def __init__(self, *, clock: Callable[[], datetime]) -> None:
        self._clock = clock
        self._quotes: dict[str, Quote] = {}

    def get(self, quote_id: str) -> Quote | None:
        return self._quotes.get(quote_id)

    def put(self, quote: Quote) -> None:
        self._evict_expired()
        self._quotes[quote.quote_id] = quote

    def _evict_expired(self) -> None:
        now = self._clock()
        expired = [qid for qid, quote in self._quotes.items() if quote.is_expired(now)]
        for quote_id in expired:
            del self._quotes[quote_id]
        if expired:
            logger.info("quotes_evicted", evicted=len(expired), remaining=len(self._quotes))

    def __len__(self) -> int:
        return len(self._quotes)
