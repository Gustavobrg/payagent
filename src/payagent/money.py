"""Money and identifier primitives shared by `policy/`, `mandates/` and `mcp_server/`.

Top-level rather than inside `mcp_server/` on purpose: `policy/` needs these types, and
having `policy/` import from the MCP layer would invert the dependency direction and break
the import-purity property asserted in `tests/test_policy_engine.py` (invariant I1 —
authorization is deterministic and depends on nothing).

Money is `int` cents plus an ISO-4217 code (CLAUDE.md: "float em valor monetário é bug").
The `Strict()` annotation is what makes that enforceable rather than aspirational: pydantic's
default lax mode would coerce `4999.0` to `4999`, silently accepting a float that reached the
API through JSON. With `Strict()` the same input is rejected with error type `int_type`.

Note the deliberate asymmetry between the advertised JSON Schema and the server-side check:
JSON Schema `"type": "integer"` accepts `4999.0` per draft 2020-12 (any number with a zero
fractional part), and `multipleOf: 1` does not help since `4999.0` satisfies it too. So the
published schema is strictly weaker than what the server enforces. That is the safe
direction, but it means a client can validate locally, pass, and still be rejected — hence
`CENTS_FIELD_NOTE`, which every monetary field's description embeds so the rejection is
explainable.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, Strict

# 1,000,000.00 in cents. A sanity ceiling to keep absurd values out of the type, NOT a
# policy limit — real limits are deterministic and live in `payagent.policy`.
MAX_CENTS = 100_000_000

CENTS_FIELD_NOTE = (
    "Integer number of minor units (cents). A value written with a decimal point is "
    "rejected even when its fractional part is zero."
)

Cents = Annotated[int, Strict(), Field(ge=0, le=MAX_CENTS)]
PositiveCents = Annotated[int, Strict(), Field(ge=1, le=MAX_CENTS)]

CurrencyCode = Annotated[str, Field(min_length=3, max_length=3, pattern=r"^[A-Z]{3}$")]

Quantity = Annotated[int, Strict(), Field(ge=1, le=99)]

# `min_length=16` is deliberate: it makes "1", "key" and "retry" invalid, so two unrelated
# operations in one session cannot collide on a lazily-chosen key and be served each other's
# cached result. The charset excludes whitespace and "/" so a key is safe as a dict key, a
# log field, and a future URL segment.
IdempotencyKey = Annotated[
    str, Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9._:\-]{16,128}$")
]

SkuId = Annotated[str, Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._\-]{1,64}$")]
# Catalog categories are human-readable lowercase names, not slugs — the real dataset has
# "pet shop", "toys and games", "jewelry and luxury watches". A slug-only pattern would reject 14
# of the 20 categories in `evals/datasets/catalog.json`, making `create_intent_mandate` unusable for
# most of the catalog and breaking POL-004 (restricted categories), which keys off this value. The
# pattern bounds the value rather than enforcing a style: it must start with a letter, so an
# all-punctuation or leading-space value is still rejected.
CategoryName = Annotated[
    str, Field(min_length=1, max_length=40, pattern=r"^[a-z][a-z0-9 _-]{0,39}$")
]
MerchantId = Annotated[str, Field(min_length=4, max_length=20, pattern=r"^[A-Z]{2,10}-[0-9]{2,6}$")]


def format_money(amount_cents: int, currency: str) -> str:
    """Render cents as a human-readable amount for mandate summaries and denial reasons.

    Integer arithmetic only — `divmod`, never a division that would reintroduce a float into
    a monetary path.
    """
    units, minor = divmod(abs(amount_cents), 100)
    sign = "-" if amount_cents < 0 else ""
    return f"{sign}{currency} {units}.{minor:02d}"
