"""Types the authorization decision is expressed in.

`PolicyDecision` is a **sealed union** (`Allow | Deny`), not a struct with an `allowed: bool`.
That choice is the core fail-closed mechanism: with a boolean there is always a default value
and a way to spell the check wrong (`if decision.allowed is not False`), and a
default-constructed decision means "allowed". With a union, `Allow` cannot exist without an
`EffectGrant`, and a caller matching on the decision has no branch to fall through into.

Nothing here is expressed as free text a model could produce. Denial reasons are authored in
this package as static English strings (CLAUDE.md: a denial returns a structured
`PolicyDenial(code, reason)`, never LLM text), and denial codes are an enum so the callers and
the eval harness agree on what happened.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class PolicyAction(StrEnum):
    """The actions that require authorization — exactly the four MCP tools with side effects.

    Read-only tools (`search_catalog`, `get_quote`) are deliberately absent: deciding what is
    *relevant* is not deciding what is *allowed*, and a read-only tool that consulted the
    policy engine would blur the boundary invariant I1 draws.
    """

    CREATE_INTENT_MANDATE = "create_intent_mandate"
    CREATE_PAYMENT_MANDATE = "create_payment_mandate"
    EXECUTE_SETTLEMENT = "execute_settlement"
    REFUND = "refund"


class DenialCode(StrEnum):
    """Why authorization was refused.

    The first two are structural (this block); the rest are the rule-driven codes the real
    engine will emit, kept here so callers and the eval harness can already switch on them
    and so the policy documents in `evals/datasets/policies/` have a code to map to.
    """

    POLICY_NOT_CONFIGURED = "POLICY_NOT_CONFIGURED"
    POLICY_ENGINE_ERROR = "POLICY_ENGINE_ERROR"
    AMOUNT_EXCEEDS_LIMIT = "AMOUNT_EXCEEDS_LIMIT"  # POL-001
    AGGREGATE_LIMIT_EXCEEDED = "AGGREGATE_LIMIT_EXCEEDED"  # POL-002
    STEP_UP_REQUIRED = "STEP_UP_REQUIRED"  # POL-003
    CATEGORY_RESTRICTED = "CATEGORY_RESTRICTED"  # POL-004
    MERCHANT_NOT_ALLOWED = "MERCHANT_NOT_ALLOWED"  # POL-005
    REFUND_NOT_ELIGIBLE = "REFUND_NOT_ELIGIBLE"  # POL-006
    MANDATE_SCOPE_EXCEEDED = "MANDATE_SCOPE_EXCEEDED"  # POL-009
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
    QUOTE_EXPIRED = "QUOTE_EXPIRED"  # POL-010


def _require_int_cents(value: object, field_name: str) -> None:
    """Reject anything that is not a plain non-negative `int`.

    `bool` is checked first because it subclasses `int`, so `isinstance(True, int)` passes and
    `True` would otherwise sail through as the amount `1`.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int number of cents, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must not be negative")


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class PolicyDenial:
    """A structured refusal. `reason` is static English authored here, never generated."""

    code: DenialCode
    reason: str
    policy_ref: str | None = None


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Everything the engine is allowed to base a decision on.

    `now` is passed in rather than read from the clock so the engine is a pure function: the
    same inputs always produce the same decision, which is what makes it testable and what
    "determinístico" in invariant I1 actually requires.

    `step_up_satisfied` is populated from server-side session state only. It must never be
    fed from a tool argument — a request field that says "step-up already done" is exactly the
    P5 evasion the step-up requirement exists to stop.
    """

    now: datetime
    amount_cents: int
    currency: str
    merchant_id: str | None = None
    category: str | None = None

    intent_mandate_id: str | None = None
    intent_mandate_expires_at: datetime | None = None
    intent_mandate_max_amount_cents: int | None = None
    intent_mandate_allowed_categories: tuple[str, ...] = ()
    intent_mandate_allowed_merchant_ids: tuple[str, ...] | None = None

    payment_mandate_id: str | None = None
    quote_id: str | None = None
    quote_expires_at: datetime | None = None

    settlement_id: str | None = None
    settled_amount_cents: int | None = None
    refund_reason_code: str | None = None

    step_up_satisfied: bool = False

    # POL-002: the sum of successful transactions in the trailing 24h, computed by the caller
    # from the settlement record (`mcp_server/handlers.py`, from `MerchantSim.charges`) — never
    # by this package, which does no I/O. Defaults to 0 for actions that do not spend (e.g. a
    # refund), so the aggregate check never fires for them.
    aggregate_spent_24h_cents: int = 0

    def __post_init__(self) -> None:
        _require_int_cents(self.amount_cents, "amount_cents")
        _require_aware(self.now, "now")
        _require_int_cents(self.aggregate_spent_24h_cents, "aggregate_spent_24h_cents")
        if self.intent_mandate_max_amount_cents is not None:
            _require_int_cents(
                self.intent_mandate_max_amount_cents, "intent_mandate_max_amount_cents"
            )
        if self.settled_amount_cents is not None:
            _require_int_cents(self.settled_amount_cents, "settled_amount_cents")


@dataclass(frozen=True, slots=True)
class EffectGrant:
    """Proof that one specific effect was authorized.

    Only `Allow` carries one, and the effect functions in `mcp_server/handlers.py` take it as
    a required positional argument — so an effect is unreachable from a code path that never
    obtained an `Allow`. It records the amount and merchant that were authorized, so the
    effect cannot quietly be performed for different ones.
    """

    action: PolicyAction
    amount_cents: int
    currency: str
    merchant_id: str | None
    decided_at: datetime

    def __post_init__(self) -> None:
        _require_int_cents(self.amount_cents, "amount_cents")
        _require_aware(self.decided_at, "decided_at")


@dataclass(frozen=True, slots=True)
class Allow:
    """Authorization granted. Cannot be constructed without a grant — that is the point."""

    grant: EffectGrant
    obligations: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class Deny:
    """Authorization refused, with a structured reason."""

    denial: PolicyDenial


PolicyDecision = Allow | Deny
