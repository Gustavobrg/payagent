"""Test doubles and deterministic helpers shared by the MCP contract suite.

A separate module rather than living in `conftest.py` so tests can import these by name instead of
receiving them as fixtures — the policy doubles get constructed inline with different inner engines,
which a fixture cannot express as cleanly.

`AlwaysVerifyingAuthority` lives here and **not** in `src/` on purpose: the property worth keeping is
that no shipped code path can produce a verified mandate, so settlement is impossible until real
signing lands (invariant I3). Following the repo's existing pattern, this is the same kind of
test-only stand-in as `DeterministicEmbedder` for the real embedder.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from payagent.mandates import MandateVerification, UnsignedMandateAuthority
from payagent.policy import (
    Allow,
    DenyAllPolicyEngine,
    EffectGrant,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
)

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


class FakeClock:
    """Mutable injectable clock. Expiry tests advance it instead of sleeping."""

    def __init__(self, now: datetime = FIXED_NOW) -> None:
        self._now = now

    def __call__(self) -> datetime:
        return self._now

    def advance(self, seconds: int) -> None:
        self._now += timedelta(seconds=seconds)


def counter_id_factory():
    """`prefix -> "PREFIX-000...N"`, so a test can assert an ID literally.

    Production uses a uuid4-based factory: mandate and settlement IDs are capability-like handles,
    and sequential IDs are enumerable.
    """
    counter = {"n": 0}

    def factory(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:016d}"

    return factory


class AllowAllPolicyEngine(PolicyEngine):
    """Test double: grants every action. Never shipped — the production default denies."""

    def decide(self, action: PolicyAction, context: PolicyContext):
        return Allow(
            grant=EffectGrant(
                action=action,
                amount_cents=context.amount_cents,
                currency=context.currency,
                merchant_id=context.merchant_id,
                decided_at=context.now,
            )
        )


class RecordingPolicyEngine(PolicyEngine):
    """Wraps another engine and records what it was asked, to assert policy was (not) consulted."""

    def __init__(self, inner: PolicyEngine | None = None) -> None:
        self.inner = inner or DenyAllPolicyEngine()
        self.calls: list[tuple[PolicyAction, PolicyContext]] = []

    def decide(self, action: PolicyAction, context: PolicyContext):
        self.calls.append((action, context))
        return self.inner.decide(action, context)


class AlwaysVerifyingAuthority(UnsignedMandateAuthority):
    """Test-only authority that verifies mandates, unlocking the happy path."""

    def verify_for_settlement(self, mandate, *, now) -> MandateVerification:
        return MandateVerification(verified=True)
