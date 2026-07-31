"""In-process merchant stand-in: accepts settlement, keeps state, and fails on demand.

Architecture.md puts real acquirer/network integration out of scope, so this is what
`execute_settlement` and `refund` actually talk to. It exists to make the settlement path
testable end to end, including the failure modes that matter.

Two design points worth stating:

**Failures never sleep.** `TIMEOUT` raises immediately. What is under test is the *handling* of
a timeout — a structured `MERCHANT_TIMEOUT`, no charge recorded, the idempotency reservation
released — none of which depends on elapsed wall time. Sleeping would add seconds per test,
make the suite flaky under load, and test the clock instead of the code. If a real
deadline-sensitive client ever appears, inject the sleep function rather than making this block.

**Dedup here is a second, independent layer.** `mcp_server/dispatch.py` already dedups by
`idempotency_key` before an effect runs; this class dedups again on its own. That redundancy is
the point: a bug in the dispatch layer still cannot produce two charges, which is what makes the
"no duplicated effect" contract test meaningful rather than tautological.

Refunds are full-amount only, mirroring the tool surface: there is no caller-chosen amount
anywhere, so a repeat refund is a state error rather than an arithmetic check.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType

from payagent.observability.logging import get_logger
from payagent.policy import EffectGrant

logger = get_logger(__name__)


class MerchantBehavior(StrEnum):
    """How the merchant responds to the next call.

    `TIMEOUT` and `TIMEOUT_AFTER_COMMIT` are deliberately distinct: the first fails before
    recording anything (safe to retry, nothing happened), the second records the charge and
    *then* fails (the response was lost, but the money moved). Only the second can produce a
    double charge on retry, so only the second exercises the dedup that prevents it.
    """

    ACCEPT = "accept"
    DECLINE = "decline"
    TIMEOUT = "timeout"
    TIMEOUT_AFTER_COMMIT = "timeout_after_commit"
    UNAVAILABLE = "unavailable"


class MerchantError(Exception):
    """Base for merchant-side failures. Messages are static — no caller data (invariant I2)."""

    code = "MERCHANT_ERROR"


class MerchantDeclined(MerchantError):
    code = "MERCHANT_DECLINED"


class MerchantTimeout(MerchantError):
    """The outcome is indeterminate: the charge may or may not have been recorded."""

    code = "MERCHANT_TIMEOUT"


class MerchantUnavailable(MerchantError):
    code = "MERCHANT_UNAVAILABLE"


class MerchantIdempotencyConflict(MerchantError):
    """Same key, different charge parameters — refused rather than charged or replayed."""

    code = "MERCHANT_IDEMPOTENCY_CONFLICT"


@dataclass(frozen=True, slots=True)
class Charge:
    """A recorded settlement. Money in cents, currency explicit."""

    charge_id: str
    idempotency_key: str
    payment_mandate_id: str
    amount_cents: int
    currency: str
    merchant_id: str
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Refund:
    """A recorded full reversal of one `Charge`."""

    refund_id: str
    idempotency_key: str
    charge_id: str
    amount_cents: int
    currency: str
    reason_code: str
    created_at: datetime


class MerchantSim:
    """Simulated merchant with in-memory state and injectable failure behaviour."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        behavior: MerchantBehavior = MerchantBehavior.ACCEPT,
        behavior_by_merchant: Mapping[str, MerchantBehavior] | None = None,
        scripted: Sequence[MerchantBehavior] | None = None,
    ) -> None:
        """Wire the simulator.

        Args:
            clock: Returns "now". Injected so expiry and timestamps are deterministic.
            id_factory: `prefix -> id`. Tests pass a counter; production passes a uuid4-based
                factory, because charge IDs are capability-like handles and sequential IDs are
                enumerable.
            behavior: Default response.
            behavior_by_merchant: Per-merchant override — lets one merchant decline while
                another accepts, which is the POL-005 allowlist scenario.
            scripted: A queue of behaviours consumed one per call, then falling back to
                `behavior`. Drives multi-step sequences like lost-response-then-retry.
        """
        self._clock = clock
        self._id_factory = id_factory
        self._behavior = behavior
        self._behavior_by_merchant = dict(behavior_by_merchant or {})
        self._scripted: deque[MerchantBehavior] = deque(scripted or ())

        self._charges: dict[str, Charge] = {}
        self._charge_by_key: dict[str, str] = {}
        self._refunds: dict[str, Refund] = {}
        self._refund_by_key: dict[str, str] = {}
        self._refunded_charges: set[str] = set()
        self._calls: list[dict] = []

    @property
    def charges(self) -> Mapping[str, Charge]:
        """Read-only view — a caller cannot fabricate merchant state through it."""
        return MappingProxyType(self._charges)

    @property
    def refunds(self) -> Mapping[str, Refund]:
        """Read-only view of recorded refunds."""
        return MappingProxyType(self._refunds)

    @property
    def calls(self) -> Sequence[dict]:
        """Audit log of every attempt, including failures. Lets a test assert nothing was tried."""
        return tuple(self._calls)

    def get_charge(self, charge_id: str) -> Charge | None:
        return self._charges.get(charge_id)

    def is_refunded(self, charge_id: str) -> bool:
        return charge_id in self._refunded_charges

    def _next_behavior(self, merchant_id: str) -> MerchantBehavior:
        if self._scripted:
            return self._scripted.popleft()
        return self._behavior_by_merchant.get(merchant_id, self._behavior)

    def _record_call(self, tool: str, key: str, merchant_id: str, amount_cents: int, outcome: str) -> None:
        self._calls.append(
            {
                "tool": tool,
                "idempotency_key": key,
                "merchant_id": merchant_id,
                "amount_cents": amount_cents,
                "outcome": outcome,
            }
        )
        logger.info(
            "merchant_call",
            tool=tool,
            merchant_id=merchant_id,
            amount_cents=amount_cents,
            outcome=outcome,
        )

    @staticmethod
    def _raise_for(behavior: MerchantBehavior) -> None:
        if behavior is MerchantBehavior.DECLINE:
            raise MerchantDeclined("The merchant declined the operation.")
        if behavior in (MerchantBehavior.TIMEOUT, MerchantBehavior.TIMEOUT_AFTER_COMMIT):
            raise MerchantTimeout("The merchant did not respond in time.")
        if behavior is MerchantBehavior.UNAVAILABLE:
            raise MerchantUnavailable("The merchant is unavailable.")

    def settle(
        self,
        grant: EffectGrant,
        *,
        idempotency_key: str,
        payment_mandate_id: str,
        amount_cents: int,
        currency: str,
        merchant_id: str,
    ) -> Charge:
        """Charge `amount_cents` for a payment mandate.

        `grant` is a required positional argument, not a convenience: it is the proof that
        `payagent.policy` authorized this specific effect, so there is no way to reach a charge
        from a code path that never obtained an `Allow`.

        Raises:
            MerchantIdempotencyConflict: `idempotency_key` was used for a different charge.
            MerchantDeclined, MerchantTimeout, MerchantUnavailable: simulated failures.
        """
        existing_id = self._charge_by_key.get(idempotency_key)
        if existing_id is not None:
            existing = self._charges[existing_id]
            if (
                existing.amount_cents != amount_cents
                or existing.currency != currency
                or existing.merchant_id != merchant_id
            ):
                self._record_call("settle", idempotency_key, merchant_id, amount_cents, "conflict")
                raise MerchantIdempotencyConflict(
                    "This idempotency key was already used for a different charge."
                )
            self._record_call("settle", idempotency_key, merchant_id, amount_cents, "replayed")
            return existing

        behavior = self._next_behavior(merchant_id)

        if behavior in (MerchantBehavior.DECLINE, MerchantBehavior.TIMEOUT, MerchantBehavior.UNAVAILABLE):
            outcome = {
                MerchantBehavior.DECLINE: "declined",
                MerchantBehavior.TIMEOUT: "timeout",
                MerchantBehavior.UNAVAILABLE: "unavailable",
            }[behavior]
            self._record_call("settle", idempotency_key, merchant_id, amount_cents, outcome)
            self._raise_for(behavior)

        charge = Charge(
            charge_id=self._id_factory("CH"),
            idempotency_key=idempotency_key,
            payment_mandate_id=payment_mandate_id,
            amount_cents=amount_cents,
            currency=currency,
            merchant_id=merchant_id,
            created_at=self._clock(),
        )
        self._charges[charge.charge_id] = charge
        self._charge_by_key[idempotency_key] = charge.charge_id

        if behavior is MerchantBehavior.TIMEOUT_AFTER_COMMIT:
            # The charge is recorded and the response is lost. A retry with the same key hits
            # `_charge_by_key` above and returns this charge instead of creating a second one.
            self._record_call(
                "settle", idempotency_key, merchant_id, amount_cents, "timeout_after_commit"
            )
            self._raise_for(behavior)

        self._record_call("settle", idempotency_key, merchant_id, amount_cents, "accepted")
        return charge

    def refund(
        self,
        grant: EffectGrant,
        *,
        idempotency_key: str,
        charge_id: str,
        reason_code: str,
    ) -> Refund:
        """Fully reverse `charge_id`. There is no amount parameter — the charge decides it.

        Raises:
            KeyError: no such charge.
            ValueError: the charge was already refunded.
            MerchantDeclined, MerchantTimeout, MerchantUnavailable: simulated failures.
        """
        existing_id = self._refund_by_key.get(idempotency_key)
        if existing_id is not None:
            existing = self._refunds[existing_id]
            if existing.charge_id != charge_id:
                raise MerchantIdempotencyConflict(
                    "This idempotency key was already used for a different refund."
                )
            self._record_call(
                "refund", idempotency_key, "", existing.amount_cents, "replayed"
            )
            return existing

        charge = self._charges.get(charge_id)
        if charge is None:
            raise KeyError(charge_id)
        if charge_id in self._refunded_charges:
            raise ValueError("This charge was already refunded.")

        behavior = self._next_behavior(charge.merchant_id)
        if behavior in (MerchantBehavior.DECLINE, MerchantBehavior.TIMEOUT, MerchantBehavior.UNAVAILABLE):
            outcome = {
                MerchantBehavior.DECLINE: "declined",
                MerchantBehavior.TIMEOUT: "timeout",
                MerchantBehavior.UNAVAILABLE: "unavailable",
            }[behavior]
            self._record_call(
                "refund", idempotency_key, charge.merchant_id, charge.amount_cents, outcome
            )
            self._raise_for(behavior)

        refund = Refund(
            refund_id=self._id_factory("RF"),
            idempotency_key=idempotency_key,
            charge_id=charge_id,
            amount_cents=charge.amount_cents,
            currency=charge.currency,
            reason_code=reason_code,
            created_at=self._clock(),
        )
        self._refunds[refund.refund_id] = refund
        self._refund_by_key[idempotency_key] = refund.refund_id
        self._refunded_charges.add(charge_id)

        if behavior is MerchantBehavior.TIMEOUT_AFTER_COMMIT:
            self._record_call(
                "refund",
                idempotency_key,
                charge.merchant_id,
                charge.amount_cents,
                "timeout_after_commit",
            )
            self._raise_for(behavior)

        self._record_call(
            "refund", idempotency_key, charge.merchant_id, charge.amount_cents, "accepted"
        )
        return refund
