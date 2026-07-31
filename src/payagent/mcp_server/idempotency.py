"""Idempotency for the four tools with side effects (CLAUDE.md: idempotent by `idempotency_key`).

Records are keyed by the tuple `(tool, key)` rather than by key alone. Per-tool namespacing makes
it impossible to serve a cached `execute_settlement` response to a `refund` call that happened to
reuse a key — a result-type confusion that would have the agent report the wrong operation as
done. Two tools colliding on a key is not itself an attack once each tool's effect is deduped
independently.

The decision worth reading carefully is **replay with a different payload**. Three options exist
and two are unsafe:

- *Return the cached result.* An attacker, or an injected chunk (P2), calls `execute_settlement`
  with the same key but a different mandate or amount, receives the previous **success** payload,
  and the agent tells the user the new operation succeeded. A settlement that never happened is
  confirmed as done — and the mirror case confirms a large charge that was actually small.
- *Perform the effect anyway.* Breaks the contract the tool advertises: two effects, one key.
- *Refuse* — what this module does. It is the only answer that is both truthful and effect-free.

Pending state exists so an in-flight duplicate is refused rather than racing. On an
**indeterminate** outcome (a merchant timeout) the caller `release()`s the reservation so a retry
with the same key can proceed; that retry is caught by the merchant's own dedup, so exactly one
effect exists either way. On a **deterministic** failure — declined, policy denied, invalid
arguments — the record is `complete`d with the error, so a replay returns the identical denial
without re-consulting policy. That also means a denied purchase cannot be retried into an allowed
one under the same key.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    """One reservation, pending or complete, with the response to replay."""

    tool: str
    key: str
    fingerprint: str
    state: Literal["pending", "complete"]
    response: dict[str, Any] | None = None
    is_error: bool = False
    created_at: datetime | None = None


class IdempotencyConflict(Exception):
    """Same `(tool, key)`, different arguments. Refused: no effect, no cached result."""


class IdempotencyInProgress(Exception):
    """Same `(tool, key)` while the first attempt is still running."""


def argument_fingerprint(payload: dict[str, Any]) -> str:
    """Hash a validated request payload, excluding `idempotency_key`.

    Computed from the **validated** model rather than the raw arguments, so an explicitly-sent
    default and an omitted field hash identically — they are the same request, and treating them
    as a conflict would reject a legitimate retry.
    """
    material = {key: value for key, value in payload.items() if key != "idempotency_key"}
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class IdempotencyStore(Protocol):
    """Reserve/complete/release for `(tool, key)` pairs."""

    def begin(
        self, tool: str, key: str, fingerprint: str, now: datetime
    ) -> IdempotencyRecord | None: ...

    def complete(
        self, tool: str, key: str, response: dict[str, Any], *, is_error: bool
    ) -> None: ...

    def release(self, tool: str, key: str) -> None: ...


class InMemoryIdempotencyStore:
    """Process-local idempotency store.

    Safe without locks only because the whole reserve → effect → complete sequence runs inside a
    single synchronous call that never yields to the event loop; see the note in `dispatch.py`.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[str, str], IdempotencyRecord] = {}

    def begin(
        self, tool: str, key: str, fingerprint: str, now: datetime
    ) -> IdempotencyRecord | None:
        """Reserve `(tool, key)`.

        Returns:
            The completed record on a matching replay, or `None` when newly reserved.

        Raises:
            IdempotencyConflict: same key, different arguments.
            IdempotencyInProgress: the first attempt has not finished.
        """
        existing = self._records.get((tool, key))
        if existing is None:
            self._records[(tool, key)] = IdempotencyRecord(
                tool=tool, key=key, fingerprint=fingerprint, state="pending", created_at=now
            )
            return None

        if existing.fingerprint != fingerprint:
            raise IdempotencyConflict(tool)

        if existing.state == "pending":
            raise IdempotencyInProgress(tool)

        return existing

    def complete(
        self, tool: str, key: str, response: dict[str, Any], *, is_error: bool
    ) -> None:
        """Store the response to replay for future calls with this key."""
        existing = self._records.get((tool, key))
        if existing is None:
            return
        self._records[(tool, key)] = IdempotencyRecord(
            tool=existing.tool,
            key=existing.key,
            fingerprint=existing.fingerprint,
            state="complete",
            response=response,
            is_error=is_error,
            created_at=existing.created_at,
        )

    def release(self, tool: str, key: str) -> None:
        """Drop a pending reservation so an indeterminate attempt can be retried."""
        existing = self._records.get((tool, key))
        if existing is not None and existing.state == "pending":
            del self._records[(tool, key)]

    def __len__(self) -> int:
        return len(self._records)
