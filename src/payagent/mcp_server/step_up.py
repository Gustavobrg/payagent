"""Simulated WebAuthn step-up: issue a challenge, resolve it on a separate channel, query state.

`PolicyContext.step_up_satisfied` (`policy/models.py`) must come from server-side session state
only — never a tool argument, which is exactly the P5 evasion POL-003 exists to stop. This module
is the seam that produces that state honestly:

- `issue_challenge` mints a challenge the same way starting a WebAuthn ceremony would.
- `resolve_challenge` is called by whatever completes that ceremony — a separate confirm-step
  endpoint in a real deployment, reached outside the MCP tool call entirely. Nothing on the
  `execute_settlement`/`create_payment_mandate` request schemas can reach this method, by
  construction (`StrictModel`'s `extra="forbid"` — see `mcp_server/schemas.py`).
- `is_satisfied` is the only thing `mcp_server/handlers.py` consults when building a
  `PolicyContext`. It answers only for the *most recently issued* challenge, so resolving a
  stale challenge cannot satisfy a step-up the agent has not actually re-prompted for.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

# WebAuthn ceremonies are seconds, not minutes; five minutes is generous headroom for a user to
# complete a passkey prompt without leaving the window open long enough to be replayed later.
DEFAULT_STEP_UP_TTL_SECONDS = 300


@dataclass(frozen=True, slots=True)
class StepUpChallenge:
    """A minted step-up challenge. `challenge_id` is the only thing the resolving channel needs."""

    challenge_id: str
    issued_at: datetime
    expires_at: datetime


class StepUpVerifier(Protocol):
    """Issues and resolves step-up challenges, and reports whether one is currently satisfied."""

    def issue_challenge(self, *, now: datetime) -> StepUpChallenge: ...

    def resolve_challenge(self, challenge_id: str, *, now: datetime) -> bool: ...

    def is_satisfied(self, *, now: datetime) -> bool: ...


class InMemoryStepUpVerifier:
    """Process-local challenge/response state for one agent session.

    Only the most recently issued challenge can ever satisfy `is_satisfied`: issuing a new one
    supersedes any earlier unresolved (or even resolved) challenge, so a step-up prompt the agent
    already moved past cannot be completed late and retroactively authorize a different request.
    """

    def __init__(
        self, *, id_factory: Callable[[str], str], ttl_seconds: int = DEFAULT_STEP_UP_TTL_SECONDS
    ) -> None:
        self._id_factory = id_factory
        self._ttl_seconds = ttl_seconds
        self._expires_at_by_challenge_id: dict[str, datetime] = {}
        self._resolved_challenge_ids: set[str] = set()
        self._last_challenge_id: str | None = None

    def issue_challenge(self, *, now: datetime) -> StepUpChallenge:
        challenge_id = self._id_factory("SU")
        expires_at = now + timedelta(seconds=self._ttl_seconds)
        self._expires_at_by_challenge_id[challenge_id] = expires_at
        self._last_challenge_id = challenge_id
        return StepUpChallenge(challenge_id=challenge_id, issued_at=now, expires_at=expires_at)

    def resolve_challenge(self, challenge_id: str, *, now: datetime) -> bool:
        expires_at = self._expires_at_by_challenge_id.get(challenge_id)
        if expires_at is None or now >= expires_at:
            return False
        self._resolved_challenge_ids.add(challenge_id)
        return True

    def is_satisfied(self, *, now: datetime) -> bool:
        challenge_id = self._last_challenge_id
        if challenge_id is None or challenge_id not in self._resolved_challenge_ids:
            return False
        return now < self._expires_at_by_challenge_id[challenge_id]
