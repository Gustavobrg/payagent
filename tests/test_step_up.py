"""Tests for the simulated WebAuthn step-up verifier (P5, POL-003).

`step_up_satisfied` must be server-side session state that nothing on the MCP tool surface can
set directly. This module is the seam that produces it: a challenge is issued, resolved through
a channel independent of any tool call, and only then does `is_satisfied` report true. The
property under test throughout is that satisfaction can *only* come from that separate
`resolve_challenge` call — never from a caller simply asserting it happened.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from payagent.mcp_server.step_up import InMemoryStepUpVerifier, StepUpChallenge

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)


def _counter_id_factory():
    counter = {"n": 0}

    def factory(prefix: str) -> str:
        counter["n"] += 1
        return f"{prefix}-{counter['n']:04d}"

    return factory


def _verifier(ttl_seconds: int = 300) -> InMemoryStepUpVerifier:
    return InMemoryStepUpVerifier(id_factory=_counter_id_factory(), ttl_seconds=ttl_seconds)


def test_not_satisfied_before_any_challenge_is_issued():
    verifier = _verifier()

    assert verifier.is_satisfied(now=FIXED_NOW) is False


def test_not_satisfied_after_issuing_but_before_resolving():
    verifier = _verifier()

    challenge = verifier.issue_challenge(now=FIXED_NOW)

    assert isinstance(challenge, StepUpChallenge)
    assert verifier.is_satisfied(now=FIXED_NOW) is False


def test_satisfied_only_after_the_separate_channel_resolves_the_challenge():
    """The core P5 property: satisfaction is produced by `resolve_challenge`, nothing else."""
    verifier = _verifier()
    challenge = verifier.issue_challenge(now=FIXED_NOW)

    resolved = verifier.resolve_challenge(challenge.challenge_id, now=FIXED_NOW)

    assert resolved is True
    assert verifier.is_satisfied(now=FIXED_NOW) is True


def test_resolving_an_unknown_challenge_id_does_nothing():
    verifier = _verifier()
    verifier.issue_challenge(now=FIXED_NOW)

    resolved = verifier.resolve_challenge("SU-does-not-exist", now=FIXED_NOW)

    assert resolved is False
    assert verifier.is_satisfied(now=FIXED_NOW) is False


def test_satisfaction_expires_with_the_challenge_window():
    verifier = _verifier(ttl_seconds=60)
    challenge = verifier.issue_challenge(now=FIXED_NOW)
    verifier.resolve_challenge(challenge.challenge_id, now=FIXED_NOW)

    still_valid = verifier.is_satisfied(now=FIXED_NOW + timedelta(seconds=59))
    expired = verifier.is_satisfied(now=FIXED_NOW + timedelta(seconds=61))

    assert still_valid is True
    assert expired is False


def test_resolving_an_expired_challenge_fails():
    verifier = _verifier(ttl_seconds=60)
    challenge = verifier.issue_challenge(now=FIXED_NOW)

    resolved = verifier.resolve_challenge(challenge.challenge_id, now=FIXED_NOW + timedelta(seconds=61))

    assert resolved is False
    assert verifier.is_satisfied(now=FIXED_NOW + timedelta(seconds=61)) is False


def test_resolving_a_stale_challenge_does_not_satisfy_a_freshly_issued_one():
    """Only the most recently issued challenge can be satisfied — an old resolved one cannot
    be replayed to authorize a step-up the agent has not actually re-prompted for."""
    verifier = _verifier()
    first = verifier.issue_challenge(now=FIXED_NOW)
    verifier.issue_challenge(now=FIXED_NOW)  # a second, unresolved challenge supersedes it

    verifier.resolve_challenge(first.challenge_id, now=FIXED_NOW)

    assert verifier.is_satisfied(now=FIXED_NOW) is False
