"""The authorization engine seam, and a default that denies everything.

Invariant I1: authorization is deterministic and lives here. Limits, allowlists, mandate
scope and step-up never reach a Colang file, a prompt, or an LLM decision — this package
makes no model call and no network call at all, which `tests/test_policy_engine.py` enforces
by parsing imports.

Callers use `evaluate()`, never `engine.decide()` directly. `evaluate` is the fail-closed
funnel: an engine that raises, returns the wrong type, or hands back a grant for a different
action than the one asked about all become a `Deny`. That matters because the real rules land
in a later block, and the failure mode of a half-written engine must be refusal, not silent
permission.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from payagent.observability.logging import get_logger
from payagent.policy.models import (
    Allow,
    DenialCode,
    Deny,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyDenial,
)

logger = get_logger(__name__)

_NOT_CONFIGURED_REASON = (
    "No authorization policy is configured for this action, so it cannot be permitted."
)
_ENGINE_ERROR_REASON = "The authorization policy could not be evaluated."
_UNUSABLE_DECISION_REASON = "The authorization policy returned an unusable decision."
_GRANT_MISMATCH_REASON = "The authorization policy returned a grant for a different action."


class PolicyEngine(ABC):
    """Decides whether one action may be performed, given a context."""

    @abstractmethod
    def decide(self, action: PolicyAction, context: PolicyContext) -> PolicyDecision:
        """Return `Allow` or `Deny` for `action` under `context`.

        Implementations must be pure and deterministic: no LLM call, no network, no I/O, and
        no clock read (the time arrives as `context.now`), so the same inputs always yield the
        same decision. Callers must go through `evaluate`, which turns any breach of that
        contract into a denial.
        """


class DenyAllPolicyEngine(PolicyEngine):
    """Denies every action. The default, and the safe answer while the rules are unwritten.

    Not a placeholder in the "fill this in later" sense — it is the correct behaviour for an
    engine with no configured policy. Fail-closed means an unconfigured payment agent buys
    nothing, rather than buying anything.
    """

    def decide(self, action: PolicyAction, context: PolicyContext) -> PolicyDecision:
        return Deny(
            denial=PolicyDenial(
                code=DenialCode.POLICY_NOT_CONFIGURED,
                reason=_NOT_CONFIGURED_REASON,
            )
        )


def evaluate(
    engine: PolicyEngine, action: PolicyAction, context: PolicyContext
) -> PolicyDecision:
    """Call `engine.decide` fail-closed. This is the only entry point callers may use.

    Three ways an engine can misbehave, all of which become `Deny`:
    raising, returning something that is not a decision, and returning an `Allow` whose grant
    names a different action than the one under evaluation (which would otherwise let an
    authorization for a cheap refund be spent on a settlement).
    """
    try:
        decision = engine.decide(action, context)
    except Exception as exc:
        logger.error("policy_engine_error", action=action.value, error=str(exc))
        return Deny(PolicyDenial(DenialCode.POLICY_ENGINE_ERROR, _ENGINE_ERROR_REASON))

    if not isinstance(decision, (Allow, Deny)):
        logger.error("policy_engine_invalid_decision", action=action.value)
        return Deny(PolicyDenial(DenialCode.POLICY_ENGINE_ERROR, _UNUSABLE_DECISION_REASON))

    if isinstance(decision, Allow) and decision.grant.action is not action:
        logger.error(
            "policy_engine_grant_action_mismatch",
            action=action.value,
            granted_action=decision.grant.action.value,
        )
        return Deny(PolicyDenial(DenialCode.POLICY_ENGINE_ERROR, _GRANT_MISMATCH_REASON))

    return decision
