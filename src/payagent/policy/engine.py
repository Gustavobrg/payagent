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
from collections.abc import Iterable

from payagent.observability.logging import get_logger
from payagent.policy.models import (
    Allow,
    DenialCode,
    Deny,
    EffectGrant,
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


_AMOUNT_REASON = "The amount exceeds the maximum permitted for a single transaction."
_CATEGORY_REASON = "This category is restricted and cannot be purchased by the autonomous agent."
_MERCHANT_REASON = "This merchant is not on the authorized settlement allowlist."
_AGGREGATE_REASON = "This transaction would exceed the 24-hour aggregate spending cap."
_STEPUP_REASON = (
    "This amount requires step-up authentication (WebAuthn/passkey) that has not been completed."
)


class RulesPolicyEngine(PolicyEngine):
    """The real rules, read from configuration handed in at construction time.

    Every threshold and list is a constructor argument, never an `os.environ` read — this
    package makes no I/O, so whatever composes the engine (`mcp_server/__main__.py` in
    production, a test fixture in `tests/`) is what actually reads `POLICY_MAX_AMOUNT_CENTS` and
    friends. That is what keeps `decide` a pure function of `(action, context)`: the same inputs
    always produce the same decision, satisfied even for the 24h aggregate check because the
    running total arrives as `context.aggregate_spent_24h_cents` rather than being looked up here.

    Checks run in the order POL-002 §5 documents for limits, with the category and merchant
    checks (POL-004, POL-005) folded in ahead of the aggregate and step-up checks: amount ceiling,
    restricted category, merchant allowlist, 24h aggregate, step-up. The first violated rule wins
    — a request that fails two rules gets one denial, not a list, matching `PolicyDenial`'s shape.

    Category and merchant are checked against every value the action puts at stake: for
    `create_intent_mandate` that means every category/merchant the mandate would *permit*
    (`intent_mandate_allowed_categories`/`_merchant_ids`), so a restricted scope is refused before
    it is ever recorded, not only when it is later spent.

    Step-up is the one check `create_intent_mandate` is exempt from: an Intent Mandate "does not
    execute any financial transfer; it merely records the user's consent for subsequent payment
    actions" (`mandates/models.py`), so its ceiling is not itself "a transaction" in POL-003's
    sense. The spend actions derived from it (`create_payment_mandate`, `execute_settlement`)
    still require step-up above the threshold — declaring a high ceiling costs nothing, spending
    against it does.
    """

    def __init__(
        self,
        *,
        max_amount_cents: int,
        stepup_threshold_cents: int,
        daily_aggregate_limit_cents: int,
        merchant_allowlist: Iterable[str],
        restricted_categories: Iterable[str],
    ) -> None:
        self._max_amount_cents = max_amount_cents
        self._stepup_threshold_cents = stepup_threshold_cents
        self._daily_aggregate_limit_cents = daily_aggregate_limit_cents
        self._merchant_allowlist = frozenset(merchant_allowlist)
        self._restricted_categories = frozenset(c.lower() for c in restricted_categories)

    def _categories_at_stake(self, action: PolicyAction, context: PolicyContext) -> tuple[str, ...]:
        if action is PolicyAction.CREATE_INTENT_MANDATE:
            return context.intent_mandate_allowed_categories
        return (context.category,) if context.category else ()

    def _merchants_at_stake(self, action: PolicyAction, context: PolicyContext) -> tuple[str, ...]:
        if action is PolicyAction.CREATE_INTENT_MANDATE:
            return context.intent_mandate_allowed_merchant_ids or ()
        return (context.merchant_id,) if context.merchant_id else ()

    def decide(self, action: PolicyAction, context: PolicyContext) -> PolicyDecision:
        if context.amount_cents > self._max_amount_cents:
            return Deny(PolicyDenial(DenialCode.AMOUNT_EXCEEDS_LIMIT, _AMOUNT_REASON, "POL-001"))

        for category in self._categories_at_stake(action, context):
            if category.lower() in self._restricted_categories:
                return Deny(
                    PolicyDenial(DenialCode.CATEGORY_RESTRICTED, _CATEGORY_REASON, "POL-004")
                )

        for merchant_id in self._merchants_at_stake(action, context):
            if merchant_id not in self._merchant_allowlist:
                return Deny(
                    PolicyDenial(DenialCode.MERCHANT_NOT_ALLOWED, _MERCHANT_REASON, "POL-005")
                )

        projected_aggregate = context.aggregate_spent_24h_cents + context.amount_cents
        if projected_aggregate > self._daily_aggregate_limit_cents:
            return Deny(
                PolicyDenial(DenialCode.AGGREGATE_LIMIT_EXCEEDED, _AGGREGATE_REASON, "POL-002")
            )

        if (
            action is not PolicyAction.CREATE_INTENT_MANDATE
            and context.amount_cents > self._stepup_threshold_cents
            and not context.step_up_satisfied
        ):
            return Deny(PolicyDenial(DenialCode.STEP_UP_REQUIRED, _STEPUP_REASON, "POL-003"))

        return Allow(
            grant=EffectGrant(
                action=action,
                amount_cents=context.amount_cents,
                currency=context.currency,
                merchant_id=context.merchant_id,
                decided_at=context.now,
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
