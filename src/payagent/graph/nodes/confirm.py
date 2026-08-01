"""`confirm` node: the step-up gate between an issued mandate and settlement.

This is deliberately the *only* place in the graph that checks step-up freshness right before
`settle` — separately from whatever `mandate` observed when it called `create_payment_mandate`.
A step-up challenge has a TTL (`payagent.mcp_server.step_up`); enough wall-clock time between
mandate issuance and this node could let it lapse, so re-checking here (rather than trusting
`mandate`'s already-past check) is a real, non-redundant control, not paranoia for its own sake.

The check is a direct call to `payagent.policy.evaluate` for `EXECUTE_SETTLEMENT` — the same
function `mcp_server/handlers.py` uses, so "does the rule require step-up right now" is answered
by the one deterministic authority in the system, not reimplemented here. It is a *pre-check*:
no `EffectGrant` from it is consumed anywhere, and `settle` still calls `execute_settlement`
through the real fail-closed path regardless of what this node concluded. `step_up_satisfied`
comes from `deps.step_up.is_satisfied(now=...)` alone — this node never reads `state.user_request`
or any other field a user could have written, which is what "não aceita confirmação vinda do
texto do usuário" means in code: there is no branch here that inspects the request text at all.

If the rule requires step-up and it is not satisfied, the graph pauses
(`status="awaiting_step_up"`) rather than failing or looping — resolving it happens entirely
outside this conversation (see `StepUpVerifier.resolve_challenge`), and re-running this node
after that resolution is how the graph resumes. Any other denial is terminal, exactly like
`mandate`'s: no workaround is attempted.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta

from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.deps import ToolDeps
from payagent.observability.logging import get_logger
from payagent.policy import Allow, DenialCode, PolicyAction, PolicyContext, evaluate

logger = get_logger(__name__)


def _aggregate_spent_24h_cents(deps: ToolDeps, *, now: datetime, currency: str) -> int:
    """Same computation `mcp_server/handlers.py` makes — duplicated deliberately (ADR-0003):
    two independent code paths agreeing is a stronger property than either alone."""
    cutoff = now - timedelta(hours=24)
    return sum(
        charge.amount_cents
        for charge in deps.merchant.charges.values()
        if charge.currency == currency and charge.created_at >= cutoff
    )


def _denial_payload(denial) -> dict:
    return {"code": denial.code.value, "reason": denial.reason, "policy_ref": denial.policy_ref}


def make_confirm_node(deps: ToolDeps) -> Callable[[PurchaseState], PurchaseState]:
    """Build the `confirm` node, with `deps` bound by closure."""

    def node(state: PurchaseState) -> PurchaseState:
        if state.status not in ("in_progress", "awaiting_step_up"):
            return state

        if state.mandate is None or not state.mandate.get("ok") or state.quote is None:
            raise IllegalTransitionError(
                "confirm requires an issued payment mandate — state.mandate is unset"
            )

        payment_data = state.mandate["data"]
        quote_data = state.quote["data"]
        now = deps.clock()

        # A pre-check mirror of what `execute_settlement`'s handler will itself evaluate —
        # only the fields `RulesPolicyEngine.decide` actually reads for this action. The
        # authoritative check still happens inside `settle`; this can only pause or refuse
        # earlier, never authorize.
        context = PolicyContext(
            now=now,
            amount_cents=state.quote_amount_cents,
            currency=quote_data["currency"],
            merchant_id=payment_data["merchant_id"],
            category=quote_data["category"],
            aggregate_spent_24h_cents=_aggregate_spent_24h_cents(
                deps, now=now, currency=quote_data["currency"]
            ),
            step_up_satisfied=deps.step_up.is_satisfied(now=now),
        )
        decision = evaluate(deps.policy, PolicyAction.EXECUTE_SETTLEMENT, context)

        if isinstance(decision, Allow):
            logger.info("confirm_ready_to_settle")
            return state.model_copy(update={"status": "in_progress", "policy_decision": None})

        denial = decision.denial
        if denial.code is DenialCode.STEP_UP_REQUIRED:
            logger.info("confirm_awaiting_step_up")
            return state.model_copy(
                update={"status": "awaiting_step_up", "policy_decision": _denial_payload(denial)}
            )

        logger.info("confirm_denied", denial_code=denial.code.value)
        return state.model_copy(
            update={"status": "settlement_denied", "policy_decision": _denial_payload(denial)}
        )

    return node
