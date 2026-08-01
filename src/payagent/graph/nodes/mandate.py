"""`mandate` node: issues an Intent Mandate, then a Payment Mandate, for the confirmed quote.

Both calls go through `dispatch_tool_call`, which means both go through
`payagent.policy.evaluate` and (for `create_payment_mandate`, via `MandateAuthority.issue_payment`)
the real signing authority wired into `deps` — whatever `PolicyEngine`/`MandateAuthority` the
composition root configured (`RulesPolicyEngine`/`Ed25519MandateAuthority` in production,
per-test doubles in the test suite). This node does not special-case any denial reason,
including a step-up denial: a `Deny` from either call ends the purchase for this run,
full stop. There is no retry-with-different-arguments path here, because there is nothing
to differ — `mandate` never chooses the amount, currency, merchant, or category itself; all
four come from `state.quote`, the one thing `quote` wrote.

`confirm` (the next node) is where a step-up-shaped pause actually happens — it re-checks
step-up freshness right before `settle`, independently of whatever this node observed, since
a step-up window can lapse between mandate issuance and settlement. See its module docstring.
"""

from __future__ import annotations

from collections.abc import Callable

from payagent.graph.nodes.tool_call import call_tool
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.deps import ToolDeps
from payagent.observability.logging import get_logger

logger = get_logger(__name__)


def make_mandate_node(deps: ToolDeps) -> Callable[[PurchaseState], PurchaseState]:
    """Build the `mandate` node, with `deps` bound by closure."""

    def node(state: PurchaseState) -> PurchaseState:
        if not state.is_active:
            return state

        if state.quote_id is None or state.quote_amount_cents is None or state.quote is None:
            raise IllegalTransitionError(
                "mandate requires a successful quote — state.quote_id/quote_amount_cents are unset"
            )
        quote_data = state.quote["data"]

        intent_envelope = call_tool(
            deps,
            "create_intent_mandate",
            {
                "idempotency_key": deps.id_factory("intent-idem"),
                # Narrowed to exactly what was quoted — least privilege, not a request field
                # the caller controls (the values come from `state.quote`, never `state.user_request`).
                "purpose": f"Purchase {quote_data['quantity']} x {quote_data['name']}"[:200],
                "max_amount_cents": state.quote_amount_cents,
                "currency": quote_data["currency"],
                "allowed_categories": [quote_data["category"]],
                "allowed_merchant_ids": [quote_data["merchant_id"]],
            },
        )
        if not intent_envelope["ok"]:
            logger.info(
                "mandate_denied",
                stage="create_intent_mandate",
                error_code=intent_envelope["error"]["code"],
            )
            return state.model_copy(
                update={
                    "status": "mandate_denied",
                    "policy_decision": intent_envelope["error"].get("denial"),
                }
            )

        payment_envelope = call_tool(
            deps,
            "create_payment_mandate",
            {
                "idempotency_key": deps.id_factory("payment-idem"),
                "intent_mandate_id": intent_envelope["data"]["intent_mandate_id"],
                "quote_id": state.quote_id,
                "expected_amount_cents": state.quote_amount_cents,
                "expected_currency": quote_data["currency"],
            },
        )
        if not payment_envelope["ok"]:
            logger.info(
                "mandate_denied",
                stage="create_payment_mandate",
                error_code=payment_envelope["error"]["code"],
            )
            return state.model_copy(
                update={
                    "status": "mandate_denied",
                    "policy_decision": payment_envelope["error"].get("denial"),
                }
            )

        logger.info(
            "mandate_issued", payment_mandate_id=payment_envelope["data"]["payment_mandate_id"]
        )
        return state.model_copy(update={"mandate": payment_envelope})

    return node
