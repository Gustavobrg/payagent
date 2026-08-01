"""`settle` node: calls `execute_settlement`, and only that, exactly once.

Invariant I3 is enforced inside `execute_settlement`'s own handler (mandate resolution,
re-binding to its quote, `MandateAuthority.verify_for_settlement`, then policy) — this node
adds nothing on top of that except the precondition that a payment mandate actually exists in
state before attempting the call at all. `expected_amount_cents`/`expected_currency` are
asserted from `state.quote_amount_cents` and the mandate's own currency — never from
`state.user_request` — so a mismatch aborts the call rather than settling a different figure
(P2).

There is exactly one `call_tool` in this module, with no loop and no retry around it. A
`Deny` from policy or a merchant decline/failure both end the run the same way: terminal
`"settlement_denied"`, `settlement_result` holding the structured envelope. Nothing here
re-attempts automatically — a merchant timeout (indeterminate outcome, safe to retry with the
same `idempotency_key`) is a case a *caller* may choose to retry by running this node again on
the same state, but this node itself never does that on its own.
"""

from __future__ import annotations

from collections.abc import Callable

from payagent.graph.nodes.tool_call import call_tool
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.deps import ToolDeps
from payagent.observability.logging import get_logger

logger = get_logger(__name__)


def make_settle_node(deps: ToolDeps) -> Callable[[PurchaseState], PurchaseState]:
    """Build the `settle` node, with `deps` bound by closure."""

    def node(state: PurchaseState) -> PurchaseState:
        if not state.is_active:
            return state

        if state.mandate is None or not state.mandate.get("ok"):
            raise IllegalTransitionError(
                "settle requires an issued payment mandate — state.mandate is unset"
            )

        payment_data = state.mandate["data"]
        envelope = call_tool(
            deps,
            "execute_settlement",
            {
                "idempotency_key": deps.id_factory("settle-idem"),
                "payment_mandate_id": payment_data["payment_mandate_id"],
                "expected_amount_cents": state.quote_amount_cents,
                "expected_currency": payment_data["currency"],
            },
        )

        if envelope["ok"]:
            logger.info("settlement_succeeded", settlement_id=envelope["data"]["settlement_id"])
            return state.model_copy(update={"status": "settled", "settlement_result": envelope})

        logger.info("settlement_denied", error_code=envelope["error"]["code"])
        return state.model_copy(
            update={"status": "settlement_denied", "settlement_result": envelope}
        )

    return node
