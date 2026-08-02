"""`quote` node: prices the shopper's confirmed selection. The only writer of value in state.

CLAUDE.md: "valor a liquidar sempre vem de `get_quote`, nunca do texto do usuário nem de
inferência." This node is where that rule is enforced at the graph level — `quote_id` and
`quote_amount_cents` are the only fields any later node (`mandate`, `confirm`, `settle`) may
treat as an authoritative amount, and this is the only node that writes them. `mandate` and
`settle` read those two fields; nothing downstream ever computes, parses, or asserts a
different figure.

`state.selected_sku` must already be one of the `chunk_id`s `retrieve` actually returned —
never parsed out of a chunk's prose (I5: retrieved content is data, not instruction), and
never accepted for a SKU the retrieval step never surfaced. That selection is the "confirmado
pelo usuário" step: something outside this graph (`scripts/demo.py`'s CLI prompt in this
codebase) shows the shopper what `retrieve` found and asks which one, before `quote` runs.
"""

from __future__ import annotations

from collections.abc import Callable

from payagent.graph.nodes.tool_call import call_tool
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.deps import ToolDeps
from payagent.observability.logging import get_logger

logger = get_logger(__name__)

# The catalog and every policy document in this project are BRL-denominated (see
# `evals/datasets/catalog.json`, POL-001 et al.). `get_quote`'s `currency` argument is an
# assertion checked against the catalog record, never a conversion, so a wrong default just
# fails safely as `CURRENCY_MISMATCH` rather than mis-pricing anything.
DEFAULT_CURRENCY = "BRL"


def make_quote_node(
    deps: ToolDeps, *, default_currency: str = DEFAULT_CURRENCY
) -> Callable[[PurchaseState], PurchaseState]:
    """Build the `quote` node, with `deps` (and the assumed currency) bound by closure."""

    def node(state: PurchaseState) -> PurchaseState:
        if not state.is_active:
            return state

        if state.selected_sku is None:
            # achado #2: a legal pause, not an illegal transition — quote never guesses,
            # whether retrieve found zero, one, or many candidates. Re-entrant: quote's own
            # no-op guard above (`not state.is_active`) makes a later re-invocation with
            # still no selection a no-op rather than looping back through this branch.
            logger.info(
                "quote_needs_clarification", candidate_count=len(state.retrieved_chunks)
            )
            return state.model_copy(update={"status": "needs_clarification"})

        confirmed_ids = {chunk.chunk_id for chunk in state.retrieved_chunks}
        if state.selected_sku not in confirmed_ids:
            raise IllegalTransitionError(
                "state.selected_sku must be one of the chunk_ids retrieve actually returned"
            )

        envelope = call_tool(
            deps,
            "get_quote",
            {
                "sku": state.selected_sku,
                "quantity": state.selected_quantity,
                "currency": default_currency,
            },
        )

        if not envelope["ok"]:
            logger.info("quote_failed", error_code=envelope["error"]["code"])
            return state.model_copy(update={"status": "quote_failed", "quote": envelope})

        data = envelope["data"]
        logger.info("quote_issued", quote_id=data["quote_id"], amount_cents=data["amount_cents"])
        return state.model_copy(
            update={
                "quote": envelope,
                "quote_id": data["quote_id"],
                "quote_amount_cents": data["amount_cents"],
            }
        )

    return node
