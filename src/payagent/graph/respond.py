"""Renders a finished `PurchaseState` as the single user-facing response string.

This is what `payagent.guardrails.rails.check_output` checks — the output guardrail seam
plugged in "after the terminal response node" (CLAUDE.md / the graph's own state-machine
docstring: whichever node the run actually stopped at, see `graph.guarded.run_guarded`).

Every branch is either `direct_response` (already LLM-authored prose, `plan`'s own job) or
built straight from a node's structured envelope/denial fields — never `state.user_request`
itself, so raw user text can't be echoed back out as if it were a grounded claim (that would
defeat the point of the output rail checking it). The `settled` branch cites `selected_sku`
(invariant I4: a factual claim about what was purchased needs a `chunk_id` citation) —
`quote` already asserted it's one of `retrieve`'s own chunk_ids, so `grounding_check` can
verify it against `state.retrieved_chunks`.
"""

from __future__ import annotations

from payagent.graph.state import PurchaseState

_STEP_UP_MESSAGE = (
    "Additional confirmation (step-up) is required before this purchase can proceed."
)
_FALLBACK_MESSAGE = "I could not complete this request."
_NOTHING_FOUND_MESSAGE = (
    "I couldn't find a matching product. Could you tell me more specifically what you're "
    "looking for?"
)


def compose_response(state: PurchaseState) -> str:
    """Render `state` (after a graph run) as the response text to show the user."""
    if state.direct_response is not None:
        return state.direct_response

    if state.status == "settled":
        data = state.settlement_result["data"]
        citation = f" [[{state.selected_sku}]]" if state.selected_sku else ""
        return (
            f"Purchase confirmed: settlement {data['settlement_id']} for "
            f"{data['amount_cents']} {data['currency']}.{citation}"
        )

    if state.status == "needs_clarification":
        if not state.retrieved_chunks:
            return _NOTHING_FOUND_MESSAGE
        # Only chunk_ids are listed, never chunk.text — that text is untrusted retrieved
        # prose (I5) and echoing it here would be exactly the round trip
        # untrusted.py::unwrap_untrusted_text's docstring warns against.
        candidate_ids = ", ".join(chunk.chunk_id for chunk in state.retrieved_chunks)
        return f"Which of these did you mean: {candidate_ids}?"

    if state.status == "quote_failed":
        error = state.quote["error"]
        return f"I could not price this item ({error['code']}): {error['message']}"

    if state.status in ("mandate_denied", "settlement_denied"):
        if state.policy_decision is not None:
            return f"This purchase was not authorized: {state.policy_decision['reason']}"
        error = (state.settlement_result or {}).get("error", {})
        return f"This purchase was not authorized ({error.get('code', 'unknown')}): {error.get('message', '')}".rstrip()

    if state.status == "awaiting_step_up":
        return _STEP_UP_MESSAGE

    return _FALLBACK_MESSAGE
