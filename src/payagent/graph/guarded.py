"""Wires the two NeMo guardrail seams around a purchase graph run.

CLAUDE.md: "Plugue o LLMRails em dois pontos: antes do nó plan e depois do nó terminal de
resposta." — `check_input` (`payagent.guardrails.rails`) runs before `PurchaseGraph.run`
ever touches `plan`, so a blocked request never reaches the state machine at all;
`check_output` runs once the run finishes, against `graph.respond.compose_response`'s
rendering of whichever node the run actually stopped at (`plan` on a direct answer,
`settle` on a full purchase, or any denial/pause in between — `PurchaseGraph.run` already
handles all of those as legal terminal states, see its own module docstring).

NeMo itself never drives the conversation here (CLAUDE.md: "Quem conduz a conversa é o
LangGraph") — both guardrail calls pass `dialog=False`, so this module's only interaction
with the graph is calling `.run()` once, exactly the way `PurchaseGraph.run` is already
meant to be called.
"""

from __future__ import annotations

from dataclasses import dataclass

from nemoguardrails import LLMRails

from payagent.graph.graph import PurchaseGraph
from payagent.graph.respond import compose_response
from payagent.graph.state import PurchaseState
from payagent.guardrails.rails import check_input, check_output
from payagent.observability.tracing import open_session_span


@dataclass(frozen=True)
class GuardedRunResult:
    """`run_guarded`'s result: the graph's final state, plus the text that actually reaches
    the user — never `compose_response`'s raw rendering directly, always what `check_output`
    returned (unchanged, DLP-masked, or replaced by a refusal)."""

    state: PurchaseState
    response: str
    input_blocked: bool


def run_guarded(
    graph: PurchaseGraph,
    rails: LLMRails,
    state: PurchaseState,
    *,
    session_id: str | None = None,
) -> GuardedRunResult:
    """Run `state` through `graph.run`, guarded by `rails` on both ends.

    Precondition matches `PurchaseGraph.run` itself: `state` must already be ready to run to
    completion (e.g. `selected_sku` set for a real purchase) — this wraps the existing state
    machine, it does not add an interactive SKU-selection step (that stays `scripts/demo.py`'s
    job, via `PurchaseGraph.step`).

    `session_id` defaults to a fresh UUID when omitted, so a caller that doesn't care about
    grouping (most tests) doesn't have to supply one. One session is one purchase attempt here
    (CLAUDE.md's flow diagram) — `open_session_span` propagates it to every span this call and
    everything it triggers (input/output rails, graph nodes, MCP tool calls) creates, for
    `scripts/cost_report.py`'s aggregation.
    """
    with open_session_span("purchase.run", session_id=session_id):
        input_result = check_input(rails, state.user_request)
        if not input_result.allowed:
            blocked_state = state.model_copy(
                update={"status": "answered_directly", "direct_response": input_result.refusal}
            )
            return GuardedRunResult(
                state=blocked_state, response=input_result.refusal, input_blocked=True
            )

        final_state = graph.run(state)
        response_text = compose_response(final_state)

        output_result = check_output(
            rails,
            user_message=state.user_request,
            bot_message=response_text,
            retrieved_chunks=final_state.retrieved_chunks,
        )
        return GuardedRunResult(
            state=final_state, response=output_result.text, input_blocked=False
        )
