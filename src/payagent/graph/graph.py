"""Minimal skeleton of the purchase graph.

Plain Python + LangChain only — no LangGraph library. The state machine invariant
(I6 in CLAUDE.md: `plan -> retrieve -> quote -> mandate -> confirm -> settle`, illegal
transition raises) is expressed here as a fixed `NODE_ORDER` a `PurchaseGraph` runs
through in sequence; there is no API to invoke a step out of order or skip ahead.

Only `retrieve` has real logic (see `payagent.graph.nodes.retrieve`). The other five
are pass-through placeholders that return the state unchanged — they land in bloco 4.
This module is the wiring/contract, not the functional graph.
"""

from __future__ import annotations

from collections.abc import Callable

from payagent.graph.nodes.retrieve import (
    DEFAULT_MAX_ITERATIONS,
    ToolCallingChatModel,
    make_retrieve_node,
)
from payagent.graph.state import PurchaseState
from payagent.rag.retriever import Retriever

NODE_ORDER: tuple[str, ...] = ("plan", "retrieve", "quote", "mandate", "confirm", "settle")

NodeStep = Callable[[PurchaseState], PurchaseState]


class IllegalTransitionError(RuntimeError):
    """Raised when the graph is asked to run to, or is built without, a step outside NODE_ORDER."""


def _passthrough(state: PurchaseState) -> PurchaseState:
    """Placeholder for plan/quote/mandate/confirm/settle: no business logic yet (bloco 4)."""
    return state


def _make_retrieve_step(
    llm: ToolCallingChatModel,
    retriever: Retriever,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> NodeStep:
    node = make_retrieve_node(llm, retriever, max_iterations=max_iterations)

    def step(state: PurchaseState) -> PurchaseState:
        update = node({"query": state.user_request})
        return state.model_copy(update=update)

    return step


class PurchaseGraph:
    """Runs `NODE_ORDER` steps in sequence over a `PurchaseState`."""

    def __init__(self, nodes: dict[str, NodeStep]):
        missing = set(NODE_ORDER) - set(nodes)
        if missing:
            raise IllegalTransitionError(f"graph is missing node(s): {sorted(missing)}")
        self._nodes = nodes

    def run(self, state: PurchaseState, *, through: str | None = None) -> PurchaseState:
        """Run all steps from `plan` onward, stopping after `through` (inclusive) if given.

        `through=None` runs the full pipeline. Raises `IllegalTransitionError` if `through`
        names anything outside `NODE_ORDER` — there's no way to target a step that doesn't
        exist in the fixed sequence.
        """
        if through is not None and through not in NODE_ORDER:
            raise IllegalTransitionError(f"unknown step: {through!r}")

        for name in NODE_ORDER:
            state = self._nodes[name](state)
            if name == through:
                break
        return state


def build_graph(
    llm: ToolCallingChatModel,
    retriever: Retriever,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> PurchaseGraph:
    """Wire the minimal skeleton: real `retrieve`, pass-through everywhere else."""
    nodes: dict[str, NodeStep] = {name: _passthrough for name in NODE_ORDER}
    nodes["retrieve"] = _make_retrieve_step(llm, retriever, max_iterations=max_iterations)
    return PurchaseGraph(nodes)
