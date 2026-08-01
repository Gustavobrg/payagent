"""The purchase graph: wiring for `plan -> retrieve -> quote -> mandate -> confirm -> settle`.

Plain Python + LangChain only — no LangGraph library. The state machine invariant (I6 in
CLAUDE.md: illegal transition raises) is expressed here as a fixed `NODE_ORDER` a
`PurchaseGraph` runs through in sequence; there is no API to invoke a step out of order or
skip ahead through `run()`. `IllegalTransitionError` itself lives in `payagent.graph.state`
(re-exported here for backward compatibility) so that module and every node module can raise
it without importing this one — nodes are built and wired here, not the other way around.

Every node past `retrieve` now has real logic (`payagent.graph.nodes.{quote,mandate,confirm,settle}`).
A node whose precondition is missing while the purchase is still active raises
`IllegalTransitionError` before attempting any tool call (each node's own docstring says
which precondition); a node that finds the purchase already resolved by an earlier step
(`state.status != "in_progress"`, or, for `confirm` only, also not `"awaiting_step_up"`)
passes the state through unchanged instead — that is a legal no-op, not an illegal transition.
"""

from __future__ import annotations

from collections.abc import Callable

from payagent.graph.nodes.confirm import make_confirm_node
from payagent.graph.nodes.mandate import make_mandate_node
from payagent.graph.nodes.plan import ChatModel as PlanChatModel
from payagent.graph.nodes.plan import make_plan_node
from payagent.graph.nodes.quote import DEFAULT_CURRENCY, make_quote_node
from payagent.graph.nodes.retrieve import (
    DEFAULT_MAX_ITERATIONS,
    ToolCallingChatModel,
    make_retrieve_node,
)
from payagent.graph.nodes.settle import make_settle_node
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.mcp_server.deps import ToolDeps
from payagent.rag.retriever import Retriever

__all__ = [
    "DEFAULT_CURRENCY",
    "DEFAULT_MAX_ITERATIONS",
    "NODE_ORDER",
    "IllegalTransitionError",
    "NodeStep",
    "PurchaseGraph",
    "build_graph",
]

NODE_ORDER: tuple[str, ...] = ("plan", "retrieve", "quote", "mandate", "confirm", "settle")

NodeStep = Callable[[PurchaseState], PurchaseState]


def _make_retrieve_step(
    llm: ToolCallingChatModel,
    retriever: Retriever,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> NodeStep:
    """Adapt `retrieve`'s dict-partial-update node shape to the `NodeStep` contract.

    `retrieve` predates `PurchaseState` gaining `status`; it also has nothing to say about
    routing, so it must not run at all once `plan` has already resolved the request
    (`status != "in_progress"`) — a greeting must not trigger a catalog search.
    """
    node = make_retrieve_node(llm, retriever, max_iterations=max_iterations)

    def step(state: PurchaseState) -> PurchaseState:
        if not state.is_active:
            return state
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

    def step(self, name: str, state: PurchaseState) -> PurchaseState:
        """Run exactly one named step, out of band from `run`.

        The seam an interactive driver (`scripts/demo.py`) needs: it pauses between `retrieve`
        and `quote` for the shopper to confirm a SKU, and may call `confirm` more than once —
        first landing on `"awaiting_step_up"`, then again after the step-up channel resolves
        the challenge — neither of which `run`'s single pass through `NODE_ORDER` supports.
        Still raises `IllegalTransitionError` for a name outside `NODE_ORDER`; each node
        itself still enforces its own precondition (see module docstrings) regardless of how
        it is invoked.
        """
        if name not in NODE_ORDER:
            raise IllegalTransitionError(f"unknown step: {name!r}")
        return self._nodes[name](state)


def build_graph(
    llm: ToolCallingChatModel | PlanChatModel,
    retriever: Retriever,
    deps: ToolDeps,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    quote_currency: str = DEFAULT_CURRENCY,
) -> PurchaseGraph:
    """Wire the full graph: real logic in every node.

    `llm` backs both `plan` (routing) and `retrieve` (the retrieval sub-agent) — one model,
    two different tool bindings, exactly as `payagent.graph.nodes.plan.ChatModel` promises
    (any model satisfying `retrieve`'s stricter `ToolCallingChatModel` also satisfies it).
    `deps` backs `quote`, `mandate`, `confirm`, and `settle` — the same `ToolDeps` the MCP
    server itself uses, so the graph and a direct MCP client reach the identical policy engine,
    mandate authority, and step-up state, never a graph-local copy of any of them.
    """
    return PurchaseGraph(
        {
            "plan": make_plan_node(llm),
            "retrieve": _make_retrieve_step(llm, retriever, max_iterations=max_iterations),
            "quote": make_quote_node(deps, default_currency=quote_currency),
            "mandate": make_mandate_node(deps),
            "confirm": make_confirm_node(deps),
            "settle": make_settle_node(deps),
        }
    )
