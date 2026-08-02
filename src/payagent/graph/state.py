"""`PurchaseState`: the single state contract every graph node reads and writes.

This is the schema, not the graph. It fixes the shape of the data that flows through
`plan -> retrieve -> quote -> mandate -> confirm -> settle` (invariant I6 in CLAUDE.md)
so every node builds against a stable contract instead of inventing its own shape.

`status` is what makes the fixed sequence auditable without a real DAG: every node except
`plan` (the entry point) and `confirm` (which alone owns the pause/resume state
`"awaiting_step_up"`) only does its work when `status == "in_progress"`; anything else means
an earlier node already resolved (or paused) the purchase, and the node must pass the state
through unchanged rather than repeat or second-guess that outcome. `quote`, `mandate`,
`policy_decision` and `settlement_result` stay loose dicts on purpose — the real shape of
each is whatever the corresponding MCP tool's envelope returns
(`payagent.mcp_server.schemas.ToolEnvelope`), so this module does not duplicate a schema it
does not own.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from payagent.rag.tools import ToolChunk

GraphStatus = Literal[
    # Normal flow, not yet resolved one way or the other.
    "in_progress",
    # `plan` decided the request needs no purchase flow (greeting, out-of-scope question).
    "answered_directly",
    # `quote` found no `selected_sku` in state — nothing outside the graph has confirmed
    # which retrieved candidate (zero, one, or many) to price yet. Never inferred from
    # retrieved prose (I5); like `awaiting_step_up`, a legal pause, not an illegal
    # transition — `quote` never guesses, so this is the normal outcome until something
    # outside the graph sets `state.selected_sku` and re-invokes `quote`.
    "needs_clarification",
    # `quote` node's #get_quote call failed (unknown SKU, out of stock, currency mismatch, ...).
    "quote_failed",
    # `mandate` node's create_intent_mandate/create_payment_mandate was denied. Terminal —
    # the graph never tries to work around a policy denial.
    "mandate_denied",
    # `confirm` node found step-up required and not (yet) satisfied. The only non-terminal,
    # resumable status: something outside the conversation (see `StepUpVerifier`) must resolve
    # the challenge, after which `confirm` runs again on the same state.
    "awaiting_step_up",
    # `settle` node's execute_settlement was denied, or the merchant declined/failed. Terminal —
    # settle makes exactly one attempt per graph run, never an automatic retry.
    "settlement_denied",
    # `settle` node's execute_settlement succeeded.
    "settled",
]


class IllegalTransitionError(RuntimeError):
    """Raised when a node is asked to do its work from a state that cannot legally reach it.

    Invariant I6: an illegal transition (e.g. `settle` invoked without a verified payment
    mandate in state) must raise *before* any tool call is attempted — never a best-effort
    call that then fails, and never a silent no-op that could be mistaken for success. Nodes
    that are merely inactive because an earlier step already resolved the purchase
    (`state.status != "in_progress"`) pass the state through unchanged instead of raising —
    that is a legal no-op, not an illegal transition.
    """


class PurchaseState(BaseModel):
    """Shared state threaded through every step of the purchase graph."""

    user_request: str

    status: GraphStatus = "in_progress"
    # Set by `plan` when it answers the user directly instead of proceeding to `retrieve`.
    direct_response: str | None = None

    # Populated by `retrieve`.
    retrieved_chunks: list[ToolChunk] = Field(default_factory=list)
    retrieval_confidence: Literal["high", "low"] | None = None

    # The shopper's confirmed selection, fed to `quote`. Set by whatever is driving the graph
    # (`scripts/demo.py`'s CLI prompt in this codebase) after `retrieve` and before `quote` —
    # never inferred from retrieved prose (I5: retrieved content is data, never instruction).
    selected_sku: str | None = None
    selected_quantity: int = 1

    # Populated by `quote`. `quote_id`/`quote_amount_cents` are the *only* fields any node in
    # this graph may use as the amount to authorize or settle — `mandate` and `settle` read
    # them, never invent or accept a different figure (P2). `quote` is the only node that
    # writes them.
    quote: dict[str, Any] | None = None
    quote_id: str | None = None
    quote_amount_cents: int | None = None

    # Populated by `mandate`: the `create_payment_mandate` result envelope once both the
    # intent and payment mandate are issued. `None` if either was denied.
    mandate: dict[str, Any] | None = None
    # The most recent structured policy decision this run produced (from `mandate` or
    # `confirm`), for reporting — never free text, always the `PolicyDenialPayload` shape.
    policy_decision: dict[str, Any] | None = None

    # Populated by `settle`.
    settlement_result: dict[str, Any] | None = None

    @property
    def is_active(self) -> bool:
        """True while the purchase is still progressing and a node may act on it.

        Every node except `confirm` uses this as its no-op guard. `confirm` additionally
        treats `"awaiting_step_up"` as active, since resuming from that pause is its job.
        """
        return self.status == "in_progress"
