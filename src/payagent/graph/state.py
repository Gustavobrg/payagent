"""`PurchaseState`: the single state contract every graph node reads and writes.

This is the schema, not the graph. It fixes the shape of the data that flows through
`plan -> retrieve -> quote -> mandate -> confirm -> settle` (invariant I6 in CLAUDE.md)
so nodes landing later (bloco 4) have a stable contract to build against instead of each
inventing its own shape. Fields owned by a node that doesn't exist yet stay `None`/empty
until that node lands — populating them early would be guessing at a schema (`quote`,
`mandate`, `policy_decision`, `settlement_result`) that its own node should define.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from payagent.rag.tools import ToolChunk


class PurchaseState(BaseModel):
    """Shared state threaded through every step of the purchase graph."""

    user_request: str

    # Populated by `retrieve` (implemented).
    retrieved_chunks: list[ToolChunk] = Field(default_factory=list)
    retrieval_confidence: Literal["high", "low"] | None = None

    # Populated by `quote`, `mandate`, `confirm`/policy checks, and `settle` respectively
    # (bloco 4). Left as a loose dict for now — the real schema for each belongs to the
    # node that produces it (e.g. money as int cents + ISO currency per CLAUDE.md), not
    # to this skeleton.
    quote: dict[str, Any] | None = None
    mandate: dict[str, Any] | None = None
    policy_decision: dict[str, Any] | None = None
    settlement_result: dict[str, Any] | None = None
