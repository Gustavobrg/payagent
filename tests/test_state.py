"""Tests for `PurchaseState`: the state contract, not any node's behavior."""

from __future__ import annotations

from payagent.graph.state import PurchaseState
from payagent.rag.tools import ToolChunk


def test_defaults_are_empty_for_steps_without_a_node_yet():
    state = PurchaseState(user_request="fone bluetooth ate R$150")

    assert state.retrieved_chunks == []
    assert state.retrieval_confidence is None
    assert state.quote is None
    assert state.mandate is None
    assert state.policy_decision is None
    assert state.settlement_result is None


def test_accepts_tool_chunks_and_is_updatable_via_model_copy():
    state = PurchaseState(user_request="fone bluetooth")
    chunk = ToolChunk(chunk_id="SKU-1", source="catalog", score=0.9, text="<untrusted>...</untrusted>")

    updated = state.model_copy(update={"retrieved_chunks": [chunk], "retrieval_confidence": "high"})

    assert updated.retrieved_chunks == [chunk]
    assert updated.retrieval_confidence == "high"
    # model_copy doesn't mutate the original.
    assert state.retrieved_chunks == []
