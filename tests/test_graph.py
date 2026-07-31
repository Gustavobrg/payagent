"""Tests for the minimal graph skeleton: node order, placeholders, illegal transitions.

Not the functional graph (plan/quote/mandate/confirm/settle are pass-throughs until
bloco 4) — these tests only lock down the wiring contract: fixed order, placeholders
that truly do nothing, `retrieve` is real, and out-of-order/unknown steps raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from payagent.graph.graph import NODE_ORDER, IllegalTransitionError, PurchaseGraph, build_graph
from payagent.graph.state import PurchaseState
from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    create_collection_if_missing,
    ingest_chunks,
)
from payagent.rag.retriever import Retriever

CATALOG_ITEMS = [
    {
        "sku_id": "SKU-SPEAKER",
        "name": "Nexus Bluetooth Speaker",
        "description": "Portable and powerful sound.",
        "price_cents": 4999,
        "currency": "BRL",
        "category": "electronics",
        "merchant_id": "MERCH-02",
        "stock": 20,
    },
]


@pytest.fixture
def retriever() -> Retriever:
    client = QdrantClient(":memory:")
    create_collection_if_missing(client, CATALOG_COLLECTION)
    chunks = chunk_catalog(CATALOG_ITEMS)
    ingest_chunks(client, CATALOG_COLLECTION, chunks, DeterministicEmbedder())
    return Retriever(client=client, use_deterministic_embedder=True)


@dataclass
class FakeChatModel:
    """Replays `responses` in order, one per `.invoke()` call. Ignores message content."""

    responses: list[AIMessage]
    _index: int = field(default=0)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self.responses[self._index]
        self._index += 1
        return response


def _tool_call_message(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _final_message() -> AIMessage:
    return AIMessage(content="")


def test_node_order_matches_the_state_machine_invariant():
    assert NODE_ORDER == ("plan", "retrieve", "quote", "mandate", "confirm", "settle")


def test_build_graph_rejects_incomplete_node_set():
    with pytest.raises(IllegalTransitionError):
        PurchaseGraph(nodes={"plan": lambda s: s})


def test_run_through_unknown_step_raises(retriever: Retriever):
    graph = build_graph(FakeChatModel(responses=[_final_message()]), retriever)
    state = PurchaseState(user_request="fone bluetooth")

    with pytest.raises(IllegalTransitionError):
        graph.run(state, through="checkout")


def test_placeholders_pass_state_through_unchanged(retriever: Retriever):
    llm = FakeChatModel(responses=[_final_message()])
    graph = build_graph(llm, retriever)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.run(state, through="plan")

    assert result == state
    assert result.retrieved_chunks == []


def test_full_pipeline_only_retrieve_populates_state(retriever: Retriever):
    llm = FakeChatModel(
        responses=[
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    graph = build_graph(llm, retriever)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.run(state)

    assert result.retrieval_confidence == "high"
    assert any(c.chunk_id == "SKU-SPEAKER" for c in result.retrieved_chunks)
    # quote/mandate/confirm/settle are still placeholders — untouched.
    assert result.quote is None
    assert result.mandate is None
    assert result.policy_decision is None
    assert result.settlement_result is None


def test_run_stops_after_through_step(retriever: Retriever):
    llm = FakeChatModel(
        responses=[
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    graph = build_graph(llm, retriever)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.run(state, through="retrieve")

    assert result.retrieval_confidence == "high"
    assert result.quote is None
