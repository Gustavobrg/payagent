"""Tests for the purchase graph's wiring: node order, `PurchaseGraph.run`/`step`, illegal
transitions, and one end-to-end happy path through every real node.

Per-node business logic (policy denial reasons, step-up pausing, "only quote writes value",
...) is covered in `tests/test_plan_node.py`, `test_quote_node.py`, `test_mandate_node.py`,
`test_confirm_node.py`, and `test_settle_node.py`. This file only locks down the graph-level
contract: fixed order, `retrieve` is real, `plan` degrades safely, and every node refuses to
run out of order.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from doubles import AllowAllPolicyEngine, AlwaysVerifyingAuthority
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
    """Replays `responses` in order, one per `.invoke()` call. Ignores message content.

    One instance backs both `plan` (routing) and `retrieve` (the sub-agent loop) when passed
    to `build_graph` — `bind_tools` ignores its argument and returns `self`, so responses are
    consumed strictly in call order regardless of which node is asking.
    """

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


def _route_to_retrieve() -> AIMessage:
    return _tool_call_message("route_decision", {"route": "retrieve"}, call_id="route-1")


def test_node_order_matches_the_state_machine_invariant():
    assert NODE_ORDER == ("plan", "retrieve", "quote", "mandate", "confirm", "settle")


def test_build_graph_rejects_incomplete_node_set():
    with pytest.raises(IllegalTransitionError):
        PurchaseGraph(nodes={"plan": lambda s: s})


def test_run_through_unknown_step_raises(retriever: Retriever, deps):
    graph = build_graph(FakeChatModel(responses=[]), retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    with pytest.raises(IllegalTransitionError):
        graph.run(state, through="checkout")


def test_step_runs_a_single_named_node(retriever: Retriever, deps):
    """The out-of-band entry point `scripts/demo.py` needs to pause and resume mid-graph."""
    graph = build_graph(FakeChatModel(responses=[_route_to_retrieve()]), retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.step("plan", state)

    assert result.status == "in_progress"


def test_step_with_an_unknown_name_raises(retriever: Retriever, deps):
    graph = build_graph(FakeChatModel(responses=[]), retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    with pytest.raises(IllegalTransitionError):
        graph.step("checkout", state)


def test_plan_degrades_to_in_progress_when_the_model_makes_no_routing_call(
    retriever: Retriever, deps
):
    """If the model doesn't call `route_decision` at all, `plan` fails toward retrieval."""
    llm = FakeChatModel(responses=[_final_message()])
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.run(state, through="plan")

    assert result.status == "in_progress"
    assert result.direct_response is None


def test_retrieve_step_populates_state_and_nothing_past_it_runs(retriever: Retriever, deps):
    llm = FakeChatModel(
        responses=[
            _route_to_retrieve(),
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    result = graph.run(state, through="retrieve")

    assert result.status == "in_progress"
    assert result.retrieval_confidence == "high"
    assert any(c.chunk_id == "SKU-SPEAKER" for c in result.retrieved_chunks)
    assert result.quote is None
    assert result.mandate is None


def test_retrieve_does_not_run_once_plan_has_answered_directly(retriever: Retriever, deps):
    llm = FakeChatModel(
        responses=[_tool_call_message("route_decision", {"route": "direct_response", "response": "Oi!"})]
    )
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(user_request="oi")

    result = graph.run(state, through="retrieve")

    assert result.status == "answered_directly"
    assert result.direct_response == "Oi!"
    assert result.retrieved_chunks == []  # retrieve never ran


def test_running_past_retrieve_without_a_confirmed_selection_is_an_illegal_transition(
    retriever: Retriever, deps
):
    """`quote` requires a human-confirmed `selected_sku` — `retrieve` alone never sets one."""
    llm = FakeChatModel(
        responses=[
            _route_to_retrieve(),
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(user_request="fone bluetooth")

    with pytest.raises(IllegalTransitionError):
        graph.run(state, through="quote")


def test_full_pipeline_settles_when_everything_is_pre_confirmed_and_permissive(
    retriever: Retriever, build_deps
):
    """One end-to-end happy path through all six real nodes."""
    llm = FakeChatModel(
        responses=[
            _route_to_retrieve(),
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(
        user_request="fone bluetooth",
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )

    result = graph.run(state)

    assert result.status == "settled"
    assert result.quote_amount_cents == 4999
    assert result.settlement_result["data"]["amount_cents"] == 4999
    assert result.mandate["data"]["payment_mandate_id"]
