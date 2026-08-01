"""Tests for graph/guarded.py — the two LLMRails seams wired around a purchase graph run:
input check before `plan` ever sees the request, output check after the run's terminal
state is rendered by `graph.respond.compose_response`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from doubles import AllowAllPolicyEngine, AlwaysVerifyingAuthority
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from payagent.graph.graph import build_graph
from payagent.graph.guarded import run_guarded
from payagent.graph.state import PurchaseState
from payagent.guardrails import provider as guardrail_provider
from payagent.guardrails.provider import GuardrailProvider, GuardrailVerdict
from payagent.guardrails.rails import build_rails
from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    create_collection_if_missing,
    ingest_chunks,
)
from payagent.rag.retriever import Retriever

VISA_TEST_PAN = "4111111111111111"

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


class _AllowAllProvider(GuardrailProvider):
    def check_input(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)

    def check_output(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)


@pytest.fixture
def rails(monkeypatch):
    monkeypatch.setattr(guardrail_provider, "OpenRouterLlamaGuard", _AllowAllProvider)
    return build_rails()


@pytest.fixture
def retriever() -> Retriever:
    client = QdrantClient(":memory:")
    create_collection_if_missing(client, CATALOG_COLLECTION)
    ingest_chunks(client, CATALOG_COLLECTION, chunk_catalog(CATALOG_ITEMS), DeterministicEmbedder())
    return Retriever(client=client, use_deterministic_embedder=True)


@dataclass
class FakeChatModel:
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


def test_input_blocked_short_circuits_before_the_graph_ever_runs(retriever, deps, rails):
    """Zero FakeChatModel responses queued — if `plan` ran at all, `.invoke()` would
    IndexError. It never gets the chance because the input rail blocks first."""
    graph = build_graph(FakeChatModel(responses=[]), retriever, deps)
    state = PurchaseState(user_request=f"meu cartao e {VISA_TEST_PAN}")

    result = run_guarded(graph, rails, state)

    assert result.input_blocked is True
    assert result.state.status == "answered_directly"
    assert result.response


def test_full_pipeline_settles_and_output_passes_the_guardrail(retriever, build_deps, rails):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    llm = FakeChatModel(
        responses=[
            _route_to_retrieve(),
            _tool_call_message("search_catalog", {"query": "bluetooth speaker"}),
            _final_message(),
        ]
    )
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(
        user_request="fone bluetooth", selected_sku="SKU-SPEAKER", selected_quantity=1
    )

    result = run_guarded(graph, rails, state)

    assert result.input_blocked is False
    assert result.state.status == "settled"
    assert "[[SKU-SPEAKER]]" in result.response


def test_greeting_short_circuits_at_plan_and_output_still_passes_through(retriever, deps, rails):
    llm = FakeChatModel(
        responses=[
            _tool_call_message("route_decision", {"route": "direct_response", "response": "Oi!"})
        ]
    )
    graph = build_graph(llm, retriever, deps)
    state = PurchaseState(user_request="oi")

    result = run_guarded(graph, rails, state)

    assert result.input_blocked is False
    assert result.state.status == "answered_directly"
    assert result.response == "Oi!"
