"""Tests for evals/harness/tracing.py — the in-process instrumentation the harness uses
instead of scraping logs (which deliberately never carry argument values, per I2)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pytest
from doubles import AllowAllPolicyEngine, AlwaysVerifyingAuthority
from langchain_core.messages import AIMessage
from qdrant_client import QdrantClient

from evals.harness.tracing import (
    ToolCallRecord,
    Tracer,
    TracingChatModel,
    patch_mcp_handler_tracing,
    patch_rag_tool_tracing,
)
from payagent.graph.nodes.retrieve import run_retrieval_subagent
from payagent.mcp_server.dispatch import dispatch_tool_call
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
        "sku_id": "SKU-HEADPHONES",
        "name": "AuraSound Pro Wireless Headphones",
        "description": "Great audio quality with noise cancellation.",
        "price_cents": 12999,
        "currency": "BRL",
        "category": "electronics",
        "merchant_id": "MERCH-01",
        "stock": 45,
    },
]


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


def _tool_call_message(name: str, args: dict) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call-1"}])


# --- TracingChatModel --------------------------------------------------------------


def test_tracing_chat_model_records_latency_and_usage():
    tracer = Tracer()
    tracer.begin("SC-001", node="plan")
    inner = FakeChatModel(
        responses=[
            AIMessage(
                content="hi",
                usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            )
        ]
    )
    model = TracingChatModel(inner, tracer)

    model.invoke([])

    assert len(tracer.llm_calls) == 1
    record = tracer.llm_calls[0]
    assert record.scenario_id == "SC-001"
    assert record.node == "plan"
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    assert record.total_tokens == 15
    assert record.latency_ms >= 0


def test_tracing_chat_model_survives_bind_tools():
    tracer = Tracer()
    tracer.begin("SC-002", node="retrieve")
    inner = FakeChatModel(responses=[AIMessage(content="done")])
    model = TracingChatModel(inner, tracer)

    bound = model.bind_tools([])
    bound.invoke([])

    assert len(tracer.llm_calls) == 1
    assert tracer.llm_calls[0].scenario_id == "SC-002"


# --- RAG tool tracing --------------------------------------------------------------


def test_patch_rag_tool_tracing_records_a_real_search_catalog_call(retriever):
    tracer = Tracer()
    tracer.begin("SC-003", node="retrieve")
    # Must actually match CATALOG_ITEMS' one item ("AuraSound Pro Wireless Headphones") —
    # Retriever.search's relevance floor (achado #5) drops zero-overlap candidates instead
    # of padding to top_k, so an unrelated query would make `outcome.chunks` empty below,
    # which is not what this test is checking (it's checking the tracing plumbing wraps a
    # real search_catalog call, not retrieval relevance).
    llm = FakeChatModel(
        responses=[
            _tool_call_message("search_catalog", {"query": "wireless headphones"}),
            AIMessage(content="done"),
        ]
    )

    with patch_rag_tool_tracing(tracer):
        outcome = run_retrieval_subagent(llm, retriever, "wireless headphones")

    assert outcome.chunks  # the fake model's tool call actually ran against real Qdrant
    rag_calls = [c for c in tracer.tool_calls if c.layer == "rag"]
    assert len(rag_calls) == 1
    assert rag_calls[0].tool == "search_catalog"
    assert rag_calls[0].args["_positional"] == ["wireless headphones"]
    assert rag_calls[0].ok is True


def test_patch_rag_tool_tracing_restores_originals_on_exit(retriever):
    import payagent.graph.nodes.retrieve as retrieve_module

    original = retrieve_module.search_catalog
    tracer = Tracer()
    with patch_rag_tool_tracing(tracer):
        assert retrieve_module.search_catalog is not original
    assert retrieve_module.search_catalog is original


# --- MCP handler tracing -------------------------------------------------------------


def test_patch_mcp_handler_tracing_records_a_real_get_quote_call(retriever, build_deps):
    deps = build_deps(policy=AllowAllPolicyEngine(), mandate_authority=AlwaysVerifyingAuthority())
    tracer = Tracer()
    tracer.begin("SC-004")

    with patch_mcp_handler_tracing(tracer):
        import anyio

        anyio.run(
            dispatch_tool_call,
            deps,
            "get_quote",
            {"sku": "SKU-HEADPHONES", "quantity": 1, "currency": "BRL"},
        )

    mcp_calls = [c for c in tracer.tool_calls if c.layer == "mcp"]
    assert len(mcp_calls) == 1
    assert mcp_calls[0].tool == "get_quote"
    assert mcp_calls[0].args["sku"] == "SKU-HEADPHONES"
    assert mcp_calls[0].ok is True


def test_patch_mcp_handler_tracing_records_a_denied_call_with_its_error_code(retriever, deps):
    tracer = Tracer()
    tracer.begin("SC-005")

    with patch_mcp_handler_tracing(tracer):
        import anyio

        anyio.run(
            dispatch_tool_call,
            deps,
            "get_quote",
            {"sku": "SKU-DOES-NOT-EXIST", "quantity": 1, "currency": "BRL"},
        )

    mcp_calls = [c for c in tracer.tool_calls if c.layer == "mcp"]
    assert len(mcp_calls) == 1
    assert mcp_calls[0].ok is False
    assert mcp_calls[0].error_code == "UNKNOWN_SKU"


# --- Thread-safety (evals.harness.run --workers > 1) --------------------------------


def test_tracer_context_is_isolated_per_thread():
    """Each worker thread must see its own scenario_id/node, never another thread's —
    this is the bug --workers > 1 would hit if `context` were one shared RunContext."""
    tracer = Tracer()
    seen: dict[str, str] = {}
    barrier = threading.Barrier(2)

    def worker(scenario_id: str) -> None:
        tracer.begin(scenario_id, node="retrieve")
        barrier.wait()  # force both threads to have set `begin` before either reads back
        time.sleep(0.01)
        seen[scenario_id] = tracer.context.scenario_id

    threads = [
        threading.Thread(target=worker, args=("SC-A",)),
        threading.Thread(target=worker, args=("SC-B",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert seen == {"SC-A": "SC-A", "SC-B": "SC-B"}


def test_tracer_record_calls_are_safe_under_concurrent_writers():
    tracer = Tracer()

    def record_many(scenario_id: str) -> None:
        tracer.begin(scenario_id)
        for _ in range(200):
            tracer.record_tool_call(
                ToolCallRecord(
                    scenario_id=tracer.context.scenario_id,
                    layer="rag",
                    tool="search_catalog",
                    args={},
                    ok=True,
                    error_code=None,
                    latency_ms=0.0,
                )
            )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(record_many, [f"SC-{i}" for i in range(8)]))

    assert len(tracer.tool_calls) == 8 * 200
    for i in range(8):
        _, calls = tracer.calls_for(f"SC-{i}")
        assert len(calls) == 200
        assert all(c.scenario_id == f"SC-{i}" for c in calls)
