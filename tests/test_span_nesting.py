"""Verifies OTel parent/child span nesting across the two seams this codebase's
instrumentation relies on:

- `mcp.tool.*` (dispatch.py) under its `graph.<node>` span — safe by inspection
  (`tool_call.call_tool`'s `anyio.run` creates its Task synchronously on the calling
  thread, so `contextvars.copy_context()` captures the ambient OTel context before the new
  event loop starts) but worth a concrete assertion rather than a narrative assumption.
- `rail.<action_name>` (guardrails/actions.py) under `rail.check_input`/`rail.check_output`
  (guardrails/rails.py) — NOT safe by inspection, since NeMo dispatches actions through its
  own internal async machinery, a different framework's event loop.
"""

from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from payagent.graph.graph import build_graph
from payagent.graph.state import PurchaseState
from payagent.observability import tracing
from payagent.rag.tools import ToolChunk


class _UnusedChatModel:
    """`build_graph` wires `plan`/`retrieve` at construction time (their own `bind_tools`
    calls run then), but this test only ever calls `graph.step("quote", ...)` — `invoke`
    is never actually reached."""

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        raise AssertionError("not expected to be called by this test")


def test_mcp_tool_span_nests_under_its_graph_node_span(deps):
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)

    graph = build_graph(_UnusedChatModel(), deps.retriever, deps)
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[ToolChunk(chunk_id="SKU-SPEAKER", source="catalog", score=0.9, text="")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )

    result = graph.step("quote", state)
    assert result.status == "in_progress"
    tracing.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "graph.quote" in spans
    assert "mcp.tool.get_quote" in spans

    graph_span = spans["graph.quote"]
    tool_span = spans["mcp.tool.get_quote"]
    assert tool_span.parent is not None
    assert tool_span.parent.span_id == graph_span.context.span_id
