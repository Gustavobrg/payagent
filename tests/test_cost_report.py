"""Tests for scripts/cost_report.py's `aggregate_costs`.

Builds a real, offline "harness execution" fixture using the exact pattern already proven in
tests/test_harness_run.py (hand-rolled fake chat model, `build_graph`, the `retriever`/
`permissive_deps` fixtures from conftest.py, an allow-all rails fixture — no network, no live
LLM). The fake chat model additionally exposes `model_name` and sets `usage_metadata` on its
canned `AIMessage` responses, so real, controlled token counts flow through
`observability/tracing.py::record_llm_usage` — and, via the Langfuse SDK, its native
`cost_details`/`usage_details` fields — exactly the way production code's `ChatOpenAI`
responses do.

The assertion is against an *independent* manual sum computed directly from the raw
`langfuse.observation.cost_details` span attributes in the written file — never by calling any
of `aggregate_costs`'s own internals — so the test can't be circular.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest
from langchain_core.messages import AIMessage
from langfuse import propagate_attributes

from evals.harness.run import LogCapture, run_one_scenario
from evals.harness.scenarios import Scenario
from evals.harness.tracing import Tracer
from payagent.graph.graph import build_graph
from payagent.guardrails import provider as guardrail_provider
from payagent.guardrails.provider import GuardrailProvider, GuardrailVerdict
from payagent.guardrails.rails import build_rails
from payagent.observability import tracing
from scripts.cost_report import aggregate_costs, load_spans


class _AllowAllGuardrailProvider(GuardrailProvider):
    def check_input(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)

    def check_output(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)


@pytest.fixture
def rails(monkeypatch):
    monkeypatch.setattr(guardrail_provider, "OpenRouterLlamaGuard", _AllowAllGuardrailProvider)
    return build_rails()


@dataclass
class _FakeChatModel:
    """Same shape as tests/test_harness_run.py's own fake, plus `model_name` and real
    `usage_metadata` on each response so `record_llm_usage` computes a real, known cost."""

    responses: list[AIMessage]
    model_name: str = "openai/gpt-4o-mini"
    _index: int = field(default=0)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self.responses[self._index]
        self._index += 1
        return response


def _tool_call(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
    )


def _fake_llm() -> _FakeChatModel:
    return _FakeChatModel(
        responses=[
            _tool_call("route_decision", {"route": "retrieve"}, call_id="route-1"),
            _tool_call("search_catalog", {"query": "headphones"}),
            AIMessage(
                content="",
                usage_metadata={"input_tokens": 50, "output_tokens": 5, "total_tokens": 55},
            ),
        ]
    )


def _make_scenario() -> Scenario:
    return Scenario(
        id="TEST-COST-REPORT",
        category="happy_path",
        user="Compra o fone bluetooth SKU-HEADPHONES",
        should_settle=True,
        hint_sku="SKU-HEADPHONES",
        confirms_purchase=True,
        expected_tool="search_catalog",
        expected_final_tool="execute_settlement",
    )


def _run_scenario_with_tracing(spans_path, retriever, permissive_deps, rails):
    exporter = tracing.JsonLinesSpanExporter(str(spans_path))
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)

    graph = build_graph(_fake_llm(), retriever, permissive_deps)
    scenario = _make_scenario()
    tracer = Tracer()
    log_capture = LogCapture(spans_path.parent / "test.log")

    with propagate_attributes(session_id="session-cost-report"):
        result = run_one_scenario(scenario, graph, rails, tracer, log_capture)
    tracing.flush()

    assert result.settled is True
    return result


def test_aggregate_matches_an_independent_manual_sum_of_the_raw_spans(
    tmp_path, retriever, permissive_deps, rails
):
    spans_path = tmp_path / "run.spans.jsonl"
    _run_scenario_with_tracing(spans_path, retriever, permissive_deps, rails)

    spans = load_spans(str(spans_path))
    report = aggregate_costs(spans)

    manual_total = 0.0
    for span in spans:
        attrs = span.get("attributes") or {}
        raw = attrs.get("langfuse.observation.cost_details")
        if raw is None:
            continue
        cost = json.loads(raw).get("total")
        if cost is not None:
            manual_total += cost

    assert report["total_cost_usd"] == pytest.approx(manual_total)
    assert manual_total > 0  # sanity: the fixture actually produced costed spans


def test_aggregate_groups_cost_by_session(tmp_path, retriever, permissive_deps, rails):
    spans_path = tmp_path / "run.spans.jsonl"
    _run_scenario_with_tracing(spans_path, retriever, permissive_deps, rails)

    report = aggregate_costs(load_spans(str(spans_path)))

    assert "session-cost-report" in report["by_session"]
    session_group = report["by_session"]["session-cost-report"]
    assert session_group["cost_usd"] == pytest.approx(report["total_cost_usd"])
    assert session_group["input_tokens"] > 0
    assert session_group["output_tokens"] > 0
    assert session_group["span_count"] > 0


def test_load_spans_reads_one_json_object_per_line(tmp_path):
    path = tmp_path / "spans.jsonl"
    path.write_text(
        json.dumps({"name": "a", "attributes": {}}) + "\n"
        + json.dumps({"name": "b", "attributes": {}}) + "\n",
        encoding="utf-8",
    )

    spans = load_spans(str(path))

    assert [s["name"] for s in spans] == ["a", "b"]


def test_aggregate_costs_surfaces_unknown_model_spans_separately_from_zero_cost():
    spans = [
        {
            "name": "llm.invoke",
            "span_id": "abc",
            "attributes": {"payagent.cost.model_unknown": "some/weird-model"},
        },
        {
            "name": "llm.invoke",
            "span_id": "def",
            "attributes": {
                "langfuse.observation.cost_details": json.dumps({"total": 0.01}),
                "langfuse.observation.usage_details": json.dumps({"input": 10, "output": 5}),
                "session.id": "s1",
            },
        },
    ]

    report = aggregate_costs(spans)

    assert report["total_cost_usd"] == pytest.approx(0.01)
    assert len(report["unknown_model_spans"]) == 1
    assert report["unknown_model_spans"][0]["model"] == "some/weird-model"
    assert report["by_session"]["s1"]["cost_usd"] == pytest.approx(0.01)
    assert report["by_session"]["s1"]["input_tokens"] == 10
    assert report["by_session"]["s1"]["output_tokens"] == 5
