"""Tests for observability/tracing.py's DLP choke point — same spirit as test_dlp.py's I2
permanent regression test: exercise the real sink, not `dlp.mask`/`mask_value` in isolation.
"""

from __future__ import annotations

import json

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from payagent.observability import tracing

VISA_TEST_PAN = "4111111111111111"


def _configured_exporter() -> InMemorySpanExporter:
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)
    return exporter


def _flush() -> None:
    tracing.flush()


def test_i2_pan_never_appears_in_a_span_attribute_via_set_safe_attributes():
    """The real sink: `set_safe_attributes` must redact a PAN before it becomes an attribute."""
    exporter = _configured_exporter()

    with tracing.start_span(
        "test.span",
        raw_pan=VISA_TEST_PAN,
        message=f"attempting charge for card {VISA_TEST_PAN}",
    ):
        pass
    _flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    serialized = str(dict(spans[0].attributes))
    assert VISA_TEST_PAN not in serialized


def test_i2_pan_never_appears_in_nested_dict_attribute():
    """Dict-valued attributes get JSON-encoded after masking, not before."""
    exporter = _configured_exporter()

    with tracing.start_span("test.span", request={"card_number": VISA_TEST_PAN, "sku": "SKU-1"}):
        pass
    _flush()

    spans = exporter.get_finished_spans()
    serialized = str(dict(spans[0].attributes))
    assert VISA_TEST_PAN not in serialized


def test_non_sensitive_attributes_pass_through_unchanged():
    exporter = _configured_exporter()

    with tracing.start_span("test.span", tool="get_quote", ok=True, amount_cents=1500):
        pass
    _flush()

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert attrs["tool"] == "get_quote"
    assert attrs["ok"] is True
    assert attrs["amount_cents"] == 1500


def test_none_valued_attributes_are_skipped_not_set():
    exporter = _configured_exporter()

    with tracing.start_span("test.span", error_code=None, tool="get_quote"):
        pass
    _flush()

    spans = exporter.get_finished_spans()
    attrs = dict(spans[0].attributes)
    assert "error_code" not in attrs
    assert attrs["tool"] == "get_quote"


def test_exception_inside_start_span_is_recorded_and_reraised():
    exporter = _configured_exporter()

    class _Boom(Exception):
        pass

    try:
        with tracing.start_span("test.span"):
            raise _Boom("kaboom")
    except _Boom:
        pass
    else:
        raise AssertionError("expected _Boom to propagate")
    _flush()

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = dict(spans[0].attributes)
    assert attrs["langfuse.observation.level"] == "ERROR"
    assert "_Boom" in attrs["langfuse.observation.status_message"]


def test_session_id_propagates_to_every_span_within_scope():
    from langfuse import propagate_attributes

    exporter = _configured_exporter()

    with tracing.start_span("test.span.root"), propagate_attributes(session_id="session-1"):
        with tracing.start_span("test.span.a"):
            pass
        with tracing.start_span("test.span.b"):
            pass
    with tracing.start_span("test.span.outside_scope"):
        pass
    _flush()

    spans = {s.name: dict(s.attributes) for s in exporter.get_finished_spans()}
    assert spans["test.span.a"]["session.id"] == "session-1"
    assert spans["test.span.b"]["session.id"] == "session-1"
    assert "session.id" not in spans["test.span.outside_scope"]


def test_record_llm_usage_computes_cost_for_a_known_model():
    exporter = _configured_exporter()

    with tracing.start_generation("llm.invoke", model="openai/gpt-4o-mini") as gen:
        tracing.record_llm_usage(
            gen, model="openai/gpt-4o-mini", input_tokens=1000, output_tokens=1000
        )
    _flush()

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["langfuse.observation.model.name"] == "openai/gpt-4o-mini"
    usage = json.loads(attrs["langfuse.observation.usage_details"])
    assert usage == {"input": 1000, "output": 1000}
    cost = json.loads(attrs["langfuse.observation.cost_details"])
    input_price, output_price = tracing.PRICE_PER_1K_TOKENS_USD["openai/gpt-4o-mini"]
    assert cost["total"] == input_price + output_price
    assert "payagent.cost.model_unknown" not in attrs


def test_record_llm_usage_surfaces_unknown_model_instead_of_zero_cost():
    exporter = _configured_exporter()

    with tracing.start_generation("llm.invoke", model="some/unpriced-model") as gen:
        tracing.record_llm_usage(
            gen, model="some/unpriced-model", input_tokens=100, output_tokens=50
        )
    _flush()

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["payagent.cost.model_unknown"] == "some/unpriced-model"
    assert "langfuse.observation.cost_details" not in attrs


def test_i2_pan_never_appears_in_native_input_output_via_set_safe_io():
    """The real sink for `set_safe_io`: a PAN in either `input` or `output` must be redacted
    before it reaches `langfuse.observation.input`/`.output`, exactly like
    `set_safe_attributes` above but for Langfuse's native fields."""
    exporter = _configured_exporter()

    with tracing.start_span("test.span") as span:
        tracing.set_safe_io(
            span,
            input={"message": f"card {VISA_TEST_PAN}"},
            output=f"charged {VISA_TEST_PAN}",
        )
    _flush()

    spans = exporter.get_finished_spans()
    serialized = str(dict(spans[0].attributes))
    assert VISA_TEST_PAN not in serialized
    assert "REDACTED" in serialized


def test_set_safe_io_none_leaves_the_field_unset_not_cleared():
    exporter = _configured_exporter()

    with tracing.start_span("test.span") as span:
        tracing.set_safe_io(span, input="only input")
    _flush()

    attrs = dict(exporter.get_finished_spans()[0].attributes)
    assert attrs["langfuse.observation.input"] == "only input"
    assert "langfuse.observation.output" not in attrs


def test_json_lines_span_exporter_writes_one_redacted_line_per_span(tmp_path):
    path = tmp_path / "spans.jsonl"
    exporter = tracing.JsonLinesSpanExporter(str(path))
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)

    with tracing.start_span("test.span", raw_pan=VISA_TEST_PAN):
        pass
    _flush()

    content = path.read_text(encoding="utf-8")
    assert VISA_TEST_PAN not in content
    lines = [line for line in content.splitlines() if line.strip()]
    assert len(lines) == 1
