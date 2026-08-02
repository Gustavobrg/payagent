"""Langfuse SDK (v4, OTel-based) instrumentation — the single sink every span in this
codebase goes through.

Uses the Langfuse Python SDK's own tracing client (`get_client()`,
`start_as_current_observation`, `LangfuseSpan`/`LangfuseGeneration`) rather than the raw
OpenTelemetry API directly — the SDK is itself OTel-based (every observation is a real OTel
span underneath), so this still produces standard OTel spans exportable to any OTel backend;
it just gets Langfuse's native cost/token calculation, session propagation, and Langfuse Cloud
export for free instead of hand-rolling an OTLP exporter and a GenAI-semconv attribute set.

`@observe` (Langfuse's function decorator) is deliberately NOT used anywhere in this codebase.
Its default behavior auto-captures a decorated function's arguments as `input` and its return
value as `output` — for a graph node that would be the entire `PurchaseState` (which can carry
mandate/settlement detail and retrieved chunk text) and for an MCP tool call it would be the
raw `request`/`response` objects `mcp_server/dispatch.py`'s own docstring explicitly forbids
ever logging. Native spans, built with only the curated, closed-set attributes each wrap point
already uses for logging, are the only shape compatible with invariant I2 here.

**Invariant I2 still applies, unchanged — only the carrier changed.** `set_safe_attributes` is
the mandatory choke point for every *custom* attribute that has no native Langfuse field (named
per GenAI semantic conventions where applicable, e.g. `mcp.tool` fields, rail-action results,
graph status) — every call site must route through it, never
`observation._otel_span.set_attribute` directly. It runs each value through
`payagent.guardrails.dlp.mask_value` before it becomes an attribute, the same redactor
`observability/logging.py`'s log sink uses. Native Langfuse fields that already exist for a
concept (`model`, `usage_details`, `cost_details`, `level`, `status_message`) are preferred over
a custom attribute when one exists, and `record_llm_usage` routes those through the same
redactor too (a no-op on the ints/floats they actually carry, but consistent — nothing here is
exempt from the discipline just because it "shouldn't" ever be sensitive). `set_safe_io` is the
same choke point for Langfuse's native `input`/`output` fields (what the dashboard's Preview
panel shows) — every wrap point that has a meaningful input/output to show calls it, always with
already-redacted or inherently-safe data, never `observation.update(input=..., output=...)`
directly.

`configure_tracing()`'s default path is `langfuse.get_client()`, which is itself already safe
with no configuration: `Langfuse.__init__` degrades to a no-op tracer (not an exception) when
`LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are unset (true in dev/test — `.env.example` ships
them blank), so nothing here ever requires a live Langfuse account or network access to run
safely, and every span-creation call in every wrap point below is a cheap no-op for any test
that never calls `configure_tracing()` at all.

`session.id` propagation (one session = one purchase, per CLAUDE.md's flow diagram) uses
Langfuse's own `propagate_attributes(session_id=...)` context manager directly at each
composition root/top-level span (`graph/guarded.py::run_guarded`, `evals/harness/run.py`,
`scripts/demo.py`, `scripts/demo_gradio.py`) — not a bespoke contextvar of this module's own —
so it is automatically stamped onto every span created afterward in the same context, including
across NeMo's async action dispatch, without any node or wrap point needing to thread an ID
through its own arguments.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

from langfuse import Langfuse, get_client, propagate_attributes
from langfuse._client.span import LangfuseObservationWrapper
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from payagent.guardrails import dlp

_client: Langfuse | None = None
_lock = threading.Lock()

# USD per 1,000 tokens. Point-in-time prices (OpenRouter, approximate) for the two models
# actually in play — `cost_report.py` must work network-free, so these are not fetched live
# and need occasional manual updates. Passed to Langfuse as `cost_details` rather than relying
# on Langfuse's own model-price catalog, since these models are called through OpenRouter and
# may not match Langfuse's internal catalog by name. Naming spells out the unit explicitly:
# per-1K vs per-1M vs per-token is a classic silent order-of-magnitude bug.
PRICE_PER_1K_TOKENS_USD: dict[str, tuple[float, float]] = {
    # (input_price_per_1k, output_price_per_1k)
    "openai/gpt-4o-mini": (0.00015, 0.0006),
    "meta-llama/llama-guard-4-12b": (0.00018, 0.00018),
}


def configure_tracing(*, client: Langfuse | None = None, force: bool = False) -> None:
    """Set the active Langfuse client. Idempotent unless `force=True`.

    `client`, when given (tests, or a composition root that wants a non-default exporter —
    see `client_with_exporter`), is used as-is. Otherwise `langfuse.get_client()` is used, which
    reads `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY`/`LANGFUSE_HOST` from the environment and
    degrades to a no-op tracer rather than raising when they're unset.
    """
    global _client
    with _lock:
        if _client is not None and not force:
            return
        _client = client if client is not None else get_client()


def _active_client() -> Langfuse:
    if _client is None:
        configure_tracing()
    return _client  # type: ignore[return-value]


def flush() -> None:
    """Force-export any buffered spans now, rather than waiting for `BatchSpanProcessor`'s
    own interval/queue-size trigger. `LangfuseResourceManager` also registers an `atexit`
    flush, so this is normally unnecessary for a long-running process — but a short-lived
    script (e.g. a one-off demo run) that wants to print a confirmation *after* spans have
    actually reached Langfuse, rather than trusting process-exit timing, should call this."""
    if _client is not None:
        _client.flush()


def client_with_exporter(exporter: SpanExporter) -> Langfuse:
    """Test-only helper: a fresh Langfuse client with tracing forced on and `exporter` as its
    sole span destination — no network. `LangfuseResourceManager` (the SDK's internal client
    registry) is a singleton keyed strictly by `public_key`, so a fixed key across tests would
    silently hand back a *previous* test's client (and its exporter) instead of this one —
    each call here uses a fresh key for that reason.
    """
    return Langfuse(
        public_key=f"test-{uuid4().hex}",
        secret_key="test",
        tracing_enabled=True,
        span_exporter=exporter,
    )


def _redact(value: Any) -> Any:
    """Run `value` through the DLP masker, JSON-encoding dict values (OTel attributes can't
    be nested objects) after masking. `dlp.mask_value` already recurses through dict/list/
    tuple/str on its own, so list/tuple values are handed to it directly rather than
    re-implementing that recursion here (it also covers nested dicts inside a list, which a
    shallow per-element comprehension would miss)."""
    if isinstance(value, dict):
        return json.dumps(dlp.mask_value(value), sort_keys=True)
    if isinstance(value, (list, tuple)):
        return dlp.mask_value(value)
    if isinstance(value, str):
        return dlp.mask_value(value)
    return value


def set_safe_attributes(observation: LangfuseObservationWrapper, **attributes: Any) -> None:
    """The mandatory choke point for custom attributes with no native Langfuse field.

    Writes directly to the underlying OTel span (`observation._otel_span` — documented by
    `LangfuseObservationWrapper` itself as "the underlying OpenTelemetry span", not a private
    implementation detail reached into by accident) since `LangfuseObservationWrapper.update()`
    silently ignores any `**kwargs` beyond its own named native fields. `None` values are
    skipped (OTel rejects them); everything else is redacted via
    `payagent.guardrails.dlp.mask_value` before it becomes an attribute (invariant I2).
    """
    for key, value in attributes.items():
        if value is None:
            continue
        observation._otel_span.set_attribute(key, _redact(value))


def set_safe_io(observation: LangfuseObservationWrapper, *, input: Any = None, output: Any = None) -> None:
    """The mandatory choke point for Langfuse's native `input`/`output` fields — what the
    dashboard's Preview panel renders. Every call site must route through this, never
    `observation.update(input=..., output=...)` directly, so DLP redaction (invariant I2)
    always runs first.

    Unlike `set_safe_attributes` (raw OTel attributes, which cannot hold nested objects, so
    `_redact` pre-JSON-encodes dicts), `input`/`output` accept structured data directly —
    Langfuse's own `update()` serializes it. `None` is a genuine "leave this field alone" (the
    SDK drops `None` entries before calling `set_attributes`), not "clear it".
    """
    observation.update(input=dlp.mask_value(input), output=dlp.mask_value(output))


@contextmanager
def start_span(name: str, **attributes: Any) -> Iterator[LangfuseObservationWrapper]:
    """Open a `LangfuseSpan` named `name`, with `attributes` set via `set_safe_attributes`.

    Exceptions are recorded on the observation (`level="ERROR"`, `status_message`) and always
    re-raised — this never swallows an error the caller's own code needs to see.
    """
    client = _active_client()
    with client.start_as_current_observation(name=name, as_type="span") as span:
        set_safe_attributes(span, **attributes)
        try:
            yield span
        except Exception as exc:
            span.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise


@contextmanager
def start_generation(name: str, *, model: str) -> Iterator[LangfuseObservationWrapper]:
    """Open a `LangfuseGeneration` named `name` for one LLM call — `plan`/`retrieve`'s main
    model, or the Llama Guard classifier — so Langfuse's native token/cost display applies.
    Call `record_llm_usage` once the response is available, inside this context.
    """
    client = _active_client()
    with client.start_as_current_observation(
        name=name, as_type="generation", model=model
    ) as generation:
        try:
            yield generation
        except Exception as exc:
            generation.update(level="ERROR", status_message=f"{type(exc).__name__}: {exc}")
            raise


@contextmanager
def open_session_span(
    name: str, *, session_id: str | None = None
) -> Iterator[LangfuseObservationWrapper]:
    """Open a root span for one purchase attempt, with `session_id` (a fresh UUID when
    omitted) propagated to every span created within it via `langfuse.propagate_attributes`.

    The composed idiom every entry point needs to open one session-scoped root span per
    purchase attempt (one session = one purchase, CLAUDE.md's flow diagram) —
    `graph/guarded.py::run_guarded`, `scripts/demo.py`, `scripts/demo_gradio.py`, and
    `evals/harness/run.py` all call this rather than each pairing `start_span` +
    `propagate_attributes` by hand.
    """
    with start_span(name) as span, propagate_attributes(session_id=session_id or str(uuid4())):
        yield span


def record_llm_usage(
    generation: LangfuseObservationWrapper,
    *,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> None:
    """Set Langfuse's native usage/cost fields on `generation`.

    `cost_details` is set only when `model` is a known entry in `PRICE_PER_1K_TOKENS_USD` and
    both token counts are available — Langfuse's own model-price catalog is not relied on,
    since these models are called through OpenRouter and may not match it by name. Otherwise
    `payagent.cost.model_unknown` (no native equivalent for "unpriced") records the model name
    so unpriced spans stay visible to `cost_report.py` instead of silently contributing zero.
    """
    usage_details = {
        k: v
        for k, v in {"input": input_tokens, "output": output_tokens}.items()
        if v is not None
    }
    prices = PRICE_PER_1K_TOKENS_USD.get(model)
    cost_details = None
    if prices is not None and input_tokens is not None and output_tokens is not None:
        input_price, output_price = prices
        cost_usd = (input_tokens / 1000) * input_price + (output_tokens / 1000) * output_price
        cost_details = {"total": cost_usd}

    generation.update(
        usage_details=dlp.mask_value(usage_details) if usage_details else None,
        cost_details=dlp.mask_value(cost_details) if cost_details else None,
    )
    if cost_details is None:
        set_safe_attributes(generation, **{"payagent.cost.model_unknown": model})


def usage_from_ai_message(message: Any) -> tuple[int | None, int | None]:
    """Best-effort (input_tokens, output_tokens) extraction from a LangChain `AIMessage`.

    Independent, small re-implementation of the same extraction
    `evals/harness/tracing.py::_usage_from_message` does — not imported from there, since
    `evals` depends on `src`, never the reverse.
    """
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens")
    meta = getattr(message, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    if token_usage:
        return token_usage.get("prompt_tokens"), token_usage.get("completion_tokens")
    return None, None


def model_name_of(llm: Any) -> str:
    """Best-effort model-name introspection for span attributes. `"unknown"` when the chat
    model object exposes neither `model_name` nor `model` (e.g. a test double)."""
    return getattr(llm, "model_name", None) or getattr(llm, "model", None) or "unknown"


def _message_to_dict(message: Any) -> dict[str, Any]:
    role = getattr(message, "type", message.__class__.__name__)
    entry: dict[str, Any] = {"role": role, "content": getattr(message, "content", str(message))}
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
        ]
    return entry


def traced_llm_invoke(llm: Any, bound_llm: Any, messages: list[Any]) -> Any:
    """Invoke `bound_llm.invoke(messages)` inside an `llm.invoke` generation span, recording
    token usage/cost and the redacted prompt/response as the generation's native `input`/
    `output` (invariant I2 via `set_safe_io`). Shared by `plan.py` and `retrieve.py`'s
    otherwise identical LLM-call wrapping — `llm` is the unbound model (for `model_name_of`),
    `bound_llm` is what actually gets called (`llm.bind_tools(...)`). Exceptions propagate to
    the caller unchanged (`start_generation` records and re-raises, never swallows).
    """
    model = model_name_of(llm)
    with start_generation("llm.invoke", model=model) as gen:
        ai_message = bound_llm.invoke(messages)
        input_tokens, output_tokens = usage_from_ai_message(ai_message)
        record_llm_usage(gen, model=model, input_tokens=input_tokens, output_tokens=output_tokens)
        set_safe_io(
            gen,
            input=[_message_to_dict(m) for m in messages],
            output=_message_to_dict(ai_message),
        )
    return ai_message


def wrap_node_span(name: str, step: Any) -> Any:
    """Wrap a `graph.graph.NodeStep` in a `graph.<name>` span. Used only by
    `PurchaseGraph.__init__`, which is the single choke point every invocation path
    (`.run()` and `.step()` alike) goes through."""

    def wrapped(state: Any) -> Any:
        with start_span(f"graph.{name}") as span:
            set_safe_attributes(
                span,
                **{
                    "payagent.graph.node_was_active": state.is_active,
                    "payagent.graph.status_before": state.status,
                },
            )
            new_state = step(state)
            set_safe_attributes(span, **{"payagent.graph.status_after": new_state.status})
            return new_state

    return wrapped


class JsonLinesSpanExporter(SpanExporter):
    """Serializes each ended span as one JSON line, appended to `path`.

    Used by `evals/harness/run.py` to leave a local trace artifact independent of whether live
    Langfuse credentials are configured, and directly by tests that need a concrete,
    inspectable trace without touching Langfuse or the network. Works against any OTel
    `ReadableSpan` — Langfuse observations are real OTel spans underneath, so this exporter is
    unchanged from before this module switched to the Langfuse SDK. Attributes on the span are
    already redacted, since custom ones only ever got there via `set_safe_attributes` and native
    Langfuse fields (`usage_details`, `cost_details`, ...) are routed through the same masker in
    `record_llm_usage`.

    `_lock` guards the file append the same way `evals/harness/run.py::LogCapture` guards its
    own file writes: a `SimpleSpanProcessor`'s `on_end()` calls `export()` synchronously on
    whichever thread ended the span, and `evals/harness/run.py --workers N > 1` runs scenarios
    on a `ThreadPoolExecutor`, so concurrent `export()` calls from different worker threads are
    real, not hypothetical.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()

    def export(self, spans: Any) -> SpanExportResult:
        lines = [json.dumps(_span_to_dict(span)) for span in spans]
        with self._lock, open(self._path, "a", encoding="utf-8") as fh:
            for line in lines:
                fh.write(line + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


def _span_to_dict(span: ReadableSpan) -> dict[str, Any]:
    context = span.context
    parent = span.parent
    return {
        "name": span.name,
        "trace_id": format(context.trace_id, "032x") if context else None,
        "span_id": format(context.span_id, "016x") if context else None,
        "parent_span_id": format(parent.span_id, "016x") if parent else None,
        "start_time": span.start_time,
        "end_time": span.end_time,
        "status": span.status.status_code.name if span.status else None,
        "attributes": dict(span.attributes or {}),
    }
