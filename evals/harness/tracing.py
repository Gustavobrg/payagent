"""In-process instrumentation for the eval harness's trace capture.

Three independent seams, none of which touch production logging (invariant I2 in
CLAUDE.md: `dispatch.py` deliberately never logs an argument dict, an individual argument
value, or `request.model_dump()` — see its own module docstring). Nothing here writes to
a log, span, or error message; it is a harness-only observation point that reads real
in-process objects the graph already builds, the same way a debugger would.

- `TracingChatModel` wraps the LLM passed into `build_graph`, so every `.invoke()` call
  made by `plan` or `retrieve` is timed and its token usage recorded, without either node
  module knowing it is being observed — it still only sees the `ChatModel` /
  `ToolCallingChatModel` Protocol both already require (`bind_tools`, `invoke`).
- `patch_rag_tool_tracing` / `patch_mcp_handler_tracing` are context managers that record
  every RAG sub-agent tool call (search_catalog, search_policies, get_sku_details,
  compare_skus — the internal tools `graph.nodes.retrieve` binds) and every MCP handler
  call (the six tools in `mcp_server.handlers.HANDLERS`) with their real arguments.
- `RunContext` is the tiny mutable holder the harness sets before each `graph.step(...)`
  call, so `TracingChatModel` knows which scenario/node an `.invoke()` belongs to without
  either Protocol growing a parameter it doesn't otherwise need.

Patching note: `graph.nodes.retrieve` does `from payagent.rag.tools import search_catalog,
...` — a `from x import y` binds a *new* name in `retrieve`'s own namespace, so patching
`payagent.rag.tools.search_catalog` would not affect calls already routed through
`retrieve`'s bound name. The patch targets below are the *consuming* module's own
attributes for exactly that reason. `mcp_server.handlers.HANDLERS`, by contrast, is a
single mutable dict object shared by reference with `dispatch.py` (`from payagent....
handlers import HANDLERS` binds the *same dict*, not a copy), so mutating its values in
place is visible to every caller without needing to patch dispatch.py at all.

Thread-safety (`evals/harness/run.py --workers N > 1` runs scenarios on a
`ThreadPoolExecutor`): `Tracer.context` is a *per-thread* `RunContext`, backed by
`threading.local` — each worker thread gets its own isolated `scenario_id`/`node`, so one
scenario's `.begin()`/`.set_node()` can never bleed into a concurrently-running scenario's
trace records. `Tracer.llm_calls`/`tool_calls` are plain lists appended to under a shared
`threading.Lock`, since the CPython GIL alone does not guarantee `list.append` ordering
guarantees callers should rely on across threads (it guarantees no corruption, not
isolation). Patches installed by `patch_rag_tool_tracing`/`patch_mcp_handler_tracing` are
process-wide (there is exactly one `graph.nodes.retrieve` module and one `HANDLERS` dict),
which is fine: the wrapped functions read `tracer.context` — the thread-local — at call
time, so concurrent calls from different worker threads each see their own scenario_id.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.tools import BaseTool

import payagent.graph.nodes.retrieve as retrieve_module
import payagent.mcp_server.handlers as handlers_module


@dataclass
class LLMCallRecord:
    scenario_id: str
    node: str
    latency_ms: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    had_tool_calls: bool


@dataclass
class ToolCallRecord:
    scenario_id: str
    layer: str  # "rag" | "mcp"
    tool: str
    args: dict[str, Any]
    ok: bool
    error_code: str | None
    latency_ms: float


@dataclass
class RunContext:
    """Mutable "where are we right now" holder, set by the harness's orchestration loop."""

    scenario_id: str = ""
    node: str = ""


@dataclass
class Tracer:
    """Accumulates every LLM and tool call recorded during a harness run.

    `context` is per-thread (see module docstring): each `ThreadPoolExecutor` worker gets
    its own `RunContext`, created lazily on first access. `llm_calls`/`tool_calls` are
    shared across threads and protected by `_lock`.
    """

    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    _local: threading.local = field(default_factory=threading.local, repr=False, compare=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    @property
    def context(self) -> RunContext:
        if not hasattr(self._local, "run_context"):
            self._local.run_context = RunContext()
        return self._local.run_context

    def begin(self, scenario_id: str, node: str = "") -> None:
        self.context.scenario_id = scenario_id
        self.context.node = node

    def set_node(self, node: str) -> None:
        self.context.node = node

    def record_llm_call(self, record: LLMCallRecord) -> None:
        with self._lock:
            self.llm_calls.append(record)

    def record_tool_call(self, record: ToolCallRecord) -> None:
        with self._lock:
            self.tool_calls.append(record)

    def calls_for(self, scenario_id: str) -> tuple[list[LLMCallRecord], list[ToolCallRecord]]:
        with self._lock:
            llm = [c for c in self.llm_calls if c.scenario_id == scenario_id]
            tool = [c for c in self.tool_calls if c.scenario_id == scenario_id]
        return llm, tool


def _usage_from_message(message: AIMessage) -> tuple[int | None, int | None, int | None]:
    usage = getattr(message, "usage_metadata", None)
    if usage:
        return usage.get("input_tokens"), usage.get("output_tokens"), usage.get("total_tokens")
    meta = getattr(message, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or {}
    if token_usage:
        return (
            token_usage.get("prompt_tokens"),
            token_usage.get("completion_tokens"),
            token_usage.get("total_tokens"),
        )
    return None, None, None


class TracingChatModel:
    """Wraps any `ChatModel`/`ToolCallingChatModel`-compatible object.

    `bind_tools` returns a new `TracingChatModel` around the inner bound model, so timing
    survives the `bound_llm = llm.bind_tools(tools)` pattern both `plan.py` and
    `retrieve.py` use before calling `.invoke()` in a loop.
    """

    def __init__(self, inner: Any, tracer: Tracer) -> None:
        self._inner = inner
        self._tracer = tracer

    def bind_tools(self, tools: list[BaseTool]) -> TracingChatModel:
        return TracingChatModel(self._inner.bind_tools(tools), self._tracer)

    def invoke(self, messages: list[BaseMessage]) -> AIMessage:
        start = time.perf_counter()
        response = self._inner.invoke(messages)
        latency_ms = (time.perf_counter() - start) * 1000
        input_tokens, output_tokens, total_tokens = _usage_from_message(response)
        self._tracer.record_llm_call(
            LLMCallRecord(
                scenario_id=self._tracer.context.scenario_id,
                node=self._tracer.context.node,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                had_tool_calls=bool(getattr(response, "tool_calls", None)),
            )
        )
        return response


_RAG_TOOL_NAMES = ("search_catalog", "search_policies", "get_sku_details", "compare_skus")


def _wrap_rag_tool(name: str, original: Callable, tracer: Tracer) -> Callable:
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        record_args = dict(kwargs)
        if len(args) > 1:
            # Every rag/tools.py function's first positional argument is the `Retriever`
            # instance (search_catalog(retriever, query, ...), get_sku_details(retriever,
            # sku_id), ...) — drop it, keep the rest positionally.
            record_args["_positional"] = list(args[1:])
        tracer.record_tool_call(
            ToolCallRecord(
                scenario_id=tracer.context.scenario_id,
                layer="rag",
                tool=name,
                args=record_args,
                ok=True,
                error_code=None,
                latency_ms=latency_ms,
            )
        )
        return result

    return wrapped


@contextmanager
def patch_rag_tool_tracing(tracer: Tracer) -> Iterator[None]:
    """Patch `graph.nodes.retrieve`'s own bound names for the four RAG tools."""
    originals = {name: getattr(retrieve_module, name) for name in _RAG_TOOL_NAMES}
    try:
        for name, fn in originals.items():
            setattr(retrieve_module, name, _wrap_rag_tool(name, fn, tracer))
        yield
    finally:
        for name, fn in originals.items():
            setattr(retrieve_module, name, fn)


def _wrap_mcp_handler(name: str, original: Callable, tracer: Tracer) -> Callable:
    def wrapped(deps, request):
        start = time.perf_counter()
        try:
            result = original(deps, request)
        except Exception as exc:
            latency_ms = (time.perf_counter() - start) * 1000
            error_code = getattr(getattr(exc, "payload", None), "code", None)
            tracer.record_tool_call(
                ToolCallRecord(
                    scenario_id=tracer.context.scenario_id,
                    layer="mcp",
                    tool=name,
                    args=request.model_dump(mode="json"),
                    ok=False,
                    error_code=error_code.value if error_code is not None else type(exc).__name__,
                    latency_ms=latency_ms,
                )
            )
            raise
        latency_ms = (time.perf_counter() - start) * 1000
        tracer.record_tool_call(
            ToolCallRecord(
                scenario_id=tracer.context.scenario_id,
                layer="mcp",
                tool=name,
                args=request.model_dump(mode="json"),
                ok=True,
                error_code=None,
                latency_ms=latency_ms,
            )
        )
        return result

    return wrapped


@contextmanager
def patch_mcp_handler_tracing(tracer: Tracer) -> Iterator[None]:
    """Mutate `mcp_server.handlers.HANDLERS`' values in place (same dict `dispatch.py` uses)."""
    originals = dict(handlers_module.HANDLERS)
    try:
        for name, fn in originals.items():
            handlers_module.HANDLERS[name] = _wrap_mcp_handler(name, fn, tracer)
        yield
    finally:
        for name, fn in originals.items():
            handlers_module.HANDLERS[name] = fn
