"""The single funnel every tool call passes through: validate, dedupe, run, envelope.

Everything cross-cutting lives here exactly once, so no handler can forget it: strict argument
validation, the idempotency reservation for side-effect tools, structured error conversion, and the
uniform result envelope.

**Logging discipline (invariant I2).** This is the one place every call passes through, so an
over-eager log line here is the worst possible leak. The allowed fields are a closed set — tool
name, argument *fingerprint*, whether a key was present, field *paths* and error *codes*, policy
action and denial code, merchant outcome, duration. Never the `arguments` dict, never an individual
argument value, never `request.model_dump()`, never `str(exc)` of a `ValidationError`, and never
`query` / `purpose` / `reason_note` / `sku`. Paths and codes carry all the diagnostic value a
caller needs without carrying their text. `idempotency_key` is the sole caller-supplied string on
the list, and only because tracing a duplicate charge is impossible without it. The DLP processor
in `observability/logging.py` is a second layer, not the first — it is a passthrough stub today, so
this discipline is the actual control.

**Atomicity.** The MCP kernel dispatches each inbound request with `tg.start_soon`, so requests run
concurrently, and every store here is an unlocked dict. That is safe *only because* the handlers
are synchronous and this function contains no `await`: a task that never yields runs the
reserve → effect → complete sequence atomically with respect to the event loop. The obvious future
"optimization" — wrapping a handler in `anyio.to_thread.run_sync` to get the retriever off the
loop — would open a real double-settlement window. If that is ever needed, it must happen outside
the idempotency critical section, or the stores must grow an `anyio.Lock`.
"""

from __future__ import annotations

import json
from typing import Any

import mcp_types as types
from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ValidationError

from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.errors import (
    ToolErrorCode,
    ToolErrorPayload,
    from_validation_error,
    tool_error,
)
from payagent.mcp_server.handlers import HANDLERS, HandlerFailure
from payagent.mcp_server.idempotency import (
    IdempotencyConflict,
    IdempotencyInProgress,
    argument_fingerprint,
)
from payagent.mcp_server.registry import TOOL_SPECS_BY_NAME, ToolSpec
from payagent.observability.logging import get_logger
from payagent.observability.tracing import set_safe_attributes, set_safe_io, start_span

logger = get_logger(__name__)

REPLAY_META_KEY = "payagent/idempotent_replay"


def _envelope_payload(
    *, ok: bool, data: BaseModel | None = None, error: ToolErrorPayload | None = None
) -> dict[str, Any]:
    """Render the uniform envelope as JSON-ready data."""
    return {
        "ok": ok,
        "data": data.model_dump(mode="json") if data is not None else None,
        "error": error.model_dump(mode="json") if error is not None else None,
    }


def _to_call_tool_result(
    payload: dict[str, Any], *, is_error: bool, replayed: bool = False
) -> types.CallToolResult:
    """Wrap an envelope in a `CallToolResult`, mirrored into `content` for text-only clients.

    A replay is signalled only out of band, in `_meta`. Keeping it out of the payload is what makes
    "same key and arguments returns the same result" literally true and testable with `==`.
    """
    return types.CallToolResult(
        content=[
            types.TextContent(
                type="text",
                text=json.dumps(payload, sort_keys=True, separators=(",", ":")),
            )
        ],
        structured_content=payload,
        is_error=is_error,
        meta={REPLAY_META_KEY: True} if replayed else None,
    )


def _error_result(error: ToolErrorPayload, *, replayed: bool = False) -> types.CallToolResult:
    return _to_call_tool_result(
        _envelope_payload(ok=False, error=error), is_error=True, replayed=replayed
    )


def _annotate_span_from_result(span: Any, result: types.CallToolResult) -> None:
    """Read the closed set of safe fields off an already-built `CallToolResult` envelope and
    set them on the span — never re-derives anything from `arguments`/`request` (I2).

    `denial` fields are `None` (and so silently dropped by `set_safe_attributes`, which
    already skips `None` values) whenever there's no policy denial in `error` — no need to
    branch on `denial`'s presence here as well.

    The same closed set is also set as the span's native `output` (via `set_safe_io`) so it
    renders in Langfuse's Preview panel — still the closed set, never the full envelope's
    `data`, honoring this module's own "never `arguments`/`request.model_dump()`" discipline.
    """
    payload = result.structured_content or {}
    error = payload.get("error") or {}
    denial = error.get("denial") or {}
    fields = {
        "ok": payload.get("ok"),
        "error.code": error.get("code"),
        "denial.code": denial.get("code"),
        "denial.policy_ref": denial.get("policy_ref"),
        "replayed": bool(result.meta and result.meta.get(REPLAY_META_KEY)),
    }
    set_safe_attributes(span, **fields)
    set_safe_io(span, output=fields)


def _run_handler(deps: ToolDeps, spec: ToolSpec, request: BaseModel) -> dict[str, Any]:
    """Invoke the handler and render its envelope, or raise `HandlerFailure`."""
    result = HANDLERS[spec.name](deps, request)
    return _envelope_payload(ok=True, data=result)


async def dispatch_tool_call(
    deps: ToolDeps, name: str, arguments: dict[str, Any] | None
) -> types.CallToolResult:
    """Validate, authorize, execute, and envelope one tool call.

    `async def` only because the MCP `on_call_tool` handler signature requires it; the body contains
    no `await`, which is what keeps the idempotency sequence atomic (see the module docstring).

    Takes no request context on purpose: that is the seam that lets the whole contract suite call
    this directly without constructing a live `ServerSession`.

    Raises:
        MCPError: only for an unknown tool name, which is a protocol-level mistake rather than a
            tool outcome the model can correct.
    """
    spec = TOOL_SPECS_BY_NAME.get(name)
    if spec is None:
        # The requested name goes to the log, never into the error message echoed back.
        logger.warning("unknown_tool", tool=name)
        raise MCPError(types.INVALID_PARAMS, "Unknown tool.")

    # Single funnel, single span: covers both the graph path (tool_call.call_tool ->
    # anyio.run) and the live MCP server path (server.py's on_call_tool). Attributes set
    # below are exactly the closed set this module's own docstring already allows in its
    # `logger.*` calls (invariant I2) — never `arguments`, `request.model_dump()`, or a
    # ValidationError's interpolated `msg`.
    with start_span(f"mcp.tool.{name}", tool=name) as span:
        try:
            request = spec.request_model.model_validate(arguments or {})
        except ValidationError as exc:
            error = from_validation_error(exc)
            logger.info(
                "tool_validation_failed",
                tool=spec.name,
                field_paths=[fe.path for fe in error.field_errors],
                field_codes=[fe.code for fe in error.field_errors],
            )
            validation_fields = {
                "field_paths": [fe.path for fe in error.field_errors],
                "field_codes": [fe.code for fe in error.field_errors],
                "ok": False,
                "error.code": error.code.value,
            }
            set_safe_attributes(span, **validation_fields)
            set_safe_io(span, input={"tool": name}, output=validation_fields)
            return _error_result(error)

        payload = request.model_dump(mode="json")
        fingerprint = argument_fingerprint(payload)
        idempotency_key: str | None = payload.get("idempotency_key")

        logger.info(
            "tool_call_start",
            tool=spec.name,
            has_idempotency_key=idempotency_key is not None,
            arg_fingerprint=fingerprint,
        )
        input_fields = {
            "tool": name,
            "arg_fingerprint": fingerprint,
            "has_idempotency_key": idempotency_key is not None,
            "idempotency_key": idempotency_key,
        }
        set_safe_attributes(
            span,
            arg_fingerprint=fingerprint,
            has_idempotency_key=idempotency_key is not None,
            idempotency_key=idempotency_key,
        )
        set_safe_io(span, input=input_fields)

        if not spec.has_side_effect:
            result = _dispatch_without_idempotency(deps, spec, request)
        else:
            assert idempotency_key is not None  # guaranteed by the schema for side-effect tools
            result = _dispatch_with_idempotency(
                deps, spec, request, idempotency_key, fingerprint
            )

        _annotate_span_from_result(span, result)
        return result


def _dispatch_without_idempotency(
    deps: ToolDeps, spec: ToolSpec, request: BaseModel
) -> types.CallToolResult:
    """Read-only path: no reservation, and no policy consultation (invariant I1's boundary)."""
    try:
        envelope = _run_handler(deps, spec, request)
    except HandlerFailure as failure:
        logger.info("tool_call_complete", tool=spec.name, ok=False, error_code=failure.payload.code.value)
        return _error_result(failure.payload)
    except Exception:
        # `exc_info` goes to the log (which the DLP processor sees); the caller gets a static
        # message, so an unexpected exception's text cannot become a response.
        logger.exception("tool_unhandled_error", tool=spec.name)
        return _error_result(tool_error(ToolErrorCode.INTERNAL_ERROR))

    logger.info("tool_call_complete", tool=spec.name, ok=True, error_code=None)
    return _to_call_tool_result(envelope, is_error=False)


def _dispatch_with_idempotency(
    deps: ToolDeps,
    spec: ToolSpec,
    request: BaseModel,
    idempotency_key: str,
    fingerprint: str,
) -> types.CallToolResult:
    """Side-effect path: reserve, run, then either complete or release the reservation."""
    now = deps.clock()

    try:
        replay = deps.idempotency.begin(spec.name, idempotency_key, fingerprint, now)
    except IdempotencyConflict:
        # Same key, different arguments. Refusing is the only truthful, effect-free answer:
        # returning the cached result would let a changed amount or mandate inherit a previous
        # success, and the agent would report an operation that never happened as done (P2).
        logger.warning(
            "idempotency_conflict",
            tool=spec.name,
            idempotency_key=idempotency_key,
            arg_fingerprint=fingerprint,
        )
        return _error_result(tool_error(ToolErrorCode.IDEMPOTENCY_KEY_REUSED))
    except IdempotencyInProgress:
        logger.warning(
            "idempotency_in_progress", tool=spec.name, idempotency_key=idempotency_key
        )
        return _error_result(tool_error(ToolErrorCode.IDEMPOTENCY_IN_PROGRESS))

    if replay is not None and replay.response is not None:
        logger.info(
            "idempotent_replay",
            tool=spec.name,
            idempotency_key=idempotency_key,
            arg_fingerprint=fingerprint,
        )
        return _to_call_tool_result(
            replay.response, is_error=replay.is_error, replayed=True
        )

    try:
        envelope = _run_handler(deps, spec, request)
    except HandlerFailure as failure:
        if failure.indeterminate:
            # The effect may or may not have landed. Release the key so a retry can proceed — the
            # merchant's own dedup catches it, so exactly one effect exists either way. Completing
            # here instead would strand the operation permanently.
            deps.idempotency.release(spec.name, idempotency_key)
        else:
            # A deterministic failure is a real outcome: store it so a replay returns the identical
            # denial rather than re-consulting policy, which also means a denied purchase cannot be
            # retried into an allowed one under the same key.
            deps.idempotency.complete(
                spec.name,
                idempotency_key,
                _envelope_payload(ok=False, error=failure.payload),
                is_error=True,
            )
        logger.info(
            "tool_call_complete",
            tool=spec.name,
            ok=False,
            error_code=failure.payload.code.value,
        )
        return _error_result(failure.payload)
    except Exception:
        # Unknown outcome: treat it like a timeout and release, rather than caching a failure that
        # might not be one.
        deps.idempotency.release(spec.name, idempotency_key)
        logger.exception("tool_unhandled_error", tool=spec.name)
        return _error_result(tool_error(ToolErrorCode.INTERNAL_ERROR))

    deps.idempotency.complete(spec.name, idempotency_key, envelope, is_error=False)
    logger.info("tool_call_complete", tool=spec.name, ok=True, error_code=None)
    return _to_call_tool_result(envelope, is_error=False)
