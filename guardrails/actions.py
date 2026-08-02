"""Custom NeMo actions for the PayAgent Guard rails — pure orchestration.

Auto-imported by `RailsConfig.from_path("guardrails")` (this file's own directory): NeMo
scans its config directory for an `actions.py`/`actions/` module and self-registers every
`@action`-decorated function it finds, by name. Nothing in `guardrails/rails/*.co` calls
`register_action` explicitly — the flow's `execute dlp_scan_input` etc. resolves here.

Every action below is a thin adapter: read `context` (NeMo's per-generation variable
store — `user_message`/`bot_message` are populated by `payagent.guardrails.rails`, see
that module), call the one function in `src/payagent/guardrails/` (or `rag/context.py`)
that actually does the detection, and shape the result into what `guardrails/rails/*.co`
expects (`$var["key"]`). No new detection logic lives here — see CLAUDE.md and each
action's docstring for which existing module it defers to.
"""

from __future__ import annotations

from nemoguardrails.actions import action

from payagent.guardrails import dlp, heuristics
from payagent.guardrails import provider as guardrail_provider
from payagent.guardrails.provider import GuardrailProvider
from payagent.observability.logging import get_logger
from payagent.observability.tracing import set_safe_attributes, set_safe_io, start_span
from payagent.rag.context import verify_citations
from payagent.rag.tools import ToolChunk

logger = get_logger(__name__)

_CARD_CATEGORIES = frozenset({dlp.PAN_CATEGORY, dlp.CVV_CATEGORY, dlp.EXPIRY_CATEGORY})

_provider: GuardrailProvider | None = None


def _get_provider() -> GuardrailProvider:
    """Lazily construct the live provider so importing this module never needs
    OPENROUTER_API_KEY (only actually calling llama_guard_check_input/output does).

    Looks up `OpenRouterLlamaGuard` on the `provider` module itself (not via a
    module-load-time `from ... import OpenRouterLlamaGuard`) so a test that monkeypatches
    `payagent.guardrails.provider.OpenRouterLlamaGuard` before the guardrail is first
    invoked still takes effect — NeMo loads this file with its own module loader (see
    module docstring), so a plain `import actions` from a test is a *different* module
    object than the one NeMo actually dispatches through; patching the shared
    `provider` module is what reaches both.
    """
    global _provider
    if _provider is None:
        _provider = guardrail_provider.OpenRouterLlamaGuard()
    return _provider


def set_guardrail_provider(provider: GuardrailProvider | None) -> None:
    """Composition-root/test seam: override the provider the llama_guard_check_* actions
    use. `None` resets to the lazy default (`OpenRouterLlamaGuard`)."""
    global _provider
    _provider = provider


@action(name="dlp_scan_input")
async def dlp_scan_input(context: dict | None = None) -> dict:
    """P1: orchestrates `payagent.guardrails.dlp.scan` over the user's message."""
    with start_span("rail.dlp_scan_input") as span:
        user_message = (context or {}).get("user_message", "")
        result = dlp.scan(user_message)
        has_card_data = any(match.category in _CARD_CATEGORIES for match in result.matches)
        response = {"has_card_data": has_card_data, "categories": list(result.categories)}
        set_safe_attributes(span, has_card_data=has_card_data, categories=list(result.categories))
        set_safe_io(span, input=user_message, output=response)
        return response


@action(name="heuristic_injection_check")
async def heuristic_injection_check(context: dict | None = None) -> dict:
    """P4: orchestrates `payagent.guardrails.heuristics.scan` over the user's message."""
    with start_span("rail.heuristic_injection_check") as span:
        user_message = (context or {}).get("user_message", "")
        result = heuristics.scan(user_message)
        response = {"blocked": result.blocked, "patterns": list(result.patterns)}
        set_safe_attributes(span, blocked=result.blocked, patterns=list(result.patterns))
        set_safe_io(span, input=user_message, output=response)
        return response


@action(name="llama_guard_check_input")
async def llama_guard_check_input(context: dict | None = None) -> dict:
    """P1-P5: orchestrates `GuardrailProvider.check_input` over the user's message."""
    with start_span("rail.llama_guard_check_input") as span:
        user_message = (context or {}).get("user_message", "")
        verdict = _get_provider().check_input(user_message)
        response = {"allowed": verdict.allowed, "policy_violations": list(verdict.categories)}
        set_safe_attributes(
            span, allowed=verdict.allowed, policy_violations=list(verdict.categories)
        )
        set_safe_io(span, input=user_message, output=response)
        return response


def _as_tool_chunk(item: ToolChunk | dict) -> ToolChunk:
    """`verify_citations` only reads `.chunk_id`, but the value crossing NeMo's wire is a
    plain `{"chunk_id": ...}` dict (see `payagent.guardrails.rails.check_output` — NeMo
    JSON-hashes every message's content, and a real `ToolChunk` isn't serializable), while
    a direct in-process caller may still pass real `ToolChunk`s. Accept both."""
    if isinstance(item, ToolChunk):
        return item
    return ToolChunk(chunk_id=item["chunk_id"], source=item.get("source", ""), score=item.get("score"), text="")


@action(name="grounding_check")
async def grounding_check(context: dict | None = None) -> dict:
    """I4: orchestrates `rag.context.verify_citations` over the bot's message."""
    with start_span("rail.grounding_check") as span:
        ctx = context or {}
        bot_message = ctx.get("bot_message", "")
        retrieved_chunks = [_as_tool_chunk(item) for item in ctx.get("retrieved_chunks", [])]
        invalid = verify_citations(bot_message, retrieved_chunks)
        response = {"grounded": not invalid, "invalid_citations": invalid}
        set_safe_attributes(span, grounded=not invalid, invalid_citations=invalid)
        set_safe_io(span, input=bot_message, output=response)
        return response


@action(name="llama_guard_check_output")
async def llama_guard_check_output(context: dict | None = None) -> dict:
    """P1/P3/P6: orchestrates `GuardrailProvider.check_output` over the bot's message."""
    with start_span("rail.llama_guard_check_output") as span:
        bot_message = (context or {}).get("bot_message", "")
        verdict = _get_provider().check_output(bot_message)
        response = {"allowed": verdict.allowed, "policy_violations": list(verdict.categories)}
        set_safe_attributes(
            span, allowed=verdict.allowed, policy_violations=list(verdict.categories)
        )
        set_safe_io(span, input=bot_message, output=response)
        return response


@action(name="dlp_mask_output")
async def dlp_mask_output(context: dict | None = None) -> dict:
    """I2: orchestrates `payagent.guardrails.dlp.mask` over the bot's message."""
    with start_span("rail.dlp_mask_output") as span:
        bot_message = (context or {}).get("bot_message", "")
        masked, redacted = dlp.mask(bot_message)
        response = {"text": masked, "redacted": redacted}
        set_safe_attributes(span, redacted=redacted)
        set_safe_io(span, input=bot_message, output=response)
        return response


@action(name="audit_log_redaction")
async def audit_log_redaction(context: dict | None = None) -> dict:
    """Audit trail for `dlp_mask_output` actually redacting something — should be rare;
    if it fires often, an earlier rail let sensitive data reach the bot message (bug
    elsewhere, per `guardrails/rails/output.co`)."""
    logger.warning("guardrail_output_redacted")
    return {}
