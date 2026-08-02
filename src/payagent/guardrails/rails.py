"""LLMRails gateway — the two guardrail seams `payagent.graph` plugs into.

`guardrails/` (repo root — `config.yml`, `prompts.yml`, `rails/*.co`, `actions.py`) is
NeMo's rails-only config directory (CLAUDE.md: dialog rails stay empty on purpose; the
LangGraph drives the conversation, NeMo only filters). This module is the only place that
constructs an `LLMRails` from it and knows how to drive it for that rails-only purpose:

- `check_input` runs just the input rails (`dlp_scan_input` -> `heuristic_injection_check`
  -> `llama_guard_check_input`) against a raw user message, before it ever reaches `plan`.
- `check_output` runs just the output rails (`dlp_mask_output` -> `grounding_check` ->
  `llama_guard_check_output`) against a composed response, before it reaches the user.

Both pass `dialog: False` in `GenerationOptions.rails`, so the (nonexistent) 'main' model is
never invoked and NeMo never tries to generate dialog on its own — the response NeMo hands
back is either the untouched input (allowed) or the canned `bot refuse ...` message (blocked)
for `check_input`, and either the rewritten `$bot_message` (allowed, possibly DLP-masked) or
the canned refusal (blocked) for `check_output`. This is NeMo's own documented pattern for
"rails only, no main LLM" (`nemoguardrails/rails/llm/options.py`'s module docstring) — the
`{"role": "assistant", ...}` message ending the list plus `rails.dialog=False` is what makes
`generate_async` treat it as the value to check rather than something to generate.

Whether a call was actually blocked is read off `response.log.activated_rails` (any rail
with `stop=True`) rather than compared against the canned refusal strings — robust to the
`.co` files' wording changing, and the same signal `nemoguardrails` itself uses internally.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemoguardrails import LLMRails, RailsConfig

from payagent.observability.tracing import set_safe_io, start_span
from payagent.rag.tools import ToolChunk

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "guardrails"

_LOG_OPTIONS = {"activated_rails": True}


def build_rails(config_path: Path | str = _CONFIG_PATH) -> LLMRails:
    """Load `guardrails/config.yml` + `rails/*.co` and construct the rails-only app.

    `guardrails/actions.py` is auto-imported by `RailsConfig.from_path` and its
    `@action`-decorated functions self-register — nothing here calls `register_action`.
    """
    config = RailsConfig.from_path(str(config_path))
    return LLMRails(config)


@dataclass(frozen=True)
class InputCheckResult:
    """`check_input`'s result. `refusal` is the canned message to show the user when blocked."""

    allowed: bool
    refusal: str | None = None


@dataclass(frozen=True)
class OutputCheckResult:
    """`check_output`'s result. `text` is always what should reach the user: the original
    message (possibly DLP-masked) when allowed, the canned refusal when blocked."""

    text: str
    blocked: bool


def _was_blocked(response: Any) -> bool:
    return any(rail.stop for rail in response.log.activated_rails)


def check_input(rails: LLMRails, user_message: str) -> InputCheckResult:
    """Run only the input rails against `user_message`. Never touches the purchase graph."""
    with start_span("rail.check_input") as span:
        response = rails.generate(
            messages=[{"role": "user", "content": user_message}],
            options={
                "rails": {"input": True, "output": False, "dialog": False, "retrieval": False},
                "log": _LOG_OPTIONS,
            },
        )
        if _was_blocked(response):
            result = InputCheckResult(allowed=False, refusal=response.response[0]["content"])
        else:
            result = InputCheckResult(allowed=True)
        set_safe_io(span, input=user_message, output={"allowed": result.allowed, "refusal": result.refusal})
        return result


def check_output(
    rails: LLMRails,
    *,
    user_message: str,
    bot_message: str,
    retrieved_chunks: Sequence[ToolChunk] = (),
) -> OutputCheckResult:
    """Run only the output rails against `bot_message`.

    `retrieved_chunks` is what `grounding_check` (guardrails/actions.py) verifies citations
    against — pass `state.retrieved_chunks` from the graph run that produced `bot_message`.
    """
    # NeMo hashes every message's content with `json.dumps` for its history cache key
    # (nemoguardrails/rails/llm/utils.py) — a raw ToolChunk isn't serializable, so only
    # what grounding_check (guardrails/actions.py) actually needs crosses the wire.
    chunk_refs = [{"chunk_id": chunk.chunk_id} for chunk in retrieved_chunks]
    with start_span("rail.check_output") as span:
        response = rails.generate(
            messages=[
                {"role": "context", "content": {"retrieved_chunks": chunk_refs}},
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": bot_message},
            ],
            options={
                "rails": {"input": False, "output": True, "dialog": False, "retrieval": False},
                "log": _LOG_OPTIONS,
            },
        )
        result = OutputCheckResult(
            text=response.response[0]["content"], blocked=_was_blocked(response)
        )
        set_safe_io(
            span,
            input={"user_message": user_message, "bot_message": bot_message},
            output={"text": result.text, "blocked": result.blocked},
        )
        return result
