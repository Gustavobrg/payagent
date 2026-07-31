"""Context assembler: turns `state.retrieved_chunks` into the context block the next
graph node (`quote`) prompts against.

Two responsibilities live here, both required by invariants in CLAUDE.md:

- **I5** — chunks arrive from `rag.tools` already wrapped in a per-chunk untrusted-content
  delimiter. This module never unwraps, rewrites, or strips that wrapper; it only decides
  *which* wrapped chunks make it into the final block and in what order. `chunk_id` stays
  visible as an attribute on the wrapper's open tag, so a chunk can always be cited by ID
  without exposing its raw (undelimited) text.
- **I4** — every factual claim the response node makes about product, price, or policy
  must cite a `chunk_id` present in the assembled context. `verify_citations` is the check
  for that: it extracts citations from a response and reports any that don't match a
  chunk actually offered. The output rail `grounding_check` (bloco 8) reuses this function
  as-is rather than re-implementing citation parsing.

Deduplication matters because the retrieval sub-agent (`graph.nodes.retrieve`) can reach
the same SKU two different ways in one turn — `search_catalog` returning a fuzzy hit and
`get_sku_details` returning the exact lookup — and both are legitimate tool calls. Without
dedup, the same product would occupy the context budget twice under two different scores.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from payagent.observability.logging import get_logger
from payagent.rag.tools import ToolChunk

logger = get_logger(__name__)

# No tokenizer dependency in this project (see pyproject.toml) — this is a conservative
# chars-per-token heuristic (English/Portuguese averages ~4 chars/token) good enough for
# a budget cutoff. It only needs to be consistent, not exact: over-counting means we trim
# a little early, which is safe; a real tokenizer can replace this later without changing
# the public API.
_CHARS_PER_TOKEN = 4
DEFAULT_MAX_CONTEXT_TOKENS = 3000

# Citation format the response node's prompt must require: the literal chunk_id inside
# double square brackets, immediately after the claim it grounds, e.g.
# "The headphones cost R$129.99 [[SKU-HEADPHONES]]." Kept here as the single source of
# truth so the response node's prompt and this module's parser never drift apart.
_CITATION_PATTERN = re.compile(r"\[\[([^\[\]]+)\]\]")

CITATION_FORMAT_INSTRUCTION = (
    "Every factual claim about a product, its price, or a policy must be immediately "
    "followed by a citation naming the exact chunk_id it comes from, in double square "
    'brackets: e.g. "The headphones cost R$129.99 [[SKU-HEADPHONES]]." Only cite a '
    "chunk_id that appears in the context above. Never cite a chunk_id you were not "
    "given, and never state a fact about product, price, or policy without a citation."
)


@dataclass(frozen=True)
class AssembledContext:
    """What `assemble_context` hands to the next node: a prompt-ready block plus its chunks."""

    text: str
    chunks: list[ToolChunk]
    dropped_duplicates: int
    dropped_budget: int


def _estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / _CHARS_PER_TOKEN))


def _dedupe(chunks: Sequence[ToolChunk]) -> tuple[list[ToolChunk], int]:
    """Keep the first chunk seen per `chunk_id`; report how many later copies were dropped.

    First-seen (rather than highest-score) keeps this deterministic regardless of which
    tool call happened to run first in a given turn.
    """
    seen: dict[str, ToolChunk] = {}
    duplicates = 0
    for chunk in chunks:
        if chunk.chunk_id in seen:
            duplicates += 1
            continue
        seen[chunk.chunk_id] = chunk
    return list(seen.values()), duplicates


def _sort_key(chunk: ToolChunk) -> tuple[int, float]:
    """Rank by score, descending; scoreless chunks (exact lookups) rank above any fuzzy hit.

    `get_sku_details`/`compare_skus` return `score=None` because there's no ranking
    involved — the caller named that exact chunk_id. That's a stronger relevance signal
    than a fuzzy-search hit, not a missing one, so it must not sort last (or get cut by
    the budget first) just because it lacks a numeric score.
    """
    if chunk.score is None:
        return (1, 0.0)
    return (0, chunk.score)


def assemble_context(
    chunks: Sequence[ToolChunk],
    *,
    max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
) -> AssembledContext:
    """Dedupe, rank, and budget-cut `chunks` into the context block for the next node.

    Order: dedupe by chunk_id first, then sort by score descending, then greedily include
    chunks (highest-ranked first) until the next one would exceed `max_tokens`. The cutoff
    is a hard stop, not best-effort bin-packing — once a chunk doesn't fit, assembly ends,
    so token usage stays monotonic with rank.
    """
    deduped, dropped_duplicates = _dedupe(chunks)
    ordered = sorted(deduped, key=_sort_key, reverse=True)

    included: list[ToolChunk] = []
    tokens_used = 0
    for chunk in ordered:
        cost = _estimate_tokens(chunk.text)
        if tokens_used + cost > max_tokens:
            break
        included.append(chunk)
        tokens_used += cost

    dropped_budget = len(ordered) - len(included)

    logger.info(
        "context_assembled",
        input_chunks=len(chunks),
        dropped_duplicates=dropped_duplicates,
        included=len(included),
        dropped_budget=dropped_budget,
        tokens_used=tokens_used,
        max_tokens=max_tokens,
    )

    return AssembledContext(
        text="\n\n".join(chunk.text for chunk in included),
        chunks=included,
        dropped_duplicates=dropped_duplicates,
        dropped_budget=dropped_budget,
    )


def extract_citations(response: str) -> list[str]:
    """Pull every `[[chunk_id]]` citation out of `response`, in the order they appear."""
    return _CITATION_PATTERN.findall(response)


def verify_citations(response: str, context_chunks: Sequence[ToolChunk]) -> list[str]:
    """Return the cited chunk_ids in `response` that don't match any chunk in `context_chunks`.

    Empty result means every citation is grounded (invariant I4). This is the function the
    `grounding_check` output rail (bloco 8) calls directly — keep its signature stable.
    """
    valid_ids = {chunk.chunk_id for chunk in context_chunks}
    invalid: list[str] = []
    seen: set[str] = set()
    for cited_id in extract_citations(response):
        if cited_id in valid_ids or cited_id in seen:
            continue
        seen.add(cited_id)
        invalid.append(cited_id)
    return invalid
