"""`ScenarioResult`: what one harness run of one scenario produces.

Kept separate from `run.py` (the orchestrator that builds these against live
infrastructure) and `metrics.py` (which only ever reads them) so metric computation stays
a pure function of data — trivially unit-testable with hand-built results, no Qdrant, no
LLM, no NeMo rails required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evals.harness.scenarios import Scenario
from evals.harness.tracing import LLMCallRecord, ToolCallRecord


@dataclass
class ScenarioResult:
    scenario: Scenario

    # What the input/output rails decided.
    input_blocked: bool = False
    output_blocked: bool = False

    # Where the graph run actually ended up. "input_blocked" / "output_blocked" / "error"
    # are harness-level statuses layered on top of `payagent.graph.state.GraphStatus` for
    # the cases that never reach (or never leave) the graph.
    final_status: str = "unknown"
    settled: bool = False

    # First and last tool actually invoked, across both the RAG sub-agent's internal tools
    # and the MCP side-effect tools — see evals/harness/tracing.py for how these are
    # captured without touching production logging.
    actual_first_tool: str | None = None
    actual_final_tool: str | None = None
    actual_final_tool_ok: bool | None = None
    actual_denial_code: str | None = None

    # For scenarios with a `hint_sku`: was it actually among the chunk_ids `retrieve`
    # surfaced? This is `param_exact_match`'s basis — a meaningful, non-circular check,
    # since the harness itself supplies `hint_sku` to `state.selected_sku` afterward (the
    # same out-of-band role `scripts/demo.py`'s CLI prompt plays); checking whether
    # `get_quote` received that value would only test the harness's own plumbing. Whether
    # retrieval actually *found* the right product for the natural-language request is the
    # real signal. `None` when the scenario has no `hint_sku` (nothing to check).
    hint_sku_found_in_retrieval: bool | None = None

    # P4 only: re-running `guardrails.heuristics.scan` against the actual retrieved chunk
    # text for `scenario.poisoned_sku`, live, in this run — not the offline check done once
    # while authoring scenarios.yaml. `None` when not applicable or the chunk never came
    # back from retrieval at all.
    p4_injection_detected_by_heuristic: bool | None = None

    # P4-specific consistency check: for scenarios that reach settlement, did the settled
    # amount/merchant ever disagree with the quote that priced them? Always expected True
    # given the current handlers.py (amount/merchant are copied from the quote, never from
    # request text) — a False here is a real regression, not a calibration nuance.
    amount_merchant_consistent: bool | None = None

    # Did the graph attempt `quote` with no `selected_sku` set (the harness's deliberate
    # probe for achado #2 in scenarios.yaml's header) and correctly pause at
    # `needs_clarification` rather than guessing? Before the achado #2 fix (2026-08-01) this
    # tracked an `IllegalTransitionError`; `quote` now handles "no selection" as a graceful
    # status transition instead of raising (see quote.py), so this checks `state.status`.
    no_autonomous_disambiguation: bool | None = None

    pan_leak_count: int = 0

    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    rail_input_latency_ms: float = 0.0
    rail_output_latency_ms: float = 0.0
    graph_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    response_text: str = ""
    error: str | None = None
    raw_trace: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return sum(c.total_tokens or 0 for c in self.llm_calls)
