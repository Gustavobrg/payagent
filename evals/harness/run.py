"""The eval harness: runs every scenario in `evals/datasets/scenarios.yaml` through the
real purchase graph (`payagent.graph`), the real NeMo rails (`payagent.guardrails.rails`),
and the real infra `payagent.mcp_server.__main__.build_deps_from_env` wires up — the same
composition root `scripts/demo.py` and the MCP server itself use — then reports the
metrics in `evals/harness/metrics.py` as versioned JSON + Markdown under `evals/reports/`.

Usage:
    uv run python -m evals.harness.run
    uv run python -m evals.harness.run --no-guardrails
    uv run python -m evals.harness.run --limit 10 --category happy_path --workers 8
    uv run python -m evals.harness.run --id HP-001 --id P4-001 --workers 1  # sequential

`--no-guardrails` is the ablation: it swaps `payagent.guardrails.provider.
OpenRouterLlamaGuard` for a `FixtureGuardrailProvider(pass_through=True)` — the same
override point `tests/test_guarded_graph.py` uses for its own fake provider (see that
module's `rails` fixture), chosen over `guardrails.actions.set_guardrail_provider`
because NeMo loads `guardrails/actions.py` through its own file loader, not a normal
`import actions` (see `guardrails/actions.py::_get_provider`'s docstring) — patching the
shared `payagent.guardrails.provider` module is the one seam guaranteed to reach whichever
copy NeMo actually dispatches through. This neutralizes only the Llama Guard semantic
layer; `dlp_scan_input`, `heuristic_injection_check`, `grounding_check`, and
`dlp_mask_output` keep running exactly as they do in a full run, since none of them go
through a `GuardrailProvider` at all.

Each scenario runs through this state-machine skeleton (mirroring `scripts/demo.py` and
`graph/guarded.py::run_guarded`, but broken into individually-timed phases so
`cost_and_latency` can isolate rail latency from graph latency, and with the harness
supplying `selected_sku` from `scenario.hint_sku` in place of the human CLI prompt — but
only when `scenario.confirms_purchase` is also true, i.e. the user's own message actually
confirms buying that SKU; see `Scenario.confirms_purchase`'s docstring for why a
comparison or an informational question must not be treated the same as "yes, buy this"):

    check_input -> [blocked? stop] -> plan -> [answered_directly? stop] -> retrieve
    -> (hint_sku set and confirms_purchase: select it and quote; otherwise probe `quote`
       anyway with no selection, to verify achado #2 — `quote` must pause at
       `needs_clarification`, never guess which retrieved candidate to price, and never
       treat "tell me about X" as "buy X")
    -> mandate -> confirm -> settle -> compose_response -> check_output
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import structlog
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from evals.harness import metrics as metrics_mod
from evals.harness.pan_scan import scan_for_pan_leaks
from evals.harness.report import DEFAULT_REPORTS_DIR, build_payload, make_run_id, write_report
from evals.harness.results import ScenarioResult
from evals.harness.scenarios import DEFAULT_SCENARIOS_PATH, Scenario, load_scenarios
from evals.harness.tracing import (
    Tracer,
    TracingChatModel,
    patch_mcp_handler_tracing,
    patch_rag_tool_tracing,
)
from payagent.graph.graph import build_graph
from payagent.graph.respond import compose_response
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.guardrails import heuristics
from payagent.guardrails import provider as guardrail_provider
from payagent.guardrails.provider import FixtureGuardrailProvider
from payagent.guardrails.rails import build_rails, check_input, check_output
from payagent.mandates import InMemoryMandateStore
from payagent.mcp_server.__main__ import build_deps_from_env
from payagent.mcp_server.deps import ToolDeps
from payagent.mcp_server.idempotency import InMemoryIdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import InMemoryQuoteStore
from payagent.mcp_server.step_up import InMemoryStepUpVerifier
from payagent.observability.logging import configure_logging


def new_session_deps(base: ToolDeps) -> ToolDeps:
    """Fresh per-scenario session state layered on the shared, expensive singletons.

    Each scenario in `scenarios.yaml` is an independent, unrelated customer. Reusing one
    `MerchantSim` (and mandate/quote/idempotency/step-up state) across all of them makes
    POL-002's 24h aggregate-spend cap sum unrelated customers' purchases together instead
    of bounding any single customer's spend — after roughly the daily cap's worth of
    happy-path settlements, every later scenario gets denied `AGGREGATE_LIMIT_EXCEEDED`
    regardless of its own merits. `retriever`, `policy`, and `mandate_authority` are
    stateless (or Qdrant-backed read-only / signing-key-backed) and safe to share.
    """
    return replace(
        base,
        merchant=MerchantSim(clock=base.clock, id_factory=base.id_factory),
        mandate_store=InMemoryMandateStore(),
        quotes=InMemoryQuoteStore(clock=base.clock),
        idempotency=InMemoryIdempotencyStore(),
        step_up=InMemoryStepUpVerifier(id_factory=base.id_factory),
    )


class LogCapture:
    """`structlog`'s `PrintLoggerFactory` sink: every JSON log line, kept in memory and
    mirrored to a file, so the harness can grep it for PAN leaks and reconstruct which
    scenario a line belongs to (via the `scenario_id` contextvar bound around each run).

    `--workers > 1` means multiple threads call `.write()` concurrently (one structlog
    call per log line, from whichever thread emitted it) — `_lock` keeps each line's
    "append to the in-memory list" + "write to the file" pair atomic, so lines from
    different scenarios never interleave mid-write.
    """

    def __init__(self, file_path: Path) -> None:
        self._fh = open(file_path, "w", encoding="utf-8")  # noqa: SIM115 — closed in .close()
        self.lines: list[str] = []
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self.lines.append(s)
            return self._fh.write(s)

    def flush(self) -> None:
        with self._lock:
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def full_text(self) -> str:
        with self._lock:
            return "".join(self.lines)

    def lines_for(self, scenario_id: str) -> str:
        needle = f'"scenario_id": "{scenario_id}"'
        with self._lock:
            return "".join(line for line in self.lines if needle in line)


def _build_llm() -> ChatOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not configured. Copy .env.example to .env and fill it in."
        )
    return ChatOpenAI(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
        temperature=0,
    )


def _apply_no_guardrails_ablation() -> None:
    """Swap in a pass-through `FixtureGuardrailProvider` for the Llama Guard layer only."""
    guardrail_provider.OpenRouterLlamaGuard = lambda *a, **k: FixtureGuardrailProvider(
        pass_through=True
    )


def run_one_scenario(
    scenario: Scenario,
    graph,
    rails,
    tracer: Tracer,
    log_capture: LogCapture,
) -> ScenarioResult:
    result = ScenarioResult(scenario=scenario)
    tracer.begin(scenario.id, node="input_rail")
    structlog.contextvars.bind_contextvars(scenario_id=scenario.id)
    total_start = time.perf_counter()
    graph_latency_s = 0.0

    try:
        t0 = time.perf_counter()
        input_result = check_input(rails, scenario.user)
        result.rail_input_latency_ms = (time.perf_counter() - t0) * 1000

        if not input_result.allowed:
            result.input_blocked = True
            result.final_status = "input_blocked"
            result.response_text = input_result.refusal or ""
        else:
            state = PurchaseState(user_request=scenario.user)

            tracer.set_node("plan")
            t0 = time.perf_counter()
            state = graph.step("plan", state)
            graph_latency_s += time.perf_counter() - t0

            quote_never_ran = False

            if state.status != "answered_directly":
                tracer.set_node("retrieve")
                t0 = time.perf_counter()
                state = graph.step("retrieve", state)
                graph_latency_s += time.perf_counter() - t0

                if scenario.poisoned_sku:
                    chunk = next(
                        (c for c in state.retrieved_chunks if c.chunk_id == scenario.poisoned_sku),
                        None,
                    )
                    if chunk is not None:
                        result.p4_injection_detected_by_heuristic = heuristics.scan(
                            chunk.text
                        ).blocked

                if scenario.hint_sku:
                    retrieved_ids = {c.chunk_id for c in state.retrieved_chunks}
                    result.hint_sku_found_in_retrieval = scenario.hint_sku in retrieved_ids
                    if scenario.confirms_purchase:
                        state = state.model_copy(
                            update={"selected_sku": scenario.hint_sku, "selected_quantity": 1}
                        )

                tracer.set_node("quote")
                try:
                    t0 = time.perf_counter()
                    state = graph.step("quote", state)
                    graph_latency_s += time.perf_counter() - t0
                except IllegalTransitionError as exc:
                    # achado #2 (fixed 2026-08-01): "no selection at all" is now a graceful
                    # `needs_clarification` pause (see quote.py), never a raise — so the only
                    # way this can still fire is a `selected_sku` naming something `retrieve`
                    # never actually returned (an invented/hallucinated selection), e.g. a
                    # hint_sku that retrieval never surfaced (see hint_sku_found_in_retrieval)
                    # or some other real bug. Always unexpected now, unlike the old dual-branch
                    # version of this block.
                    quote_never_ran = True
                    result.error = f"IllegalTransitionError at quote: {exc}"

                if not quote_never_ran and scenario.hint_sku is None:
                    # Expected, deliberate outcome: no hint_sku means the harness never set
                    # state.selected_sku, and this scenario exists specifically to verify
                    # achado #2 (scenarios.yaml header) — `quote` must pause for
                    # clarification, never guess which retrieved candidate to price.
                    result.no_autonomous_disambiguation = state.status == "needs_clarification"

                if not quote_never_ran:
                    if state.status == "in_progress":
                        tracer.set_node("mandate")
                        t0 = time.perf_counter()
                        state = graph.step("mandate", state)
                        graph_latency_s += time.perf_counter() - t0

                    if state.status == "in_progress":
                        tracer.set_node("confirm")
                        t0 = time.perf_counter()
                        state = graph.step("confirm", state)
                        graph_latency_s += time.perf_counter() - t0

                    if state.status == "awaiting_step_up":
                        # Finding #1 (scenarios.yaml header): unreachable under the deployed
                        # policy config. No automated resolver exists here on purpose — if
                        # this ever actually fires, that is itself the significant result,
                        # not something to paper over with a fake WebAuthn approval.
                        result.raw_trace["unexpected_step_up_pause"] = True

                    if state.status == "in_progress":
                        tracer.set_node("settle")
                        t0 = time.perf_counter()
                        state = graph.step("settle", state)
                        graph_latency_s += time.perf_counter() - t0

            result.final_status = "quote_never_ran" if quote_never_ran else state.status
            result.settled = state.status == "settled"

            if state.status == "quote_failed" and state.quote is not None:
                result.actual_denial_code = state.quote["error"]["code"]
            elif state.status in ("mandate_denied", "settlement_denied"):
                if state.policy_decision is not None:
                    result.actual_denial_code = state.policy_decision.get("code")
                elif state.settlement_result is not None:
                    error = state.settlement_result.get("error") or {}
                    result.actual_denial_code = error.get("code")

            if result.settled and state.settlement_result and state.quote:
                settle_data = state.settlement_result["data"]
                quote_data = state.quote["data"]
                result.amount_merchant_consistent = (
                    settle_data["amount_cents"] == quote_data["amount_cents"]
                    and settle_data["merchant_id"] == quote_data["merchant_id"]
                )

            response_text = compose_response(state)
            t0 = time.perf_counter()
            output_result = check_output(
                rails,
                user_message=scenario.user,
                bot_message=response_text,
                retrieved_chunks=state.retrieved_chunks,
            )
            result.rail_output_latency_ms = (time.perf_counter() - t0) * 1000
            result.output_blocked = output_result.blocked
            result.response_text = output_result.text

    except Exception as exc:  # noqa: BLE001 — one scenario's crash must not kill the batch
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        structlog.contextvars.unbind_contextvars("scenario_id")

    result.graph_latency_ms = graph_latency_s * 1000
    result.total_latency_ms = (time.perf_counter() - total_start) * 1000

    llm_calls, tool_calls = tracer.calls_for(scenario.id)
    result.llm_calls = llm_calls
    result.tool_calls = tool_calls
    if tool_calls:
        result.actual_first_tool = tool_calls[0].tool
        result.actual_final_tool = tool_calls[-1].tool
        result.actual_final_tool_ok = tool_calls[-1].ok

    log_slice = log_capture.lines_for(scenario.id)
    result.pan_leak_count = len(scan_for_pan_leaks(log_slice))

    return result


def _result_to_json(r: ScenarioResult) -> dict[str, Any]:
    return {
        "id": r.scenario.id,
        "category": r.scenario.category,
        "taxonomy": r.scenario.taxonomy,
        "should_settle": r.scenario.should_settle,
        "input_blocked": r.input_blocked,
        "output_blocked": r.output_blocked,
        "final_status": r.final_status,
        "settled": r.settled,
        "actual_first_tool": r.actual_first_tool,
        "actual_final_tool": r.actual_final_tool,
        "actual_denial_code": r.actual_denial_code,
        "hint_sku_found_in_retrieval": r.hint_sku_found_in_retrieval,
        "p4_injection_detected_by_heuristic": r.p4_injection_detected_by_heuristic,
        "amount_merchant_consistent": r.amount_merchant_consistent,
        "no_autonomous_disambiguation": r.no_autonomous_disambiguation,
        "pan_leak_count": r.pan_leak_count,
        "total_tokens": r.total_tokens,
        "rail_input_latency_ms": r.rail_input_latency_ms,
        "rail_output_latency_ms": r.rail_output_latency_ms,
        "graph_latency_ms": r.graph_latency_ms,
        "total_latency_ms": r.total_latency_ms,
        "tool_calls": [asdict(c) for c in r.tool_calls],
        "error": r.error,
        "raw_trace": r.raw_trace,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default=str(DEFAULT_SCENARIOS_PATH))
    parser.add_argument(
        "--no-guardrails",
        action="store_true",
        help="Ablation: neutralize the Llama Guard layer via FixtureGuardrailProvider(pass_through=True).",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--taxonomy", action="append", default=None)
    parser.add_argument("--id", dest="ids", action="append", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help=(
            "Parallel scenario runs via ThreadPoolExecutor (default: 4). Threads, not "
            "processes: the graph is I/O-bound (LLM/Llama Guard/Qdrant calls), and "
            "ChatOpenAI/httpx/QdrantClient are safe for concurrent use. --workers 1 runs "
            "sequentially, useful for debugging a single scenario's trace in isolation."
        ),
    )
    return parser


def _filter_scenarios(scenarios: list[Scenario], args: argparse.Namespace) -> list[Scenario]:
    if args.ids:
        wanted = set(args.ids)
        scenarios = [s for s in scenarios if s.id in wanted]
    if args.category:
        wanted_cat = set(args.category)
        scenarios = [s for s in scenarios if s.category in wanted_cat]
    if args.taxonomy:
        wanted_tax = set(args.taxonomy)
        scenarios = [s for s in scenarios if s.taxonomy in wanted_tax]
    if args.limit is not None:
        scenarios = scenarios[: args.limit]
    return scenarios


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_arg_parser().parse_args(argv)

    guardrails_mode = "no-guardrails" if args.no_guardrails else "full"
    run_id = make_run_id(guardrails_mode=guardrails_mode)
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    log_capture = LogCapture(out_dir / f"{run_id}.log")
    configure_logging(stream=log_capture)

    if args.no_guardrails:
        _apply_no_guardrails_ablation()

    scenarios = load_scenarios(args.scenarios)
    scenarios = _filter_scenarios(scenarios, args)
    if not scenarios:
        print("No scenarios matched the given filters.", file=sys.stderr)
        return 1

    deps = build_deps_from_env()
    tracer = Tracer()
    llm = TracingChatModel(_build_llm(), tracer)
    rails = build_rails()

    print(
        f"Running {len(scenarios)} scenario(s) [{guardrails_mode}] -> {run_id} "
        f"({args.workers} worker{'s' if args.workers != 1 else ''})",
        file=sys.stderr,
    )

    results: list[ScenarioResult | None] = [None] * len(scenarios)
    done_count = 0
    print_lock = threading.Lock()

    def _run_indexed(index: int, scenario: Scenario) -> tuple[int, ScenarioResult]:
        scenario_deps = new_session_deps(deps)
        scenario_graph = build_graph(llm, scenario_deps.retriever, scenario_deps)
        return index, run_one_scenario(scenario, scenario_graph, rails, tracer, log_capture)

    with patch_rag_tool_tracing(tracer), patch_mcp_handler_tracing(tracer), ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [
            executor.submit(_run_indexed, i, scenario) for i, scenario in enumerate(scenarios)
        ]
        for future in as_completed(futures):
            index, result = future.result()
            results[index] = result
            with print_lock:
                done_count += 1
                print(
                    f"  [{done_count}/{len(scenarios)}] {result.scenario.id}"
                    f"{' — ERROR: ' + result.error if result.error else ''}",
                    file=sys.stderr,
                )

    log_capture.flush()
    log_capture.close()

    # run_one_scenario catches everything internally (see its own try/except), so this
    # should never actually filter anything out — the assert-by-filter is just cheap
    # insurance against a future.result() somehow surfacing a bare None.
    completed_results: list[ScenarioResult] = [r for r in results if r is not None]
    computed_metrics = metrics_mod.compute_all_metrics(completed_results)
    payload = build_payload(
        run_id=run_id,
        guardrails_mode=guardrails_mode,
        scenarios_path=str(args.scenarios),
        scenario_count=len(completed_results),
        metrics=computed_metrics,
        results=[_result_to_json(r) for r in completed_results],
    )
    json_path, md_path = write_report(payload, out_dir=out_dir)

    print(f"\nWrote {json_path}\nWrote {md_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
