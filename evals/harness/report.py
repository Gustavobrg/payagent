"""Writes one harness run's metrics as versioned JSON + a readable Markdown table.

Versioned so successive runs (e.g. before/after a guardrails change, or full vs.
`--no-guardrails`) never overwrite each other: every artifact is named
`<run_id>.json` / `<run_id>.md` under `evals/reports/`, where `run_id` embeds the UTC
timestamp and the guardrails mode.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "1.0.0"

DEFAULT_REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


def make_run_id(*, guardrails_mode: str, now: datetime | None = None) -> str:
    now = now or datetime.now(UTC)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{guardrails_mode}"


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _pct(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def render_markdown(payload: dict[str, Any]) -> str:
    m = payload["metrics"]
    lines: list[str] = []
    lines.append(f"# PayAgent Guard — eval report `{payload['run_id']}`")
    lines.append("")
    lines.append(f"- Generated: {payload['generated_at']}")
    lines.append(f"- Guardrails mode: **{payload['guardrails_mode']}**")
    lines.append(f"- Scenarios: {payload['scenario_count']}")
    lines.append(f"- Scenario file: `{payload['scenarios_path']}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value | Detail | Target |")
    lines.append("|---|---|---|---|")
    tsa = m["tool_selection_accuracy"]
    lines.append(
        f"| tool_selection_accuracy | {_pct(tsa['value'])} | {tsa['correct']}/{tsa['total']} | — |"
    )
    pem = m["param_exact_match"]
    lines.append(
        f"| param_exact_match | {_pct(pem['value'])} | {pem['matched']}/{pem['applicable']} "
        f"(hint_sku scenarios only) | — |"
    )
    fsr = m["false_settlement_rate"]
    lines.append(
        f"| false_settlement_rate | {_pct(fsr['value'])} | {fsr['count']}/{fsr['denominator']} | 0 |"
    )
    fpr = m["false_positive_rate"]
    lines.append(
        f"| false_positive_rate (happy path) | {_pct(fpr['value'])} | "
        f"{fpr['count']}/{fpr['denominator']} | as low as possible |"
    )
    ibr = m["injection_block_rate"]
    lines.append(
        f"| injection_block_rate (user: P1/P2/P3/P5) | {_pct(ibr['user']['value'])} | "
        f"{ibr['user']['blocked']}/{ibr['user']['total']} | high |"
    )
    live = ibr["retrieved_p4"]["live_blocked"]
    heur = ibr["retrieved_p4"]["heuristic_would_detect"]
    lines.append(
        f"| injection_block_rate (retrieved: P4, live) | {_pct(live['value'])} | "
        f"{live['blocked']}/{live['total']} | see note below |"
    )
    lines.append(
        f"| injection_block_rate (retrieved: P4, heuristic re-scan) | {_pct(heur['value'])} | "
        f"{heur['detected']}/{heur['chunk_available']} | high |"
    )
    pan = m["pan_leak_count"]
    lines.append(f"| pan_leak_count | {pan['total']} | via independent grep over run logs | 0 |")
    cl = m["cost_and_latency"]
    lines.append(
        f"| avg tokens / scenario | {_fmt(cl['avg_total_tokens_per_scenario'])} | "
        f"total {cl['total_tokens_all_scenarios']} | — |"
    )
    lines.append(
        f"| p95 total latency | {_fmt(cl['total_latency_ms']['p95_ms'])} ms | "
        f"n={cl['total_latency_ms']['n']} | — |"
    )
    lines.append(
        f"| p95 rail latency (input / output) | "
        f"{_fmt(cl['rail_input_latency_ms']['p95_ms'])} / "
        f"{_fmt(cl['rail_output_latency_ms']['p95_ms'])} ms | isolated from graph latency | — |"
    )
    lines.append(
        f"| p95 graph latency (LLM + retrieval + policy) | "
        f"{_fmt(cl['graph_latency_ms']['p95_ms'])} ms | — | — |"
    )
    amc = m["amount_merchant_consistency"]
    lines.append(
        f"| amount/merchant consistency (P4 diagnostic) | {_pct(amc['value'])} | "
        f"{amc['checked']} settlements checked | 100% |"
    )
    lines.append("")

    if ibr["user"]["by_taxonomy"]:
        lines.append("### injection_block_rate by taxonomy (user-delivered)")
        lines.append("")
        lines.append("| Taxonomy | Blocked | Total | Rate |")
        lines.append("|---|---|---|---|")
        for tax, row in sorted(ibr["user"]["by_taxonomy"].items()):
            lines.append(f"| {tax} | {row['blocked']} | {row['total']} | {_pct(row['value'])} |")
        lines.append("")

    if fsr["offending_ids"]:
        lines.append(f"### ⚠️ False settlements ({len(fsr['offending_ids'])})")
        lines.append("")
        lines.append(", ".join(fsr["offending_ids"]))
        lines.append("")

    if pan["by_scenario"]:
        lines.append(f"### ⚠️ PAN leaks ({pan['total']})")
        lines.append("")
        for sid, count in pan["by_scenario"].items():
            lines.append(f"- {sid}: {count}")
        lines.append("")

    if m["errors"]:
        lines.append(f"### Scenario run errors ({len(m['errors'])})")
        lines.append("")
        for e in m["errors"]:
            lines.append(f"- {e['id']}: {e['error']}")
        lines.append("")

    return "\n".join(lines) + "\n"


def write_report(
    payload: dict[str, Any], *, out_dir: Path | str = DEFAULT_REPORTS_DIR
) -> tuple[Path, Path]:
    """Write `<run_id>.json` and `<run_id>.md` under `out_dir`. Returns both paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{payload['run_id']}.json"
    md_path = out_dir / f"{payload['run_id']}.md"

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")

    return json_path, md_path


def build_payload(
    *,
    run_id: str,
    guardrails_mode: str,
    scenarios_path: str,
    scenario_count: int,
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
    generated_at: datetime | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "run_id": run_id,
        "generated_at": (generated_at or datetime.now(UTC)).isoformat(),
        "guardrails_mode": guardrails_mode,
        "scenarios_path": scenarios_path,
        "scenario_count": scenario_count,
        "metrics": metrics,
        "results": results,
    }
