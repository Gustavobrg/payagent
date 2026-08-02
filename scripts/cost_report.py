"""Aggregates cost per session from exported OTel spans.

Reads the JSON-lines format `payagent.observability.tracing.JsonLinesSpanExporter` writes
(one JSON object per span) — the local trace artifact `evals/harness/run.py` leaves at
`evals/reports/<run_id>.spans.jsonl` alongside its existing `.json`/`.md`/`.log` report.

Cost and token usage come from Langfuse's own native generation fields
(`langfuse.observation.cost_details` / `langfuse.observation.usage_details`, set by
`observability/tracing.py::record_llm_usage` via the Langfuse SDK's `.update(cost_details=...,
usage_details=...)`) rather than a bespoke flat attribute — both are JSON-encoded strings on
the span, per Langfuse's own OTel attribute mapping.

Grouping is by `session.id` only. In this codebase one session is one purchase attempt
(CLAUDE.md's flow diagram — `graph/guarded.py::run_guarded` opens exactly one session per
call), so a separate "per transaction" dimension would just repeat the same grouping under a
different name; `session.id`, propagated via `langfuse.propagate_attributes` at every
composition root, is the one grouping key that already answers both.

`aggregate_costs` is pure (no I/O), mirroring `evals/harness/metrics.py`'s style: it takes
already-parsed span dicts and returns a plain, JSON-serializable summary. Spans that carry a
`payagent.cost.model_unknown` attribute (an unpriced model — see
`observability/tracing.py::record_llm_usage`) are collected separately rather than silently
folded into a total as if they cost nothing.

Usage:
    uv run python scripts/cost_report.py evals/reports/<run_id>.spans.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any

SESSION_ATTR = "session.id"
COST_DETAILS_ATTR = "langfuse.observation.cost_details"
USAGE_DETAILS_ATTR = "langfuse.observation.usage_details"
UNKNOWN_MODEL_ATTR = "payagent.cost.model_unknown"


def _new_group() -> dict[str, Any]:
    return {"cost_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "span_count": 0}


def aggregate_costs(spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Group cost-bearing spans by `session.id`.

    Returns `{"by_session": {...}, "total_cost_usd": float, "unknown_model_spans": [...]}`. A
    span contributes to a group only if it carries a parseable `langfuse.observation.
    cost_details` with a `"total"` entry; spans with a `payagent.cost.model_unknown` attribute
    instead are listed in `unknown_model_spans`, never counted as zero cost.
    """
    by_session: dict[str, dict[str, Any]] = defaultdict(_new_group)
    unknown_model_spans: list[dict[str, Any]] = []
    total_cost_usd = 0.0

    for span in spans:
        attributes = span.get("attributes") or {}

        unknown_model = attributes.get(UNKNOWN_MODEL_ATTR)
        if unknown_model:
            unknown_model_spans.append(
                {
                    "name": span.get("name"),
                    "span_id": span.get("span_id"),
                    "model": unknown_model,
                }
            )
            continue

        cost_details_raw = attributes.get(COST_DETAILS_ATTR)
        if cost_details_raw is None:
            continue
        cost_details = json.loads(cost_details_raw)
        cost = cost_details.get("total")
        if cost is None:
            continue

        usage_details_raw = attributes.get(USAGE_DETAILS_ATTR)
        usage_details = json.loads(usage_details_raw) if usage_details_raw else {}
        input_tokens = usage_details.get("input") or 0
        output_tokens = usage_details.get("output") or 0

        total_cost_usd += cost

        session_id = attributes.get(SESSION_ATTR)
        if session_id is not None:
            group = by_session[session_id]
            group["cost_usd"] += cost
            group["input_tokens"] += input_tokens
            group["output_tokens"] += output_tokens
            group["span_count"] += 1

    return {
        "by_session": dict(by_session),
        "total_cost_usd": total_cost_usd,
        "unknown_model_spans": unknown_model_spans,
    }


def load_spans(path: str) -> list[dict[str, Any]]:
    spans = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                spans.append(json.loads(line))
    return spans


def render_report(report: dict[str, Any]) -> str:
    lines = ["# Cost report", ""]
    lines.append(f"Total cost: ${report['total_cost_usd']:.6f}")
    lines.append("")

    lines.append("## By session")
    lines.append("")
    lines.append("| session.id | cost_usd | input_tokens | output_tokens | spans |")
    lines.append("|---|---|---|---|---|")
    for session_id, group in sorted(report["by_session"].items()):
        lines.append(
            f"| {session_id} | {group['cost_usd']:.6f} | {group['input_tokens']} | "
            f"{group['output_tokens']} | {group['span_count']} |"
        )
    lines.append("")

    if report["unknown_model_spans"]:
        lines.append(f"## ⚠️ Unpriced-model spans ({len(report['unknown_model_spans'])})")
        lines.append("")
        for entry in report["unknown_model_spans"]:
            lines.append(f"- {entry['name']} ({entry['span_id']}): model={entry['model']}")
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spans_path", help="Path to a *.spans.jsonl file.")
    args = parser.parse_args(argv)

    spans = load_spans(args.spans_path)
    report = aggregate_costs(spans)
    print(render_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
