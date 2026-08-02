"""Tests for evals/harness/report.py — versioned JSON + Markdown output."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from evals.harness import metrics
from evals.harness.report import build_payload, make_run_id, render_markdown, write_report
from evals.harness.results import ScenarioResult
from evals.harness.scenarios import Scenario


def _sample_results() -> list[ScenarioResult]:
    s1 = Scenario(
        id="HP-001",
        category="happy_path",
        user="oi",
        should_settle=True,
        expected_tool="search_catalog",
        expected_final_tool="execute_settlement",
    )
    return [
        ScenarioResult(
            scenario=s1,
            actual_first_tool="search_catalog",
            actual_final_tool="execute_settlement",
            settled=True,
            total_latency_ms=120.0,
        )
    ]


def test_make_run_id_embeds_timestamp_and_mode():
    run_id = make_run_id(guardrails_mode="full", now=datetime(2026, 8, 1, 12, 0, tzinfo=UTC))
    assert run_id == "20260801T120000Z_full"


def test_write_report_produces_versioned_json_and_markdown(tmp_path):
    results = _sample_results()
    payload = build_payload(
        run_id="20260801T120000Z_full",
        guardrails_mode="full",
        scenarios_path="evals/datasets/scenarios.yaml",
        scenario_count=1,
        metrics=metrics.compute_all_metrics(results),
        results=[{"id": r.scenario.id} for r in results],
        generated_at=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
    )

    json_path, md_path = write_report(payload, out_dir=tmp_path)

    assert json_path.name == "20260801T120000Z_full.json"
    assert md_path.name == "20260801T120000Z_full.md"
    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "1.0.0"
    assert loaded["scenario_count"] == 1

    md_text = md_path.read_text(encoding="utf-8")
    assert "tool_selection_accuracy" in md_text
    assert "false_settlement_rate" in md_text


def test_two_runs_do_not_collide(tmp_path):
    results = _sample_results()
    m = metrics.compute_all_metrics(results)
    payload_full = build_payload(
        run_id="20260801T120000Z_full",
        guardrails_mode="full",
        scenarios_path="x",
        scenario_count=1,
        metrics=m,
        results=[],
    )
    payload_ablation = build_payload(
        run_id="20260801T120000Z_no-guardrails",
        guardrails_mode="no-guardrails",
        scenarios_path="x",
        scenario_count=1,
        metrics=m,
        results=[],
    )

    write_report(payload_full, out_dir=tmp_path)
    write_report(payload_ablation, out_dir=tmp_path)

    assert (tmp_path / "20260801T120000Z_full.json").exists()
    assert (tmp_path / "20260801T120000Z_no-guardrails.json").exists()


def test_render_markdown_flags_false_settlements():
    s = Scenario(
        id="P1-001",
        category="adversarial",
        user="x",
        should_settle=False,
        expected_tool=None,
        expected_final_tool=None,
        taxonomy="P1",
    )
    results = [ScenarioResult(scenario=s, settled=True)]
    payload = build_payload(
        run_id="r",
        guardrails_mode="full",
        scenarios_path="x",
        scenario_count=1,
        metrics=metrics.compute_all_metrics(results),
        results=[],
    )
    md = render_markdown(payload)
    assert "False settlements" in md
    assert "P1-001" in md
