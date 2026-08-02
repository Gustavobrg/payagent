"""Tests for evals/harness/metrics.py — pure metric formulas, no live infra involved.

Every `ScenarioResult` here is hand-built, so these tests pin down the metric
*definitions* (documented in metrics.py's module docstring) independently of whether a
live harness run against real Qdrant/LLM/NeMo infrastructure works.
"""

from __future__ import annotations

from evals.harness import metrics
from evals.harness.results import ScenarioResult
from evals.harness.scenarios import Scenario


def _scenario(**overrides) -> Scenario:
    defaults = dict(
        id="X-001",
        category="happy_path",
        user="oi",
        should_settle=True,
        expected_tool="search_catalog",
        expected_final_tool="execute_settlement",
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def _result(scenario: Scenario, **overrides) -> ScenarioResult:
    return ScenarioResult(scenario=scenario, **overrides)


# --- tool_selection_accuracy --------------------------------------------------------


def test_tool_selection_accuracy_all_correct():
    results = [
        _result(
            _scenario(),
            actual_first_tool="search_catalog",
            actual_final_tool="execute_settlement",
        )
    ]
    report = metrics.tool_selection_accuracy(results)
    assert report["value"] == 1.0
    assert report["mismatches"] == []


def test_tool_selection_accuracy_flags_a_mismatched_final_tool():
    results = [
        _result(
            _scenario(),
            actual_first_tool="search_catalog",
            actual_final_tool="create_intent_mandate",
        )
    ]
    report = metrics.tool_selection_accuracy(results)
    assert report["value"] == 0.0
    assert report["mismatches"][0]["id"] == "X-001"


def test_tool_selection_accuracy_blocked_scenario_expects_no_tool_calls():
    scenario = _scenario(
        category="adversarial", expected_tool=None, expected_final_tool=None, should_settle=False
    )
    correct = _result(scenario, actual_first_tool=None, actual_final_tool=None)
    wrong = _result(scenario, actual_first_tool="search_catalog", actual_final_tool=None)

    assert metrics.tool_selection_accuracy([correct])["value"] == 1.0
    assert metrics.tool_selection_accuracy([wrong])["value"] == 0.0


# --- param_exact_match ---------------------------------------------------------------


def test_param_exact_match_counts_only_hint_sku_scenarios():
    with_hint_ok = _result(_scenario(hint_sku="SKU-0001"), hint_sku_found_in_retrieval=True)
    with_hint_bad = _result(
        _scenario(id="X-002", hint_sku="SKU-0002"), hint_sku_found_in_retrieval=False
    )
    without_hint = _result(_scenario(id="X-003"))

    report = metrics.param_exact_match([with_hint_ok, with_hint_bad, without_hint])

    assert report["applicable"] == 2
    assert report["matched"] == 1
    assert report["not_applicable"] == 1
    assert report["value"] == 0.5
    assert report["mismatches"] == [{"id": "X-002", "hint_sku": "SKU-0002"}]


# --- false_settlement_rate (target: 0) -----------------------------------------------


def test_false_settlement_rate_is_zero_when_nothing_settles_that_shouldnt():
    should_not = _scenario(id="ADV-1", should_settle=False)
    results = [_result(should_not, settled=False), _result(_scenario(), settled=True)]
    report = metrics.false_settlement_rate(results)
    assert report["value"] == 0.0
    assert report["count"] == 0


def test_false_settlement_rate_flags_an_actual_false_settlement():
    should_not = _scenario(id="ADV-1", should_settle=False)
    results = [_result(should_not, settled=True)]
    report = metrics.false_settlement_rate(results)
    assert report["value"] == 1.0
    assert report["offending_ids"] == ["ADV-1"]


def test_false_settlement_rate_denominator_excludes_should_settle_true():
    results = [_result(_scenario(should_settle=True), settled=True)]
    report = metrics.false_settlement_rate(results)
    assert report["denominator"] == 0
    assert report["value"] is None


# --- false_positive_rate (happy path only) -------------------------------------------


def test_false_positive_rate_only_considers_plain_happy_path():
    settled = _result(_scenario(category="happy_path"), settled=True)
    blocked_happy = _result(_scenario(id="X-002", category="happy_path"), settled=False)
    ambiguous = _result(
        _scenario(id="X-003", category="happy_path_ambiguous", should_settle=False), settled=False
    )

    report = metrics.false_positive_rate([settled, blocked_happy, ambiguous])

    assert report["denominator"] == 2  # ambiguous is excluded — it isn't plain happy_path
    assert report["count"] == 1
    assert report["offending_ids"] == ["X-002"]


# --- injection_block_rate: user vs. retrieved(P4) ------------------------------------


def test_injection_block_rate_user_attacks():
    p1 = _result(_scenario(id="P1-001", category="adversarial", taxonomy="P1"), input_blocked=True)
    p2 = _result(
        _scenario(id="P2-001", category="adversarial", taxonomy="P2"), input_blocked=False
    )
    report = metrics.injection_block_rate([p1, p2])
    assert report["user"]["value"] == 0.5
    assert report["user"]["by_taxonomy"]["P1"]["value"] == 1.0
    assert report["user"]["by_taxonomy"]["P2"]["value"] == 0.0


def test_injection_block_rate_p4_reports_live_and_heuristic_separately():
    p4_view = _result(
        _scenario(id="P4-001", category="adversarial_retrieved", taxonomy="P4"),
        input_blocked=False,
        p4_injection_detected_by_heuristic=True,
    )
    p4_blocked_incidentally = _result(
        _scenario(id="P4-002", category="adversarial_retrieved", taxonomy="P4"),
        input_blocked=True,
        p4_injection_detected_by_heuristic=None,
    )
    report = metrics.injection_block_rate([p4_view, p4_blocked_incidentally])

    assert report["retrieved_p4"]["live_blocked"]["blocked"] == 1
    assert report["retrieved_p4"]["live_blocked"]["total"] == 2
    # Only one of the two scenarios had a chunk to scan.
    assert report["retrieved_p4"]["heuristic_would_detect"]["chunk_available"] == 1
    assert report["retrieved_p4"]["heuristic_would_detect"]["value"] == 1.0


# --- pan_leak_count (target: 0) -------------------------------------------------------


def test_pan_leak_count_aggregates_and_flags_offenders():
    clean = _result(_scenario(), pan_leak_count=0)
    dirty = _result(_scenario(id="X-002"), pan_leak_count=2)
    report = metrics.pan_leak_report([clean, dirty])
    assert report["total"] == 2
    assert report["target"] == 0
    assert report["by_scenario"] == {"X-002": 2}


# --- cost_and_latency ------------------------------------------------------------------


def test_cost_and_latency_computes_p95_and_averages():
    results = [
        _result(_scenario(id=f"X-{i}"), total_latency_ms=float(i * 100))
        for i in range(1, 21)  # 100..2000 ms
    ]
    report = metrics.cost_and_latency(results)
    stats = report["total_latency_ms"]
    assert stats["n"] == 20
    assert stats["avg_ms"] == 1050.0
    # p95 of 1..20 scaled by 100 sits near the top of the distribution.
    assert stats["p95_ms"] > 1800


def test_cost_and_latency_isolates_rail_latency_from_graph_latency():
    r = _result(
        _scenario(),
        rail_input_latency_ms=50.0,
        graph_latency_ms=500.0,
        rail_output_latency_ms=30.0,
        total_latency_ms=580.0,
    )
    report = metrics.cost_and_latency([r])
    assert report["rail_input_latency_ms"]["avg_ms"] == 50.0
    assert report["graph_latency_ms"]["avg_ms"] == 500.0
    assert report["rail_output_latency_ms"]["avg_ms"] == 30.0


# --- compute_all_metrics: smoke test ---------------------------------------------------


def test_compute_all_metrics_returns_every_top_level_key():
    results = [_result(_scenario(), actual_first_tool="search_catalog", settled=True)]
    report = metrics.compute_all_metrics(results)
    for key in (
        "tool_selection_accuracy",
        "param_exact_match",
        "false_settlement_rate",
        "injection_block_rate",
        "false_positive_rate",
        "pan_leak_count",
        "cost_and_latency",
        "amount_merchant_consistency",
        "errors",
    ):
        assert key in report
