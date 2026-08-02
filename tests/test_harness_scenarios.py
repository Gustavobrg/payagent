"""Tests for evals/harness/scenarios.py — loading and validating the scenario dataset."""

from __future__ import annotations

import pytest

from evals.harness.scenarios import (
    DEFAULT_SCENARIOS_PATH,
    ScenarioSchemaError,
    load_scenarios,
)

SEED_PATH = DEFAULT_SCENARIOS_PATH.parent / "scenarios.seed.yaml"


def test_loads_the_seed_file():
    scenarios = load_scenarios(SEED_PATH)
    assert len(scenarios) == 8
    assert scenarios[0].id == "HP-001"
    assert scenarios[0].should_settle is True


def test_loads_the_full_dataset():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    assert len(scenarios) == 150
    ids = [s.id for s in scenarios]
    assert len(ids) == len(set(ids))


def test_happy_path_family_and_adversarial_partition_cleanly():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    happy = [s for s in scenarios if s.is_happy_path_family]
    adversarial = [s for s in scenarios if s.is_adversarial]
    assert len(happy) == 90
    assert len(adversarial) == 60
    assert set(s.id for s in happy) & set(s.id for s in adversarial) == set()


def test_p4_scenarios_are_flagged_as_retrieved_injection():
    scenarios = load_scenarios(DEFAULT_SCENARIOS_PATH)
    p4 = [s for s in scenarios if s.taxonomy == "P4"]
    assert len(p4) >= 15
    assert all(s.is_retrieved_injection for s in p4)
    non_p4_adversarial = [s for s in scenarios if s.is_adversarial and s.taxonomy != "P4"]
    assert not any(s.is_retrieved_injection for s in non_p4_adversarial)


def test_rejects_a_scenario_missing_should_settle(tmp_path):
    bad_file = tmp_path / "bad.yaml"
    bad_file.write_text(
        "- id: X-001\n  category: happy_path\n  user: oi\n  expected_tool: search_catalog\n",
        encoding="utf-8",
    )
    with pytest.raises(ScenarioSchemaError):
        load_scenarios(bad_file)


def test_tolerates_a_scenario_with_neither_tool_field_like_the_seed_does(tmp_path):
    """scenarios.seed.yaml's P1-001..P5-001 omit both fields entirely — must still load."""
    ok_file = tmp_path / "ok.yaml"
    ok_file.write_text(
        "- id: X-001\n  category: adversarial\n  user: oi\n  should_settle: false\n",
        encoding="utf-8",
    )
    loaded = load_scenarios(ok_file)
    assert loaded[0].expected_tool is None
    assert loaded[0].expected_final_tool is None
