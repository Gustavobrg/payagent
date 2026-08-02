"""Tests for evals/harness/pan_scan.py — the harness's independent PAN-leak audit.

Written independently of guardrails/dlp.py on purpose (see the module docstring); these
tests exercise that independent implementation directly, not dlp.py's.
"""

from __future__ import annotations

from evals.harness.pan_scan import scan_for_pan_leaks

VISA_TEST_PAN = "4111111111111111"


def test_no_leak_in_clean_text():
    assert scan_for_pan_leaks("policy_decision action=execute_settlement allowed=true") == []


def test_finds_a_luhn_valid_pan():
    leaks = scan_for_pan_leaks(f"tool_call arg_fingerprint={VISA_TEST_PAN}")
    assert len(leaks) == 1
    assert leaks[0].span == VISA_TEST_PAN


def test_ignores_a_long_digit_run_that_fails_luhn():
    not_a_card = "1234567890123456"  # 16 digits, fails Luhn
    assert scan_for_pan_leaks(f"trace_id={not_a_card}") == []


def test_finds_a_grouped_pan_with_spaces():
    leaks = scan_for_pan_leaks("card on file: 4111 1111 1111 1111 confirmed")
    assert len(leaks) == 1


def test_finds_multiple_leaks_in_one_blob():
    other_valid_pan = "5500000000000004"
    text = f"first={VISA_TEST_PAN} second={other_valid_pan}"
    leaks = scan_for_pan_leaks(text)
    assert len(leaks) == 2
