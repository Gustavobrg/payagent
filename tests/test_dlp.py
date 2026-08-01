"""Tests for guardrails/dlp.py — deterministic PAN/CVV/expiry detection plus Presidio PII.

I2 (CLAUDE.md): no PAN, CVV, or expiry may reach a log, span, exception, or error message.
Detection here must be regex + Luhn, never a model call — see docs/guardrail-taxonomy.md P1.

Only known test card numbers are used (Visa/Mastercard/Amex test PANs); no real card data
touches this repository.
"""

from __future__ import annotations

import io
import sys

from payagent.guardrails.dlp import DlpScanResult, luhn_is_valid, mask, scan

VISA_TEST_PAN = "4111111111111111"
MASTERCARD_TEST_PAN = "5555555555554444"
NOT_A_CARD_16_DIGITS = "1234567890123456"  # fails Luhn — e.g. a tracking number


def test_known_test_pan_is_detected():
    result = scan(f"my card number is {VISA_TEST_PAN}, please charge it")

    assert isinstance(result, DlpScanResult)
    assert result.has_sensitive_data is True
    assert "PAN" in result.categories


def test_pan_with_spaces_is_detected():
    spaced = "4111 1111 1111 1111"

    result = scan(f"card: {spaced}")

    assert result.has_sensitive_data is True
    assert "PAN" in result.categories


def test_pan_with_hyphens_is_detected():
    hyphenated = "4111-1111-1111-1111"

    result = scan(f"card: {hyphenated}")

    assert result.has_sensitive_data is True
    assert "PAN" in result.categories


def test_second_test_pan_is_also_detected():
    result = scan(f"mastercard {MASTERCARD_TEST_PAN} on file")

    assert result.has_sensitive_data is True
    assert "PAN" in result.categories


def test_number_that_fails_luhn_is_not_a_false_positive():
    assert luhn_is_valid(NOT_A_CARD_16_DIGITS) is False

    result = scan(f"tracking number {NOT_A_CARD_16_DIGITS} left the warehouse today")

    assert result.has_sensitive_data is False
    assert result.categories == ()


def test_isolated_cvv_is_detected():
    result = scan("the CVV is 737, please confirm")

    assert result.has_sensitive_data is True
    assert "CVV" in result.categories


def test_isolated_cvv_is_detected_with_portuguese_keyword():
    result = scan("o codigo de seguranca do cartao e 048")

    assert result.has_sensitive_data is True
    assert "CVV" in result.categories


def test_legitimate_16_digit_number_that_is_not_a_card_does_not_trigger():
    result = scan(f"your package tracking code is {NOT_A_CARD_16_DIGITS}")

    assert result.has_sensitive_data is False


def test_scan_with_no_sensitive_data_returns_empty_result():
    result = scan("I would like to buy a pair of running shoes under R$ 250")

    assert result.has_sensitive_data is False
    assert result.categories == ()


def test_mask_redacts_pan_and_reports_redaction():
    text = f"card number {VISA_TEST_PAN} was declined"

    masked, redacted = mask(text)

    assert redacted is True
    assert VISA_TEST_PAN not in masked
    assert "PAN" in masked


def test_mask_is_a_noop_on_clean_text():
    text = "please cancel my last order"

    masked, redacted = mask(text)

    assert redacted is False
    assert masked == text


def test_mask_redacts_cvv():
    text = "cvv 737 confirmed"

    masked, redacted = mask(text)

    assert redacted is True
    assert "737" not in masked


# --- I2 permanent regression test — keep this in the suite forever. ---------------


def test_i2_pan_never_appears_anywhere_in_a_serialized_log_line():
    """A PAN passed to the structured logger must never survive to the serialized line.

    This exercises the real sink (payagent.observability.logging), not dlp.mask in
    isolation — I2 is about what actually reaches a log, not about the DLP module alone.
    """
    from payagent.observability import logging as obs_logging

    stream = io.StringIO()
    obs_logging.configure_logging(stream=stream)
    try:
        logger = obs_logging.get_logger("test_i2")
        logger.info(
            "card_charge_attempted",
            message=f"attempting charge for card {VISA_TEST_PAN}",
            raw_pan=VISA_TEST_PAN,
        )
    finally:
        obs_logging.configure_logging(stream=sys.stdout)

    output = stream.getvalue()
    assert VISA_TEST_PAN not in output
