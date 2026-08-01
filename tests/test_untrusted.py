"""Tests for the untrusted-content wrapper and its display-only inverse."""

from __future__ import annotations

from payagent.guardrails.untrusted import unwrap_untrusted_text, wrap_untrusted


def test_unwrap_recovers_the_original_text():
    original = "AuraSound Pro Wireless Headphones. Great audio quality with noise cancellation."

    wrapped = wrap_untrusted(original, source="catalog", chunk_id="SKU-HEADPHONES", score=0.87)

    assert unwrap_untrusted_text(wrapped) == original


def test_unwrap_recovers_multi_line_text():
    original = "Line one.\nLine two.\nLine three."

    wrapped = wrap_untrusted(original, source="policies", chunk_id="POL-001")

    assert unwrap_untrusted_text(wrapped) == original


def test_unwrap_is_a_no_op_on_text_that_was_never_wrapped():
    plain = "just some text, not wrapped at all"

    assert unwrap_untrusted_text(plain) == plain


def test_unwrap_never_raises_on_garbage_input():
    assert unwrap_untrusted_text("") == ""
    assert unwrap_untrusted_text("<untrusted-retrieved-content:abc") == (
        "<untrusted-retrieved-content:abc"
    )
