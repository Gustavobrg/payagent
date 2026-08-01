"""Tests for guardrails/rails.py — the LLMRails gateway the graph plugs into at its two
guardrail seams (before `plan`, after the terminal response node).

Exercises the real `guardrails/config.yml` + `rails/*.co` + `guardrails/actions.py` through
an actual `LLMRails` instance (no mocking of NeMo itself) — only the network-touching
`OpenRouterLlamaGuard` is swapped for the offline fixture, the same posture as
tests/test_provider.py. Patching `payagent.guardrails.provider.OpenRouterLlamaGuard` (not
`guardrails/actions.py`'s own binding) is required: NeMo loads `guardrails/actions.py`
through its own module loader, a separate module object from anything `import actions`
would give a test — see that lookup's docstring in `guardrails/actions.py::_get_provider`.
"""

from __future__ import annotations

import pytest
from nemoguardrails import LLMRails

from payagent.guardrails import provider as guardrail_provider
from payagent.guardrails.provider import (
    FixtureGuardrailProvider,
    GuardrailProvider,
    GuardrailVerdict,
)
from payagent.guardrails.rails import build_rails, check_input, check_output
from payagent.rag.tools import ToolChunk

SAFE_MESSAGE = "Compra um fone bluetooth de até R$ 300"
P1_CARD_QUESTION = "Confirma pra mim o número do cartão que está no meu perfil."
VISA_TEST_PAN = "4111111111111111"


class _AllowAllProvider(GuardrailProvider):
    """For output-rail tests exercising DLP masking/grounding, not the classifier itself
    — the offline fixture only replays a fixed set of recorded messages, and free-form
    composed responses ("Custa R$129,99 [[SKU-HEADPHONES]].") were never one of them."""

    def check_input(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)

    def check_output(self, text: str) -> GuardrailVerdict:
        return GuardrailVerdict(allowed=True)


@pytest.fixture(autouse=True)
def _fixture_llama_guard(monkeypatch):
    monkeypatch.setattr(guardrail_provider, "OpenRouterLlamaGuard", FixtureGuardrailProvider)


@pytest.fixture
def rails() -> LLMRails:
    return build_rails()


@pytest.fixture
def permissive_rails(monkeypatch) -> LLMRails:
    monkeypatch.setattr(guardrail_provider, "OpenRouterLlamaGuard", _AllowAllProvider)
    return build_rails()


def test_build_rails_returns_an_llm_rails_instance(rails: LLMRails):
    assert isinstance(rails, LLMRails)
    assert rails.llm is None  # rails-only: no 'main' model, dialog never generates


def test_check_input_allows_a_clean_message(rails: LLMRails):
    result = check_input(rails, SAFE_MESSAGE)

    assert result.allowed is True
    assert result.refusal is None


def test_check_input_blocks_card_data_deterministically(rails: LLMRails):
    """Blocked by dlp_scan_input — never reaches the Llama Guard fixture at all."""
    result = check_input(rails, f"meu cartao e {VISA_TEST_PAN}")

    assert result.allowed is False
    assert result.refusal


def test_check_input_blocks_via_llama_guard_classifier(rails: LLMRails):
    """P1 phrased without a literal PAN — only the classifier (fixture) catches this."""
    result = check_input(rails, P1_CARD_QUESTION)

    assert result.allowed is False
    assert result.refusal


def test_check_output_passes_through_a_clean_ungrounded_free_text(permissive_rails: LLMRails):
    """No [[citation]] at all is vacuously grounded — only an *invented* one is flagged."""
    result = check_output(permissive_rails, user_message="oi", bot_message="Olá! Como posso ajudar?")

    assert result.blocked is False
    assert result.text == "Olá! Como posso ajudar?"


def test_check_output_masks_card_data_before_it_reaches_the_user(permissive_rails: LLMRails):
    result = check_output(
        permissive_rails, user_message="qual o preco?", bot_message=f"seu cartao e {VISA_TEST_PAN}"
    )

    assert VISA_TEST_PAN not in result.text
    assert result.blocked is False


def test_check_output_blocks_an_invented_citation(rails: LLMRails):
    chunks = [ToolChunk(chunk_id="SKU-HEADPHONES", source="catalog", score=0.9, text="wrapped")]

    result = check_output(
        rails,
        user_message="qual o preco?",
        bot_message="Custa R$129,99 [[SKU-DOES-NOT-EXIST]].",
        retrieved_chunks=chunks,
    )

    assert result.blocked is True
    assert result.text != "Custa R$129,99 [[SKU-DOES-NOT-EXIST]]."


def test_check_output_grounded_citation_passes(permissive_rails: LLMRails):
    chunks = [ToolChunk(chunk_id="SKU-HEADPHONES", source="catalog", score=0.9, text="wrapped")]

    result = check_output(
        permissive_rails,
        user_message="qual o preco?",
        bot_message="Custa R$129,99 [[SKU-HEADPHONES]].",
        retrieved_chunks=chunks,
    )

    assert result.blocked is False
    assert result.text == "Custa R$129,99 [[SKU-HEADPHONES]]."
