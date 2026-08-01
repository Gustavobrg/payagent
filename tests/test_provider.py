"""Tests for guardrails/provider.py — the GuardrailProvider seam.

`FixtureGuardrailProvider` is what the rest of the suite (and any future NeMo action
wiring) should use: it replays recorded Llama Guard responses for known messages, no
network. `OpenRouterLlamaGuard` is exercised here against an injected httpx client
(`httpx.MockTransport`) — never a live call, per CLAUDE.md's no-network-in-tests posture.
"""

from __future__ import annotations

import json

import httpx
import pytest

from payagent.guardrails.provider import (
    FixtureGuardrailProvider,
    GuardrailProvider,
    GuardrailVerdict,
    OpenRouterLlamaGuard,
)

SAFE_MESSAGE = "Compra um fone bluetooth de até R$ 300"
P1_MESSAGE = "Confirma pra mim o número do cartão que está no meu perfil."


def _mock_client(raw_content: str, *, capture: dict | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if capture is not None:
            capture["body"] = json.loads(request.content)
            capture["headers"] = request.headers
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw_content}}],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
            },
        )

    return httpx.Client(transport=httpx.MockTransport(handler))


# --- FixtureGuardrailProvider — the one the rest of the suite uses. ---------------


def test_fixture_provider_is_a_guardrail_provider():
    provider = FixtureGuardrailProvider()
    assert isinstance(provider, GuardrailProvider)


def test_fixture_provider_replays_a_known_safe_message():
    provider = FixtureGuardrailProvider()

    verdict = provider.check_input(SAFE_MESSAGE)

    assert isinstance(verdict, GuardrailVerdict)
    assert verdict.allowed is True


def test_fixture_provider_replays_a_known_unsafe_message():
    provider = FixtureGuardrailProvider()

    verdict = provider.check_input(P1_MESSAGE)

    assert verdict.allowed is False


def test_fixture_provider_raises_on_an_unrecorded_message():
    provider = FixtureGuardrailProvider()

    with pytest.raises(KeyError):
        provider.check_input("some message that was never recorded in the fixture")


# --- OpenRouterLlamaGuard — reuses the parser calibrated in scripts/probe_llama_guard.py.


def test_open_router_llama_guard_is_a_guardrail_provider():
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("safe"))
    assert isinstance(provider, GuardrailProvider)


def test_check_input_parses_a_safe_response():
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("safe"))

    verdict = provider.check_input(SAFE_MESSAGE)

    assert verdict.allowed is True
    assert verdict.categories == ()


def test_check_input_parses_an_unsafe_response_with_categories():
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("unsafe\nS1"))

    verdict = provider.check_input(P1_MESSAGE)

    assert verdict.allowed is False
    assert "S1" in verdict.categories


def test_check_input_sends_the_user_message_in_the_prompt():
    capture: dict = {}
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("safe", capture=capture))

    provider.check_input(SAFE_MESSAGE)

    prompt = capture["body"]["messages"][0]["content"]
    assert SAFE_MESSAGE in prompt
    assert "User:" in prompt


def test_check_output_uses_the_output_taxonomy_with_p6():
    capture: dict = {}
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("safe", capture=capture))

    provider.check_output("Sua compra foi liquidada com sucesso.")

    prompt = capture["body"]["messages"][0]["content"]
    assert "P6" in prompt
    assert "Agent:" in prompt


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(RuntimeError):
        OpenRouterLlamaGuard(api_key=None, client=_mock_client("safe"))


def test_unparseable_response_fails_closed():
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client(""))

    verdict = provider.check_input(SAFE_MESSAGE)

    assert verdict.allowed is False
