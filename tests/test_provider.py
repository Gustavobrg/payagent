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
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from payagent.guardrails.provider import (
    FixtureGuardrailProvider,
    GuardrailProvider,
    GuardrailVerdict,
    OpenRouterLlamaGuard,
)
from payagent.observability import tracing

SAFE_MESSAGE = "Compra um fone bluetooth de até R$ 300"
P1_MESSAGE = "Confirma pra mim o número do cartão que está no meu perfil."
VISA_TEST_PAN = "4111111111111111"


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


# --- pass-through mode — the eval harness's --no-guardrails ablation. --------------


def test_fixture_provider_pass_through_does_not_require_a_fixture_file():
    """No fixture file needed at all in pass-through mode — it never consults one."""
    provider = FixtureGuardrailProvider(pass_through=True, fixture_path="/nonexistent/path.json")

    verdict = provider.check_input("anything at all, never recorded anywhere")

    assert isinstance(verdict, GuardrailVerdict)
    assert verdict.allowed is True
    assert verdict.categories == ()


def test_fixture_provider_pass_through_allows_any_input_regardless_of_content():
    provider = FixtureGuardrailProvider(pass_through=True)

    verdict = provider.check_input("Meu cartao e 4111 1111 1111 1111")

    assert verdict.allowed is True


def test_fixture_provider_pass_through_allows_any_output():
    provider = FixtureGuardrailProvider(pass_through=True)

    verdict = provider.check_output("qualquer resposta do agente")

    assert verdict.allowed is True


def test_fixture_provider_without_pass_through_still_requires_a_fixture():
    """Default behavior (pass_through=False) is unchanged: still needs recorded fixtures."""
    provider = FixtureGuardrailProvider()

    verdict = provider.check_input(SAFE_MESSAGE)

    assert verdict.allowed is True


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


# --- OTel instrumentation: one span per Llama Guard call, with usage/cost, DLP-safe. ----


def test_check_input_emits_a_rail_llama_guard_generation_with_token_usage_and_cost():
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)
    provider = OpenRouterLlamaGuard(api_key="test-key", client=_mock_client("safe"))

    provider.check_input(SAFE_MESSAGE)
    tracing.flush()

    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert "rail.llama_guard" in spans
    attrs = dict(spans["rail.llama_guard"].attributes)
    assert attrs["langfuse.observation.model.name"] == "meta-llama/llama-guard-4-12b"
    usage = json.loads(attrs["langfuse.observation.usage_details"])
    assert usage == {"input": 12, "output": 3}
    cost = json.loads(attrs["langfuse.observation.cost_details"])
    assert cost["total"] > 0
    assert attrs["allowed"] is True


def test_check_input_raw_response_containing_a_pan_is_redacted_on_the_span():
    """P1 edge case: if the classifier ever echoes a PAN back in its raw response, the span
    attribute must not carry it — same I2 discipline as every other sink."""
    exporter = InMemorySpanExporter()
    tracing.configure_tracing(client=tracing.client_with_exporter(exporter), force=True)
    provider = OpenRouterLlamaGuard(
        api_key="test-key", client=_mock_client(f"unsafe\nP1 card {VISA_TEST_PAN}")
    )

    provider.check_input(P1_MESSAGE)
    tracing.flush()

    spans = exporter.get_finished_spans()
    serialized = str([dict(s.attributes) for s in spans])
    assert VISA_TEST_PAN not in serialized
