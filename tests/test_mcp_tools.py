"""Contract tests for the six MCP tools, driven through `dispatch_tool_call`.

Five required categories per tool — valid schema accepted, invalid schema rejected with a structured
error, float in a monetary field rejected, idempotency, and policy denial — plus the invariant
checks that matter more than any of them: no caller input in an error message (I2), no settlement
without a verified mandate (I3), the amount always from the quote (P2), no crypto material in a
response (P3), and retrieved prose delimited and inert (P4/I5).

Tests are `async def` with no decorator: `asyncio_mode = "auto"` is already configured.
`dispatch_tool_call` takes no request context precisely so it can be called directly here, without
building a live `ServerSession`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from doubles import (
    AllowAllPolicyEngine,
    AlwaysVerifyingAuthority,
    RecordingPolicyEngine,
    counter_id_factory,
)
from mcp.shared.exceptions import MCPError
from pydantic import ValidationError

from payagent.mcp_server.dispatch import REPLAY_META_KEY, dispatch_tool_call
from payagent.mcp_server.merchant_sim import MerchantBehavior, MerchantSim
from payagent.mcp_server.registry import TOOL_SPECS
from payagent.mcp_server.schemas import ExecuteSettlementRequest
from payagent.mcp_server.step_up import InMemoryStepUpVerifier
from payagent.policy import DenyAllPolicyEngine, RulesPolicyEngine

SIDE_EFFECT_TOOLS = [
    "create_intent_mandate",
    "create_payment_mandate",
    "execute_settlement",
    "refund",
]

# A test-card-shaped string. Used to prove it never comes back in an error payload (I2).
TEST_PAN = "4111111111111111"

KEY_INTENT = "idem-key-intent-000001"
KEY_PAYMENT = "idem-key-payment-00001"
KEY_SETTLE = "idem-key-settle-000001"
KEY_REFUND = "idem-key-refund-000001"


def _envelope(result):
    return result.structured_content


def _error(result) -> dict:
    payload = _envelope(result)
    assert payload["error"] is not None, payload
    return payload["error"]


def _data(result) -> dict:
    payload = _envelope(result)
    assert payload["error"] is None, payload
    return payload["data"]


VALID_ARGUMENTS = {
    "search_catalog": {"query": "bluetooth speaker"},
    "get_quote": {"sku": "SKU-SPEAKER", "quantity": 1, "currency": "BRL"},
    "create_intent_mandate": {
        "idempotency_key": KEY_INTENT,
        "purpose": "Buy a bluetooth speaker",
        "max_amount_cents": 50000,
        "currency": "BRL",
        "allowed_categories": ["electronics"],
    },
    "create_payment_mandate": {
        "idempotency_key": KEY_PAYMENT,
        "intent_mandate_id": "IM-0000000000000001",
        "quote_id": "QT-0000000000000000000000000000000a",
        "expected_amount_cents": 4999,
        "expected_currency": "BRL",
    },
    "execute_settlement": {
        "idempotency_key": KEY_SETTLE,
        "payment_mandate_id": "PM-0000000000000002",
        "expected_amount_cents": 4999,
        "expected_currency": "BRL",
    },
    "refund": {
        "idempotency_key": KEY_REFUND,
        "settlement_id": "CH-0000000000000001",
        "expected_amount_cents": 4999,
        "expected_currency": "BRL",
        "reason_code": "defective",
    },
}


async def _quote(deps, *, sku: str = "SKU-SPEAKER", quantity: int = 1) -> dict:
    result = await dispatch_tool_call(
        deps, "get_quote", {"sku": sku, "quantity": quantity, "currency": "BRL"}
    )
    return _data(result)


async def _intent(deps, *, key: str = KEY_INTENT, max_amount_cents: int = 50000) -> dict:
    result = await dispatch_tool_call(
        deps,
        "create_intent_mandate",
        {
            "idempotency_key": key,
            "purpose": "Buy a bluetooth speaker",
            "max_amount_cents": max_amount_cents,
            "currency": "BRL",
            "allowed_categories": ["electronics"],
        },
    )
    return _data(result)


async def _payment_mandate(deps, *, key: str = KEY_PAYMENT) -> dict:
    intent = await _intent(deps)
    quote = await _quote(deps)
    result = await dispatch_tool_call(
        deps,
        "create_payment_mandate",
        {
            "idempotency_key": key,
            "intent_mandate_id": intent["intent_mandate_id"],
            "quote_id": quote["quote_id"],
            "expected_amount_cents": quote["amount_cents"],
            "expected_currency": quote["currency"],
        },
    )
    return _data(result)


async def _settlement(deps, *, key: str = KEY_SETTLE) -> dict:
    mandate = await _payment_mandate(deps)
    result = await dispatch_tool_call(
        deps,
        "execute_settlement",
        {
            "idempotency_key": key,
            "payment_mandate_id": mandate["payment_mandate_id"],
            "expected_amount_cents": mandate["amount_cents"],
            "expected_currency": mandate["currency"],
        },
    )
    return _data(result)


# ================================================ (1) valid schema accepted


async def test_search_catalog_accepts_a_valid_call(deps):
    result = await dispatch_tool_call(deps, "search_catalog", VALID_ARGUMENTS["search_catalog"])

    data = _data(result)
    assert result.is_error is not True
    assert data["returned"] == len(data["results"])
    assert any(hit["sku"] == "SKU-SPEAKER" for hit in data["results"])


async def test_get_quote_accepts_a_valid_call_and_computes_the_amount(deps):
    data = await _quote(deps, quantity=3)

    assert data["unit_price_cents"] == 4999
    assert data["amount_cents"] == 14997  # integer multiplication, server-side
    assert data["currency"] == "BRL"
    assert data["merchant_id"] == "MERCH-02"


@pytest.mark.parametrize("tool", SIDE_EFFECT_TOOLS)
async def test_side_effect_tools_accept_the_schema_then_are_denied_by_policy(deps, tool: str):
    """Under the production engine a valid call is denied — but for authorization, not validation.

    `field_errors == []` is what distinguishes "the schema accepted it" from "the schema rejected
    it" when every call is going to be denied anyway.
    """
    result = await dispatch_tool_call(deps, tool, VALID_ARGUMENTS[tool])

    error = _error(result)
    assert result.is_error is True
    assert error["field_errors"] == []
    # `create_payment_mandate`/`execute_settlement`/`refund` reference records that don't exist in
    # this bare fixture, so they stop earlier than policy — the point is that they got past the
    # schema.
    assert error["code"] != "INVALID_ARGUMENTS"


async def test_create_intent_mandate_succeeds_when_policy_allows(permissive_deps):
    data = await _intent(permissive_deps)

    assert data["status"] == "issued"
    assert data["intent_mandate_id"].startswith("IM-")
    assert data["max_amount_cents"] == 50000
    assert permissive_deps.mandate_store.intent_count == 1


async def test_create_payment_mandate_succeeds_and_copies_the_quoted_amount(permissive_deps):
    data = await _payment_mandate(permissive_deps)

    assert data["status"] == "issued"
    assert data["amount_cents"] == 4999
    assert data["sku"] == "SKU-SPEAKER"
    assert data["intent_mandate_id"].startswith("IM-")  # POL-009 parent reference present


async def test_execute_settlement_succeeds_and_charges_exactly_once(permissive_deps):
    data = await _settlement(permissive_deps)

    assert data["status"] == "settled"
    assert data["amount_cents"] == 4999
    assert len(permissive_deps.merchant.charges) == 1


async def test_refund_succeeds_and_reverses_the_full_amount(permissive_deps):
    settlement = await _settlement(permissive_deps)

    result = await dispatch_tool_call(
        permissive_deps,
        "refund",
        {
            "idempotency_key": KEY_REFUND,
            "settlement_id": settlement["settlement_id"],
            "expected_amount_cents": settlement["amount_cents"],
            "expected_currency": settlement["currency"],
            "reason_code": "defective",
        },
    )

    data = _data(result)
    assert data["status"] == "refunded"
    assert data["amount_cents"] == settlement["amount_cents"]


# ============================================ (2) invalid schema rejected, structured


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
async def test_unknown_argument_is_rejected_as_extra_forbidden(deps, spec):
    """`additionalProperties: false` is advertised, so it must actually be enforced."""
    arguments = {**VALID_ARGUMENTS[spec.name], "bogus_field": 1}

    result = await dispatch_tool_call(deps, spec.name, arguments)

    error = _error(result)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert ("bogus_field", "extra_forbidden") in [
        (fe["path"], fe["code"]) for fe in error["field_errors"]
    ]


@pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)
async def test_each_missing_required_field_is_reported_by_path(deps, spec):
    required = spec.input_schema().get("required", [])
    for field in required:
        arguments = {k: v for k, v in VALID_ARGUMENTS[spec.name].items() if k != field}

        result = await dispatch_tool_call(deps, spec.name, arguments)

        error = _error(result)
        assert error["code"] == "INVALID_ARGUMENTS", (spec.name, field)
        assert field in [fe["path"] for fe in error["field_errors"]], (spec.name, field)


@pytest.mark.parametrize(
    ("tool", "arguments", "expected_path", "expected_code"),
    [
        ("search_catalog", {"query": "x", "top_k": 0}, "top_k", "greater_than_equal"),
        ("search_catalog", {"query": "x", "top_k": 99}, "top_k", "less_than_equal"),
        ("search_catalog", {"query": ""}, "query", "string_too_short"),
        (
            "search_catalog",
            {"query": "x", "min_price_cents": 500, "max_price_cents": 100},
            "(root)",
            "value_error",
        ),
        ("get_quote", {"sku": "SKU-SPEAKER", "currency": "brl"}, "currency", "string_pattern_mismatch"),
        ("get_quote", {"sku": "SKU-SPEAKER", "currency": "BRLL"}, "currency", "string_too_long"),
        ("get_quote", {"sku": "SKU-SPEAKER", "currency": "BRL", "quantity": 0}, "quantity", "greater_than_equal"),
        (
            "refund",
            {**VALID_ARGUMENTS["refund"], "reason_code": "whatever"},
            "reason_code",
            "literal_error",
        ),
        (
            "execute_settlement",
            {**VALID_ARGUMENTS["execute_settlement"], "idempotency_key": "short"},
            "idempotency_key",
            "string_too_short",
        ),
        (
            "create_intent_mandate",
            {**VALID_ARGUMENTS["create_intent_mandate"], "allowed_categories": []},
            "allowed_categories",
            "too_short",
        ),
        (
            "create_intent_mandate",
            {**VALID_ARGUMENTS["create_intent_mandate"], "expires_in_seconds": 5},
            "expires_in_seconds",
            "greater_than_equal",
        ),
    ],
)
async def test_constraint_violations_report_the_expected_path_and_code(
    deps, tool: str, arguments: dict, expected_path: str, expected_code: str
):
    result = await dispatch_tool_call(deps, tool, arguments)

    error = _error(result)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert (expected_path, expected_code) in [
        (fe["path"], fe["code"]) for fe in error["field_errors"]
    ]


@pytest.mark.parametrize("arguments", [None, {}], ids=["none", "empty"])
async def test_missing_arguments_are_rejected_not_defaulted(deps, arguments):
    result = await dispatch_tool_call(deps, "get_quote", arguments)

    assert _error(result)["code"] == "INVALID_ARGUMENTS"


async def test_validation_error_never_echoes_the_rejected_value(deps):
    """I2 in one test: a PAN pasted into any field must not come back in the error payload.

    Pydantic includes the rejected input in `errors()` by default and interpolates it into `msg` for
    several error types, so this guards the whole class of leak rather than one field.
    """
    result = await dispatch_tool_call(
        deps,
        "get_quote",
        {"sku": TEST_PAN + "!!", "currency": TEST_PAN, "quantity": 1},
    )

    serialized = json.dumps(result.model_dump(by_alias=True, mode="json"))
    assert _error(result)["code"] == "INVALID_ARGUMENTS"
    assert TEST_PAN not in serialized


async def test_unknown_tool_raises_a_protocol_error_without_echoing_the_name(deps):
    with pytest.raises(MCPError) as exc_info:
        await dispatch_tool_call(deps, "drain_the_account", {})

    assert "drain_the_account" not in str(exc_info.value)


# ============================================ (3) float in a monetary field rejected

MONEY_FIELDS = [
    ("search_catalog", "min_price_cents"),
    ("search_catalog", "max_price_cents"),
    ("create_intent_mandate", "max_amount_cents"),
    ("create_payment_mandate", "expected_amount_cents"),
    ("execute_settlement", "expected_amount_cents"),
    ("refund", "expected_amount_cents"),
]


@pytest.mark.parametrize(("tool", "field"), MONEY_FIELDS, ids=lambda v: str(v))
@pytest.mark.parametrize("bad_value", [4999.0, 49.99, "4999", True], ids=["float-int", "float", "str", "bool"])
async def test_non_integer_monetary_values_are_rejected(deps, tool: str, field: str, bad_value):
    """CLAUDE.md: float in a monetary value is a bug. `4999.0` must fail as loudly as `49.99`."""
    arguments = {**VALID_ARGUMENTS[tool], field: bad_value}

    result = await dispatch_tool_call(deps, tool, arguments)

    error = _error(result)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert (field, "int_type") in [(fe["path"], fe["code"]) for fe in error["field_errors"]]


@pytest.mark.parametrize(("tool", "field"), MONEY_FIELDS, ids=lambda v: str(v))
async def test_negative_monetary_values_are_rejected(deps, tool: str, field: str):
    result = await dispatch_tool_call(deps, tool, {**VALID_ARGUMENTS[tool], field: -1})

    error = _error(result)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert field in [fe["path"] for fe in error["field_errors"]]


@pytest.mark.parametrize("field", ["quantity", "top_k"])
async def test_non_integer_counts_are_rejected_too(deps, field: str):
    """The same `Strict()` class as the money fields, so it regresses the same way."""
    tool = "get_quote" if field == "quantity" else "search_catalog"
    arguments = {**VALID_ARGUMENTS[tool], field: 1.0}

    result = await dispatch_tool_call(deps, tool, arguments)

    assert (field, "int_type") in [
        (fe["path"], fe["code"]) for fe in _error(result)["field_errors"]
    ]


async def test_refund_rejects_a_caller_supplied_amount(deps):
    """Full-refund-only: `amount_cents` is not a field, so sending it is a schema violation."""
    result = await dispatch_tool_call(
        deps, "refund", {**VALID_ARGUMENTS["refund"], "amount_cents": 1}
    )

    assert ("amount_cents", "extra_forbidden") in [
        (fe["path"], fe["code"]) for fe in _error(result)["field_errors"]
    ]


# ================================================================ (4) idempotency


async def test_intent_mandate_replay_returns_the_identical_result(permissive_deps):
    first = await dispatch_tool_call(
        permissive_deps, "create_intent_mandate", VALID_ARGUMENTS["create_intent_mandate"]
    )
    second = await dispatch_tool_call(
        permissive_deps, "create_intent_mandate", VALID_ARGUMENTS["create_intent_mandate"]
    )

    assert second.structured_content == first.structured_content
    assert permissive_deps.mandate_store.intent_count == 1
    assert second.meta == {REPLAY_META_KEY: True}


async def test_settlement_replay_returns_the_identical_result_and_charges_once(permissive_deps):
    mandate = await _payment_mandate(permissive_deps)
    arguments = {
        "idempotency_key": KEY_SETTLE,
        "payment_mandate_id": mandate["payment_mandate_id"],
        "expected_amount_cents": mandate["amount_cents"],
        "expected_currency": mandate["currency"],
    }

    first = await dispatch_tool_call(permissive_deps, "execute_settlement", arguments)
    second = await dispatch_tool_call(permissive_deps, "execute_settlement", arguments)

    assert second.structured_content == first.structured_content
    assert len(permissive_deps.merchant.charges) == 1


async def test_settlement_replay_with_different_arguments_is_refused_and_performs_no_effect(
    permissive_deps,
):
    """The P2 case: returning the cached success here would confirm a charge that never happened."""
    mandate = await _payment_mandate(permissive_deps)
    arguments = {
        "idempotency_key": KEY_SETTLE,
        "payment_mandate_id": mandate["payment_mandate_id"],
        "expected_amount_cents": mandate["amount_cents"],
        "expected_currency": mandate["currency"],
    }
    await dispatch_tool_call(permissive_deps, "execute_settlement", arguments)
    calls_before = len(permissive_deps.merchant.calls)

    result = await dispatch_tool_call(
        permissive_deps,
        "execute_settlement",
        {**arguments, "expected_amount_cents": 500000},
    )

    assert _error(result)["code"] == "IDEMPOTENCY_KEY_REUSED"
    assert len(permissive_deps.merchant.charges) == 1
    assert len(permissive_deps.merchant.calls) == calls_before


async def test_different_keys_with_the_same_arguments_perform_two_effects(permissive_deps):
    """Proves the key, not the payload, is the dedup axis."""
    await _intent(permissive_deps, key="idem-key-intent-000001")
    await _intent(permissive_deps, key="idem-key-intent-000002")

    assert permissive_deps.mandate_store.intent_count == 2


async def test_a_policy_denial_replays_without_re_consulting_policy(build_deps):
    """A denied purchase cannot be retried into an allowed one under the same key."""
    engine = RecordingPolicyEngine(DenyAllPolicyEngine())
    deps = build_deps(policy=engine)

    first = await dispatch_tool_call(
        deps, "create_intent_mandate", VALID_ARGUMENTS["create_intent_mandate"]
    )
    calls_after_first = len(engine.calls)
    second = await dispatch_tool_call(
        deps, "create_intent_mandate", VALID_ARGUMENTS["create_intent_mandate"]
    )

    assert _error(first)["code"] == "POLICY_DENIED"
    assert second.structured_content == first.structured_content
    assert len(engine.calls) == calls_after_first  # policy was not asked twice


async def test_a_merchant_timeout_releases_the_key_so_a_retry_charges_exactly_once(
    build_deps, clock
):
    """The lost-response case: release on an indeterminate outcome, and let merchant dedup finish it."""
    merchant = MerchantSim(
        clock=clock,
        id_factory=counter_id_factory(),
        scripted=[MerchantBehavior.TIMEOUT_AFTER_COMMIT, MerchantBehavior.ACCEPT],
    )
    deps = build_deps(
        policy=AllowAllPolicyEngine(),
        mandate_authority=AlwaysVerifyingAuthority(),
        merchant=merchant,
    )
    mandate = await _payment_mandate(deps)
    arguments = {
        "idempotency_key": KEY_SETTLE,
        "payment_mandate_id": mandate["payment_mandate_id"],
        "expected_amount_cents": mandate["amount_cents"],
        "expected_currency": mandate["currency"],
    }

    timed_out = await dispatch_tool_call(deps, "execute_settlement", arguments)
    retried = await dispatch_tool_call(deps, "execute_settlement", arguments)

    assert _error(timed_out)["code"] == "MERCHANT_TIMEOUT"
    assert _error(timed_out)["retryable"] is True
    assert _data(retried)["status"] == "settled"
    assert len(merchant.charges) == 1


# ============================================================= (5) policy denial


@pytest.mark.parametrize("tool", SIDE_EFFECT_TOOLS)
async def test_side_effect_tools_are_denied_and_perform_no_effect(build_deps, tool: str):
    """Every side-effect tool must be reachable-but-refused, with nothing left behind."""
    engine = RecordingPolicyEngine(DenyAllPolicyEngine())
    deps = build_deps(policy=engine, mandate_authority=AlwaysVerifyingAuthority())

    # Build the prerequisites under a permissive engine, then deny the tool under test.
    if tool in ("create_payment_mandate", "execute_settlement", "refund"):
        setup = build_deps(
            policy=AllowAllPolicyEngine(),
            mandate_authority=AlwaysVerifyingAuthority(),
            mandate_store=deps.mandate_store,
            quotes=deps.quotes,
            merchant=deps.merchant,
        )
        if tool == "create_payment_mandate":
            intent = await _intent(setup)
            quote = await _quote(setup)
            arguments = {
                "idempotency_key": KEY_PAYMENT,
                "intent_mandate_id": intent["intent_mandate_id"],
                "quote_id": quote["quote_id"],
                "expected_amount_cents": quote["amount_cents"],
                "expected_currency": quote["currency"],
            }
        elif tool == "execute_settlement":
            mandate = await _payment_mandate(setup)
            arguments = {
                "idempotency_key": KEY_SETTLE,
                "payment_mandate_id": mandate["payment_mandate_id"],
                "expected_amount_cents": mandate["amount_cents"],
                "expected_currency": mandate["currency"],
            }
        else:
            settlement = await _settlement(setup)
            arguments = {
                "idempotency_key": KEY_REFUND,
                "settlement_id": settlement["settlement_id"],
                "expected_amount_cents": settlement["amount_cents"],
                "expected_currency": settlement["currency"],
                "reason_code": "defective",
            }
        charges_before = len(deps.merchant.charges)
    else:
        arguments = VALID_ARGUMENTS[tool]
        charges_before = 0

    result = await dispatch_tool_call(deps, tool, arguments)

    error = _error(result)
    assert error["code"] == "POLICY_DENIED"
    assert error["denial"] == {
        "code": "POLICY_NOT_CONFIGURED",
        "reason": (
            "No authorization policy is configured for this action, so it cannot be permitted."
        ),
        "policy_ref": None,
    }
    assert len(deps.merchant.charges) == charges_before


async def test_read_only_tools_never_consult_the_policy_engine(build_deps):
    """The I1 boundary: deciding relevance is not deciding permission."""
    engine = RecordingPolicyEngine()
    deps = build_deps(policy=engine)

    await dispatch_tool_call(deps, "search_catalog", VALID_ARGUMENTS["search_catalog"])
    await dispatch_tool_call(deps, "get_quote", VALID_ARGUMENTS["get_quote"])

    assert engine.calls == []


async def test_policy_sees_the_quoted_amount_not_the_asserted_one(build_deps):
    """Whatever the caller asserts, the engine is asked about the amount the server recorded."""
    engine = RecordingPolicyEngine(AllowAllPolicyEngine())
    deps = build_deps(policy=engine, mandate_authority=AlwaysVerifyingAuthority())

    await _payment_mandate(deps)

    payment_calls = [ctx for action, ctx in engine.calls if action.value == "create_payment_mandate"]
    assert payment_calls
    assert payment_calls[0].amount_cents == 4999


async def test_policy_never_receives_step_up_satisfied_from_a_request(build_deps):
    """P5: step-up state is server-side only, and no request model can supply it."""
    engine = RecordingPolicyEngine(AllowAllPolicyEngine())
    deps = build_deps(policy=engine, mandate_authority=AlwaysVerifyingAuthority())

    await _settlement(deps)

    assert all(ctx.step_up_satisfied is False for _, ctx in engine.calls)


# =================================================== invariants beyond the five


async def test_settlement_is_refused_when_the_mandate_cannot_be_verified(build_deps):
    """I3 defence in depth: policy allows, but the shipped authority verifies nothing."""
    deps = build_deps(policy=AllowAllPolicyEngine())  # default UnsignedMandateAuthority
    mandate = await _payment_mandate(deps)

    result = await dispatch_tool_call(
        deps,
        "execute_settlement",
        {
            "idempotency_key": KEY_SETTLE,
            "payment_mandate_id": mandate["payment_mandate_id"],
            "expected_amount_cents": mandate["amount_cents"],
            "expected_currency": mandate["currency"],
        },
    )

    assert _error(result)["code"] == "MANDATE_NOT_VERIFIED"
    assert deps.merchant.calls == ()


async def test_settlement_is_refused_when_the_quote_has_expired(permissive_deps, clock):
    """POL-010: expiry is re-checked at settlement time, not only at mandate time."""
    mandate = await _payment_mandate(permissive_deps)
    clock.advance(permissive_deps.quote_ttl_seconds + 1)

    result = await dispatch_tool_call(
        permissive_deps,
        "execute_settlement",
        {
            "idempotency_key": KEY_SETTLE,
            "payment_mandate_id": mandate["payment_mandate_id"],
            "expected_amount_cents": mandate["amount_cents"],
            "expected_currency": mandate["currency"],
        },
    )

    assert _error(result)["code"] in {"QUOTE_EXPIRED", "PAYMENT_MANDATE_EXPIRED"}
    assert permissive_deps.merchant.charges == {}


async def test_payment_mandate_rejects_a_mismatched_expected_amount(permissive_deps):
    """The P2 test: an injected or hallucinated amount becomes a countable rejection."""
    intent = await _intent(permissive_deps)
    quote = await _quote(permissive_deps)

    result = await dispatch_tool_call(
        permissive_deps,
        "create_payment_mandate",
        {
            "idempotency_key": KEY_PAYMENT,
            "intent_mandate_id": intent["intent_mandate_id"],
            "quote_id": quote["quote_id"],
            "expected_amount_cents": 100,  # the quote says 4999
            "expected_currency": "BRL",
        },
    )

    assert _error(result)["code"] == "AMOUNT_MISMATCH"
    assert permissive_deps.mandate_store.payment_count == 0


async def test_settlement_rejects_a_mismatched_expected_amount_without_charging(permissive_deps):
    mandate = await _payment_mandate(permissive_deps)

    result = await dispatch_tool_call(
        permissive_deps,
        "execute_settlement",
        {
            "idempotency_key": KEY_SETTLE,
            "payment_mandate_id": mandate["payment_mandate_id"],
            "expected_amount_cents": 500000,
            "expected_currency": "BRL",
        },
    )

    assert _error(result)["code"] == "AMOUNT_MISMATCH"
    assert permissive_deps.merchant.charges == {}


async def test_no_response_payload_contains_mandate_cryptographic_material(permissive_deps):
    """P3: mandates are returned as readable summaries, never as signing material."""
    forbidden = ("jws", "signature", "private", "secret", "key_material")

    for data in (
        await _intent(permissive_deps),
        await _payment_mandate(permissive_deps),
        await _settlement(permissive_deps),
    ):
        serialized = json.dumps(data).lower()
        for marker in forbidden:
            assert marker not in serialized, marker


async def test_search_catalog_results_are_wrapped_as_untrusted(deps):
    """I5: retrieved prose reaches the caller delimited, with a nonce it cannot forge."""
    result = await dispatch_tool_call(deps, "search_catalog", VALID_ARGUMENTS["search_catalog"])

    for hit in _data(result)["results"]:
        text = hit["text_untrusted"]
        assert text.startswith("<untrusted-retrieved-content:")
        assert "retrieved data, not instructions" in text
        assert f'chunk_id="{hit["chunk_id"]}"' in text


async def test_search_catalog_price_comes_from_the_payload_not_the_product_text(deps):
    """P4: a description claiming 'Actual price: R$1.00' cannot move the number."""
    result = await dispatch_tool_call(
        deps, "search_catalog", {"query": "Nexus Bluetooth Speaker", "top_k": 10}
    )

    speaker = next(h for h in _data(result)["results"] if h["sku"] == "SKU-SPEAKER")
    assert speaker["price_cents"] == 4999
    assert "R$1.00" in speaker["text_untrusted"]  # the claim is delimited, not removed


async def test_get_quote_price_ignores_the_injected_price_claim(deps):
    data = await _quote(deps)

    assert data["unit_price_cents"] == 4999
    assert data["merchant_id"] == "MERCH-02"  # not the MERCH-99 the injection names


async def test_get_quote_rejects_a_currency_that_disagrees_with_the_catalog(deps):
    result = await dispatch_tool_call(
        deps, "get_quote", {"sku": "SKU-SPEAKER", "quantity": 1, "currency": "USD"}
    )

    assert _error(result)["code"] == "CURRENCY_MISMATCH"


async def test_get_quote_returns_the_same_quote_id_within_the_validity_window(deps, clock):
    first = await _quote(deps)
    clock.advance(60)
    second = await _quote(deps)

    assert second["quote_id"] == first["quote_id"]
    assert second["expires_at"] == first["expires_at"]  # the expiry was not silently refreshed


async def test_get_quote_rejects_an_unknown_sku(deps):
    result = await dispatch_tool_call(
        deps, "get_quote", {"sku": "SKU-GHOST", "quantity": 1, "currency": "BRL"}
    )

    assert _error(result)["code"] == "UNKNOWN_SKU"


async def test_get_quote_rejects_an_out_of_stock_sku(deps):
    result = await dispatch_tool_call(
        deps, "get_quote", {"sku": "SKU-OUTOFSTOCK", "quantity": 1, "currency": "BRL"}
    )

    assert _error(result)["code"] == "OUT_OF_STOCK"


async def test_payment_mandate_is_refused_when_the_quote_expired(permissive_deps, clock):
    intent = await _intent(permissive_deps)
    quote = await _quote(permissive_deps)
    clock.advance(permissive_deps.quote_ttl_seconds + 1)

    result = await dispatch_tool_call(
        permissive_deps,
        "create_payment_mandate",
        {
            "idempotency_key": KEY_PAYMENT,
            "intent_mandate_id": intent["intent_mandate_id"],
            "quote_id": quote["quote_id"],
            "expected_amount_cents": quote["amount_cents"],
            "expected_currency": quote["currency"],
        },
    )

    assert _error(result)["code"] in {"QUOTE_EXPIRED", "QUOTE_NOT_FOUND"}
    assert permissive_deps.mandate_store.payment_count == 0


async def test_a_second_refund_on_the_same_settlement_is_refused(permissive_deps):
    settlement = await _settlement(permissive_deps)
    base = {
        "settlement_id": settlement["settlement_id"],
        "expected_amount_cents": settlement["amount_cents"],
        "expected_currency": settlement["currency"],
        "reason_code": "defective",
    }
    await dispatch_tool_call(
        permissive_deps, "refund", {**base, "idempotency_key": "idem-key-refund-000001"}
    )

    result = await dispatch_tool_call(
        permissive_deps, "refund", {**base, "idempotency_key": "idem-key-refund-000002"}
    )

    assert _error(result)["code"] == "SETTLEMENT_ALREADY_REFUNDED"


async def test_refund_of_an_unknown_settlement_is_refused(permissive_deps):
    result = await dispatch_tool_call(permissive_deps, "refund", VALID_ARGUMENTS["refund"])

    assert _error(result)["code"] == "SETTLEMENT_NOT_FOUND"


async def test_every_result_mirrors_its_envelope_into_text_content(deps):
    """Clients that ignore `structuredContent` must still get the full payload."""
    result = await dispatch_tool_call(deps, "search_catalog", VALID_ARGUMENTS["search_catalog"])

    assert json.loads(result.content[0].text) == result.structured_content


# ============================================================ P5: step-up cannot come from a tool call


def test_execute_settlement_schema_has_no_field_that_can_assert_step_up():
    """P5 structurally: `extra='forbid'` means a caller cannot smuggle step-up state into the call."""
    with pytest.raises(ValidationError):
        ExecuteSettlementRequest(
            idempotency_key=KEY_SETTLE,
            payment_mandate_id="PM-0000000000000002",
            expected_amount_cents=4999,
            expected_currency="BRL",
            step_up_satisfied=True,  # type: ignore[call-arg]
        )


@pytest.mark.parametrize("tool", VALID_ARGUMENTS.keys())
@pytest.mark.parametrize("field_name", ["step_up_token", "user_confirmed"])
async def test_no_tool_call_can_smuggle_step_up_through_any_argument_shape(
    deps, tool: str, field_name: str
):
    """P5, black-box: this field must not exist on any of the six tools' schemas.

    If it ever does, this starts passing schema validation — which is the regression this test
    exists to catch — and step-up would be one caller-supplied argument away from bypassed,
    regardless of how `deps.step_up` or the policy engine behave.
    """
    arguments = {**VALID_ARGUMENTS[tool], field_name: True}

    result = await dispatch_tool_call(deps, tool, arguments)

    error = _error(result)
    assert error["code"] == "INVALID_ARGUMENTS"
    assert any(fe["path"] == field_name for fe in error["field_errors"])


def test_handlers_never_call_policy_engine_decide_directly():
    """I1's boundary at the call site: only `payagent.policy.evaluate` may reach `engine.decide`.

    A direct `.decide(` call in `handlers.py` would bypass the fail-closed funnel — `evaluate()`
    is what turns a raising, malformed, or grant-mismatched engine into a `Deny` — so a
    misbehaving or half-written engine's raw return value could reach a handler unguarded.
    Grepping the source is deliberate: an import-based check could be fooled by an alias
    (`from payagent.policy.engine import PolicyEngine as X; X.decide(...)`), and the property
    that actually matters is that the literal call never appears.
    """
    source = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "payagent"
        / "mcp_server"
        / "handlers.py"
    ).read_text(encoding="utf-8")

    assert ".decide(" not in source
    assert "evaluate(" in source


async def test_settlement_above_the_stepup_threshold_is_denied_without_out_of_band_resolution(
    build_deps,
):
    """POL-003 through the real engine: no request field can stand in for the WebAuthn ceremony."""
    engine = RulesPolicyEngine(
        max_amount_cents=1_000_000,
        stepup_threshold_cents=1000,
        daily_aggregate_limit_cents=1_000_000,
        merchant_allowlist=frozenset({"MERCH-01", "MERCH-02"}),
        restricted_categories=frozenset(),
    )
    deps = build_deps(
        policy=engine,
        mandate_authority=AlwaysVerifyingAuthority(),
        step_up=InMemoryStepUpVerifier(id_factory=counter_id_factory()),
    )
    intent = await _intent(deps, max_amount_cents=1_000_000)
    quote = await _quote(deps)
    payment = await dispatch_tool_call(
        deps,
        "create_payment_mandate",
        {
            "idempotency_key": KEY_PAYMENT,
            "intent_mandate_id": intent["intent_mandate_id"],
            "quote_id": quote["quote_id"],
            "expected_amount_cents": quote["amount_cents"],
            "expected_currency": quote["currency"],
        },
    )

    assert _error(payment)["denial"]["code"] == "STEP_UP_REQUIRED"
    assert deps.mandate_store.payment_count == 0


async def test_settlement_above_the_stepup_threshold_succeeds_after_the_separate_channel_resolves_it(
    build_deps, clock
):
    """Same scenario, but the step-up ceremony completes out of band before the retry."""
    step_up = InMemoryStepUpVerifier(id_factory=counter_id_factory())
    engine = RulesPolicyEngine(
        max_amount_cents=1_000_000,
        stepup_threshold_cents=1000,
        daily_aggregate_limit_cents=1_000_000,
        merchant_allowlist=frozenset({"MERCH-01", "MERCH-02"}),
        restricted_categories=frozenset(),
    )
    deps = build_deps(
        policy=engine, mandate_authority=AlwaysVerifyingAuthority(), step_up=step_up
    )
    intent = await _intent(deps, max_amount_cents=1_000_000)
    quote = await _quote(deps)

    # The step-up ceremony happens on a channel entirely separate from the tool call.
    challenge = step_up.issue_challenge(now=clock())
    step_up.resolve_challenge(challenge.challenge_id, now=clock())

    payment = await dispatch_tool_call(
        deps,
        "create_payment_mandate",
        {
            "idempotency_key": "idem-key-payment-00002",
            "intent_mandate_id": intent["intent_mandate_id"],
            "quote_id": quote["quote_id"],
            "expected_amount_cents": quote["amount_cents"],
            "expected_currency": quote["currency"],
        },
    )

    assert _data(payment)["status"] == "issued"
