"""Tests for the deterministic policy engine (invariant I1).

The engine decides authorization; nothing here calls an LLM, touches the network, or reads
a clock. Two properties matter more than any individual rule and are tested structurally
rather than by convention:

- **Fail-closed.** Every path that isn't an explicit `Allow` must end as a `Deny` — including
  an engine that raises, returns the wrong type, or grants the wrong action.
- **Purity.** `policy/` must not import an LLM client, an HTTP client, a vector store, or
  either of its own callers. Asserted by parsing the AST, so it survives refactors.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path

import pytest

from payagent.policy import (
    Allow,
    DenialCode,
    Deny,
    DenyAllPolicyEngine,
    EffectGrant,
    PolicyAction,
    PolicyContext,
    PolicyDenial,
    PolicyEngine,
    evaluate,
)

FIXED_NOW = datetime(2026, 7, 31, 12, 0, tzinfo=UTC)

POLICY_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "payagent" / "policy"

# Modules `policy/` must never reach for: LLM clients, network clients, the vector store,
# the guardrail framework, and its own callers (which would invert the dependency).
FORBIDDEN_IMPORT_ROOTS = {
    "langchain",
    "langchain_core",
    "langchain_openai",
    "openai",
    "anthropic",
    "httpx",
    "requests",
    "qdrant_client",
    "nemoguardrails",
    "mcp",
    "mcp_types",
}
FORBIDDEN_FIRST_PARTY = {"payagent.rag", "payagent.mcp_server", "payagent.graph"}


def _context(**overrides) -> PolicyContext:
    base = {"now": FIXED_NOW, "amount_cents": 12999, "currency": "BRL"}
    return PolicyContext(**{**base, **overrides})


class RaisingEngine(PolicyEngine):
    def decide(self, action, context):
        raise RuntimeError("upstream policy store unreachable")


class NoneReturningEngine(PolicyEngine):
    def decide(self, action, context):
        return None  # type: ignore[return-value]


class WrongActionGrantEngine(PolicyEngine):
    """Allows, but the grant names a different action than the one asked about."""

    def decide(self, action, context):
        other = (
            PolicyAction.REFUND
            if action is not PolicyAction.REFUND
            else PolicyAction.EXECUTE_SETTLEMENT
        )
        return Allow(
            grant=EffectGrant(
                action=other,
                amount_cents=context.amount_cents,
                currency=context.currency,
                merchant_id=context.merchant_id,
                decided_at=context.now,
            )
        )


@pytest.mark.parametrize("action", list(PolicyAction))
def test_deny_all_engine_denies_every_action(action: PolicyAction):
    """Parametrized over the enum so a new action added in a later block cannot default to allow."""
    decision = evaluate(DenyAllPolicyEngine(), action, _context())

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.POLICY_NOT_CONFIGURED


def test_allow_cannot_be_constructed_without_a_grant():
    """`Allow` has no default: there is no zero-argument decision object to fall into."""
    with pytest.raises(TypeError):
        Allow()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "engine", [RaisingEngine(), NoneReturningEngine(), WrongActionGrantEngine()]
)
def test_evaluate_converts_a_misbehaving_engine_into_a_denial(engine: PolicyEngine):
    decision = evaluate(engine, PolicyAction.EXECUTE_SETTLEMENT, _context())

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.POLICY_ENGINE_ERROR


def test_evaluate_passes_through_a_well_formed_allow():
    class AllowingEngine(PolicyEngine):
        def decide(self, action, context):
            return Allow(
                grant=EffectGrant(
                    action=action,
                    amount_cents=context.amount_cents,
                    currency=context.currency,
                    merchant_id=context.merchant_id,
                    decided_at=context.now,
                )
            )

    decision = evaluate(AllowingEngine(), PolicyAction.REFUND, _context())

    assert isinstance(decision, Allow)
    assert decision.grant.action is PolicyAction.REFUND
    assert decision.grant.amount_cents == 12999


def test_policy_context_rejects_a_float_amount():
    with pytest.raises(TypeError, match="int"):
        _context(amount_cents=129.99)


def test_policy_context_rejects_a_bool_amount():
    """`True` is an `int` subclass in Python — an explicit bool check is the only thing that catches it."""
    with pytest.raises(TypeError, match="int"):
        _context(amount_cents=True)


def test_policy_context_rejects_a_negative_amount():
    with pytest.raises(ValueError, match="negative"):
        _context(amount_cents=-1)


def test_policy_context_rejects_a_naive_datetime():
    """A naive `now` makes expiry comparisons silently wrong rather than loudly broken."""
    with pytest.raises(ValueError, match="timezone-aware"):
        _context(now=datetime(2026, 7, 31, 12, 0))


def test_effect_grant_rejects_a_float_amount():
    with pytest.raises(TypeError, match="int"):
        EffectGrant(
            action=PolicyAction.REFUND,
            amount_cents=1.0,  # type: ignore[arg-type]
            currency="BRL",
            merchant_id=None,
            decided_at=FIXED_NOW,
        )


def test_denial_reason_is_static_text_not_a_template():
    """Denial reasons are authored in `policy/`, never generated — so they carry no caller input."""
    denial = DenyAllPolicyEngine().decide(PolicyAction.REFUND, _context()).denial

    assert isinstance(denial, PolicyDenial)
    assert denial.reason
    assert "{" not in denial.reason
    assert "%s" not in denial.reason


def _imported_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize(
    "module_path", sorted(POLICY_PACKAGE.glob("*.py")), ids=lambda p: p.name
)
def test_policy_package_imports_no_llm_network_or_caller(module_path: Path):
    """Invariant I1 as a test: `policy/` is deterministic and depends on nothing that can decide for it."""
    for name in _imported_names(module_path):
        root = name.split(".")[0]
        assert root not in FORBIDDEN_IMPORT_ROOTS, f"{module_path.name} imports {name}"
        for forbidden in FORBIDDEN_FIRST_PARTY:
            assert not name.startswith(forbidden), f"{module_path.name} imports {name}"
