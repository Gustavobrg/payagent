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
    RulesPolicyEngine,
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


# ============================================================ RulesPolicyEngine (the real rules)

# Deliberately far apart from the default max so amount-limit and step-up can be tested in
# isolation from one another, mirroring POL-002 §6: the step-up threshold only ever matters
# above the standard ceiling, which in production means it never fires. A high max here lets a
# test push the amount past the step-up threshold without also tripping AMOUNT_EXCEEDS_LIMIT.
_MAX_AMOUNT = 50_000
_STEPUP_THRESHOLD = 100_000
_DAILY_CAP = 200_000
_ALLOWED_MERCHANTS = frozenset({"MERCH-01", "MERCH-02"})
_RESTRICTED_CATEGORIES = frozenset({"alcoholic beverages", "jewelry and luxury watches"})


def _rules_engine(**overrides) -> RulesPolicyEngine:
    defaults = dict(
        max_amount_cents=_MAX_AMOUNT,
        stepup_threshold_cents=_STEPUP_THRESHOLD,
        daily_aggregate_limit_cents=_DAILY_CAP,
        merchant_allowlist=_ALLOWED_MERCHANTS,
        restricted_categories=_RESTRICTED_CATEGORIES,
    )
    return RulesPolicyEngine(**{**defaults, **overrides})


def test_rules_engine_allows_a_well_formed_purchase_within_every_limit():
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(amount_cents=4999, merchant_id="MERCH-02", category="electronics"),
    )

    assert isinstance(decision, Allow)
    assert decision.grant.amount_cents == 4999


def test_rules_engine_denies_amount_above_the_per_transaction_limit():
    """POL-001: any amount over POLICY_MAX_AMOUNT_CENTS is refused."""
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(amount_cents=_MAX_AMOUNT + 1, merchant_id="MERCH-02", category="electronics"),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.AMOUNT_EXCEEDS_LIMIT
    assert decision.denial.policy_ref == "POL-001"


def test_rules_engine_allows_amount_exactly_at_the_limit():
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(amount_cents=_MAX_AMOUNT, merchant_id="MERCH-02", category="electronics"),
    )

    assert isinstance(decision, Allow)


def test_rules_engine_denies_merchant_outside_the_allowlist():
    """POL-005: settlement is refused for any merchant_id not on the allowlist."""
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(amount_cents=1000, merchant_id="MERCH-99", category="electronics"),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.MERCHANT_NOT_ALLOWED
    assert decision.denial.policy_ref == "POL-005"


def test_rules_engine_denies_a_restricted_category_regardless_of_the_amount():
    """POL-004: category restriction is not a limit, it applies at any amount."""
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(amount_cents=1, merchant_id="MERCH-02", category="alcoholic beverages"),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.CATEGORY_RESTRICTED
    assert decision.denial.policy_ref == "POL-004"


def test_rules_engine_denies_an_intent_mandate_scoped_to_a_restricted_category():
    """The restriction also blocks the intent from ever declaring the scope, not just spending it."""
    decision = _rules_engine().decide(
        PolicyAction.CREATE_INTENT_MANDATE,
        _context(
            amount_cents=1000,
            intent_mandate_allowed_categories=("electronics", "jewelry and luxury watches"),
        ),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.CATEGORY_RESTRICTED


def test_rules_engine_denies_an_intent_mandate_scoped_to_a_disallowed_merchant():
    decision = _rules_engine().decide(
        PolicyAction.CREATE_INTENT_MANDATE,
        _context(
            amount_cents=1000,
            intent_mandate_allowed_categories=("electronics",),
            intent_mandate_allowed_merchant_ids=("MERCH-99",),
        ),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.MERCHANT_NOT_ALLOWED


def test_rules_engine_denies_when_the_24h_aggregate_would_be_exceeded():
    """POL-002: the running 24h total plus this transaction must not exceed the daily cap."""
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(
            amount_cents=1000,
            merchant_id="MERCH-02",
            category="electronics",
            aggregate_spent_24h_cents=_DAILY_CAP,
        ),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.AGGREGATE_LIMIT_EXCEEDED
    assert decision.denial.policy_ref == "POL-002"


def test_rules_engine_allows_when_the_24h_aggregate_stays_within_the_cap():
    decision = _rules_engine().decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(
            amount_cents=1000,
            merchant_id="MERCH-02",
            category="electronics",
            aggregate_spent_24h_cents=_DAILY_CAP - 1000,
        ),
    )

    assert isinstance(decision, Allow)


def test_rules_engine_denies_settlement_above_the_stepup_threshold_without_step_up():
    """POL-003: step-up is mandatory above the threshold and cannot be implied."""
    decision = _rules_engine(max_amount_cents=1_000_000).decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(
            amount_cents=_STEPUP_THRESHOLD + 1,
            merchant_id="MERCH-02",
            category="electronics",
            step_up_satisfied=False,
        ),
    )

    assert isinstance(decision, Deny)
    assert decision.denial.code is DenialCode.STEP_UP_REQUIRED
    assert decision.denial.policy_ref == "POL-003"


def test_rules_engine_allows_settlement_above_the_stepup_threshold_when_satisfied():
    decision = _rules_engine(max_amount_cents=1_000_000).decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(
            amount_cents=_STEPUP_THRESHOLD + 1,
            merchant_id="MERCH-02",
            category="electronics",
            step_up_satisfied=True,
        ),
    )

    assert isinstance(decision, Allow)


def test_rules_engine_does_not_require_step_up_to_declare_an_intent_mandate_ceiling():
    """An intent authorizes future spend; it does not itself move money (mandates/models.py)."""
    decision = _rules_engine(max_amount_cents=1_000_000).decide(
        PolicyAction.CREATE_INTENT_MANDATE,
        _context(
            amount_cents=_STEPUP_THRESHOLD + 1,
            intent_mandate_allowed_categories=("electronics",),
            step_up_satisfied=False,
        ),
    )

    assert isinstance(decision, Allow)


def test_rules_engine_does_not_require_step_up_at_or_below_the_threshold():
    decision = _rules_engine(max_amount_cents=1_000_000).decide(
        PolicyAction.EXECUTE_SETTLEMENT,
        _context(
            amount_cents=_STEPUP_THRESHOLD,
            merchant_id="MERCH-02",
            category="electronics",
            step_up_satisfied=False,
        ),
    )

    assert isinstance(decision, Allow)


def _read_env_example_int(name: str) -> int:
    env_example = Path(__file__).resolve().parent.parent / ".env.example"
    for line in env_example.read_text(encoding="utf-8").splitlines():
        if line.startswith(f"{name}="):
            return int(line.split("=", 1)[1].strip())
    raise AssertionError(f"{name} not found in .env.example")


def test_env_example_keeps_step_up_reachable():
    """Regression guard for scenarios.yaml achado #1: `decide()` checks the amount ceiling
    (POL-001) before the step-up threshold (POL-003), so any documented default where
    `POLICY_STEPUP_THRESHOLD_CENTS >= POLICY_MAX_AMOUNT_CENTS` makes POL-003 dead code —
    every amount large enough to require step-up is already denied by AMOUNT_EXCEEDS_LIMIT
    first. This does not exercise `RulesPolicyEngine` itself (that's already covered above);
    it only pins the *documented default config* so this specific misconfiguration can't
    silently come back.
    """
    max_amount = _read_env_example_int("POLICY_MAX_AMOUNT_CENTS")
    stepup_threshold = _read_env_example_int("POLICY_STEPUP_THRESHOLD_CENTS")

    assert stepup_threshold < max_amount
