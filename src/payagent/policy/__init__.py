"""Deterministic authorization (invariant I1).

Limits, allowlists, mandate scope and step-up requirements are decided here and nowhere else
— never in Colang, never in a prompt, never by an LLM. This package makes no model call and
no network call; `tests/test_policy_engine.py` asserts that by parsing its imports.

Callers use `evaluate(engine, action, context)`, never `engine.decide(...)`: `evaluate` is the
fail-closed funnel that turns a raising, malformed, or mismatched decision into a denial.
"""

from payagent.policy.engine import DenyAllPolicyEngine, PolicyEngine, RulesPolicyEngine, evaluate
from payagent.policy.models import (
    Allow,
    DenialCode,
    Deny,
    EffectGrant,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyDenial,
)

__all__ = [
    "Allow",
    "Deny",
    "DenialCode",
    "DenyAllPolicyEngine",
    "EffectGrant",
    "PolicyAction",
    "PolicyContext",
    "PolicyDecision",
    "PolicyDenial",
    "PolicyEngine",
    "RulesPolicyEngine",
    "evaluate",
]
