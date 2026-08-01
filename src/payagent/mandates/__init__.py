"""Signed-credential records for the `usuário → agente → merchant` delegation chain.

An Intent Mandate declares scope; a Payment Mandate authorizes one amount against one quote and
references its parent intent (POL-009 §3). No record here has a field able to hold cryptographic
material — that is P3 enforced by the type system, so a response cannot leak signing material.

Only `MandateSummary` may be returned to a caller. The authority that will sign and verify
(JWS/Ed25519/did:key) lands in a later block; `UnsignedMandateAuthority` verifies nothing, so
no code path in `src/` can settle (invariant I3, upheld by refusal).
"""

from payagent.mandates.authority import (
    Ed25519MandateAuthority,
    MandateAuthority,
    MandateVerification,
    UnsignedMandateAuthority,
)
from payagent.mandates.did_key import did_key_from_ed25519_public_key
from payagent.mandates.models import (
    IntentMandate,
    MandateSummary,
    PaymentMandate,
    summarize,
)
from payagent.mandates.store import (
    InMemoryMandateSignatureStore,
    InMemoryMandateStore,
    MandateSignatureStore,
    MandateStore,
)

__all__ = [
    "Ed25519MandateAuthority",
    "InMemoryMandateSignatureStore",
    "InMemoryMandateStore",
    "IntentMandate",
    "MandateAuthority",
    "MandateSignatureStore",
    "MandateStore",
    "MandateSummary",
    "MandateVerification",
    "PaymentMandate",
    "UnsignedMandateAuthority",
    "did_key_from_ed25519_public_key",
    "summarize",
]
