"""Everything the tool handlers need, injected rather than constructed.

Same discipline as `Retriever.__init__` and `make_retrieve_node`: nothing in this package reads
`os.environ` or builds a network client except `__main__.py`. That is what lets the whole contract
suite run against an in-memory Qdrant, a fake clock and an in-process merchant with no HTTP, no
sleeping, and no API key.

`clock` and `id_factory` are injected for the same reason expiry is tested by advancing a clock
rather than sleeping: a handler that read `datetime.now()` directly could not be tested for quote
expiry without waiting 30 minutes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from payagent.mandates import MandateAuthority, MandateStore
from payagent.mcp_server.idempotency import IdempotencyStore
from payagent.mcp_server.merchant_sim import MerchantSim
from payagent.mcp_server.quotes import DEFAULT_QUOTE_TTL_SECONDS, QuoteStore
from payagent.mcp_server.step_up import StepUpVerifier
from payagent.policy import PolicyEngine
from payagent.rag.retriever import Retriever


@dataclass(frozen=True, slots=True)
class ToolDeps:
    """Injected collaborators for the six MCP tools.

    Args:
        retriever: Vector store access. Used via `mcp_server.catalog`, never via `rag.tools` —
            the external tool surface stays separate from the retrieval sub-agent's.
        policy: Authorization engine. Reached only through `payagent.policy.evaluate`.
        mandate_authority: Issues mandates and verifies them before settlement.
        mandate_store: Where issued mandates are resolved by ID.
        quotes: Where issued quotes are resolved by ID — the only source of a payable amount.
        merchant: Settlement target.
        idempotency: Reserve/complete/release records keyed by `(tool, idempotency_key)`.
        clock: Returns "now". Injected so expiry is deterministic.
        id_factory: `prefix -> id`. Production passes a uuid4-based factory: mandate and
            settlement IDs are capability-like handles, and sequential IDs are enumerable.
        step_up: Server-side WebAuthn challenge/response state (POL-003). The only source of
            `PolicyContext.step_up_satisfied` — never a tool argument (P5).
        quote_ttl_seconds: Quote validity window. Defaults to POL-010's 30 minutes.
    """

    retriever: Retriever
    policy: PolicyEngine
    mandate_authority: MandateAuthority
    mandate_store: MandateStore
    quotes: QuoteStore
    merchant: MerchantSim
    idempotency: IdempotencyStore
    clock: Callable[[], datetime]
    id_factory: Callable[[str], str]
    step_up: StepUpVerifier
    quote_ttl_seconds: int = DEFAULT_QUOTE_TTL_SECONDS
