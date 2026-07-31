"""The six tools as data: name, models, description, annotations.

One table so the tool list, the dispatch map, and the eval harness cannot drift apart — adding a
tool means adding a row, not editing four places.

This is the layer that authors `inputSchema` by hand (via `strict_input_schema`) rather than
letting the SDK derive it. The SDK's generator never emits `additionalProperties: false` and its
argument model silently drops unknown keys, so deriving the schema would mean advertising a
contract the server does not enforce.

`ToolAnnotations` are hints, but they are what a client surfaces in a confirmation prompt, so they
are set honestly rather than conveniently: `get_quote` is **not** read-only because it records a
quote, and `execute_settlement` is the only `destructive_hint=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mcp_types as types
from pydantic import BaseModel

from payagent.mcp_server import descriptions
from payagent.mcp_server.schemas import (
    CreateIntentMandateRequest,
    CreateIntentMandateResult,
    CreatePaymentMandateRequest,
    CreatePaymentMandateResult,
    ExecuteSettlementRequest,
    ExecuteSettlementResult,
    GetQuoteRequest,
    GetQuoteResult,
    RefundRequest,
    RefundResult,
    SearchCatalogRequest,
    SearchCatalogResult,
    ToolEnvelope,
    strict_input_schema,
)
from payagent.policy import PolicyAction

SEARCH_CATALOG = "search_catalog"
GET_QUOTE = "get_quote"
CREATE_INTENT_MANDATE = "create_intent_mandate"
CREATE_PAYMENT_MANDATE = "create_payment_mandate"
EXECUTE_SETTLEMENT = "execute_settlement"
REFUND = "refund"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Everything the server needs to advertise and validate one tool.

    `policy_action` is `None` exactly for the read-only tools. That is not a convenience flag: it
    is how `dispatch.py` decides whether a call needs an idempotency reservation and a policy
    evaluation at all, so the read-only/side-effect split has one definition.
    """

    name: str
    title: str
    description: str
    request_model: type[BaseModel]
    result_model: type[BaseModel]
    policy_action: PolicyAction | None
    annotations: types.ToolAnnotations

    @property
    def has_side_effect(self) -> bool:
        return self.policy_action is not None

    def input_schema(self) -> dict[str, Any]:
        return strict_input_schema(self.request_model)

    def output_schema(self) -> dict[str, Any]:
        return strict_input_schema(ToolEnvelope[self.result_model])  # type: ignore[name-defined]

    def to_mcp_tool(self) -> types.Tool:
        return types.Tool(
            name=self.name,
            title=self.title,
            description=self.description,
            input_schema=self.input_schema(),
            output_schema=self.output_schema(),
            annotations=self.annotations,
        )


TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name=SEARCH_CATALOG,
        title="Search product catalog",
        description=descriptions.SEARCH_CATALOG_DESCRIPTION,
        request_model=SearchCatalogRequest,
        result_model=SearchCatalogResult,
        policy_action=None,
        annotations=types.ToolAnnotations(
            title="Search product catalog",
            read_only_hint=True,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    ToolSpec(
        name=GET_QUOTE,
        title="Quote a purchase",
        description=descriptions.GET_QUOTE_DESCRIPTION,
        request_model=GetQuoteRequest,
        result_model=GetQuoteResult,
        policy_action=None,
        annotations=types.ToolAnnotations(
            title="Quote a purchase",
            # Not read-only: it records a quote. Annotated honestly rather than conveniently.
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    ToolSpec(
        name=CREATE_INTENT_MANDATE,
        title="Create intent mandate",
        description=descriptions.CREATE_INTENT_MANDATE_DESCRIPTION,
        request_model=CreateIntentMandateRequest,
        result_model=CreateIntentMandateResult,
        policy_action=PolicyAction.CREATE_INTENT_MANDATE,
        annotations=types.ToolAnnotations(
            title="Create intent mandate",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    ToolSpec(
        name=CREATE_PAYMENT_MANDATE,
        title="Create payment mandate",
        description=descriptions.CREATE_PAYMENT_MANDATE_DESCRIPTION,
        request_model=CreatePaymentMandateRequest,
        result_model=CreatePaymentMandateResult,
        policy_action=PolicyAction.CREATE_PAYMENT_MANDATE,
        annotations=types.ToolAnnotations(
            title="Create payment mandate",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=False,
        ),
    ),
    ToolSpec(
        name=EXECUTE_SETTLEMENT,
        title="Settle a payment (irreversible)",
        description=descriptions.EXECUTE_SETTLEMENT_DESCRIPTION,
        request_model=ExecuteSettlementRequest,
        result_model=ExecuteSettlementResult,
        policy_action=PolicyAction.EXECUTE_SETTLEMENT,
        annotations=types.ToolAnnotations(
            title="Settle a payment (irreversible)",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,
            open_world_hint=True,
        ),
    ),
    ToolSpec(
        name=REFUND,
        title="Refund a settlement",
        description=descriptions.REFUND_DESCRIPTION,
        request_model=RefundRequest,
        result_model=RefundResult,
        policy_action=PolicyAction.REFUND,
        annotations=types.ToolAnnotations(
            title="Refund a settlement",
            read_only_hint=False,
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,
        ),
    ),
)

TOOL_SPECS_BY_NAME: dict[str, ToolSpec] = {spec.name: spec for spec in TOOL_SPECS}


def list_tools_result() -> types.ListToolsResult:
    """The full tool list. Pure — no deps, no pagination, safe to call per request."""
    return types.ListToolsResult(tools=[spec.to_mcp_tool() for spec in TOOL_SPECS])
