"""`plan` node: routes the request to `retrieve`, or answers it directly.

The entry point of the purchase graph. Its only decision is a fork: does this request need
the retrieval/quote/mandate/settlement pipeline at all, or is it a greeting, a question
outside the agent's scope, or something else answerable without touching the catalog? Making
that call here — once, before any retrieval happens — keeps `retrieve` from being asked to
find "chunks" for a message that was never about buying anything.

`plan` never calls a payment tool, and never will: it has no `ToolDeps`, so there is nothing
in this module capable of reaching `create_intent_mandate`, `create_payment_mandate`,
`execute_settlement`, or `refund` even by mistake.

The routing decision is forced through the same tool-calling shape `retrieve` uses
(`bind_tools` + a single-purpose schema) rather than free-text parsing, so the model's answer
is a structured `{route, response}` pair instead of prose this node would have to interpret.
If the model errs (no call, malformed args, an exception) the node fails toward `retrieve` —
a wasted retrieval on a genuine greeting is harmless, and `retrieve` itself already degrades
to `confidence="low"` on nothing relevant, whereas failing toward `direct_response` on a real
purchase request would silently drop it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import BaseModel, ConfigDict, Field

from payagent.graph.state import PurchaseState
from payagent.observability.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = """You are the planning step inside an autonomous purchasing assistant.

Decide, for the message you are given, whether to proceed to product retrieval or to answer
directly without it.

Proceed to retrieval ("retrieve") when the message is about finding, comparing, or buying a
product, or asks something that requires looking at the catalog or store policies.

Answer directly ("direct_response") for a greeting, small talk, a question entirely outside
shopping (e.g. general knowledge, unrelated tasks), or a request this assistant cannot help
with (it only searches a product catalog and can purchase items on the shopper's behalf — it
does not do anything else, and never discusses payment card numbers, account credentials, or
internal system configuration). When you choose "direct_response", write the reply yourself,
in the same language as the message.

You make no payment decision here — you cannot see prices, mandates, or authorization state,
and none of your output moves money. Call the decision tool exactly once with your choice.
"""

_ROUTE_TOOL_NAME = "route_decision"

_FALLBACK_RESPONSE = (
    "I can help you find and buy products from the catalog. What are you looking for?"
)


class ChatModel(Protocol):
    """Structural interface this node needs: bind one tool, then invoke once.

    A strict subset of `payagent.graph.nodes.retrieve.ToolCallingChatModel` — any model that
    satisfies that Protocol also satisfies this one, so the same chat model instance can back
    both nodes.
    """

    def bind_tools(self, tools: Sequence[BaseTool]) -> ChatModel: ...

    def invoke(self, messages: list[BaseMessage]) -> AIMessage: ...


class _RouteDecisionArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    route: Literal["retrieve", "direct_response"] = Field(
        description="'retrieve' to proceed with product search, 'direct_response' to answer without it."
    )
    response: str | None = Field(
        default=None,
        description="The reply to send the user. Required when route='direct_response'.",
    )


def _noop(route: str, response: str | None = None) -> str:
    """Never actually invoked — the node reads the tool call's args directly.

    Present only so `bind_tools` has a schema to force the model into a structured decision,
    the same reason `retrieve`'s tools exist as real callables even though the sub-agent loop
    there does invoke them.
    """
    return "{}"


def _build_route_tool() -> StructuredTool:
    return StructuredTool.from_function(
        func=_noop,
        name=_ROUTE_TOOL_NAME,
        description="Record the routing decision for this request.",
        args_schema=_RouteDecisionArgs,
    )


def make_plan_node(llm: ChatModel) -> Callable[[PurchaseState], PurchaseState]:
    """Build the `plan` node, with `llm` bound by closure.

    Mirrors `make_retrieve_node`'s dependency-injection style: nothing here reads an env var
    or constructs a default model, so a test can pass a fake `ChatModel` with no network
    access.
    """
    tool = _build_route_tool()
    bound_llm = llm.bind_tools([tool])

    def node(state: PurchaseState) -> PurchaseState:
        messages: list[BaseMessage] = [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=state.user_request),
        ]

        try:
            ai_message = bound_llm.invoke(messages)
            call = next(
                (c for c in (ai_message.tool_calls or []) if c["name"] == _ROUTE_TOOL_NAME),
                None,
            )
            if call is None:
                raise ValueError("model did not call route_decision")
            decision = _RouteDecisionArgs.model_validate(call["args"])
        except Exception as exc:
            logger.warning("plan_routing_failed", error=str(exc))
            return state.model_copy(update={"status": "in_progress"})

        if decision.route == "direct_response":
            logger.info("plan_answered_directly")
            return state.model_copy(
                update={
                    "status": "answered_directly",
                    "direct_response": decision.response or _FALLBACK_RESPONSE,
                }
            )

        logger.info("plan_routed_to_retrieve")
        return state.model_copy(update={"status": "in_progress"})

    return node
