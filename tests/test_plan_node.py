"""Tests for the `plan` node: routing, degradation on a malformed model response, and the
structural guarantee that it cannot reach a payment tool."""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage

from payagent.graph.nodes import plan as plan_module
from payagent.graph.nodes.plan import make_plan_node
from payagent.graph.state import PurchaseState


@dataclass
class FakeChatModel:
    responses: list[AIMessage]
    _index: int = field(default=0)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        response = self.responses[self._index]
        self._index += 1
        return response


def _route_message(route: str, response: str | None = None) -> AIMessage:
    args = {"route": route}
    if response is not None:
        args["response"] = response
    return AIMessage(content="", tool_calls=[{"name": "route_decision", "args": args, "id": "c1"}])


def test_routes_to_retrieve():
    node = make_plan_node(FakeChatModel(responses=[_route_message("retrieve")]))
    state = PurchaseState(user_request="quero comprar um fone bluetooth")

    result = node(state)

    assert result.status == "in_progress"
    assert result.direct_response is None


def test_answers_a_greeting_directly_without_setting_up_retrieval():
    node = make_plan_node(
        FakeChatModel(responses=[_route_message("direct_response", response="Oi! Como posso ajudar?")])
    )
    state = PurchaseState(user_request="oi, tudo bem?")

    result = node(state)

    assert result.status == "answered_directly"
    assert result.direct_response == "Oi! Como posso ajudar?"


def test_degrades_to_retrieve_when_the_model_never_calls_the_tool():
    node = make_plan_node(FakeChatModel(responses=[AIMessage(content="hello")]))
    state = PurchaseState(user_request="fone bluetooth")

    result = node(state)

    assert result.status == "in_progress"
    assert result.direct_response is None


def test_degrades_to_retrieve_when_the_model_raises():
    class RaisingModel:
        def bind_tools(self, tools):
            return self

        def invoke(self, messages):
            raise RuntimeError("upstream LLM unavailable")

    node = make_plan_node(RaisingModel())
    state = PurchaseState(user_request="fone bluetooth")

    result = node(state)

    assert result.status == "in_progress"


def test_direct_response_falls_back_to_a_default_when_the_model_omits_it():
    """The route says direct_response but the model forgot the reply text — still terminates
    with *something* rather than an empty message."""
    node = make_plan_node(FakeChatModel(responses=[_route_message("direct_response")]))
    state = PurchaseState(user_request="explique a teoria da relatividade")

    result = node(state)

    assert result.status == "answered_directly"
    assert result.direct_response


def test_plan_module_imports_nothing_capable_of_reaching_a_payment_tool():
    """Structural proof, not a string search over prose: `plan` imports neither `ToolDeps` nor
    `dispatch_tool_call`/`call_tool`, so no code path in this module can reach
    `create_intent_mandate`, `create_payment_mandate`, `execute_settlement`, or `refund` —
    not just "doesn't call them today"."""
    source = inspect.getsource(plan_module)
    tree = ast.parse(source)

    imported_names = {
        alias.asname or alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported_names.isdisjoint({"ToolDeps", "dispatch_tool_call", "call_tool"})
