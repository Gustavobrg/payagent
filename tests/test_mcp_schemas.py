"""Tests on the advertised tool contract itself — no fixtures, no Qdrant, no server.

These assert properties of the six schemas and descriptions rather than of any call. The most
load-bearing one is `test_no_advertised_schema_contains_an_open_object`: it is what catches a
`dict`-typed field sneaking back in, since a pydantic `dict` renders as an open
`{"type": "object"}` that `extra="forbid"` does not reach inside — a hole in the middle of a
schema that otherwise looks strict.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from payagent.mcp_server import descriptions
from payagent.mcp_server.errors import _MESSAGES, ToolErrorCode
from payagent.mcp_server.registry import TOOL_SPECS, ToolSpec, list_tools_result
from payagent.mcp_server.schemas import (
    CreateIntentMandateRequest,
    SearchCatalogRequest,
    ToolEnvelope,
    strict_input_schema,
)
from payagent.rag.ingest import load_catalog

SIDE_EFFECT_TOOLS = {
    "create_intent_mandate",
    "create_payment_mandate",
    "execute_settlement",
    "refund",
}
READ_ONLY_TOOLS = {"search_catalog", "get_quote"}

# The only monetary request fields allowed to exist. None of them can set an amount to be moved:
# `expected_*` are assertions against a server-side record, `max_amount_cents` is a mandate
# ceiling, and `min/max_price_cents` are search filters.
ALLOWED_MONEY_REQUEST_FIELDS = {
    "expected_amount_cents",
    "max_amount_cents",
    "min_price_cents",
    "max_price_cents",
}

ALL_SPECS = pytest.mark.parametrize("spec", TOOL_SPECS, ids=lambda s: s.name)


def _walk(node: Any):
    """Yield every dict node in a JSON Schema, depth-first."""
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def _properties_of(schema: dict) -> dict[str, dict]:
    """Root properties, resolving the `anyOf: [T, null]` shape optional fields render as."""
    resolved = {}
    for name, prop in schema.get("properties", {}).items():
        if "anyOf" in prop:
            non_null = [b for b in prop["anyOf"] if b.get("type") != "null"]
            resolved[name] = non_null[0] if non_null else prop
        else:
            resolved[name] = prop
    return resolved


def test_exactly_the_six_architecture_tools_are_advertised():
    assert {spec.name for spec in TOOL_SPECS} == SIDE_EFFECT_TOOLS | READ_ONLY_TOOLS
    assert len(TOOL_SPECS) == 6


@ALL_SPECS
def test_input_schema_root_is_a_closed_object(spec: ToolSpec):
    schema = spec.input_schema()

    assert schema["type"] == "object"
    assert schema["additionalProperties"] is False


@ALL_SPECS
def test_no_advertised_schema_contains_an_open_object(spec: ToolSpec):
    """Every object node, including $defs / items / anyOf branches, must forbid extras.

    This is the test that fails if someone reintroduces a `dict`-typed field.
    """
    for schema in (spec.input_schema(), spec.output_schema()):
        for node in _walk(schema):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False, node


@ALL_SPECS
def test_idempotency_key_is_required_exactly_on_the_side_effect_tools(spec: ToolSpec):
    required = set(spec.input_schema().get("required", []))

    if spec.name in SIDE_EFFECT_TOOLS:
        assert "idempotency_key" in required
    else:
        assert "idempotency_key" not in spec.input_schema()["properties"]


@ALL_SPECS
def test_side_effect_tools_map_to_a_policy_action(spec: ToolSpec):
    """The read-only/side-effect split has one definition, and dispatch reads it from here."""
    assert spec.has_side_effect is (spec.name in SIDE_EFFECT_TOOLS)


@ALL_SPECS
def test_every_monetary_field_is_advertised_as_a_bounded_integer(spec: ToolSpec):
    for name, prop in _properties_of(spec.input_schema()).items():
        if not name.endswith("_cents"):
            continue
        assert prop["type"] == "integer", name
        assert "minimum" in prop, name
        assert "maximum" in prop, name


@ALL_SPECS
def test_no_request_schema_accepts_a_settleable_amount(spec: ToolSpec):
    """The P2 property, surface-wide: no field can set an amount to be moved.

    A new caller-chosen amount field cannot be added without this failing.
    """
    for name in spec.input_schema()["properties"]:
        if name.endswith("_cents"):
            assert name in ALLOWED_MONEY_REQUEST_FIELDS, name


def test_refund_has_no_amount_parameter():
    """Full-refund-only: a caller-supplied amount was the one unbacked monetary input, so it's gone."""
    properties = TOOL_SPECS[-1].input_schema()["properties"]

    assert TOOL_SPECS[-1].name == "refund"
    assert "amount_cents" not in properties
    assert "expected_amount_cents" in properties


@ALL_SPECS
def test_currency_fields_advertise_the_iso_4217_pattern(spec: ToolSpec):
    for name, prop in _properties_of(spec.input_schema()).items():
        if "currency" not in name:
            continue
        assert prop["pattern"] == "^[A-Z]{3}$", name
        assert prop["maxLength"] == 3, name


@ALL_SPECS
def test_monetary_and_idempotency_fields_carry_a_description(spec: ToolSpec):
    """A field a model must use correctly and cannot see the semantics of is a selection failure."""
    for name, prop in spec.input_schema()["properties"].items():
        if name.endswith("_cents") or name == "idempotency_key":
            assert prop.get("description"), name


def test_every_category_in_the_real_catalog_is_an_accepted_value():
    """Regression: the category pattern must accept the dataset it filters over.

    14 of the 20 categories in `evals/datasets/catalog.json` contain spaces ("pet shop", "toys and
    games"). A slug-only pattern rejected them, which made `create_intent_mandate` unusable for most
    of the catalog and would have broken POL-004 (restricted categories), since that rule keys off
    this exact value.
    """
    categories = sorted({item["category"] for item in load_catalog()})

    assert len(categories) > 1
    for category in categories:
        SearchCatalogRequest(query="x", category=category)
        CreateIntentMandateRequest(
            idempotency_key="a" * 16,
            purpose="demo",
            max_amount_cents=1,
            currency="BRL",
            allowed_categories=[category],
        )


@pytest.mark.parametrize(
    "junk", ["", " leading space", "UPPERCASE", "has\nnewline", "x" * 41, "-leading-dash"]
)
def test_the_category_pattern_still_rejects_junk(junk: str):
    """Widening for spaces must not turn the field into a free-text hole."""
    with pytest.raises(ValidationError):
        SearchCatalogRequest(query="x", category=junk)


@ALL_SPECS
def test_output_schema_is_the_result_envelope(spec: ToolSpec):
    """Advertising the bare result would make a conformant client reject every policy denial."""
    assert spec.output_schema() == strict_input_schema(ToolEnvelope[spec.result_model])


@ALL_SPECS
def test_tool_survives_conversion_to_the_mcp_type(spec: ToolSpec):
    tool = spec.to_mcp_tool()

    assert tool.name == spec.name
    assert tool.description == spec.description
    assert tool.input_schema["additionalProperties"] is False
    assert tool.annotations is not None


def test_list_tools_result_serializes_input_schema_under_the_wire_alias():
    """`input_schema` must reach the client as `inputSchema`, still closed."""
    payload = list_tools_result().model_dump(by_alias=True)

    assert len(payload["tools"]) == 6
    for tool in payload["tools"]:
        assert tool["inputSchema"]["additionalProperties"] is False


@ALL_SPECS
def test_description_follows_the_five_slot_scaffold(spec: ToolSpec):
    """Consistent positional structure is what the tool_selection_accuracy eval leans on."""
    for slot in ("Effect:", "Use it when:", "Do not use it to:"):
        assert slot in spec.description, f"{spec.name} is missing {slot!r}"


@ALL_SPECS
def test_side_effect_descriptions_state_the_idempotency_and_policy_contract(spec: ToolSpec):
    if not spec.has_side_effect:
        return

    assert "idempotency_key" in spec.description
    assert "policy" in spec.description.lower()


def test_settlement_description_marks_itself_irreversible():
    spec = next(s for s in TOOL_SPECS if s.name == "execute_settlement")

    assert "IRREVERSIBLE" in spec.description
    assert "no undo" in spec.description


@pytest.mark.parametrize(
    "tool_name", ["get_quote", "create_payment_mandate", "execute_settlement"]
)
def test_amount_provenance_is_stated_wherever_an_amount_is_involved(tool_name: str):
    """P2: every tool near an amount must say the amount comes from the quote."""
    spec = next(s for s in TOOL_SPECS if s.name == tool_name)

    assert "quote" in spec.description.lower()


def test_search_catalog_description_marks_retrieved_text_as_data():
    """P4: the external interface must tell its consumer that product prose is not instructions."""
    spec = next(s for s in TOOL_SPECS if s.name == "search_catalog")

    assert "text_untrusted" in spec.description
    assert "not instructions" in spec.description or "as data" in spec.description


def test_server_instructions_state_the_sequence_and_the_three_rules():
    text = descriptions.SERVER_INSTRUCTIONS

    assert "search_catalog -> get_quote" in text
    assert "execute_settlement" in text
    assert "final" in text  # a denial is final (P5)


@pytest.mark.parametrize("code", list(ToolErrorCode))
def test_every_error_code_has_a_static_message(code: ToolErrorCode):
    """An interpolated message is all it takes to reopen the I2 leak, so none may have a placeholder."""
    message = _MESSAGES[code]

    assert message
    assert "{" not in message
    assert "%s" not in message


def test_annotations_mark_only_settlement_destructive():
    by_name = {spec.name: spec.annotations for spec in TOOL_SPECS}

    assert by_name["execute_settlement"].destructive_hint is True
    for name in set(by_name) - {"execute_settlement"}:
        assert by_name[name].destructive_hint is False


def test_get_quote_is_not_annotated_read_only_because_it_records_a_quote():
    by_name = {spec.name: spec.annotations for spec in TOOL_SPECS}

    assert by_name["search_catalog"].read_only_hint is True
    assert by_name["get_quote"].read_only_hint is False
