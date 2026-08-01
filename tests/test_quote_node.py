"""Tests for the `quote` node: it is the only node that writes a value into `PurchaseState`,
requires a confirmed selection from among what `retrieve` actually returned, and fails safely
(never raises) when `get_quote` itself refuses the request."""

from __future__ import annotations

import pytest

from payagent.graph.nodes.quote import make_quote_node
from payagent.graph.state import IllegalTransitionError, PurchaseState
from payagent.rag.tools import ToolChunk


def _chunk(chunk_id: str, source: str = "catalog") -> ToolChunk:
    return ToolChunk(chunk_id=chunk_id, source=source, score=0.9, text="<untrusted>...</untrusted>")


def test_writes_quote_id_and_amount_cents_from_the_real_quote(deps):
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[_chunk("SKU-SPEAKER")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=1,
    )

    result = node(state)

    assert result.status == "in_progress"
    assert result.quote_id is not None
    assert result.quote_amount_cents == 4999
    assert result.quote["data"]["quote_id"] == result.quote_id


def test_computes_amount_as_unit_price_times_quantity(deps):
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="3 fones bluetooth",
        retrieved_chunks=[_chunk("SKU-SPEAKER")],
        selected_sku="SKU-SPEAKER",
        selected_quantity=3,
    )

    result = node(state)

    assert result.quote_amount_cents == 4999 * 3


def test_raises_when_nothing_was_selected(deps):
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="fone bluetooth", retrieved_chunks=[_chunk("SKU-SPEAKER")]
    )

    with pytest.raises(IllegalTransitionError):
        node(state)


def test_raises_when_the_selection_was_never_retrieved(deps):
    """A `selected_sku` that `retrieve` never surfaced must not reach `get_quote` at all — it
    could only have gotten there by being invented, not confirmed."""
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[_chunk("SKU-OTHER")],
        selected_sku="SKU-NEVER-RETRIEVED",
    )

    with pytest.raises(IllegalTransitionError):
        node(state)


def test_is_a_no_op_when_the_purchase_already_resolved(deps):
    node = make_quote_node(deps)
    state = PurchaseState(user_request="oi", status="answered_directly", direct_response="Oi!")

    result = node(state)

    assert result == state
    assert result.quote_amount_cents is None


def test_unknown_sku_fails_the_quote_without_raising(deps):
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="fone bluetooth",
        retrieved_chunks=[_chunk("SKU-DOES-NOT-EXIST")],
        selected_sku="SKU-DOES-NOT-EXIST",
    )

    result = node(state)

    assert result.status == "quote_failed"
    assert result.quote["error"]["code"] == "UNKNOWN_SKU"
    assert result.quote_amount_cents is None


def test_out_of_stock_fails_the_quote_without_raising(deps):
    node = make_quote_node(deps)
    state = PurchaseState(
        user_request="noise machine",
        retrieved_chunks=[_chunk("SKU-OUTOFSTOCK")],
        selected_sku="SKU-OUTOFSTOCK",
    )

    result = node(state)

    assert result.status == "quote_failed"
    assert result.quote["error"]["code"] == "OUT_OF_STOCK"
