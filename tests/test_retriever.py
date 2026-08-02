"""Tests for RAG retriever: vector search, filtering, and reranking."""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    create_collection_if_missing,
    ingest_chunks,
)
from payagent.rag.retriever import DeterministicReranker, RetrievedChunk, Retriever

CATALOG_ITEMS = [
    {
        "sku_id": "SKU-HEADPHONES",
        "name": "AuraSound Pro Wireless Headphones",
        "description": "Great audio quality with noise cancellation.",
        "price_cents": 12999,
        "currency": "BRL",
        "category": "electronics",
        "merchant_id": "MERCH-01",
        "stock": 45,
    },
    {
        "sku_id": "SKU-SPEAKER",
        "name": "Nexus Bluetooth Speaker",
        "description": "Portable and powerful sound.",
        "price_cents": 4999,
        "currency": "BRL",
        "category": "electronics",
        "merchant_id": "MERCH-02",
        "stock": 20,
    },
    {
        "sku_id": "SKU-MONITOR",
        "name": "Nebula 27 inch 4K Gaming Monitor",
        "description": "High refresh rate display.",
        "price_cents": 44999,
        "currency": "BRL",
        "category": "electronics",
        "merchant_id": "MERCH-03",
        "stock": 0,
    },
]


@pytest.fixture
def catalog_collection_with_data() -> tuple[QdrantClient, str]:
    """In-memory Qdrant catalog collection, ingested via the real ingest.py pipeline."""
    client = QdrantClient(":memory:")
    create_collection_if_missing(client, CATALOG_COLLECTION)

    chunks = chunk_catalog(CATALOG_ITEMS)
    ingest_chunks(client, CATALOG_COLLECTION, chunks, DeterministicEmbedder())

    return client, CATALOG_COLLECTION


@pytest.fixture
def retriever(catalog_collection_with_data: tuple[QdrantClient, str]) -> Retriever:
    """Retriever wired directly to the in-memory catalog client (no live Qdrant connection)."""
    client, _ = catalog_collection_with_data
    return Retriever(client=client, use_deterministic_embedder=True)


def test_search_bluetooth_speaker(retriever: Retriever):
    """Test search for 'fone bluetooth' returns relevant candidates with high scores."""
    results = retriever.search(CATALOG_COLLECTION, "fone bluetooth", top_k=8)

    assert len(results) > 0, "Search should return at least one result"
    assert all(isinstance(r, RetrievedChunk) for r in results)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"
    assert all(0 <= r.score <= 1 for r in results), "Scores should be normalized to [0, 1]"

    # Headphones and speaker both carry audio-related keywords; monitor doesn't.
    chunk_ids = [r.chunk_id for r in results]
    assert "SKU-HEADPHONES" in chunk_ids or "SKU-SPEAKER" in chunk_ids


def test_search_with_category_filter(retriever: Retriever):
    """Test search with category filter."""
    results = retriever.search(
        CATALOG_COLLECTION,
        "speaker",
        filters={"category": "electronics"},
        top_k=5,
    )

    assert len(results) > 0
    assert all(r.source == "catalog" for r in results)


def test_search_with_price_range_filter(retriever: Retriever):
    """Test search with price range filter (API acceptance test).

    Note: In-memory Qdrant doesn't enforce payload indexes, so this tests
    that the filter API works, not that filtering is correct. Real filtering
    requires a server Qdrant instance.
    """
    results = retriever.search(
        CATALOG_COLLECTION,
        "audio equipment",
        filters={"price_cents_range": (0, 50000)},
        top_k=5,
    )

    assert len(results) > 0


def test_build_filter_merchant_id_shape(retriever: Retriever):
    """Test that a merchant_id filter builds the expected Qdrant Filter (unit-level, no Qdrant enforcement needed)."""
    qdrant_filter = retriever._build_filter("catalog", {"merchant_id": "MERCH-01"})

    assert qdrant_filter == Filter(
        must=[FieldCondition(key="merchant_id", match=MatchValue(value="MERCH-01"))]
    )


def test_search_rejects_unknown_filter_key(retriever: Retriever):
    """Test that an unrecognized filter key raises instead of silently widening the search."""
    with pytest.raises(ValueError, match="Unknown filter key"):
        retriever.search(CATALOG_COLLECTION, "widget", filters={"not_a_real_field": "x"})


def test_reranking_score_ordering(retriever: Retriever):
    """Test that reranking produces properly ordered scores."""
    results = retriever.search(CATALOG_COLLECTION, "bluetooth", top_k=8)

    for i in range(len(results) - 1):
        assert results[i].score >= results[i + 1].score, (
            f"Scores should be descending: {results[i].score} >= {results[i + 1].score}"
        )


def test_deterministic_reranker_keyword_overlap():
    """Test that DeterministicReranker scores by keyword overlap, no network calls."""
    reranker = DeterministicReranker()

    scores = reranker.rerank(
        "bluetooth speaker",
        [
            "Bluetooth speaker for outdoors",
            "Wired mechanical keyboard",
            "Bluetooth headphones with noise cancellation",
        ],
    )

    assert len(scores) == 3
    assert scores[0] > scores[1]  # Full overlap beats no overlap.
    assert scores[0] > scores[2]  # Full overlap beats partial overlap.


# --- achado #5 (scenarios.yaml header): retrieval had no relevance floor at all, so a query
# for something genuinely absent from the catalog still returned top_k unrelated real
# products instead of signaling "nothing matches." -------------------------------------


def test_search_excludes_candidates_with_zero_measured_relevance(retriever: Retriever):
    """A query sharing no vocabulary at all with any catalog item must not pad results out
    to top_k with unrelated products — every DeterministicReranker score is exactly 0.0 for
    zero keyword overlap, and the default `min_score` floor excludes those.
    """
    results = retriever.search(CATALOG_COLLECTION, "purple dinosaur umbrella parade", top_k=8)

    assert results == []


def test_search_keeps_a_genuine_match_at_the_default_threshold(retriever: Retriever):
    """The default floor must not be so aggressive it rejects an actual keyword match."""
    results = retriever.search(CATALOG_COLLECTION, "bluetooth speaker", top_k=8)

    assert any(r.chunk_id == "SKU-SPEAKER" for r in results)


def test_search_honors_an_explicit_min_score_override(retriever: Retriever):
    """Callers can raise the floor above the default to demand stronger matches."""
    results = retriever.search(CATALOG_COLLECTION, "bluetooth", top_k=8, min_score=1.1)

    assert results == []


def test_retrieved_chunk_structure(retriever: Retriever):
    """Test that RetrievedChunk has all required fields."""
    results = retriever.search(CATALOG_COLLECTION, "AuraSound", top_k=3)

    assert len(results) > 0

    chunk = results[0]
    assert isinstance(chunk, RetrievedChunk)
    assert isinstance(chunk.chunk_id, str)
    assert isinstance(chunk.source, str)
    assert isinstance(chunk.text, str)
    assert isinstance(chunk.score, float)
    assert len(chunk.chunk_id) > 0
    assert len(chunk.text) > 0
    assert 0 <= chunk.score <= 1
