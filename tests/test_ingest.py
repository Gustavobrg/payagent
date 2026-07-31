"""Tests for RAG ingest pipeline (catalog and policies)."""

from __future__ import annotations

from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from payagent.rag.ingest import (
    CATALOG_COLLECTION,
    DeterministicEmbedder,
    chunk_catalog,
    chunk_policies,
)


def test_chunk_catalog():
    """Test catalog chunking: each SKU is one chunk."""
    items = [
        {
            "sku_id": "SKU-0001",
            "name": "Widget A",
            "description": "A simple widget.",
            "price_cents": 1000,
            "currency": "BRL",
            "category": "electronics",
            "merchant_id": "MERCH-01",
            "stock": 10,
        },
        {
            "sku_id": "SKU-0002",
            "name": "Widget B",
            "description": "Another widget.",
            "price_cents": 2000,
            "currency": "BRL",
            "category": "electronics",
            "merchant_id": "MERCH-02",
            "stock": 0,
        },
    ]

    chunks = chunk_catalog(items)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "SKU-0001"
    assert chunks[0]["source"] == "catalog"
    assert chunks[0]["text"] == "Widget A. A simple widget."
    assert chunks[0]["payload"]["sku"] == "SKU-0001"
    assert chunks[0]["payload"]["price_cents"] == 1000
    assert chunks[0]["payload"]["in_stock"] is True
    assert "poisoned" not in chunks[0]["payload"]  # Must not leak.

    assert chunks[1]["payload"]["in_stock"] is False


def test_chunk_policies(tmp_path: Path):
    """Test policy chunking: split by ## sections."""
    policy_text = """---
id: POL-001
title: "Test Policy"
category: test
---

# Header

Some intro text.

## Section One

Body of section one.

## Section Two

Body of section two.
"""

    docs = [("POL-001", policy_text)]
    chunks = chunk_policies(docs)

    assert len(chunks) == 2
    assert chunks[0]["chunk_id"] == "POL-001_section_1"
    assert chunks[0]["source"] == "policies"
    assert chunks[0]["payload"]["doc"] == "POL-001"
    assert chunks[0]["payload"]["section"] == "Section One"
    assert "Section One" in chunks[0]["text"]
    assert "Body of section one" in chunks[0]["text"]

    assert chunks[1]["payload"]["section"] == "Section Two"


def test_embedding_deterministic():
    """Test deterministic embedder produces consistent vectors."""
    from payagent.rag.ingest import EMBEDDING_DIM

    embedder = DeterministicEmbedder()

    texts = ["hello world", "foo bar", "hello world"]
    vecs = embedder.embed_batch(texts)

    assert len(vecs) == 3
    assert all(len(v) == EMBEDDING_DIM for v in vecs)
    # Same input should produce same output.
    assert vecs[0] == vecs[2]
    # Different input should (likely) produce different output.
    assert vecs[0] != vecs[1]


def test_idempotent_uuid5():
    """Test that uuid5 produces deterministic IDs."""
    chunk_id_1 = "SKU-0001"
    chunk_id_2 = "SKU-0001"
    chunk_id_3 = "SKU-0002"

    id1 = int(uuid5(NAMESPACE_URL, chunk_id_1).int % (2**63 - 1))
    id2 = int(uuid5(NAMESPACE_URL, chunk_id_2).int % (2**63 - 1))
    id3 = int(uuid5(NAMESPACE_URL, chunk_id_3).int % (2**63 - 1))

    assert id1 == id2  # Same chunk_id => same point ID.
    assert id1 != id3  # Different chunk_id => different point ID.


def test_ingest_idempotence_in_memory():
    """Test that re-ingesting does not duplicate points (using in-memory Qdrant)."""
    from qdrant_client import QdrantClient

    from payagent.rag.ingest import create_collection_if_missing, ingest_chunks

    # Use in-memory Qdrant client (no external server needed).
    client = QdrantClient(":memory:")
    embedder = DeterministicEmbedder()

    # Create collection.
    create_collection_if_missing(client, CATALOG_COLLECTION, recreate=False)

    # First ingest: 2 chunks.
    chunks_1 = [
        {
            "chunk_id": "SKU-0001",
            "source": "catalog",
            "text": "Widget A",
            "payload": {
                "chunk_id": "SKU-0001",
                "source": "catalog",
                "sku": "SKU-0001",
                "name": "Widget A",
                "category": "electronics",
                "price_cents": 1000,
                "currency": "BRL",
                "merchant_id": "MERCH-01",
                "in_stock": True,
            },
        },
        {
            "chunk_id": "SKU-0002",
            "source": "catalog",
            "text": "Widget B",
            "payload": {
                "chunk_id": "SKU-0002",
                "source": "catalog",
                "sku": "SKU-0002",
                "name": "Widget B",
                "category": "electronics",
                "price_cents": 2000,
                "currency": "BRL",
                "merchant_id": "MERCH-02",
                "in_stock": True,
            },
        },
    ]

    ingest_chunks(client, CATALOG_COLLECTION, chunks_1, embedder)
    count_after_first = client.count(CATALOG_COLLECTION).count
    assert count_after_first == 2

    # Second ingest: same chunks (same chunk_ids => same UUIDs => upsert, no duplication).
    ingest_chunks(client, CATALOG_COLLECTION, chunks_1, embedder)
    count_after_second = client.count(CATALOG_COLLECTION).count
    assert count_after_second == 2

    # Third ingest: add a new chunk.
    chunks_2_new = [
        {
            "chunk_id": "SKU-0003",
            "source": "catalog",
            "text": "Widget C",
            "payload": {
                "chunk_id": "SKU-0003",
                "source": "catalog",
                "sku": "SKU-0003",
                "name": "Widget C",
                "category": "electronics",
                "price_cents": 3000,
                "currency": "BRL",
                "merchant_id": "MERCH-03",
                "in_stock": True,
            },
        },
    ]

    ingest_chunks(client, CATALOG_COLLECTION, chunks_2_new, embedder)
    count_after_third = client.count(CATALOG_COLLECTION).count
    assert count_after_third == 3
