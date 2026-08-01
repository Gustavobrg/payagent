"""Example: RAG retriever in action.

This demonstrates how to use the Retriever to search for products in the catalog.
Before running, ensure:
  1. Qdrant is running (docker compose up -d)
  2. Catalog has been ingested (uv run python -m payagent.rag.ingest)

Usage:
  uv run python examples/retriever_demo.py
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from payagent.rag.retriever import Retriever

load_dotenv()


def main():
    """Run retriever demo."""
    # Initialize retriever connected to Qdrant.
    qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    retriever = Retriever(qdrant_url=qdrant_url)

    print("=" * 70)
    print("RAG Retriever Demo: Search Catalog for Products")
    print("=" * 70)

    # Example 1: Search for "fone bluetooth" (Portuguese for "bluetooth headphones").
    print("\n[Example 1] Searching for: 'fone bluetooth'")
    print("-" * 70)

    query = "fone bluetooth"
    results = retriever.search("catalog", query, top_k=5)

    if results:
        print(f"Found {len(results)} result(s):\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk.source.upper()} - SKU {chunk.chunk_id}")
            print(f"   Score: {chunk.score:.3f}")
            print(f"   Text: {chunk.text[:100]}...")
            print()
    else:
        print("No results found.")

    # Example 2: Search with filters (category + price range).
    print("[Example 2] Searching for: 'monitor' in electronics, price under BRL 200")
    print("-" * 70)

    query = "monitor display"
    filters = {
        "category": "electronics",
        "price_cents_range": (0, 20000),  # Up to BRL 200.
    }
    results = retriever.search("catalog", query, filters=filters, top_k=5)

    if results:
        print(f"Found {len(results)} result(s):\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk.chunk_id}: {chunk.text[:80]}...")
            print(f"   Score: {chunk.score:.3f}\n")
    else:
        print("No results found.")

    # Example 3: Search policies collection.
    print("[Example 3] Searching policies for: 'transaction limit'")
    print("-" * 70)

    query = "transaction limit"
    results = retriever.search("policies", query, top_k=3)

    if results:
        print(f"Found {len(results)} result(s):\n")
        for i, chunk in enumerate(results, 1):
            print(f"{i}. {chunk.chunk_id}")
            print(f"   Score: {chunk.score:.3f}")
            print(f"   Text: {chunk.text[:120]}...")
            print()
    else:
        print("No results found.")

    print("=" * 70)
    print("Demo complete!")


if __name__ == "__main__":
    main()
