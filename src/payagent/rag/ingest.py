"""Ingest catalog and policies into Qdrant for retrieval-augmented generation.

Two collections:
  - catalog: Each SKU from evals/datasets/catalog.json is one chunk (atomic).
  - policies: Each policy document (evals/datasets/policies/*.md) is chunked by ## sections,
             with the document title prefixed to each chunk.

Idempotency: point IDs are uuid5(NAMESPACE_URL, chunk_id), so re-ingestion never duplicates.

Embeddings via OpenRouter (nvidia/nemotron-3-embed-1b:free), batched (size=64) with retry.

Usage:
  uv run python -m payagent.rag.ingest
  uv run python -m payagent.rag.ingest --recreate
"""

from __future__ import annotations

import argparse
import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from payagent.observability.logging import get_logger

logger = get_logger(__name__)

# Dataset paths relative to project root.
CATALOG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "evals" / "datasets" / "catalog.json"
POLICIES_DIR = Path(__file__).resolve().parent.parent.parent.parent / "evals" / "datasets" / "policies"

# Qdrant collection names and vector size.
CATALOG_COLLECTION = "catalog"
POLICIES_COLLECTION = "policies"
EMBEDDING_DIM = 2048  # nvidia/nemotron-3-embed-1b output dimension


def catalog_point_id(chunk_id: str) -> int:
    """Deterministic Qdrant point ID for a chunk_id.

    Defined here, next to the ingestion that assigns it, because every exact-lookup path
    (`rag.tools.get_sku_details`, `mcp_server.catalog.lookup_sku`) has to reproduce the same
    scheme to find a point without embedding a query. One definition means one edit if the
    scheme ever changes.
    """
    return int(uuid5(NAMESPACE_URL, chunk_id).int % (2**63 - 1))


class Embedder(ABC):
    """Abstract interface for embedding text."""

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of embedding vectors."""


class OpenRouterEmbedder(Embedder):
    """Embeddings via OpenRouter (nvidia/nemotron-3-embed-1b:free)."""

    def __init__(self, model: str = "nvidia/nemotron-3-embed-1b:free", batch_size: int = 64):
        self.model = model
        self.batch_size = batch_size

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using OpenRouter embedding API, with retries."""
        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        client = httpx.Client(base_url="https://openrouter.ai/api/v1")
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info("embedding_batch", batch_size=len(batch), offset=i)

            attempts = 0
            while attempts < 3:
                try:
                    response = client.post(
                        "/embeddings",
                        headers={"Authorization": f"Bearer {api_key}"},
                        json={"model": self.model, "input": batch},
                        timeout=60,
                    )
                    response.raise_for_status()
                    data = response.json()

                    embeddings = sorted(data["data"], key=lambda x: x["index"])
                    batch_vecs = [e["embedding"] for e in embeddings]

                    if len(batch_vecs) != len(batch):
                        raise RuntimeError(
                            f"Expected {len(batch)} embeddings, got {len(batch_vecs)}"
                        )

                    all_embeddings.extend(batch_vecs)
                    break
                except Exception as exc:
                    attempts += 1
                    if attempts >= 3:
                        logger.error("embedding_failed", batch_size=len(batch), error=str(exc))
                        raise
                    logger.warning(
                        "embedding_retry", batch_size=len(batch), attempt=attempts, error=str(exc)
                    )

        return all_embeddings


class DeterministicEmbedder(Embedder):
    """Test embedder: hash-based deterministic embeddings (all zeros except a hash-based value)."""

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings = []
        for text in texts:
            h = hash(text) % 100
            vec = [float(h / 100.0)] + [0.0] * (EMBEDDING_DIM - 1)
            embeddings.append(vec)
        return embeddings


def connect_qdrant(url: str = "http://localhost:6333", timeout: int = 60) -> QdrantClient:
    """Connect to Qdrant, falling back to a compatibility-check-disabled client on version mismatch."""
    try:
        return QdrantClient(url=url, timeout=timeout)
    except Exception as exc:
        if "incompatible" in str(exc).lower():
            logger.warning(
                "qdrant_version_mismatch",
                hint="Update Qdrant server to match client version, or downgrade qdrant-client.",
            )
            return QdrantClient(url=url, timeout=timeout, check_compatibility=False)
        raise


def build_embedder(use_deterministic: bool = False, model: str | None = None) -> Embedder:
    """Build an Embedder: deterministic for tests, OpenRouter otherwise."""
    if use_deterministic:
        logger.info("using_deterministic_embedder")
        return DeterministicEmbedder()

    embedding_model = model or os.environ.get("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free")
    logger.info("using_openrouter_embedder", model=embedding_model)
    return OpenRouterEmbedder(model=embedding_model)


def load_catalog() -> list[dict]:
    """Load catalog from JSON. Each entry is a complete SKU."""
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found: {CATALOG_PATH}")
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    logger.info("catalog_loaded", count=len(raw))
    return raw


def load_policies() -> list[tuple[str, str]]:
    """Load policy documents. Returns list of (doc_id, full_text)."""
    if not POLICIES_DIR.exists():
        raise FileNotFoundError(f"Policies directory not found: {POLICIES_DIR}")

    docs = []
    for md_file in sorted(POLICIES_DIR.glob("*.md")):
        doc_id = md_file.stem
        text = md_file.read_text(encoding="utf-8")
        docs.append((doc_id, text))

    logger.info("policies_loaded", count=len(docs))
    return docs


def chunk_catalog(items: list[dict]) -> list[dict]:
    """Convert catalog items to chunks. Each SKU is one chunk.

    Chunk structure: {
        "chunk_id": str,
        "source": "catalog",
        "text": str (name + description for embedding),
        "payload": {...}  (indexable metadata)
    }
    """
    chunks = []
    for item in items:
        chunk_id = item["sku_id"]
        # Text to embed: name + description.
        text = f"{item['name']}. {item['description']}"
        # Payload: excludes poisoned flag (never leaks to retriever).
        payload = {
            "chunk_id": chunk_id,
            "source": "catalog",
            "text": text,
            "sku": chunk_id,
            "name": item["name"],
            "category": item["category"],
            "price_cents": item["price_cents"],  # int for range filtering
            "currency": item["currency"],
            "merchant_id": item["merchant_id"],
            "in_stock": item["stock"] > 0,
        }
        chunks.append(
            {
                "chunk_id": chunk_id,
                "source": "catalog",
                "text": text,
                "payload": payload,
            }
        )
    return chunks


def chunk_policies(docs: list[tuple[str, str]]) -> list[dict]:
    """Convert policy documents to chunks by ## section.

    Chunk structure: {
        "chunk_id": str,
        "source": "policies",
        "text": str (section body prefixed with doc title),
        "payload": {...}
    }
    """
    chunks = []
    section_counter = {}

    for doc_id, full_text in docs:
        # Parse frontmatter and body.
        lines = full_text.split("\n")
        body_start = 0
        if lines[0] == "---":
            for i in range(1, len(lines)):
                if lines[i] == "---":
                    body_start = i + 1
                    break

        body = "\n".join(lines[body_start:])

        # Split by ## sections.
        # Pattern: ^## (section title)
        pattern = r"^## (.+)$"
        sections = re.split(pattern, body, flags=re.MULTILINE)

        # sections[0] is preamble (before first ##), then alternates title, content, title, content, ...
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                section_title = sections[i].strip()
                section_body = sections[i + 1].strip()

                if not section_body:
                    continue

                # Generate unique chunk_id.
                if doc_id not in section_counter:
                    section_counter[doc_id] = 0
                section_counter[doc_id] += 1
                chunk_id = f"{doc_id}_section_{section_counter[doc_id]}"

                # Text to embed: doc title + section title + body.
                text = f"{doc_id}: {section_title}\n\n{section_body}"

                payload = {
                    "chunk_id": chunk_id,
                    "source": "policies",
                    "text": text,
                    "doc": doc_id,
                    "section": section_title,
                }

                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": "policies",
                        "text": text,
                        "payload": payload,
                    }
                )

    return chunks


def create_collection_if_missing(
    client: QdrantClient, collection_name: str, recreate: bool = False
) -> None:
    """Create collection with vector params and payload indices."""
    try:
        client.get_collection(collection_name)
        if recreate:
            logger.info("recreating_collection", name=collection_name)
            client.delete_collection(collection_name)
        else:
            logger.info("collection_exists", name=collection_name)
            return
    except Exception:
        pass

    logger.info("creating_collection", name=collection_name)
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Create indices on payload fields.
    if collection_name == CATALOG_COLLECTION:
        client.create_payload_index(
            collection_name,
            field_name="category",
            field_schema="keyword",
        )
        client.create_payload_index(
            collection_name,
            field_name="price_cents",
            field_schema="integer",
        )
        client.create_payload_index(
            collection_name,
            field_name="merchant_id",
            field_schema="keyword",
        )
    elif collection_name == POLICIES_COLLECTION:
        client.create_payload_index(
            collection_name,
            field_name="doc",
            field_schema="keyword",
        )
        client.create_payload_index(
            collection_name,
            field_name="section",
            field_schema="keyword",
        )

    logger.info("collection_created", name=collection_name)


def ingest_chunks(
    client: QdrantClient, collection_name: str, chunks: list[dict], embedder: Embedder
) -> None:
    """Ingest chunks into Qdrant collection. Idempotent via uuid5."""
    if not chunks:
        logger.warning("no_chunks_to_ingest", collection=collection_name)
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedder.embed_batch(texts)

    if len(embeddings) != len(chunks):
        raise RuntimeError(f"Embedding mismatch: {len(embeddings)} vs {len(chunks)}")

    points = []
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        # Idempotent ID: uuid5(NAMESPACE_URL, chunk_id).
        point_id = catalog_point_id(chunk["chunk_id"])

        point = PointStruct(
            id=point_id,
            vector=embedding,
            payload=chunk["payload"],
        )
        points.append(point)

    # Upsert: if ID exists, update; otherwise insert.
    logger.info(
        "upserting_points",
        collection=collection_name,
        count=len(points),
    )
    client.upsert(collection_name=collection_name, points=points)

    final_count = client.count(collection_name).count
    logger.info("ingest_complete", collection=collection_name, total_points=final_count)


def ingest_catalog(
    client: QdrantClient, embedder: Embedder, recreate: bool = False
) -> None:
    """Load, chunk, and ingest catalog."""
    create_collection_if_missing(client, CATALOG_COLLECTION, recreate=recreate)
    items = load_catalog()
    chunks = chunk_catalog(items)
    ingest_chunks(client, CATALOG_COLLECTION, chunks, embedder)


def ingest_policies(
    client: QdrantClient, embedder: Embedder, recreate: bool = False
) -> None:
    """Load, chunk, and ingest policies."""
    create_collection_if_missing(client, POLICIES_COLLECTION, recreate=recreate)
    docs = load_policies()
    chunks = chunk_policies(docs)
    ingest_chunks(client, POLICIES_COLLECTION, chunks, embedder)


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qdrant-url",
        default=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        help="Qdrant server URL",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate collections",
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("EMBEDDING_MODEL", "nvidia/nemotron-3-embed-1b:free"),
        help="OpenRouter embedding model",
    )
    parser.add_argument(
        "--use-deterministic-embedder",
        action="store_true",
        help="Use deterministic embedder (for testing)",
    )

    args = parser.parse_args()

    embedder = build_embedder(use_deterministic=args.use_deterministic_embedder, model=args.embedding_model)
    client = connect_qdrant(args.qdrant_url)
    logger.info("qdrant_connected", url=args.qdrant_url)

    # Ingest both collections.
    try:
        ingest_catalog(client, embedder, recreate=args.recreate)
        ingest_policies(client, embedder, recreate=args.recreate)
        logger.info("ingest_success")
    except Exception as exc:
        logger.error("ingest_failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
