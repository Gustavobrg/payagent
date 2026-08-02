"""Retrieve and rerank chunks from Qdrant based on semantic similarity and relevance.

This module provides the search() function that retrieves relevant chunks from
Qdrant collections (catalog, policies) given a user query. It handles embedding,
vector search, optional payload filtering, and reranking.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, Range

from payagent.observability.logging import get_logger
from payagent.rag.ingest import Embedder, build_embedder, connect_qdrant

logger = get_logger(__name__)

DEFAULT_RERANK_MODEL = "cohere/rerank-4-fast"

# achado #5 (scenarios.yaml header): `search` used to always return up to `top_k` candidates
# regardless of relevance, so a query for something genuinely absent from the catalog still
# came back with `top_k` unrelated real products instead of signaling "nothing matches."
# 0.0 is a deliberately conservative floor: it excludes only candidates the reranker measured
# as having literally zero relevance (a `DeterministicReranker` keyword-overlap score of
# exactly 0.0 for tests and the eval harness's `use_deterministic_embedder` path), never a
# candidate with any positive signal. `OpenRouterReranker` (the live Cohere cross-encoder) is
# expected to almost never return an exact 0.0 for a real query, so this floor is a safety
# net for the unambiguous case, not a calibrated relevance cutoff — raising it for production
# needs live-model calibration (same caveat as `guardrails/provider.py`'s Llama Guard note:
# not done blind, would need a probe script run against the real reranker first).
DEFAULT_MIN_RELEVANCE_SCORE = 0.0

# Exact-match filter keys per collection: filter dict key -> Qdrant payload field name.
_EXACT_MATCH_FIELDS = {
    "catalog": {"category": "category", "merchant_id": "merchant_id"},
    "policies": {"doc": "doc", "section": "section"},
}
# Range filter keys per collection: filter dict key -> Qdrant payload field name.
_RANGE_FIELDS = {
    "catalog": {"price_cents_range": "price_cents"},
}


@dataclass
class RetrievedChunk:
    """A chunk retrieved and ranked from Qdrant."""

    chunk_id: str
    source: str  # "catalog" or "policies"
    text: str
    score: float  # Reranked relevance score from the cross-encoder (higher = more relevant).


class Reranker(ABC):
    """Abstract interface for scoring (query, document) pairs."""

    @abstractmethod
    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Score each document's relevance to query. Returns scores aligned to input order."""


class OpenRouterReranker(Reranker):
    """Cross-encoder reranking via OpenRouter's /rerank endpoint."""

    def __init__(self, model: str = DEFAULT_RERANK_MODEL):
        self.model = model

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        """Rerank documents using OpenRouter's rerank API, with retries."""
        if not documents:
            return []

        import httpx

        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")

        client = httpx.Client(base_url="https://openrouter.ai/api/v1")

        attempts = 0
        while True:
            attempts += 1
            try:
                response = client.post(
                    "/rerank",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json={
                        "model": self.model,
                        "query": query,
                        "documents": documents,
                        "top_n": len(documents),
                    },
                    timeout=30,
                )
                response.raise_for_status()
                data = response.json()
                break
            except Exception as exc:
                if attempts >= 3:
                    logger.error("rerank_failed", documents=len(documents), error=str(exc))
                    raise
                logger.warning(
                    "rerank_retry", documents=len(documents), attempt=attempts, error=str(exc)
                )

        scores = [0.0] * len(documents)
        for item in data["results"]:
            scores[item["index"]] = item["relevance_score"]
        return scores


class DeterministicReranker(Reranker):
    """Test reranker: keyword-overlap heuristic (no network calls).

    Tokenizes on `\\w+` rather than plain `.split()` so trailing punctuation ("Speaker."
    vs. query word "speaker") doesn't produce a false-zero score — a real cross-encoder
    isn't punctuation-sensitive either, and `Retriever.search`'s `min_score` floor
    (achado #5) depends on a 0.0 score actually meaning "no relevance," not "word boundary
    quirk," or every single-word query would spuriously get zero results.
    """

    _WORD_RE = re.compile(r"\w+")

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        query_words = set(self._WORD_RE.findall(query.lower()))
        if not query_words:
            return [0.0] * len(documents)

        scores = []
        for doc in documents:
            doc_words = set(self._WORD_RE.findall(doc.lower()))
            overlap = len(query_words & doc_words)
            scores.append(overlap / len(query_words))
        return scores


class Retriever:
    """Retriever with embedding, vector search, filtering, and reranking."""

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        client: QdrantClient | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        use_deterministic_embedder: bool = False,
    ):
        """Initialize retriever with a Qdrant client, embedder, and reranker.

        Args:
            qdrant_url: Qdrant server URL (ignored if `client` is provided).
            client: Pre-built QdrantClient (e.g. an in-memory client for tests).
                    If None, connects to `qdrant_url`.
            embedder: Custom embedder (Embedder interface).
                      If None, uses OpenRouterEmbedder by default.
            reranker: Custom reranker (Reranker interface).
                      If None, uses OpenRouterReranker by default.
            use_deterministic_embedder: For testing, use hash-based embedder and
                      keyword-overlap reranker (no network calls).
        """
        self.client = client or connect_qdrant(qdrant_url)
        self.embedder = embedder or build_embedder(use_deterministic=use_deterministic_embedder)
        self.reranker = reranker or (
            DeterministicReranker() if use_deterministic_embedder else OpenRouterReranker()
        )

    def search(
        self,
        collection: str,
        query: str,
        filters: dict | None = None,
        top_k: int = 8,
        min_score: float = DEFAULT_MIN_RELEVANCE_SCORE,
    ) -> list[RetrievedChunk]:
        """Search collection by query with optional filtering and reranking.

        Args:
            collection: Collection name ("catalog" or "policies").
            query: User query string.
            filters: Optional payload filters:
                     - For catalog: {"category": "electronics", "price_cents_range": (1000, 50000),
                                   "merchant_id": "MERCH-01"}
                     - For policies: {"doc": "POL-001", "section": "Limits"}
            top_k: Number of results to return (default 8).
            min_score: Reranked-score floor (exclusive); candidates at or below it are
                       dropped rather than padding the result out to `top_k`. See
                       `DEFAULT_MIN_RELEVANCE_SCORE`'s module-level comment for why the
                       default is conservative.

        Returns:
            List of RetrievedChunk sorted by reranked score (descending), score > min_score.

        Raises:
            ValueError: If `filters` contains a key not recognized for `collection`.
        """
        query_embedding = self.embedder.embed_batch([query])[0]
        qdrant_filter = self._build_filter(collection, filters)

        # Retrieve 2x top_k candidates so reranking has room to reorder before truncating.
        search_result = self.client.query_points(
            collection_name=collection,
            query=query_embedding,
            query_filter=qdrant_filter,
            limit=top_k * 2,
        )

        candidates = []
        for scored_point in search_result.points:
            payload = scored_point.payload
            if not payload:
                continue
            candidates.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "source": payload.get("source"),
                    "text": payload.get("text", ""),
                }
            )

        reranked = self._rerank(query, candidates)
        relevant = [item for item in reranked if item["reranked_score"] > min_score]

        result = [
            RetrievedChunk(
                chunk_id=item["chunk_id"],
                source=item["source"],
                text=item["text"],
                score=item["reranked_score"],
            )
            for item in relevant[:top_k]
        ]

        logger.info(
            "search_complete",
            collection=collection,
            query_length=len(query),
            candidates=len(candidates),
            below_min_score=len(reranked) - len(relevant),
            returned=len(result),
        )

        return result

    def _build_filter(self, collection: str, filters: dict | None) -> Filter | None:
        """Build a Qdrant filter from a filters dict.

        Raises ValueError if `filters` has a key not recognized for `collection`,
        so a typo or a cross-collection key never silently widens the search scope.
        """
        if not filters:
            return None

        exact_fields = _EXACT_MATCH_FIELDS.get(collection, {})
        range_fields = _RANGE_FIELDS.get(collection, {})

        unknown_keys = set(filters) - set(exact_fields) - set(range_fields)
        if unknown_keys:
            raise ValueError(f"Unknown filter key(s) for collection '{collection}': {unknown_keys}")

        conditions = [
            FieldCondition(key=field_name, match=MatchValue(value=filters[filter_key]))
            for filter_key, field_name in exact_fields.items()
            if filter_key in filters
        ]
        for filter_key, field_name in range_fields.items():
            if filter_key in filters:
                min_value, max_value = filters[filter_key]
                conditions.append(FieldCondition(key=field_name, range=Range(gte=min_value, lte=max_value)))

        return Filter(must=conditions) if conditions else None

    def _rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Rerank candidates using the configured cross-encoder reranker."""
        if not candidates:
            return candidates

        texts = [c["text"] for c in candidates]
        scores = self.reranker.rerank(query, texts)

        for candidate, score in zip(candidates, scores, strict=True):
            candidate["reranked_score"] = score

        return sorted(candidates, key=lambda x: x["reranked_score"], reverse=True)
