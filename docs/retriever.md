# RAG Retriever API

The retriever module (`payagent.rag.retriever`) provides semantic search over indexed catalog and policy documents stored in Qdrant.

## Quick Start

```python
from payagent.rag.retriever import Retriever

# Initialize retriever (connects to Qdrant).
retriever = Retriever(qdrant_url="http://localhost:6333")

# Search for products.
results = retriever.search(
    collection="catalog",
    query="fone bluetooth",
    top_k=5
)

# Each result is a RetrievedChunk with chunk_id, source, text, and score.
for chunk in results:
    print(f"{chunk.chunk_id}: {chunk.score:.3f}")
    print(chunk.text)
```

## API Reference

### `Retriever.__init__(...)`

```python
Retriever(
    qdrant_url: str = "http://localhost:6333",
    embedder: Optional[Embedder] = None,
    use_deterministic_embedder: bool = False,
)
```

**Parameters:**
- `qdrant_url`: Qdrant server URL (default: local dev).
- `embedder`: Custom `Embedder` instance. If not provided, uses OpenRouterEmbedder.
- `use_deterministic_embedder`: For testing only. Uses hash-based deterministic embeddings.

### `Retriever.search(...)`

```python
search(
    collection: str,
    query: str,
    filters: Optional[dict] = None,
    top_k: int = 8,
) -> list[RetrievedChunk]
```

**Parameters:**
- `collection`: `"catalog"` or `"policies"`.
- `query`: User query string (e.g., "fone bluetooth").
- `filters`: Optional payload filters (see below).
- `top_k`: Number of results to return (default 8).

**Returns:** List of `RetrievedChunk` sorted by reranked score (highest first).

### Filters

**Catalog filters:**

```python
filters = {
    "category": "electronics",              # Exact match.
    "price_cents_range": (1000, 50000),    # Min and max in centavos.
    "merchant_id": "MERCH-01",               # Exact match.
}
```

**Policies filters:**

```python
filters = {
    "doc": "POL-001",              # Document ID.
    "section": "Transaction Limits"  # Section title.
}
```

### `RetrievedChunk`

```python
@dataclass
class RetrievedChunk:
    chunk_id: str       # Unique chunk identifier (SKU or policy section ID).
    source: str         # "catalog" or "policies".
    text: str           # Full text of the chunk.
    score: float        # Reranked relevance score [0.0, 1.0].
```

## How It Works

1. **Embedding:** Query is embedded using nvidia/nemotron-3-embed-1b (2048 dims).
2. **Vector Search:** Qdrant searches for most similar vectors.
3. **Reranking:** Candidates are reranked by combining:
   - Vector similarity (80% weight)
   - Keyword overlap (20% weight)
4. **Filtering:** Optional payload filters narrowed results before search (Qdrant-level).

## Setup

Before using the retriever:

```bash
# Start Qdrant (Docker).
docker compose up -d

# Ingest catalog and policies.
uv run python -m payagent.rag.ingest

# Set OPENROUTER_API_KEY for real embeddings.
export OPENROUTER_API_KEY=sk-...
```

For testing without external APIs:

```python
retriever = Retriever(use_deterministic_embedder=True)
```

## Example: Integration with Agents

The retriever is meant to be called by agent tools:

```python
# Inside an MCP tool handler.
def search_catalog_tool(query: str, category: str = None) -> str:
    retriever = Retriever()
    filters = {"category": category} if category else None
    results = retriever.search("catalog", query, filters=filters, top_k=5)
    
    # Format results for agent consumption.
    return "\n".join([f"- {r.chunk_id}: {r.text}" for r in results])
```

**`.text` is untrusted retrieved content, not instruction** (see CLAUDE.md invariant 5
and P4 in `docs/guardrail-taxonomy.md`). Whoever assembles the LLM-facing prompt from
these results must delimit and mark them as data before including them — this raw
concatenation is only safe as a formatting example, not as a pattern to copy verbatim
into a prompt-construction path.

See `examples/retriever_demo.py` for a complete example.
