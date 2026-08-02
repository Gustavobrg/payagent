# PayAgent Guard

> Autonomous purchasing agent with agentic RAG, MCP tools, signed mandates, and layered
> guardrails (NeMo Guardrails + Llama Guard) — reference implementation.

<!-- Numbers from evals/reports/20260801T235606Z_full (150 scenarios, guardrails: full). -->
<!-- Rerun `uv run python -m evals.harness.run` and update this table when it moves. -->

| Metric | Result |
|---|---|
| False settlements (150 scenarios) | 0% (0/86) |
| Injection blocked — via user (P1/P2/P3/P5) | 93.2% (41/44) |
| Injection blocked — via retrieved content (P4, live) | 6.2% (1/16) 🔴 open investigation, see [Known issues](#known-issues) |
| False positive on legitimate purchases | 0.0% (0/60) |
| PANs in logs | 0 |
| Avg tokens per transaction | ~8,932 (no $/token pricing wired up — see [Known issues](#known-issues)) |
| Latency added by guardrails (p95) | 1,349 ms in / 1,153 ms out (isolated from graph latency) |

## The problem

An autonomous purchasing agent combines three dangerous things on the same surface: it reads
untrusted content (product descriptions, policy text, anything retrieved) that can carry
embedded instructions; it makes authorization decisions that are normally a human's call
(limits, merchant allowlist, strong confirmation above a threshold); and it moves real money.
An LLM-based guardrail is a probabilistic classifier — fine for blocking offensive content,
not acceptable as the sole control for "can I spend R$5,000 at this merchant?". This project's
bet is to split the two: the guardrail decides what's *reasonable*, deterministic code decides
what's *permitted*, and settlement never happens without going through the latter.

## Architecture

An explicit state machine, not free-form ReAct — every node has a fixed input/output
contract, and an out-of-order transition raises before any tool call is attempted:

```
plan → retrieve → quote → mandate → confirm → settle
```

- **plan** interprets the request and decides whether to proceed to `retrieve` or answer
  directly (greeting, out-of-scope question) — never calls a payment tool.
- **retrieve** is a bounded tool-calling sub-agent over four read-only tools
  (`search_catalog`, `search_policies`, `get_sku_details`, `compare_skus`).
- **quote** is the only node that writes a value into state (`quote_id`,
  `quote_amount_cents`), always from `get_quote` — never from the user's text.
- **mandate** issues an Intent Mandate and a Payment Mandate (JWS Ed25519, `did:key`)
  through the real policy engine; any `Deny` is terminal, no workaround attempted.
- **confirm** is the step-up gate: it re-evaluates `PolicyContext.step_up_satisfied` right
  before settlement, against the `StepUpVerifier` — it never accepts confirmation coming
  from the user's text.
- **settle** calls `execute_settlement` exactly once; a policy or merchant denial is
  terminal, with no automatic retry.

Full diagram, the six-MCP-tool table, and the stack: [`Architecture.md`](Architecture.md).
P1–P5 risk categories and which layer covers each one:
[`docs/guardrail-taxonomy.md`](docs/guardrail-taxonomy.md).

## Design decisions
See `docs/adr/`. The three central ones:
- an explicit state machine instead of a free-roaming ReAct agent
- deterministic authorization **outside** the guardrail
- a risk taxonomy specific to this domain, not Llama Guard's default one

## What's actually here

| Layer | Status |
|---|---|
| `policy/` — per-transaction limit, 24h aggregate, merchant allowlist, restricted category, step-up | ✅ real rules (`RulesPolicyEngine`), fail-closed (`DenyAllPolicyEngine`) when unconfigured |
| `mandates/` — Intent/Payment Mandate and signing | ✅ Ed25519 JWS + `did:key` (`Ed25519MandateAuthority`), `UnsignedMandateAuthority` fallback |
| `mcp_server/` — all six tools, strict schemas, idempotency, simulated step-up | ✅ |
| `graph/` — `plan → retrieve → quote → mandate → confirm → settle` | ✅ every node has real logic |
| `rag/` — retriever, reranking, search sub-agent | ✅ |
| Demos (`scripts/demo.py` CLI, `scripts/demo_gradio.py` web) | ✅ |
| `guardrails/dlp.py` (PAN/CVV/expiry + generic PII redaction in logs) | ✅ regex+Luhn for card data, Presidio for generic PII |
| NeMo rails (`guardrails/*.co`) | ✅ wired into a running rails-only `LLMRails` (`build_rails()`); dialog rails deliberately empty, LangGraph drives the conversation |
| Llama Guard / P1–P5 taxonomy classifier | ✅ live via OpenRouter (`OpenRouterLlamaGuard`, `meta-llama/llama-guard-4-12b`) — safe/unsafe verdict works, but the model returns its own S1–S14 codes, not the custom P1–P5 taxonomy (labeling gap, not a blocking one) |
| `evals/harness/` | ✅ 150-scenario dataset, full harness (`evals/harness/run.py`), reports under `evals/reports/` |
| OpenTelemetry / Langfuse | ⏳ dependency installed, no instrumentation yet |
| `docs/threat-model.md`, `docs/deployment.md` | ⏳ placeholders (block 11) |

## Running it

```bash
uv sync
cp .env.example .env   # fill in at least OPENROUTER_API_KEY
docker compose up -d   # local Qdrant (Langfuse uses the cloud version, see .env.example)

# populate the catalog and policies in Qdrant
uv run python -m payagent.rag.ingest

# test suite
uv run pytest -q

# one purchase end to end, from the CLI
uv run python scripts/demo.py "bluetooth headphones under R$150"

# same thing, with a web UI
uv run python scripts/demo_gradio.py

# standalone MCP server (stdio)
uv run python -m payagent.mcp_server
```


## Known issues

| Issue | Status |
|---|---|
| `injection_block_rate`, retrieved content (P4, live): 6.2% (1/16) — vs. 100% (15/15) on an offline heuristic re-scan of the same poisoned SKUs, identical across two independent full runs | 🔴 open investigation — see [ADR-0006](docs/adr/0006-eval.md) |

Exact reproducibility of the 1/16 across two otherwise-different runs points toward a
consistent code-path issue (most likely the P4 scan not being invoked at the right point
in context assembly) rather than flaky retrieval ranking — next step is instrumenting a
`retrieved: bool` field per P4 scenario to separate "never retrieved" from "retrieved but
not blocked" before attempting a fix. This does not block CI: per ADR-0006 the merge gate
only trips on `false_settlement_rate > 0` or `pan_leak_count > 0`, both of which have
stayed at 0 throughout.

## Scope and non-scope
- No integration with a real acquirer or card network. Card numbers are test numbers only.
- IaC for AKS/APIM is written and validated with `terraform plan`; not deployed, for cost.
