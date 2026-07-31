#!/usr/bin/env bash
# Recria a arvore do projeto a partir deste kit. Rode na pasta que voce quer usar.
set -euo pipefail

command -v uv >/dev/null || { echo "uv nao encontrado: https://docs.astral.sh/uv/"; exit 1; }

for d in docs/adr guardrails/rails evals/datasets evals/harness evals/reports \
         src/payagent/graph src/payagent/policy src/payagent/mandates src/payagent/rag \
         src/payagent/mcp_server src/payagent/guardrails src/payagent/observability \
         .github/workflows scripts tests; do
  mkdir -p "$d"
done
find src -type d -exec touch {}/__init__.py \;
touch evals/harness/__init__.py

uv python pin 3.11
uv sync            # resolve e gera uv.lock a partir do pyproject.toml do kit

echo
echo "versao resolvida do nemoguardrails (confirme antes de escrever Colang):"
uv run python -c "import importlib.metadata as m; print(m.version('nemoguardrails'))"
echo
echo "estrutura pronta. proximo passo: PROMPTS.md, bloco 1."