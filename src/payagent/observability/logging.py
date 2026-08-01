"""Logging estruturado em JSON — único ponto de entrada para logging do projeto.

Nenhum código de negócio deve usar `logging`/`print` diretamente: importe `get_logger`
daqui. Todo event_dict passa por `dlp.mask_value` antes do sink (invariante I2 do
CLAUDE.md) — ver `guardrails/dlp.py` para a detecção real (regex + Luhn para PAN/CVV,
Presidio para PII).
"""

from __future__ import annotations

import sys
from typing import Any, TextIO

import structlog

from payagent.guardrails import dlp

_configured = False


def _dlp_redact_processor(
    logger: object, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return {key: dlp.mask_value(value) for key, value in event_dict.items()}


def configure_logging(stream: TextIO | None = None) -> None:
    """Configura o structlog. Idempotente quando chamado sem argumento.

    Sem `stream`, configura uma única vez com `sys.stdout` — é o que todo `get_logger` faz.

    Com `stream` explícito, **reconfigura mesmo se já estiver configurado**. Isso não é
    conveniência: o servidor MCP precisa de `sys.stderr` porque no transporte stdio o stdout
    pertence ao JSON-RPC, e uma linha de log ali corrompe a conexão. Como todo módulo chama
    `get_logger` no nível do módulo, o simples `import` de `payagent.rag.retriever` já configuraria
    com stdout antes de `main()` rodar — então "chame antes de qualquer logger" é impossível de
    cumprir e a trava tem que ceder para um stream explícito.

    Não dependa do `stdio_server` do SDK redirecionar fd 1 para fd 2: ele só faz isso depois de
    entrar no contexto, e a construção das dependências (que loga) acontece antes.
    """
    global _configured
    if _configured and stream is None:
        return

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            _dlp_redact_processor,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(stream or sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Devolve um logger estruturado, configurando o structlog se necessário."""
    configure_logging()
    return structlog.get_logger(name)
