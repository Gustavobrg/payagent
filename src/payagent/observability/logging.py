"""Logging estruturado em JSON — único ponto de entrada para logging do projeto.

Nenhum código de negócio deve usar `logging`/`print` diretamente: importe `get_logger`
daqui. Todo event_dict passa por `dlp.mask_value` antes do sink (invariante I2 do
CLAUDE.md) — hoje `dlp.mask` é um passthrough (stub), mas o wrapper já está no lugar
certo para quando o Bloco 7 implementar a redação de verdade.
"""

from __future__ import annotations

import sys
from typing import Any

import structlog

from payagent.guardrails import dlp

_configured = False


def _dlp_redact_processor(
    logger: object, method_name: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    return {key: dlp.mask_value(value) for key, value in event_dict.items()}


def configure_logging() -> None:
    """Configura o structlog uma única vez. Idempotente."""
    global _configured
    if _configured:
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
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )
    _configured = True


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """Devolve um logger estruturado, configurando o structlog se necessário."""
    configure_logging()
    return structlog.get_logger(name)
