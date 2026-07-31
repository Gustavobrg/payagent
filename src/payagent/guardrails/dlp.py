"""Scan/mask de dado sensível (PAN, CVV, validade, PII).

Stub passthrough por enquanto — a implementação real (regex + Luhn para dados de
cartão, Presidio para PII) entra no Bloco 7 (ver PROMPTS.md). A interface abaixo já é
o contrato final: todo sink de log, span ou mensagem de erro deve passar por `mask`
antes de escrever (invariante I2 do CLAUDE.md), nunca o dado bruto.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DlpFinding:
    """Resultado de uma varredura DLP sobre um texto."""

    has_sensitive_data: bool
    categories: tuple[str, ...] = field(default_factory=tuple)


def scan(text: str) -> DlpFinding:
    """Detecta PAN/CVV/validade/PII em `text`. Stub: nunca encontra nada."""
    return DlpFinding(has_sensitive_data=False)


def mask(text: str) -> str:
    """Redige dado sensível em `text`. Stub: passthrough — devolve o texto intacto."""
    return text


def mask_value(value: object) -> object:
    """Aplica `mask` recursivamente a strings dentro de dict/list/tuple.

    Usado pelo wrapper de logging para redigir um event_dict inteiro antes do sink
    (invariante I2). Outros tipos passam direto.
    """
    if isinstance(value, str):
        return mask(value)
    if isinstance(value, dict):
        return {key: mask_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(mask_value(val) for val in value)
    return value
