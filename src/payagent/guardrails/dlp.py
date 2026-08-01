"""Scan/mask de dado sensível (PAN, CVV, validade, PII) — invariante I2 do CLAUDE.md.

Detecção de cartão (PAN/CVV/validade) é regex + validação Luhn, código nosso, sem chamada de
modelo — precisa rodar antes (e independente) de qualquer classificador (P1 em
docs/guardrail-taxonomy.md), e roda antes da PII genérica abaixo em `scan`/`mask`.
`CREDIT_CARD` fica de fora de `PII_ENTITIES` de propósito: Presidio tem seu próprio
reconhecedor de cartão, mas usá-lo faria a detecção de PAN depender indiretamente do
Presidio em vez de ficar 100% no nosso regex+Luhn — a mesma razão pela qual `policy/` não
faz chamada de LLM.

PII genérica (nome, email, telefone, endereço, IBAN, ...) usa o `AnalyzerEngine` do
Presidio com o modelo `en_core_web_sm` (dependência real do projeto — `uv add`, resolvido
no `uv.lock`, sem download em tempo de execução). Isso habilita tanto os reconhecedores
por padrão (email, telefone, IBAN) quanto os baseados em NER (PERSON, LOCATION).

`scan` decide bloqueio (input rail); `mask` reescreve e sinaliza se redigiu (output rail e
o wrapper de logging em `observability/logging.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import SpacyNlpEngine

PAN_CATEGORY = "PAN"
CVV_CATEGORY = "CVV"
EXPIRY_CATEGORY = "EXPIRY"

_SPACY_MODEL = "en_core_web_sm"

PII_ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "PERSON",
    "LOCATION",
    "IBAN_CODE",
    "CRYPTO",
)

# 13-19 digits (ISO/IEC 7812 PAN length range), optionally separated by spaces/hyphens in
# groups, e.g. "4111 1111 1111 1111" or "4111-1111-1111-1111". Luhn is what actually decides
# whether this is a card number — the regex alone would flag any long digit run.
_PAN_CANDIDATE_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,}\d(?!\d)")

_CVV_KEYWORD_RE = re.compile(
    r"(?i)\b(?:cvv2?|cvc2?|c[oó]digo\s+de\s+seguran[cç]a|security\s+code)\b"
    r"[^0-9]{0,20}(\d{3,4})\b"
)

_EXPIRY_RE = re.compile(r"(?<!\d)(0[1-9]|1[0-2])\s*/\s*(\d{2}|\d{4})(?!\d)")


def luhn_is_valid(digits: str) -> bool:
    """Standard Luhn checksum over a string of decimal digits."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


@dataclass(frozen=True)
class DlpMatch:
    """A single sensitive-data span found by `scan`."""

    category: str
    start: int
    end: int


@dataclass(frozen=True)
class DlpScanResult:
    """Result of a DLP scan over one piece of text."""

    has_sensitive_data: bool
    categories: tuple[str, ...] = field(default_factory=tuple)
    matches: tuple[DlpMatch, ...] = field(default_factory=tuple)


def _find_pan_matches(text: str) -> list[DlpMatch]:
    matches = []
    for m in _PAN_CANDIDATE_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_is_valid(digits):
            matches.append(DlpMatch(PAN_CATEGORY, m.start(), m.end()))
    return matches


def _find_cvv_matches(text: str) -> list[DlpMatch]:
    return [DlpMatch(CVV_CATEGORY, m.start(1), m.end(1)) for m in _CVV_KEYWORD_RE.finditer(text)]


def _find_expiry_matches(text: str) -> list[DlpMatch]:
    return [DlpMatch(EXPIRY_CATEGORY, m.start(), m.end()) for m in _EXPIRY_RE.finditer(text)]


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    engine = SpacyNlpEngine(models=[{"lang_code": "en", "model_name": _SPACY_MODEL}])
    engine.load()
    return AnalyzerEngine(nlp_engine=engine, supported_languages=["en"])


def _find_pii_matches(text: str) -> list[DlpMatch]:
    results = _analyzer().analyze(text=text, language="en", entities=list(PII_ENTITIES))
    return [DlpMatch(r.entity_type, r.start, r.end) for r in results]


def scan(text: str) -> DlpScanResult:
    """Detecta PAN, CVV, validade e PII genérica em `text`."""
    matches = sorted(
        (
            *_find_pan_matches(text),
            *_find_cvv_matches(text),
            *_find_expiry_matches(text),
            *_find_pii_matches(text),
        ),
        key=lambda m: m.start,
    )
    categories = tuple(sorted({m.category for m in matches}))
    return DlpScanResult(
        has_sensitive_data=bool(matches),
        categories=categories,
        matches=tuple(matches),
    )


def mask(text: str) -> tuple[str, bool]:
    """Redige spans sensíveis em `text`. Devolve `(texto_redigido, foi_redigido)`."""
    result = scan(text)
    if not result.matches:
        return text, False

    pieces = []
    cursor = 0
    for m in result.matches:
        if m.start < cursor:
            continue  # overlapping match already covered by a previous redaction
        pieces.append(text[cursor : m.start])
        pieces.append(f"[{m.category}_REDACTED]")
        cursor = m.end
    pieces.append(text[cursor:])
    return "".join(pieces), True


def mask_value(value: object) -> object:
    """Aplica `mask` recursivamente a strings dentro de dict/list/tuple.

    Usado pelo wrapper de logging para redigir um event_dict inteiro antes do sink
    (invariante I2). Outros tipos passam direto.
    """
    if isinstance(value, str):
        masked, _ = mask(value)
        return masked
    if isinstance(value, dict):
        return {key: mask_value(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(mask_value(val) for val in value)
    return value
