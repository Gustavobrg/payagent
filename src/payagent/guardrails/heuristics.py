"""Cheap regex-based prompt-injection heuristics (P4) — deterministic, no model call.

Same reasoning as `dlp.py`'s PAN/CVV detection (see its module docstring): both
`heuristic_injection_check` (`guardrails/rails/input.co`) and the per-chunk scan in
`rag.context.assemble_context` need something that runs before, and independently of, the
Llama Guard classifier — cheap enough to run on every retrieved chunk, not just the user's
own message.

This is not a substitute for the classifier: known phrasing only, tuned for precision over
recall on the calibration set (docs/guardrail-taxonomy.md's note on false positives — a
payment agent that blocks legitimate purchases is useless). Patterns are bilingual (PT/EN,
matching the taxonomy's own examples) and grouped by what they target: overriding prior
instructions, switching persona, claiming a bypass/test mode, telling the reader to treat
third-party text as a command, or manipulating settlement amount/beneficiary/step-up state
from inside retrieved prose — the attack shapes in evals/datasets/poisoned_skus.json.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore_prior_instructions": re.compile(
        r"(?i)\bignor[ae][a-z]*\b[^.\n]{0,40}\b(instru[cç][oõ]es|regras|comandos|prompt)\b"
        r"|\bignore\b[^.\n]{0,40}\b(previous|prior|above|earlier)\b[^.\n]{0,20}"
        r"\b(instructions|rules|prompt)\b"
    ),
    "persona_switch": re.compile(
        r"(?i)\b(assuma|finja ser|aja como|torne-se)\b[^.\n]{0,30}"
        r"\b(outra persona|outro papel|novo assistente)\b"
        r"|\b(pretend to be|act as|you are now|roleplay as)\b"
    ),
    "bypass_mode": re.compile(
        r"(?i)\bmodo (de )?(desenvolvedor|teste|debug)\b"
        r"|\b(developer|debug|test)\s+mode\b"
        r"|\bcleared in the test harness\b|\bj[aá] foi liberado no teste\b"
    ),
    "treat_as_command": re.compile(
        r"(?i)\b(trate|considere)\b[^.\n]{0,30}\b(texto|conte[uú]do)\b[^.\n]{0,30}"
        r"\bcomo (um )?(comando|instru[cç][aã]o)\b"
        r"|\btreat\b[^.\n]{0,30}\b(this|the following|that)\b[^.\n]{0,20}"
        r"\bas an? (command|instruction)\b"
    ),
    "settlement_override": re.compile(
        r"(?i)\bregardless of the (displayed |quoted )?(quote|price|total|amount)\b"
        r"|\bsettlement beneficiary\b|\bupdate the beneficiary\b"
        r"|\bauto-approve\b[^.\n]{0,30}\btransactions\b"
        r"|\bliquide?\b[^.\n]{0,30}\bpara (o )?merchant\b"
    ),
    "confirmation_bypass": re.compile(
        r"(?i)\bbypass\b[^.\n]{0,30}\b(confirmation|step-?up|authoriz\w*)\b"
        r"|\balready (been )?(authorized|cleared|approved)\b"
        r"|\bj[aá] autorizei\b|\bpular a confirma[cç][aã]o\b"
    ),
    "credential_exfiltration": re.compile(
        r"(?i)\bprint (the|your) full\b[^.\n]{0,20}\bmandate\b"
        r"|\bJWS signature\b|\bassinatura JWS\b"
        r"|\bsystem prompt\b|\bprompt de sistema\b"
    ),
    "fake_system_role": re.compile(r"(?i)\bsystem\s*:\s*\S"),
}


@dataclass(frozen=True)
class HeuristicScanResult:
    """Result of `scan`: whether any known injection pattern matched, and which ones."""

    blocked: bool
    patterns: tuple[str, ...] = field(default_factory=tuple)


def scan(text: str) -> HeuristicScanResult:
    """Check `text` against the known P4 injection patterns."""
    matched = tuple(name for name, pattern in _PATTERNS.items() if pattern.search(text))
    return HeuristicScanResult(blocked=bool(matched), patterns=matched)
