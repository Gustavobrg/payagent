"""Independent PAN-leak audit over raw log text — the harness's own "grep the logs" check.

Deliberately does NOT import `payagent.guardrails.dlp`. The point of this scan is to be a
second, independently-written detector: if `dlp.py`'s regex or its Luhn check ever had a
bug, asking `dlp.scan()` to grade its own log output would just reproduce the same blind
spot. `handlers.py::handle_execute_settlement` already uses this "re-derive the same fact
a different way" pattern deliberately (its own docstring: "so a permissive engine still
cannot settle...") — this module applies the same idea to the audit side.

`pan_leak_count` is computed over the **raw text actually written to the log sink** (after
`_dlp_redact_processor` in `observability/logging.py` has already run), so a nonzero count
here means a real leak past the production DLP layer, not a hypothetical one.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 13-19 digits, optionally grouped by spaces or hyphens — same length range dlp.py uses
# (ISO/IEC 7812), written independently rather than imported.
_DIGIT_RUN_RE = re.compile(r"(?<!\d)(?:\d[ -]?){12,}\d(?!\d)")


def _luhn_is_valid(digits: str) -> bool:
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
class PanLeak:
    span: str
    offset: int


def scan_for_pan_leaks(text: str) -> list[PanLeak]:
    """Return every Luhn-valid 13-19 digit run found in `text`."""
    leaks: list[PanLeak] = []
    for match in _DIGIT_RUN_RE.finditer(text):
        digits = re.sub(r"[ -]", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_is_valid(digits):
            leaks.append(PanLeak(span=match.group(0), offset=match.start()))
    return leaks
