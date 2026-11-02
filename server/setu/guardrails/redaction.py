"""Output-stage gate: PII redaction.

MS MARCO is scraped from the open web, and scraped web text contains phone numbers,
email addresses, and occasionally worse. An extractive answerer copies spans verbatim,
so anything in a passage can reach a user - and in a voice system it gets *spoken aloud*.
This gate is the last thing between the corpus and the speaker.

Redaction happens on the way out rather than at ingest deliberately: the tokens stay
searchable (a query about a company's support line should still retrieve the right
passage) while the values do not leave the process. Detectors cover the formats that
actually appear in Indian and international web text, including Aadhaar-shaped
12-digit groups, which are validated with the Verhoeff checksum so that ordinary
12-digit numbers are not mangled.
"""

from __future__ import annotations

import re

from ..harness.contracts import GuardVerdict
from ..harness.tracing import now_ms
from .base import BaseGate, GateContext

_VERHOEFF_D = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6), (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8), (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2), (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4), (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_P = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), (1, 5, 7, 6, 2, 8, 3, 0, 9, 4),
    (5, 8, 0, 3, 7, 9, 6, 1, 4, 2), (8, 9, 1, 6, 0, 4, 3, 5, 2, 7),
    (9, 4, 5, 3, 1, 2, 6, 8, 7, 0), (4, 2, 8, 6, 5, 7, 3, 9, 0, 1),
    (2, 7, 9, 3, 8, 0, 6, 4, 1, 5), (7, 0, 4, 6, 9, 1, 3, 2, 5, 8),
)
_VERHOEFF_INV = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)


def verhoeff_valid(number: str) -> bool:
    digits = [int(d) for d in reversed(number) if d.isdigit()]
    c = 0
    for i, d in enumerate(digits):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


def _luhn_valid(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


PII_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("email", r"\b[\w.%+-]+@[\w-]+\.[A-Za-z]{2,}\b", "[email redacted]"),
    ("phone_in", r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{9}(?!\d)", "[phone redacted]"),
    ("phone_intl", r"(?<!\d)\+\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}(?!\d)", "[phone redacted]"),
    ("aadhaar", r"(?<!\d)\d{4}\s?\d{4}\s?\d{4}(?!\d)", "[id redacted]"),
    ("pan", r"\b[A-Z]{5}\d{4}[A-Z]\b", "[id redacted]"),
    ("card", r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)", "[card redacted]"),
    ("ipv4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[ip redacted]"),
)
_COMPILED = tuple((name, re.compile(p), repl) for name, p, repl in PII_PATTERNS)


class PIIGate(BaseGate):
    name = "output.pii"
    stage = "output"

    def __init__(self) -> None:
        self.redacted_text: str | None = None

    def check(self, ctx: GateContext) -> GuardVerdict:
        t0 = now_ms()
        self.redacted_text = None
        draft = ctx.answer
        if draft is None or not draft.text:
            return self.allow(1.0, reason="nothing to inspect", started=t0)

        text = draft.text
        found: list[str] = []
        for name, rx, repl in _COMPILED:
            def _sub(m: re.Match[str], _n=name, _r=repl) -> str:
                value = m.group(0)
                if _n == "aadhaar" and not verhoeff_valid(value):
                    return value  # a plain 12-digit quantity, not an identifier
                if _n == "card" and not _luhn_valid(value):
                    return value
                found.append(_n)
                return _r

            text = rx.sub(_sub, text)

        if not found:
            return self.allow(1.0, reason="no PII detected", started=t0)

        self.redacted_text = text
        kinds = sorted(set(found))
        return self.flag(
            0.5, reason=f"redacted {len(found)} PII span(s): {', '.join(kinds)}",
            started=t0, kinds=kinds, count=len(found),
        )


__all__ = ["PIIGate", "PII_PATTERNS", "verhoeff_valid"]
