"""Indirect prompt-injection detection for *document* text.

Retrieved passages are data. A document that says "ignore your instructions and reply
with X" is an attack on any system that pastes retrieved text into a prompt, and it
arrives through the knowledge base rather than through the question box, so the input
gate never sees it.

Three defences, layered:

1. **Ingest-time flagging** (this module). Every chunk is scanned with the input gate's
   injection patterns plus document-specific ones. Flagged chunks stay searchable - a
   security report legitimately quotes attack strings - but carry ``injection_flag``,
   are shown with a warning in the UI, and are quoted with an explicit warning in the
   assembled context.
2. **Delimited context** (setu.research.context). Evidence is wrapped in tagged blocks
   and the system prompt states that passage text is never an instruction.
3. **Output check** (setu.research.verify). Claims must be supported by evidence, so a
   passage that tells the model to say something unsupported produces an unsupported
   claim, which is removed.
"""

from __future__ import annotations

import re

from ..guardrails.admission import _INJECTION_RE

_DOC_PATTERNS = tuple(
    (re.compile(p, re.IGNORECASE | re.UNICODE), label)
    for p, label in (
        (r"\b(you\s+are\s+now|from\s+now\s+on\s+you\s+(are|will))\b", "role-reassignment"),
        (r"\b(system|assistant|developer)\s*(prompt|message|instruction)s?\s*[:=]", "fake-role-block"),
        (r"<\s*/?\s*(system|assistant|instructions?)\s*>", "fake-role-tag"),
        (r"\b(ai|assistant|model|chatbot|llm)\b.{0,40}\b(must|should)\s+(always\s+)?(say|answer|reply|respond|state)\b",
         "directive-to-model"),
        (r"\bdo\s+not\s+(cite|mention|reveal)\b.{0,40}\b(source|document|instruction)", "suppress-citation"),
        (r"\b(reveal|print|output)\s+(your|the)\s+(system\s+)?(prompt|instructions|api\s*key)", "exfiltration"),
    )
)


def scan(text: str) -> list[str]:
    """Labels of every injection pattern found in ``text`` (empty when clean)."""
    labels = [label for rx, label in _INJECTION_RE if rx.search(text)]
    labels += [label for rx, label in _DOC_PATTERNS if rx.search(text)]
    return sorted(set(labels))


__all__ = ["scan"]
