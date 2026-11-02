"""Query understanding: classification, follow-up resolution, decomposition, entities.

Two tiers, so the system works with or without an LLM key and always says which ran:

* **Rules** (always): intent cues, anaphora detection, relative-period arithmetic
  ("last year" after "FY25" becomes FY24), multi-part splitting and capitalised-entity
  extraction matched against the titles of documents actually in the selected
  collections.
* **LLM rewrite** (when a key is set, and only for follow-ups the rules flag): a fast
  model rewrites the follow-up into a standalone question using the last two turns.
  Only the previous resolved questions and their one-line summaries are sent - the full
  conversation is never pasted into a prompt.

The resolved query is returned to the client and stored with the message, so a user can
always see what SETU actually searched for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from ..config import settings
from ..text import STOPWORDS, content_tokens, normalize
from .llm import LLMClient, LLMError

_ANAPHORA = re.compile(
    r"\b(it|its|it's|they|their|them|this|that|these|those|he|she|his|her|the\s+company|the\s+firm|"
    r"the\s+same|the\s+former|the\s+latter|there)\b"
    r"|(उसका|उसकी|उसके|इसका|इसकी|उन्होंने|वह|ये|वे)"
    r"|(ಅದು|ಅದರ|ಇದು|ಇದರ|ಅವರ)",
    re.IGNORECASE | re.UNICODE,
)
_FOLLOWUP_LEAD = re.compile(
    r"^\s*(and|also|what\s+about|how\s+about|and\s+what|so|then|compared\s+to|versus|vs\.?|why\??$|how\s+come)\b",
    re.IGNORECASE,
)
_RELATIVE = re.compile(r"\b(last|previous|prior|next|following)\s+(year|quarter|fy|fiscal\s+year)\b", re.IGNORECASE)
_COMPARE = re.compile(r"\b(compare|comparison|versus|vs\.?|differ|difference|differences|contrast|"
                      r"better|worse|higher|lower|relative\s+to|against)\b", re.IGNORECASE)
_ANALYSIS = re.compile(r"\b(swot|pestle|pestel|porter|five\s+forces|risk(s)?\s+analysis|strategy|strategic|"
                       r"recommend|recommendation|outlook|implication)\b", re.IGNORECASE)
_SCENARIO = re.compile(r"\b(what\s+(if|happens\s+if|would\s+happen)|suppose|assuming|if\s+.{0,40}\b(increase|decrease|"
                       r"rise|fall|grow|drop)s?\b|scenario|hypothetical)\b", re.IGNORECASE)
_CONTRADICT = re.compile(r"\b(contradict|contradictory|conflict|disagree|inconsistent|discrepanc)", re.IGNORECASE)
_COMPARE_PAIR = re.compile(
    r"(?:difference|differences|compare|comparison|contrast)\s+(?:between\s+|of\s+)?(?P<a>.{3,80}?)\s+"
    r"(?:and|with|to|vs\.?|versus)\s+(?P<b>.{3,80}?)\s*\??$"
    r"|(?P<c>.{3,80}?)\s+(?:vs\.?|versus|different\s+from|compared\s+(?:to|with))\s+(?P<d>.{3,80}?)\s*\??$",
    re.IGNORECASE,
)
_FY = re.compile(r"\b(FY|Q[1-4]\s*FY)\s*'?(\d{2,4})\b", re.IGNORECASE)
_YEAR = re.compile(r"\b(19|20)(\d{2})\b")
_ENTITY = re.compile(r"\b([A-Z][A-Za-z0-9&.\-]+(?:\s+(?:[A-Z][A-Za-z0-9&.\-]+|of|and|&))*)")
_SPLIT = re.compile(r"\?\s+(?=\S)|;\s+|\.\s+(?=(?:what|how|why|which|who|when|where|is|are|does|do|can)\b)",
                    re.IGNORECASE)
_WH_LEADS = {"what", "how", "why", "which", "who", "when", "where", "is", "are", "does", "do", "can", "compare",
             "list", "give", "show", "explain", "describe", "identify", "find", "summarise", "summarize", "tell"}


_CALENDAR = {m.lower() for m in (
    "January February March April May June July August September October November December "
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday Quarter Year Results Revenue"
).split()}


@dataclass
class Understanding:
    original: str
    normalized: str
    resolved: str
    method: str = "none"
    is_followup: bool = False
    intent: str = "question"
    multi_part: bool = False
    sub_queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    entity_documents: dict[str, list[str]] = field(default_factory=dict)
    periods: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original": self.original, "normalized": self.normalized, "resolved": self.resolved,
            "method": self.method, "is_followup": self.is_followup, "intent": self.intent,
            "multi_part": self.multi_part, "sub_queries": self.sub_queries, "entities": self.entities,
            "entity_documents": self.entity_documents, "periods": self.periods, "notes": self.notes,
        }


def extract_entities(text: str) -> list[str]:
    out: list[str] = []
    spans = [s for m in _ENTITY.finditer(text)
             for s in re.split(r"\s+(?:and|&|vs\.?|versus|with)\s+", m.group(1))]
    for raw in spans:
        span = raw.strip(" .-&")
        words = span.split()
        while words and (words[0].lower() in _WH_LEADS or words[0].lower() in STOPWORDS):
            words = words[1:]
        while words and words[-1].lower() in {"of", "and", "&"}:
            words = words[:-1]
        span = " ".join(words)
        if len(span) < 2 or span.lower() in STOPWORDS or span.lower() in _WH_LEADS:
            continue
        if re.fullmatch(r"(FY|Q[1-4])\s*\d+", span, re.IGNORECASE) or span.lower() in _CALENDAR:
            continue
        if span not in out:
            out.append(span)
    return out[:8]


def periods_in(text: str) -> list[str]:
    out = [m.group(0).upper().replace(" ", "") for m in _FY.finditer(text)]
    out += [m.group(0) for m in _YEAR.finditer(text)]
    return list(dict.fromkeys(out))


def shift_period(p: str, delta: int) -> str | None:
    m = re.fullmatch(r"(Q[1-4])?FY'?(\d{2,4})", p, re.IGNORECASE)
    if m:
        q, y = m.group(1) or "", m.group(2)
        n = int(y) + delta
        return f"{q}FY{n:0{len(y)}d}"
    if re.fullmatch(r"(19|20)\d{2}", p):
        return str(int(p) + delta)
    return None


def classify_intent(q: str) -> str:
    if _SCENARIO.search(q):
        return "scenario"
    if _CONTRADICT.search(q):
        return "contradiction"
    if _COMPARE.search(q):
        return "comparison"
    if _ANALYSIS.search(q):
        return "analysis"
    return "question"


def split_parts(q: str) -> list[str]:
    parts = [p.strip(" ?.;") for p in _SPLIT.split(q) if p and p.strip(" ?.;")]
    parts = [p for p in parts if len(content_tokens(p)) >= 2]
    return parts[:4] if len(parts) > 1 else []


class QueryUnderstanding:
    def __init__(self, llm: LLMClient | None = None) -> None:
        self.llm = llm

    def detect_followup(self, q: str, history: list[dict[str, Any]]) -> bool:
        if not history:
            return False
        n_tokens = len(q.split())
        return bool(
            _FOLLOWUP_LEAD.search(q)
            or (_ANAPHORA.search(q) and n_tokens <= 16)
            or (_RELATIVE.search(q) and n_tokens <= 14)
            or n_tokens <= 3
        )

    def resolve_rules(self, q: str, history: list[dict[str, Any]]) -> tuple[str, list[str]]:
        prev = history[-1]
        prev_q = prev.get("resolved") or prev.get("question") or ""
        notes: list[str] = []
        ents = [e for e in extract_entities(prev_q) if e.lower() not in q.lower()]
        resolved = q.rstrip("?. ")
        prev_periods = periods_in(prev_q)
        rel = _RELATIVE.search(q)
        if rel and prev_periods:
            delta = -1 if rel.group(1).lower() in ("last", "previous", "prior") else 1
            shifted = [s for s in (shift_period(p, delta) for p in prev_periods) if s]
            if shifted:
                resolved = _RELATIVE.sub(" and ".join(shifted), resolved)
                resolved += f" (compared with {' and '.join(prev_periods)})"
                notes.append(f"relative period resolved: {rel.group(0)} -> {', '.join(shifted)}")
        if ents:
            # Put the carried entity where the pronoun was when there is one; otherwise append.
            m = _ANAPHORA.search(resolved)
            if m:
                resolved = resolved[:m.start()] + ents[0] + resolved[m.end():]
                if len(ents) > 1:
                    resolved += f" ({', '.join(ents[1:3])})"
            else:
                resolved += f" - {', '.join(ents[:3])}"
            notes.append(f"carried entities from previous turn: {', '.join(ents[:3])}")
        topic = " ".join(content_tokens(prev_q)[:6])
        if not ents and topic and len(q.split()) <= 6:
            resolved += f" (context: {topic})"
            notes.append("short follow-up: appended previous topic terms")
        return normalize(resolved, casefold=False), notes

    async def resolve_llm(self, q: str, history: list[dict[str, Any]]) -> str:
        assert self.llm is not None
        turns = "\n".join(
            f"Q{i + 1}: {h.get('resolved') or h.get('question')}\nA{i + 1} (summary): {(h.get('summary') or '')[:300]}"
            for i, h in enumerate(history[-2:])
        )
        system = (
            "You rewrite a follow-up question into a standalone search question. Resolve pronouns and "
            "relative periods (e.g. 'last year' relative to the fiscal year discussed) using the "
            "conversation. Keep the user's language. Do not answer the question. Do not add facts. "
            'Return JSON: {"standalone": str}'
        )
        out, _meta = await self.llm.json(
            system=system, user=f"CONVERSATION:\n{turns}\n\nFOLLOW-UP: {q}",
            model=settings.research_fast_model, max_tokens=500, timeout_s=15.0, attempts=2,
        )
        s = str(out.get("standalone") or "").strip()
        if not s or len(s) > 600:
            raise LLMError("rewrite was empty or implausibly long")
        return s

    async def understand(self, question: str, history: list[dict[str, Any]], *,
                         doc_titles: dict[str, str] | None = None) -> Understanding:
        norm = normalize(question, casefold=False)
        u = Understanding(original=question, normalized=norm, resolved=norm)
        u.is_followup = self.detect_followup(norm, history)
        if u.is_followup:
            rule_q, notes = self.resolve_rules(norm, history)
            u.notes.extend(notes)
            u.resolved, u.method = rule_q, "rules"
            if self.llm is not None and self.llm.enabled:
                try:
                    u.resolved = await self.resolve_llm(norm, history)
                    u.method = "llm"
                    u.notes.append("standalone question rewritten by the fast LLM; rule-based form kept as fallback")
                except LLMError as exc:
                    u.notes.append(f"LLM rewrite unavailable ({exc}); using rule-based resolution")
        u.intent = classify_intent(u.resolved)
        u.periods = periods_in(u.resolved)
        u.entities = extract_entities(u.resolved)

        # Entities that name documents in the selected collections become document filters
        # for their own sub-query - that is what makes "compare A and B" retrieve evidence
        # about A *and* B instead of whichever one dominates the embedding.
        if doc_titles:
            for ent in u.entities:
                key = ent.lower()
                hits = [did for did, title in doc_titles.items() if key in title.lower()]
                if hits:
                    u.entity_documents[ent] = hits[:20]

        parts = split_parts(u.resolved)
        pair = _COMPARE_PAIR.search(u.resolved) if u.intent in ("comparison", "contradiction") else None
        if parts:
            u.multi_part = True
            u.sub_queries = parts
        elif pair and len(u.entity_documents) < 2:
            a, b = (pair.group("a") or pair.group("c")), (pair.group("b") or pair.group("d"))
            a, b = a.strip(" ?.,"), b.strip(" ?.,")
            if a and b and a.lower() != b.lower():
                u.sub_queries = [a, b]
                u.notes.append(f"comparison decomposed into topics: '{a}' | '{b}'")
        elif u.intent in ("comparison", "contradiction") and len(u.entity_documents) >= 2:
            topic_tokens = [t for t in content_tokens(u.resolved)
                            if not any(t in e.lower() for e in u.entity_documents)
                            and t not in {"compare", "comparison", "versus", "vs", "difference", "differences"}]
            topic = " ".join(topic_tokens) or u.resolved
            u.sub_queries = [f"{ent} {topic}" for ent in u.entity_documents]
            u.notes.append(f"comparison decomposed into {len(u.sub_queries)} entity sub-queries")
        return u


__all__ = ["QueryUnderstanding", "Understanding", "classify_intent", "extract_entities", "periods_in",
           "shift_period", "split_parts"]
