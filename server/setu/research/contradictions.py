"""Contradiction detection across retrieved sources.

Two detectors, reported separately so a reader knows how much to trust each:

* **deterministic** (always): sentences from *different documents* that state a value
  for the same metric and the same period (e.g. "GDP growth ... FY27 ... 6.6 percent" vs
  "... FY27 ... 7.1 per cent") with values that differ beyond rounding. This is precise
  and narrow: it only fires on numeric disagreements it can locate exactly.
* **model** (when the LLM reported one): accepted only if every statement cites a
  passage that is in the evidence and its quote actually occurs in that passage
  (normalised substring match). Unverifiable model "contradictions" are dropped.

Neither detector decides which source is right. Both surface the passages, the
publication dates and the candidate reasons (different dates, forecast vs outturn,
different definitions or scope).
"""

from __future__ import annotations

import re
from typing import Any

from ..text import normalize, split_sentences
from .models import Contradiction, EvidenceItem, Statement

_METRICS = {
    "gdp growth": r"\b(gdp|economic)\s+growth|\bgrowth\s+(rate|forecast|projection)|\breal\s+gdp|\bgdp\b",
    "inflation": r"\binflation|\bcpi\b",
    "repo rate": r"\brepo\s+rate|\bpolicy\s+rate",
    "revenue growth": r"\brevenue\s+(growth|grew|increase|rose)|\brevenues?\s+grew",
    "revenue": r"\brevenue\b|\bturnover\b",
    "operating margin": r"\boperating\s+margin|\bebit\s+margin",
    "net profit": r"\bnet\s+(profit|income)|\bprofit\s+after\s+tax|\bpat\b",
    "headcount": r"\bheadcount|\bemployees\b",
    "fiscal deficit": r"\bfiscal\s+deficit",
    "current account deficit": r"\bcurrent\s+account\s+deficit",
}
_METRIC_RE = {k: re.compile(v, re.IGNORECASE) for k, v in _METRICS.items()}
_PERIOD = re.compile(r"\b((?:Q[1-4]\s*)?FY\s*'?\d{2,4}(?:-\d{2})?|20\d{2}(?:-\d{2})?)\b", re.IGNORECASE)
_VALUE = re.compile(r"(-?\d+(?:\.\d+)?)\s*(%|per\s*cent|percent|bps)", re.IGNORECASE)


def _norm_period(p: str) -> str:
    p = re.sub(r"\s+|'", "", p.upper())
    m = re.fullmatch(r"(Q[1-4])?FY(\d{2,4})(?:-\d{2})?", p)
    if m:
        y = m.group(2)
        y = y[-2:]
        return f"{m.group(1) or ''}FY{y}"
    m = re.fullmatch(r"(20\d{2})-(\d{2})", p)
    if m:  # 2025-26 is fiscal year FY26 in Indian usage
        return f"FY{m.group(2)}"
    return p


def _statements(item: EvidenceItem) -> list[tuple[str, str, float, str]]:
    out = []
    for s in split_sentences(item.text):
        t = s.text
        periods = {_norm_period(p) for p in _PERIOD.findall(t)}
        values = _VALUE.findall(t)
        if len(periods) != 1 or not values:
            continue
        for metric, rx in _METRIC_RE.items():
            if rx.search(t):
                val = float(values[0][0])
                out.append((metric, next(iter(periods)), val, t.strip()))
                break
    return out


def detect_numeric(evidence: list[EvidenceItem], *, tolerance: float = 0.15) -> list[Contradiction]:
    seen: dict[tuple[str, str], list[tuple[float, EvidenceItem, str]]] = {}
    for e in evidence:
        for metric, period, val, sent in _statements(e):
            seen.setdefault((metric, period), []).append((val, e, sent))
    out: list[Contradiction] = []
    for (metric, period), rows in seen.items():
        by_doc: dict[str, tuple[float, EvidenceItem, str]] = {}
        for val, e, sent in rows:
            by_doc.setdefault(e.document_id or e.doc_id, (val, e, sent))
        if len(by_doc) < 2:
            continue
        vals = sorted(by_doc.values(), key=lambda r: r[0])
        lo, hi = vals[0], vals[-1]
        if hi[0] - lo[0] <= tolerance:
            continue
        dates = [r[1].published_at for r in (lo, hi)]
        reasons = ["the sources may refer to different editions or revision rounds"]
        if dates[0] and dates[1] and dates[0] != dates[1]:
            reasons.insert(0, f"they were published on different dates ({dates[0]} vs {dates[1]})")
        if any(re.search(r"\b(project|forecast|expect|estimate|outlook)", r[2], re.IGNORECASE) for r in (lo, hi)):
            reasons.append("at least one figure is a projection, which is revised over time")
        out.append(Contradiction(
            topic=f"{metric} for {period}",
            statements=[Statement(marker=r[1].marker, quote=r[2][:400], title=r[1].title,
                                  published_at=r[1].published_at, value=f"{r[0]:g}") for r in (lo, hi)],
            explanation=("The sources report different values (" + f"{lo[0]:g} vs {hi[0]:g}"
                         + "). Possible reasons: " + "; ".join(reasons) + ". SETU does not decide which is correct."),
            detector="deterministic",
        ))
    return out[:5]


def verify_model(raw: list[dict[str, Any]], evidence: dict[int, EvidenceItem]) -> tuple[list[Contradiction], int]:
    accepted: list[Contradiction] = []
    rejected = 0
    for c in raw:
        stmts = []
        ok = True
        for s in c.get("statements") or []:
            try:
                m = int(s.get("marker"))
            except (TypeError, ValueError):
                ok = False
                break
            quote = str(s.get("quote") or "").strip()
            e = evidence.get(m)
            if e is None or not quote:
                ok = False
                break
            q = normalize(quote).strip(" .\"'")
            if len(q) < 8 or q[:60] not in normalize(e.text):
                ok = False
                break
            stmts.append(Statement(marker=m, quote=quote[:400], title=e.title, published_at=e.published_at))
        docs = {evidence[s.marker].document_id or evidence[s.marker].doc_id for s in stmts}
        if ok and len(stmts) >= 2 and len(docs) >= 2:
            accepted.append(Contradiction(topic=str(c.get("topic") or "disagreement")[:160], statements=stmts,
                                          explanation=str(c.get("explanation") or "")[:800], detector="model"))
        else:
            rejected += 1
    return accepted, rejected


def merge(det: list[Contradiction], model: list[Contradiction]) -> list[Contradiction]:
    out = list(det)
    keys = {frozenset(s.marker for s in c.statements) for c in det}
    for c in model:
        k = frozenset(s.marker for s in c.statements)
        if k not in keys:
            out.append(c)
            keys.add(k)
    return out[:8]


__all__ = ["detect_numeric", "merge", "verify_model"]
