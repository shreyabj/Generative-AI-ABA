"""Output-stage gate: claim-level attribution and hallucination detection.

This gate verifies the answer against the evidence the answer *itself* cited. It does
not ask "is this answer plausible" - it asks, sentence by sentence, "is this supported".

Per claim (one sentence of the answer, citation markers stripped):

**Lexical containment.** What fraction of the claim's content tokens appear in its cited
chunks? This is asymmetric on purpose - Jaccard would penalise a short claim citing a long
passage, which is the normal and correct case. For the reflex path containment is ~1.0 by
construction, and that is the point: the check confirms a structural property rather than
guessing at one.

**Numeric consistency.** Every number in the claim must appear in the cited evidence.
This is the highest-value single check in the whole system. Quantities are where
hallucination does real damage ("89 calories" vs "189 calories"), a wrong number is
invisible to a fluency-based reviewer, and verifying it is a set intersection. A claim
that fails this is rejected outright regardless of how well it scores elsewhere.

**Entity consistency.** Capitalised multi-character tokens in the claim must appear in the
evidence, catching invented names and places.

**Semantic entailment (deliberate path only).** A paraphrase can be perfectly grounded and
still score low on containment, so for generated answers we additionally embed each claim
and take its max cosine against the cited chunks' cached sentence vectors. This costs one
encoder call and is skipped entirely for the reflex path - where it would be both
redundant and unaffordable inside the 200 ms budget.

Coverage is the length-weighted mean of per-claim support. Below ``groundedness_min_coverage``
the answer is discarded and replaced with a localised abstention: the system says it could
not verify an answer instead of shipping one it cannot stand behind.
"""

from __future__ import annotations

import re

import numpy as np

from ..config import settings
from ..embed import get_encoder
from ..harness.contracts import Claim, GuardVerdict
from ..harness.tracing import now_ms
from ..index.corpus import CorpusIndex
from ..text import containment, content_tokens, numbers_in, split_sentences, tokenize
from .base import BaseGate, GateContext

_MARKER_RE = re.compile(r"\s*\[\d+\]")
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")


class GroundednessGate(BaseGate):
    name = "output.groundedness"
    stage = "output"

    def __init__(
        self,
        index: CorpusIndex,
        min_coverage: float | None = None,
        min_claim: float | None = None,
    ) -> None:
        self.index = index
        self.min_coverage = settings.groundedness_min_coverage if min_coverage is None else min_coverage
        self.min_claim = settings.groundedness_min_claim if min_claim is None else min_claim
        self.claims: list[Claim] = []

    # --------------------------------------------------------------------- check
    def check(self, ctx: GateContext) -> GuardVerdict:
        t0 = now_ms()
        self.claims = []
        draft = ctx.answer
        if draft is None or not draft.text.strip():
            return self.block(0.0, reason="no answer to verify", started=t0)
        if not draft.citations:
            return self.block(0.0, reason="answer carries no citations", started=t0)

        by_marker = {c.marker: c for c in draft.citations}
        evidence_tokens: dict[int, set[str]] = {}
        evidence_numbers: dict[int, set[str]] = {}
        for m, cit in by_marker.items():
            row = self.index.row_of(cit.chunk_id)
            text = self.index.record(row).text if row is not None else cit.snippet
            evidence_tokens[m] = set(tokenize(text))
            evidence_numbers[m] = set(numbers_in(text))

        all_tokens: set[str] = set().union(*evidence_tokens.values()) if evidence_tokens else set()
        all_numbers: set[str] = set().union(*evidence_numbers.values()) if evidence_numbers else set()

        raw_claims = split_sentences(draft.text)
        semantic_scores = self._semantic(draft, raw_claims, by_marker)

        supports: list[float] = []
        weights: list[float] = []
        numeric_failures: list[str] = []
        entity_failures: list[str] = []

        for i, sent in enumerate(raw_claims):
            body = _MARKER_RE.sub("", sent.text).strip()
            if len(body) < 3:
                continue
            markers = [int(m) for m in re.findall(r"\[(\d+)\]", sent.text) if int(m) in by_marker]
            scope_tokens = (
                set().union(*(evidence_tokens[m] for m in markers)) if markers else all_tokens
            )
            scope_numbers = (
                set().union(*(evidence_numbers[m] for m in markers)) if markers else all_numbers
            )

            claim_tokens = set(content_tokens(body))
            lexical = containment(claim_tokens, scope_tokens) if claim_tokens else 0.0

            # Per-claim support is scored against the claim's *own* citations, but a hard
            # numeric block is judged against *all* retrieved evidence. A number that
            # appears in chunk 2 while the claim cites chunk 1 is a citation imprecision
            # worth penalising; it is not a hallucination worth discarding the answer for.
            # A number that appears nowhere in the evidence is.
            claim_numbers = set(numbers_in(body))
            unsupported_nums = claim_numbers - scope_numbers
            hallucinated_nums = claim_numbers - all_numbers
            numeric_ok = not (settings.numeric_consistency and hallucinated_nums)
            if hallucinated_nums:
                numeric_failures.extend(sorted(hallucinated_nums))

            claim_entities = {e.casefold() for e in _ENTITY_RE.findall(body)}
            unsupported_ents = {e for e in claim_entities if e not in scope_tokens}
            if unsupported_ents:
                entity_failures.extend(sorted(unsupported_ents))

            semantic = semantic_scores.get(i, 0.0)
            support = max(lexical, semantic) if semantic else lexical
            if unsupported_ents:
                support *= 0.72
            if unsupported_nums and not hallucinated_nums:
                support *= 0.85  # right number, imprecise citation
            if not numeric_ok:
                support = 0.0  # a fabricated number invalidates the claim outright

            self.claims.append(
                Claim(text=body, support=round(float(support), 4), cited=markers, numeric_ok=numeric_ok)
            )
            supports.append(float(support))
            weights.append(float(len(body)))

        if not supports:
            return self.block(0.0, reason="answer contained no verifiable claims", started=t0)

        w = np.asarray(weights, dtype=np.float32)
        coverage = float(np.average(np.asarray(supports, dtype=np.float32), weights=w))
        weakest = float(min(supports))

        detail = {
            "claims": len(supports),
            "coverage": round(coverage, 4),
            "weakest_claim": round(weakest, 4),
            "unsupported_numbers": sorted(set(numeric_failures))[:6],
            "unsupported_entities": sorted(set(entity_failures))[:6],
            "semantic_checked": bool(semantic_scores),
        }

        if numeric_failures:
            return self.block(
                0.0,
                reason=f"unsupported quantity in answer: {', '.join(sorted(set(numeric_failures))[:3])}",
                threshold=self.min_coverage, started=t0, **detail,
            )
        if coverage < self.min_coverage:
            return self.block(
                coverage, reason=f"claim coverage {coverage:.3f} below the {self.min_coverage} floor",
                threshold=self.min_coverage, started=t0, **detail,
            )
        if weakest < self.min_claim:
            return self.flag(
                coverage,
                reason=f"one claim is weakly supported ({weakest:.3f}) - answer kept, trust reduced",
                threshold=self.min_claim, started=t0, **detail,
            )
        return self.allow(
            coverage, reason=f"all {len(supports)} claim(s) attributable, coverage {coverage:.3f}",
            threshold=self.min_coverage, started=t0, **detail,
        )

    # ================================================================= internals
    def _semantic(self, draft, raw_claims, by_marker) -> dict[int, float]:
        """Entailment-style check. Only for generated answers - see module docstring."""
        if draft.path != "deliberate" or not raw_claims:
            return {}
        bodies: list[str] = []
        keep: list[int] = []
        for i, sent in enumerate(raw_claims):
            body = _MARKER_RE.sub("", sent.text).strip()
            if len(body) >= 12:
                bodies.append(body)
                keep.append(i)
        if not bodies:
            return {}
        try:
            cvecs = get_encoder().encode(bodies)
        except Exception:  # noqa: BLE001 - never fail the turn on an optional signal
            return {}

        pools: dict[int, np.ndarray] = {}
        for m, cit in by_marker.items():
            row = self.index.row_of(cit.chunk_id)
            if row is None:
                continue
            vecs = self.index.sentence_vectors(self.index.record(row))
            if vecs.size:
                pools[m] = vecs

        out: dict[int, float] = {}
        for vec, idx in zip(cvecs, keep, strict=True):
            markers = [int(m) for m in re.findall(r"\[(\d+)\]", raw_claims[idx].text) if int(m) in pools]
            pool = np.vstack([pools[m] for m in markers]) if markers else (
                np.vstack(list(pools.values())) if pools else None
            )
            if pool is None:
                continue
            out[idx] = float(np.max(pool @ vec))
        return out


__all__ = ["GroundednessGate"]
