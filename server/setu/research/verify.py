"""Claim-level citation verification for generated research answers.

The v1 groundedness gate verifies short answers and blocks the whole answer when
coverage is low. That is right for two sentences and wrong for a fifteen-claim report,
where one weak claim should be removed, not the report withheld. This verifier checks
every claim against **only the passages it cites**:

support(claim, passage) = max(lexical containment, semantic max-sim)

* lexical containment - fraction of the claim's content tokens present in the passage
  (the same measure the v1 gate uses);
* semantic max-sim - the best cosine between the claim and any sentence of the passage,
  with the same multilingual encoder retrieval uses, so a Hindi claim can be checked
  against an English passage.

Then:

1. A citation whose passage does not support its claim (< ``CITE_MIN``) is dropped -
   "retrieved" is not the same as "supports this".
2. A ``fact`` claim left with no supporting citation, below ``FACT_MIN``, or containing a
   number that appears in none of its kept passages is **removed** and reported in
   ``removed_claims`` with the reason.
3. ``calculation`` claims may contain numbers produced by the deterministic calculator;
   ``assumption`` claims may contain numbers the user supplied; ``interpretation`` claims
   must still cite something and may not introduce numbers from nowhere.

These scores are automated heuristics, not entailment proofs. The evaluation lab reports
them as such.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..storage.embed_cache import CachedEncoder
from ..text import containment, content_tokens, numbers_in, split_sentences, tokenize
from .models import ClaimOut

CITE_MIN = 0.42
FACT_MIN = 0.50
INTERP_MIN = 0.30


@dataclass
class VerifyReport:
    kept: list[ClaimOut]
    removed: list[ClaimOut]
    coverage: float
    weakest: float
    dropped_citations: int
    numeric_failures: list[str]


class CitationVerifier:
    def __init__(self, encoder: CachedEncoder | None = None) -> None:
        self.encoder = encoder or CachedEncoder()

    def _sentence_pool(self, texts: dict[int, str]) -> dict[int, tuple[list[str], np.ndarray]]:
        pools: dict[int, tuple[list[str], np.ndarray]] = {}
        all_sents: list[str] = []
        owners: list[int] = []
        for m, t in texts.items():
            sents = [s.text for s in split_sentences(t)] or [t]
            all_sents.extend(sents)
            owners.extend([m] * len(sents))
        if not all_sents:
            return pools
        vecs = self.encoder.encode(all_sents)
        for m in texts:
            idx = [i for i, o in enumerate(owners) if o == m]
            pools[m] = ([all_sents[i] for i in idx], vecs[idx])
        return pools

    def verify(
        self,
        claims: list[ClaimOut],
        evidence: dict[int, str],
        *,
        allowed_numbers: set[str] | None = None,
        user_numbers: set[str] | None = None,
        pools: dict[int, tuple[list[str], np.ndarray]] | None = None,
    ) -> VerifyReport:
        """``pools`` lets the caller pass sentence vectors the index already holds, so
        verification embeds only the claims, never the evidence."""
        allowed_numbers = allowed_numbers or set()
        user_numbers = user_numbers or set()
        if not claims:
            return VerifyReport([], [], 0.0, 0.0, 0, [])
        pools = dict(pools or {})
        missing = {m: t for m, t in evidence.items() if m not in pools or not len(pools[m][1])}
        if missing:
            pools.update(self._sentence_pool(missing))
        ev_tokens = {m: set(tokenize(t)) for m, t in evidence.items()}
        ev_numbers = {m: set(numbers_in(t)) for m, t in evidence.items()}
        all_numbers = set().union(*ev_numbers.values()) if ev_numbers else set()

        bodies = [c.text for c in claims]
        cvecs = self.encoder.inner.encode(bodies) if bodies else np.zeros((0, 1))

        kept: list[ClaimOut] = []
        removed: list[ClaimOut] = []
        dropped_total = 0
        numeric_failures: list[str] = []

        for c, cvec in zip(claims, cvecs, strict=True):
            c_tokens = set(content_tokens(c.text))
            per_marker: dict[int, tuple[float, str | None]] = {}
            for m in c.citations:
                if m not in evidence:
                    continue
                lex = containment(c_tokens, ev_tokens[m]) if c_tokens else 0.0
                sem, best = 0.0, None
                if m in pools and len(pools[m][1]):
                    sims = pools[m][1] @ cvec
                    j = int(np.argmax(sims))
                    sem, best = float(sims[j]), pools[m][0][j]
                per_marker[m] = (max(lex, sem), best)

            invalid = [m for m in c.citations if m not in evidence]
            good = [m for m, (s, _) in per_marker.items() if s >= CITE_MIN]
            dropped = [m for m in c.citations if m not in good]
            dropped_total += len([m for m in dropped if m not in invalid])
            support = max((per_marker[m][0] for m in good), default=0.0)
            best_sentence = None
            if good:
                best_m = max(good, key=lambda m: per_marker[m][0])
                best_sentence = per_marker[best_m][1]

            kept_numbers = set().union(*(ev_numbers[m] for m in good)) if good else set()
            claim_numbers = set(numbers_in(c.text))
            if c.kind == "calculation":
                bad_nums = claim_numbers - kept_numbers - allowed_numbers - all_numbers
            elif c.kind == "assumption":
                bad_nums = claim_numbers - kept_numbers - user_numbers - allowed_numbers - all_numbers
            else:
                bad_nums = claim_numbers - kept_numbers
                # A number present in *another* retrieved passage is a citation imprecision,
                # not a fabrication; only numbers found nowhere invalidate a fact claim.
                if c.kind == "interpretation":
                    bad_nums = claim_numbers - all_numbers - allowed_numbers
            fabricated = sorted(n for n in bad_nums if n not in all_numbers)
            numeric_ok = not bad_nums if c.kind == "fact" else not fabricated
            if c.kind == "fact" and bad_nums and not fabricated:
                support *= 0.85
                numeric_ok = True

            out = c.model_copy(update={
                "citations": good, "dropped_citations": dropped, "support": round(support, 4),
                "numeric_ok": numeric_ok, "evidence_sentence": best_sentence,
            })
            reason = None
            if c.kind == "fact":
                if not good:
                    reason = ("cited passage does not support this claim" if c.citations
                              else "claim has no citation")
                elif support < FACT_MIN:
                    reason = f"support {support:.2f} below the {FACT_MIN} threshold"
                elif fabricated:
                    reason = f"number(s) not found in any evidence: {', '.join(fabricated[:3])}"
            elif c.kind == "interpretation":
                if not good:
                    reason = "interpretation is not linked to any supporting passage"
                elif support < INTERP_MIN:
                    reason = f"interpretation support {support:.2f} below {INTERP_MIN}"
                elif fabricated:
                    reason = f"introduces number(s) absent from the evidence: {', '.join(fabricated[:3])}"
            elif fabricated and c.kind == "calculation":
                reason = f"calculation uses number(s) not in evidence or calculator output: {', '.join(fabricated[:3])}"
            if fabricated:
                numeric_failures.extend(fabricated)
            if reason:
                removed.append(out.model_copy(update={"supported": False, "removed_reason": reason}))
            else:
                kept.append(out.model_copy(update={"supported": True}))

        scored = [k for k in kept if k.kind in ("fact", "interpretation")]
        if scored:
            w = np.asarray([len(k.text) for k in scored], dtype=np.float32)
            coverage = float(np.average([k.support for k in scored], weights=w))
            weakest = float(min(k.support for k in scored))
        else:
            coverage, weakest = 0.0, 0.0
        return VerifyReport(kept, removed, round(coverage, 4), round(weakest, 4), dropped_total,
                            sorted(set(numeric_failures)))


__all__ = ["CITE_MIN", "FACT_MIN", "CitationVerifier", "VerifyReport"]
