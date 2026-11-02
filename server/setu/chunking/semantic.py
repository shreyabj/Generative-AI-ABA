"""L1 - Semantic coherence chunker (embedding-drift boundary detection).

Fixed-size splitting cuts wherever the character counter happens to land, which
routinely severs a definition from its example. This chunker instead asks the
encoder where the topic actually changes.

Algorithm
---------
1. Embed every sentence of the document (done once at ingest, reused by all layers).
2. Build a *drift* signal ``d_i = 1 - cos(s_i, s_{i+1})`` over adjacent sentences.
   Smooth it with a width-3 Hann-ish kernel so a single parenthetical aside does
   not register as a topic shift.
3. Cut where drift exceeds ``mu + z * sigma`` of that document's own drift
   distribution. The threshold is **per document**, not global - a tightly
   written passage and a rambling one need different sensitivities, and a global
   constant is exactly the naive choice the brief warns against.
4. Enforce ``semantic_max_chars`` by cutting at the largest remaining drift peak
   inside any oversized block (recursive bisection), and merge blocks that end up
   below a third of the target.

Each chunk records its measured ``cohesion`` (mean adjacent similarity), which
the retriever later uses as a mild quality prior.
"""

from __future__ import annotations

import numpy as np

from ..config import settings
from ..harness.contracts import Layer
from ..text import split_sentences
from .base import Chunk, Document, adjacent_cohesion, stable_id

_SMOOTH = np.array([0.25, 0.5, 0.25], dtype=np.float32)


class SemanticChunker:
    layer = Layer.SEMANTIC
    name = "semantic-drift"

    def __init__(self, max_chars: int | None = None, z: float | None = None) -> None:
        self.max_chars = max_chars or settings.semantic_max_chars
        self.z = settings.semantic_z if z is None else z
        self.min_chars = max(120, self.max_chars // 3)

    def split(self, doc: Document) -> list[Chunk]:
        sents = doc.sentences or split_sentences(doc.text)
        if not sents:
            return []
        if len(sents) == 1:
            return [self._make(doc, sents, 0, 1, 0, 1.0)]

        vecs = doc.sent_vecs
        if vecs is None or len(vecs) != len(sents):
            # No sentence vectors available: fall back to a length-balanced split
            # so the layer still exists rather than silently vanishing.
            bounds = self._length_balanced(sents)
        else:
            drift = 1.0 - np.sum(vecs[:-1] * vecs[1:], axis=1)  # (n-1,)
            drift = self._smooth(drift)
            mu, sigma = float(drift.mean()), float(drift.std())
            threshold = mu + self.z * sigma
            cuts = [i + 1 for i, d in enumerate(drift) if d >= threshold]
            bounds = self._to_bounds(cuts, len(sents))
            bounds = self._enforce_max(bounds, sents, drift)
            bounds = self._merge_small(bounds, sents)

        out: list[Chunk] = []
        for ordinal, (lo, hi) in enumerate(bounds):
            coh = adjacent_cohesion(vecs[lo:hi] if vecs is not None else None)
            out.append(self._make(doc, sents, lo, hi, ordinal, coh))
        return out

    # ---------------------------------------------------------------- internals
    @staticmethod
    def _smooth(sig: np.ndarray) -> np.ndarray:
        if len(sig) < 3:
            return sig
        padded = np.pad(sig, 1, mode="edge")
        return np.convolve(padded, _SMOOTH, mode="valid")

    @staticmethod
    def _to_bounds(cuts: list[int], n: int) -> list[tuple[int, int]]:
        edges = [0, *sorted(set(c for c in cuts if 0 < c < n)), n]
        return [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]

    def _chars(self, sents, lo: int, hi: int) -> int:
        return sents[hi - 1].end - sents[lo].start

    def _enforce_max(self, bounds, sents, drift: np.ndarray) -> list[tuple[int, int]]:
        """Recursively bisect oversized blocks at their strongest interior peak."""
        out: list[tuple[int, int]] = []
        stack = list(reversed(bounds))
        while stack:
            lo, hi = stack.pop()
            if self._chars(sents, lo, hi) <= self.max_chars or hi - lo < 2:
                out.append((lo, hi))
                continue
            interior = range(lo + 1, hi)  # candidate cut positions
            best, best_d = None, -1.0
            for c in interior:
                d = float(drift[c - 1]) if c - 1 < len(drift) else 0.0
                # Prefer peaks near the middle so we do not shave off one sentence.
                balance = 1.0 - abs((c - lo) / (hi - lo) - 0.5) * 1.4
                score = d * max(balance, 0.05)
                if score > best_d:
                    best, best_d = c, score
            if best is None:
                out.append((lo, hi))
                continue
            stack.append((best, hi))
            stack.append((lo, best))
        return sorted(out)

    def _merge_small(self, bounds, sents) -> list[tuple[int, int]]:
        if len(bounds) < 2:
            return bounds
        out = [bounds[0]]
        for lo, hi in bounds[1:]:
            plo, phi = out[-1]
            small = self._chars(sents, lo, hi) < self.min_chars
            fits = self._chars(sents, plo, hi) <= self.max_chars
            if small and fits:
                out[-1] = (plo, hi)
            else:
                out.append((lo, hi))
        return out

    def _length_balanced(self, sents) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        lo = 0
        for i in range(1, len(sents) + 1):
            if self._chars(sents, lo, i) >= self.max_chars or i == len(sents):
                out.append((lo, i))
                lo = i
        return out or [(0, len(sents))]

    def _make(self, doc: Document, sents, lo: int, hi: int, ordinal: int, cohesion: float) -> Chunk:
        start, end = sents[lo].start, sents[hi - 1].end
        text = doc.text[start:end].strip()
        return Chunk(
            chunk_id=stable_id(doc.doc_id, "L1", ordinal, start),
            doc_id=doc.doc_id,
            layer=self.layer,
            text=text,
            embed_text=text,
            lang=doc.lang,
            char_start=start,
            char_end=end,
            sent_span=(lo, hi),
            cohesion=cohesion,
            meta={"chunker": self.name, "ordinal": ordinal, "n_sentences": hi - lo},
        )


__all__ = ["SemanticChunker"]
