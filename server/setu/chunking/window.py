"""L2 - Sliding window chunker with *cohesion-adaptive* overlap.

A fixed window layer is worth keeping: it is the only view that guarantees no
region of the corpus is under-represented, and it is robust when sentence
splitting fails (which happens on badly punctuated machine translation).

What is not worth keeping is a fixed overlap. Constant 20 % overlap over-indexes
tightly-written passages (wasting index space and diluting scores with
near-duplicates) and under-protects rambling ones (where the answer straddles a
boundary). So overlap here is a function of local cohesion:

    overlap = lerp(overlap_max, overlap_min, cohesion)

High cohesion (neighbouring sentences are semantically close) means a cut is cheap
-> small overlap. Low cohesion means a cut is expensive -> large overlap. Every
chunk records the ``overlap_ratio`` actually used, so the effect is auditable in
the UI rather than a claim in a README.

Windows always snap to sentence boundaries; the character budget picks *which*
boundary, it never cuts mid-sentence.
"""

from __future__ import annotations

import numpy as np

from ..config import settings
from ..harness.contracts import Layer
from ..text import split_sentences
from .base import Chunk, Document, adjacent_cohesion, stable_id


class AdaptiveWindowChunker:
    layer = Layer.WINDOW
    name = "window-adaptive-overlap"

    def __init__(
        self,
        window_chars: int | None = None,
        overlap_min: float | None = None,
        overlap_max: float | None = None,
    ) -> None:
        self.window_chars = window_chars or settings.window_chars
        self.overlap_min = settings.window_overlap_min if overlap_min is None else overlap_min
        self.overlap_max = settings.window_overlap_max if overlap_max is None else overlap_max

    def split(self, doc: Document) -> list[Chunk]:
        sents = doc.sentences or split_sentences(doc.text)
        if not sents:
            return []

        vecs = doc.sent_vecs
        total_chars = sents[-1].end - sents[0].start
        if total_chars <= self.window_chars:
            coh = adjacent_cohesion(vecs)
            return [self._make(doc, sents, 0, len(sents), 0, 0.0, coh)]

        chunks: list[Chunk] = []
        lo = 0
        ordinal = 0
        guard = 0
        while lo < len(sents) and guard < 4096:
            guard += 1
            hi = lo + 1
            while hi < len(sents) and (sents[hi - 1].end - sents[lo].start) < self.window_chars:
                hi += 1

            local_vecs = vecs[lo:hi] if vecs is not None else None
            cohesion = adjacent_cohesion(local_vecs)
            ratio = self._overlap_for(cohesion)

            chunks.append(self._make(doc, sents, lo, hi, ordinal, ratio, cohesion))
            ordinal += 1

            if hi >= len(sents):
                break

            # Step back by `ratio` of the window, measured in characters, then
            # snap to the nearest sentence start inside the window.
            win_start, win_end = sents[lo].start, sents[hi - 1].end
            target = win_end - int((win_end - win_start) * ratio)
            next_lo = hi
            for k in range(lo + 1, hi):
                if sents[k].start >= target:
                    next_lo = k
                    break
            lo = max(next_lo, lo + 1)
        return chunks

    # ---------------------------------------------------------------- internals
    def _overlap_for(self, cohesion: float) -> float:
        """Map cohesion in [0, 1] to an overlap ratio, inverted and clamped."""
        c = float(np.clip((cohesion + 1.0) / 2.0, 0.0, 1.0))  # cosine can be negative
        return float(self.overlap_max + (self.overlap_min - self.overlap_max) * c)

    def _make(
        self, doc: Document, sents, lo: int, hi: int, ordinal: int, ratio: float, cohesion: float
    ) -> Chunk:
        start, end = sents[lo].start, sents[hi - 1].end
        text = doc.text[start:end].strip()
        return Chunk(
            chunk_id=stable_id(doc.doc_id, "L2", ordinal, start),
            doc_id=doc.doc_id,
            layer=self.layer,
            text=text,
            embed_text=text,
            lang=doc.lang,
            char_start=start,
            char_end=end,
            sent_span=(lo, hi),
            overlap_ratio=ratio,
            cohesion=cohesion,
            meta={"chunker": self.name, "ordinal": ordinal, "window_chars": self.window_chars},
        )


__all__ = ["AdaptiveWindowChunker"]
