"""L0 - Atomic / proposition chunker.

The finest layer of the lattice. One chunk is one assertion, which is what makes
extractive answering and claim-level groundedness verification possible: if the
retriever hands back an atom, that atom *is* the answer span, so there is nothing
left to hallucinate.

Two behaviours stop this layer from degenerating into useless fragments:

* **Forward accretion.** Sentences shorter than ``atomic_min_chars`` are glued to
  the next sentence rather than indexed alone. "Yes." is not a retrievable unit.
* **Clause fission.** Sentences longer than ``atomic_max_chars`` are split at
  discourse boundaries (semicolons, coordinating conjunctions, Devanagari danda)
  so a 600-character run-on sentence still produces precise atoms.
"""

from __future__ import annotations

import re

from ..config import settings
from ..harness.contracts import Layer
from ..text import split_sentences
from .base import Chunk, Document, stable_id

# Clause boundaries that survive machine translation reasonably well.
_CLAUSE_RE = re.compile(
    r"(?<=[;:])\s+"
    r"|\s+(?:however|therefore|whereas|although|because|moreover|furthermore|meanwhile|"
    r"instead|nevertheless|additionally|consequently)\s+"
    r"|\s+(?:लेकिन|परंतु|क्योंकि|इसलिए|तथापि|जबकि)\s+"
    r"|\s+(?:কিন্তু|কারণ|তাই|যদিও)\s+"
    r"|\s+(?:ஆனால்|எனவே|ஏனெனில்|இருப்பினும்)\s+",
    re.IGNORECASE | re.UNICODE,
)


class AtomicChunker:
    layer = Layer.ATOMIC
    name = "atomic-proposition"

    def __init__(self, min_chars: int | None = None, max_chars: int | None = None) -> None:
        self.min_chars = min_chars or settings.atomic_min_chars
        self.max_chars = max_chars or settings.atomic_max_chars

    def split(self, doc: Document) -> list[Chunk]:
        sents = doc.sentences or split_sentences(doc.text)
        if not sents:
            return []

        # --- pass 1: accrete undersized sentences forwards -------------------
        groups: list[tuple[int, int]] = []  # [first, last) sentence indices
        i = 0
        while i < len(sents):
            j = i + 1
            length = sents[i].n_chars
            while j < len(sents) and length < self.min_chars:
                length += sents[j].n_chars
                j += 1
            groups.append((i, j))
            i = j
        # A trailing group that is still too small merges backwards.
        if len(groups) > 1:
            lo, hi = groups[-1]
            if sum(sents[k].n_chars for k in range(lo, hi)) < self.min_chars:
                plo, _ = groups[-2]
                groups[-2:] = [(plo, hi)]

        # --- pass 2: emit, fissioning oversized groups ----------------------
        chunks: list[Chunk] = []
        for lo, hi in groups:
            start = sents[lo].start
            end = sents[hi - 1].end
            text = doc.text[start:end].strip()
            if not text:
                continue
            if len(text) <= self.max_chars:
                chunks.append(self._make(doc, text, start, end, (lo, hi), len(chunks)))
                continue
            for piece_text, piece_start in self._fission(text, start):
                chunks.append(
                    self._make(
                        doc, piece_text, piece_start, piece_start + len(piece_text),
                        (lo, hi), len(chunks), fissioned=True,
                    )
                )
        return chunks

    # ---------------------------------------------------------------- internals
    def _fission(self, text: str, base: int) -> list[tuple[str, int]]:
        parts: list[tuple[str, int]] = []
        cursor = 0
        for seg in _CLAUSE_RE.split(text):
            if not seg:
                continue
            idx = text.find(seg, cursor)
            if idx < 0:
                idx = cursor
            seg_stripped = seg.strip()
            if seg_stripped:
                offset = idx + (len(seg) - len(seg.lstrip()))
                parts.append((seg_stripped, base + offset))
            cursor = idx + len(seg)

        if not parts:
            parts = [(text, base)]

        # Hard-cap anything the discourse split could not tame.
        capped: list[tuple[str, int]] = []
        for seg_text, seg_start in parts:
            while len(seg_text) > self.max_chars:
                cut = seg_text.rfind(" ", 0, self.max_chars)
                if cut <= 0:
                    cut = self.max_chars
                capped.append((seg_text[:cut].strip(), seg_start))
                seg_start += cut
                seg_text = seg_text[cut:].lstrip()
            if seg_text:
                capped.append((seg_text, seg_start))

        # Re-merge fragments that fission made too small.
        out: list[tuple[str, int]] = []
        for seg_text, seg_start in capped:
            if out and len(seg_text) < self.min_chars:
                prev_text, prev_start = out[-1]
                if len(prev_text) + len(seg_text) <= self.max_chars:
                    out[-1] = (f"{prev_text} {seg_text}", prev_start)
                    continue
            out.append((seg_text, seg_start))
        return out

    def _make(
        self,
        doc: Document,
        text: str,
        start: int,
        end: int,
        sent_span: tuple[int, int],
        ordinal: int,
        *,
        fissioned: bool = False,
    ) -> Chunk:
        return Chunk(
            chunk_id=stable_id(doc.doc_id, "L0", ordinal, start),
            doc_id=doc.doc_id,
            layer=self.layer,
            text=text,
            embed_text=text,
            lang=doc.lang,
            char_start=start,
            char_end=end,
            sent_span=sent_span,
            meta={"chunker": self.name, "ordinal": ordinal, "fissioned": fissioned},
        )


__all__ = ["AtomicChunker"]
