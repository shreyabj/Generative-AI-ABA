"""Document segmentation: long documents become sections before the lattice runs.

The chunk lattice was designed around MS MARCO passages (~300 characters), where its
L3 "document" node is a meaningful whole-passage embedding. The encoder, however, reads
at most 128 tokens. For a 60-page annual report a single document vector would be the
embedding of its first paragraph, and the semantic and window layers would be computed
over a sentence table of thousands of rows.

So long documents are cut into sections first - on headings, then on page breaks, then
on paragraph boundaries - each between roughly ``section_target_chars`` and
``section_max_chars``. Every section is then a lattice ``Document`` in its own right:
L3 means "section" for a long document and "whole document" for a short one, and the
lattice's existing parent links still give small-to-search, big-to-read.

Every section records its heading and the page span it covers (``page_map``: page
numbers with character offsets *relative to the section*), which is how a chunk found
much later reports its page without the lattice knowing what a page is.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chunking.base import Document
from ..config import settings
from ..text import detect_lang, split_sentences
from .extract import Extracted


@dataclass
class Section:
    index: int
    start: int
    end: int
    heading: str
    page_map: list[list[int]] = field(default_factory=list)  # [page, start, end] relative

    @property
    def page_start(self) -> int | None:
        return self.page_map[0][0] if self.page_map else None

    @property
    def page_end(self) -> int | None:
        return self.page_map[-1][0] if self.page_map else None


def _paragraphs(text: str) -> list[tuple[int, int]]:
    out, pos = [], 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out


def _split_long(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Split one oversized paragraph at sentence boundaries."""
    sents = split_sentences(text[start:end])
    if not sents:
        return [(start, end)]
    out, cur = [], start
    for s in sents:
        s_end = start + s.end
        if s_end - cur > limit and start + s.start > cur:
            out.append((cur, start + s.start))
            cur = start + s.start
    out.append((cur, end))
    return out


def segment(ex: Extracted, *, target: int | None = None, maximum: int | None = None) -> list[Section]:
    target = target or settings.section_target_chars
    maximum = maximum or settings.section_max_chars
    text = ex.text
    if len(text) <= maximum:
        return [_finish(Section(0, 0, len(text), ex.headings[0][0] if ex.headings else ""), ex)]

    heading_at = {pos: h for h, pos in ex.headings}
    page_starts = {start for _, start, _ in ex.pages}

    units: list[tuple[int, int]] = []
    for a, b in _paragraphs(text):
        units.extend(_split_long(text, a, b, maximum) if b - a > maximum else [(a, b)])

    sections: list[Section] = []
    cur_start: int | None = None
    cur_end = 0
    current_heading = ""
    section_heading = ""

    def close() -> None:
        nonlocal cur_start
        if cur_start is not None and cur_end > cur_start:
            sections.append(Section(len(sections), cur_start, cur_end, section_heading))
        cur_start = None

    for a, b in units:
        is_heading = a in heading_at
        if is_heading:
            current_heading = heading_at[a]
        size = (cur_end - cur_start) if cur_start is not None else 0
        boundary = (
            cur_start is not None
            and (
                (is_heading and size >= target * 0.35)
                or size + (b - a) > maximum
                or (size >= target and (a in page_starts or is_heading))
            )
        )
        if boundary:
            close()
        if cur_start is None:
            cur_start = a
            section_heading = current_heading
        cur_end = b
    close()
    return [_finish(s, ex) for s in sections]


def _finish(sec: Section, ex: Extracted) -> Section:
    sec.page_map = [
        [p, max(s, sec.start) - sec.start, min(e, sec.end) - sec.start]
        for p, s, e in ex.pages if e > sec.start and s < sec.end
    ]
    if not sec.heading:
        # Most recent heading before the section start, if any.
        prior = [h for h, pos in ex.headings if pos <= sec.start]
        sec.heading = prior[-1] if prior else ""
    return sec


def to_documents(ex: Extracted, sections: list[Section], *, document_id: str, collection_id: str,
                 base_meta: dict) -> list[Document]:
    """Lattice documents for each section, carrying full provenance in ``meta``."""
    docs: list[Document] = []
    doc_title = base_meta.get("doc_title") or ex.title or base_meta.get("filename") or document_id
    for sec in sections:
        body = ex.text[sec.start:sec.end].strip()
        if len(body) < 20:
            continue
        lead = len(ex.text[sec.start:sec.end]) - len(ex.text[sec.start:sec.end].lstrip())
        page_map = [[p, max(0, s - lead), max(0, e - lead)] for p, s, e in sec.page_map]
        title = f"{doc_title} — {sec.heading}" if sec.heading and sec.heading != doc_title else doc_title
        docs.append(Document(
            doc_id=f"{document_id}#s{sec.index:03d}",
            text=body,
            lang=detect_lang(body),
            meta={
                **base_meta,
                "document_id": document_id,
                "collection_id": collection_id,
                "doc_title": doc_title,
                "title": title,
                "section": sec.heading or None,
                "section_index": sec.index,
                "page_start": sec.page_start,
                "page_end": sec.page_end,
                "page_map": page_map,
            },
        ))
    return docs


__all__ = ["Section", "segment", "to_documents"]
