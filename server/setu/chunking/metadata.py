"""L3 - Document chunker with metadata-aware embedding.

The coarsest layer: one chunk per passage. Its job is recall on broad,
descriptive questions where the answer is the gestalt of the passage rather than
any single span.

The interesting part is the *embedding text*. MS MARCO-XI gives us structured
context for free - the originating query, its ``query_type`` (DESCRIPTION,
NUMERIC, ENTITY, LOCATION, PERSON), the language pair, whether the human
annotator marked the passage as answer-bearing, and the passage's rank within its
result list. We fold a compact header of that into what the encoder sees:

    [hi | NUMERIC | rank 2/10] <passage text>

Two things this buys us:

* **Disambiguation.** Machine-translated passages about different topics can be
  lexically similar; a query-type token nudges them apart in vector space.
* **Type matching.** A "how many" question embeds closer to a NUMERIC-tagged
  passage than to a DESCRIPTION-tagged one, for free, with no extra model.

The header never appears in ``text``, so citations and the spoken answer stay clean.
Set ``SETU_METADATA_PREFIX=false`` to ablate it - ``scripts/ablate.py`` reports
the retrieval delta.
"""

from __future__ import annotations

from ..config import settings
from ..harness.contracts import Layer
from ..text import split_sentences, truncate
from .base import Chunk, Document, adjacent_cohesion, stable_id


def metadata_header(doc: Document) -> str:
    bits: list[str] = [doc.lang]
    qt = doc.query_type
    if qt and qt != "UNKNOWN":
        bits.append(qt)
    rank = doc.meta.get("passage_rank")
    total = doc.meta.get("passage_count")
    if rank is not None and total:
        bits.append(f"rank {int(rank) + 1}/{int(total)}")
    if doc.meta.get("is_selected"):
        bits.append("answer-bearing")
    title = doc.title
    head = f"[{' | '.join(bits)}]"
    if title:
        head += f" {truncate(title, 90)}"
    return head


def decorate(text: str, doc: Document) -> str:
    if not settings.metadata_prefix:
        return text
    return f"{metadata_header(doc)}\n{text}"


class DocumentChunker:
    layer = Layer.DOCUMENT
    name = "document-metadata"

    def split(self, doc: Document) -> list[Chunk]:
        text = doc.text.strip()
        if not text:
            return []
        sents = doc.sentences or split_sentences(text)
        return [
            Chunk(
                chunk_id=stable_id(doc.doc_id, "L3", 0),
                doc_id=doc.doc_id,
                layer=self.layer,
                text=text,
                embed_text=decorate(text, doc),
                lang=doc.lang,
                char_start=0,
                char_end=len(doc.text),
                sent_span=(0, len(sents)),
                cohesion=adjacent_cohesion(doc.sent_vecs),
                meta={
                    "chunker": self.name,
                    "ordinal": 0,
                    "n_sentences": len(sents),
                    "metadata_prefixed": settings.metadata_prefix,
                    "header": metadata_header(doc),
                },
            )
        ]


__all__ = ["DocumentChunker", "decorate", "metadata_header"]
