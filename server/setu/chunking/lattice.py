"""The Polymorphic Chunk Lattice.

A single document is not split *one* way. It is split four ways at once, and the
resulting nodes are wired into a containment hierarchy:

        L3  DOCUMENT   [hi | NUMERIC | rank 1/10] whole passage
             |
        L1  SEMANTIC   drift-bounded coherence blocks   <- L2 WINDOW overlays this level
             |
        L0  ATOMIC     one proposition per node

Why four layers instead of one good one
---------------------------------------
Different questions need different granularity, and you cannot know which until
the question arrives. "How many calories in a banana" wants an atom. "Explain the
causes of the 1943 Bengal famine" wants a coherence block. "What is BGP" wants a
metadata-tagged document. A single chunk size is a bet that all queries look the
same; the lattice refuses that bet and lets :mod:`setu.retrieval.router` choose at
query time, per query, in well under a millisecond.

Parent links make precision and context compatible
--------------------------------------------------
Retrieval runs on the fine layers, where signal is concentrated and scores are not
diluted by surrounding prose. Whatever wins is then *expanded* through ``parent_id``
to its containing coherence block, so the answer composer reads real context instead
of an orphaned fragment. Small-to-search, big-to-read.

Deduplication
-------------
Short passages would otherwise produce four identical nodes. Layers are emitted
coarse to fine and a node is dropped when a coarser node already carries
byte-identical text, so the lattice degenerates gracefully to a single document node
for one-line passages. ``L3`` is always retained, which guarantees the router has a
backstop layer for every document.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from ..embed import Encoder, get_encoder
from ..harness.contracts import Layer
from ..text import split_sentences
from .atomic import AtomicChunker
from .base import Chunk, Document, LatticeStats
from .metadata import DocumentChunker
from .semantic import SemanticChunker
from .window import AdaptiveWindowChunker

# Coarse -> fine. Dedup and parent resolution both rely on this order.
LAYER_ORDER: tuple[Layer, ...] = (Layer.DOCUMENT, Layer.SEMANTIC, Layer.WINDOW, Layer.ATOMIC)


def _dedup_and_link(layers: dict[Layer, list[Chunk]]) -> tuple[list[Chunk], dict[str, int]]:
    """Drop byte-identical finer nodes, then wire ``parent_id`` by containment."""
    kept: dict[Layer, list[Chunk]] = {}
    seen_text: set[str] = set()
    dropped: dict[str, int] = {}

    for layer in LAYER_ORDER:
        out: list[Chunk] = []
        for ch in layers.get(layer, []):
            if layer is not Layer.DOCUMENT and ch.text in seen_text:
                dropped[layer.value] = dropped.get(layer.value, 0) + 1
                continue
            seen_text.add(ch.text)
            out.append(ch)
        kept[layer] = out

    doc_nodes = kept.get(Layer.DOCUMENT, [])
    doc_parent = doc_nodes[0].chunk_id if doc_nodes else None
    semantic = sorted(kept.get(Layer.SEMANTIC, []), key=lambda c: c.char_start)

    def enclosing_semantic(ch: Chunk) -> str | None:
        """Semantic block with the largest character overlap with ``ch``."""
        best_id, best_ov = None, 0
        for s in semantic:
            ov = min(ch.char_end, s.char_end) - max(ch.char_start, s.char_start)
            if ov > best_ov:
                best_id, best_ov = s.chunk_id, ov
        return best_id

    for ch in kept.get(Layer.SEMANTIC, []):
        ch.parent_id = doc_parent
    for ch in kept.get(Layer.WINDOW, []):
        ch.parent_id = doc_parent  # windows straddle semantic blocks by design
    for ch in kept.get(Layer.ATOMIC, []):
        ch.parent_id = enclosing_semantic(ch) or doc_parent

    return [ch for layer in LAYER_ORDER for ch in kept.get(layer, [])], dropped


class ChunkLattice:
    """Builds the multi-view chunk set for a corpus."""

    def __init__(self, encoder: Encoder | None = None, layers: set[str] | None = None) -> None:
        from ..config import settings

        self.encoder = encoder or get_encoder()
        enabled = layers if layers is not None else settings.layer_set
        all_chunkers = (
            DocumentChunker(),
            SemanticChunker(),
            AdaptiveWindowChunker(),
            AtomicChunker(),
        )
        self.chunkers = tuple(c for c in all_chunkers if c.layer.value in enabled) or all_chunkers
        self.stats = LatticeStats()

    # ------------------------------------------------------------------- prepare
    def prepare(self, docs: Sequence[Document]) -> None:
        """Sentence-split and sentence-embed a batch of documents in one pass.

        Sentence vectors are computed exactly once and shared by the semantic
        chunker (drift detection), the window chunker (cohesion measurement) and
        the answer composer (span scoring at query time). This is the main reason
        building four views is not four times the cost of building one.
        """
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for doc in docs:
            if doc.sentences is None:
                doc.sentences = split_sentences(doc.text)
            start = len(flat)
            flat.extend(s.text for s in doc.sentences)
            spans.append((start, len(flat)))

        if not flat:
            return
        vecs = self.encoder.encode(flat)
        for doc, (lo, hi) in zip(docs, spans, strict=True):
            doc.sent_vecs = vecs[lo:hi] if hi > lo else None

    # --------------------------------------------------------------------- build
    def build_one(self, doc: Document) -> list[Chunk]:
        if doc.sentences is None:
            doc.sentences = split_sentences(doc.text)
        layers: dict[Layer, list[Chunk]] = {}
        for chunker in self.chunkers:
            try:
                layers[chunker.layer] = chunker.split(doc)
            except Exception:  # noqa: BLE001 - one bad layer must not lose the document
                layers[chunker.layer] = []
        chunks, dropped = _dedup_and_link(layers)
        for layer, n in dropped.items():
            self.stats.deduped[layer] = self.stats.deduped.get(layer, 0) + n
        return chunks

    def build(self, docs: Sequence[Document], *, batch: int = 256) -> Iterable[list[Chunk]]:
        """Yield the chunk list for each document, preparing in batches."""
        for i in range(0, len(docs), batch):
            group = list(docs[i : i + batch])
            self.prepare(group)
            for doc in group:
                chunks = self.build_one(doc)
                self._accumulate(doc, chunks)
                yield chunks

    def build_all(self, docs: Sequence[Document], *, batch: int = 256) -> list[Chunk]:
        out: list[Chunk] = []
        for chunks in self.build(docs, batch=batch):
            out.extend(chunks)
        return out

    # --------------------------------------------------------------------- stats
    def _accumulate(self, doc: Document, chunks: list[Chunk]) -> None:
        st = self.stats
        st.documents += 1
        st.chunks += len(chunks)
        st.per_lang[doc.lang] = st.per_lang.get(doc.lang, 0) + 1
        n_layers = len({ch.layer for ch in chunks})
        st.layers_per_doc[n_layers] = st.layers_per_doc.get(n_layers, 0) + 1
        for ch in chunks:
            key = ch.layer.value
            st.per_layer[key] = st.per_layer.get(key, 0) + 1
            n = st.per_layer[key]
            st.avg_chars[key] = st.avg_chars.get(key, 0.0) + (ch.n_chars - st.avg_chars.get(key, 0.0)) / n
            st.avg_overlap[key] = (
                st.avg_overlap.get(key, 0.0) + (ch.overlap_ratio - st.avg_overlap.get(key, 0.0)) / n
            )
            if ch.parent_id:
                st.parent_links += 1


__all__ = ["LAYER_ORDER", "ChunkLattice"]
