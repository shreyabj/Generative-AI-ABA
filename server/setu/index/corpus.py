"""The corpus index - one loadable artifact holding every structure the serve path needs.

Contents
--------
chunks        Row-aligned chunk table. Row ``i`` is the identity used by the vector
              store, the BM25 matrix, the layer masks and the reranker. No id
              translation anywhere on the hot path.
vectors       (n_chunks, dim) L2-normalised float32 chunk embeddings.
sent_vectors  (n_sentences, dim) sentence embeddings for the whole corpus, plus a
              per-document span. These power late-interaction reranking and
              extractive span selection at query time *without* re-embedding
              anything, which is what keeps the reflex path in single-digit ms.
layer_masks   Precomputed boolean masks per lattice layer, so a layer-restricted
              search is a view, not a filter loop.
bm25          Sparse inverted index over the same rows.

Everything is memory-mapped-friendly numpy plus JSONL, so a cold start is a couple
of file reads rather than a re-embedding run.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..chunking.base import Chunk, Document
from ..config import settings
from ..embed import Encoder, get_encoder
from ..harness.contracts import Layer
from ..text import Sentence, split_sentences
from .bm25 import BM25Index
from .store import FlatStore

LAYERS: tuple[Layer, ...] = (Layer.ATOMIC, Layer.SEMANTIC, Layer.WINDOW, Layer.DOCUMENT)


@dataclass(slots=True)
class ChunkRecord:
    """Flattened chunk row (what the API returns and the composer reads)."""

    row: int
    chunk_id: str
    doc_id: str
    layer: Layer
    text: str
    lang: str
    char_start: int
    char_end: int
    parent_id: str | None
    sent_span: tuple[int, int]
    overlap_ratio: float
    cohesion: float
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_chunk(cls, row: int, ch: Chunk) -> ChunkRecord:
        return cls(
            row=row,
            chunk_id=ch.chunk_id,
            doc_id=ch.doc_id,
            layer=ch.layer,
            text=ch.text,
            lang=ch.lang,
            char_start=ch.char_start,
            char_end=ch.char_end,
            parent_id=ch.parent_id,
            sent_span=tuple(ch.sent_span),  # type: ignore[arg-type]
            overlap_ratio=ch.overlap_ratio,
            cohesion=ch.cohesion,
            meta=dict(ch.meta),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "layer": self.layer.value,
            "text": self.text,
            "lang": self.lang,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "parent_id": self.parent_id,
            "sent_span": list(self.sent_span),
            "overlap_ratio": self.overlap_ratio,
            "cohesion": self.cohesion,
            "meta": self.meta,
        }

    @classmethod
    def from_json(cls, row: int, d: dict[str, Any]) -> ChunkRecord:
        return cls(
            row=row,
            chunk_id=d["chunk_id"],
            doc_id=d["doc_id"],
            layer=Layer(d["layer"]),
            text=d["text"],
            lang=d["lang"],
            char_start=d["char_start"],
            char_end=d["char_end"],
            parent_id=d.get("parent_id"),
            sent_span=tuple(d.get("sent_span") or (0, 0)),  # type: ignore[arg-type]
            overlap_ratio=d.get("overlap_ratio", 0.0),
            cohesion=d.get("cohesion", 0.0),
            meta=d.get("meta") or {},
        )


@dataclass(slots=True)
class DocRecord:
    doc_id: str
    text: str
    lang: str
    meta: dict[str, Any]
    sentences: list[Sentence]
    sent_offset: int = 0
    sent_count: int = 0


class CorpusIndex:
    """Owns every retrieval structure and knows how to save/load itself."""

    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim or settings.embed_dim
        self.chunks: list[ChunkRecord] = []
        self.docs: dict[str, DocRecord] = {}
        self.vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self.sent_vectors: np.ndarray = np.zeros((0, self.dim), dtype=np.float32)
        self.cohesion: np.ndarray = np.zeros(0, dtype=np.float32)
        self.layer_masks: dict[Layer, np.ndarray] = {}
        self.bm25 = BM25Index()
        self.flat = FlatStore(self.dim)
        self.vector_store: Any = self.flat
        self.by_chunk_id: dict[str, int] = {}
        self.by_doc: dict[str, list[int]] = {}
        self._by_group_lang: dict[tuple[str, str], list[str]] | None = None
        self.manifest: dict[str, Any] = {}
        self.backend_error: str | None = None

    # ==================================================================== build
    def build(
        self,
        docs: Sequence[Document],
        chunk_groups: Iterable[list[Chunk]],
        encoder: Encoder | None = None,
        *,
        progress: Any | None = None,
    ) -> CorpusIndex:
        encoder = encoder or get_encoder()

        # ---- documents + corpus-wide sentence vector table -------------------
        sent_blocks: list[np.ndarray] = []
        cursor = 0
        for doc in docs:
            sents = doc.sentences or split_sentences(doc.text)
            rec = DocRecord(
                doc_id=doc.doc_id, text=doc.text, lang=doc.lang, meta=dict(doc.meta),
                sentences=list(sents), sent_offset=cursor, sent_count=len(sents),
            )
            self.docs[doc.doc_id] = rec
            cursor += len(sents)
            if doc.sent_vecs is not None and len(doc.sent_vecs) == len(sents):
                sent_blocks.append(np.asarray(doc.sent_vecs, dtype=np.float32))
            else:
                vecs = encoder.encode([s.text for s in sents]) if sents else np.zeros((0, self.dim), np.float32)
                sent_blocks.append(vecs)
        self.sent_vectors = (
            np.vstack(sent_blocks) if sent_blocks else np.zeros((0, self.dim), dtype=np.float32)
        )

        # ---- chunk table ------------------------------------------------------
        texts_to_embed: list[str] = []
        for group in chunk_groups:
            for ch in group:
                row = len(self.chunks)
                rec = ChunkRecord.from_chunk(row, ch)
                self.chunks.append(rec)
                self.by_chunk_id[rec.chunk_id] = row
                self.by_doc.setdefault(rec.doc_id, []).append(row)
                texts_to_embed.append(ch.embed_text)

        n = len(self.chunks)
        if n == 0:
            raise ValueError("no chunks produced - is the corpus empty?")

        # Embed in reported batches. A single blocking encode() call over tens of
        # thousands of chunks is several silent minutes, which reads as a hang.
        blocks: list[np.ndarray] = []
        step = max(settings.embed_batch * 8, 512)
        for start in range(0, n, step):
            blocks.append(encoder.encode(texts_to_embed[start : start + step]))
            if progress:
                progress(min(start + step, n))
        self.vectors = np.vstack(blocks) if len(blocks) > 1 else blocks[0]
        self.dim = int(self.vectors.shape[1])
        self.cohesion = np.asarray([c.cohesion for c in self.chunks], dtype=np.float32)
        self._build_masks()

        self.bm25 = BM25Index().fit(texts_to_embed)
        self.flat = FlatStore(self.dim)
        self.flat.set_matrix(self.vectors)
        self.vector_store = self.flat

        self.manifest = {
            "setu_version": __version__,
            "embed_model": encoder.model_name,
            "embed_backend": encoder.backend,
            "dim": self.dim,
            "n_docs": len(self.docs),
            "n_chunks": n,
            "n_sentences": int(self.sent_vectors.shape[0]),
            "per_layer": {lay.value: int(mask.sum()) for lay, mask in self.layer_masks.items()},
            "per_lang": self._lang_counts(),
            "chunking": {
                "atomic_min_chars": settings.atomic_min_chars,
                "atomic_max_chars": settings.atomic_max_chars,
                "semantic_max_chars": settings.semantic_max_chars,
                "semantic_z": settings.semantic_z,
                "window_chars": settings.window_chars,
                "window_overlap": [settings.window_overlap_min, settings.window_overlap_max],
                "metadata_prefix": settings.metadata_prefix,
            },
        }
        return self

    def _build_masks(self) -> None:
        n = len(self.chunks)
        layer_ids = np.zeros(n, dtype=np.int8)
        order = {lay: i for i, lay in enumerate(LAYERS)}
        for i, c in enumerate(self.chunks):
            layer_ids[i] = order[c.layer]
        self.layer_masks = {lay: (layer_ids == i) for lay, i in order.items()}
        self._layer_ids = layer_ids

    def _lang_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.docs.values():
            out[d.lang] = out.get(d.lang, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    # ============================================================ vector backend
    def fingerprint(self) -> str:
        """Content identity of the row table: changes whenever any chunk or its order does."""
        import hashlib

        h = hashlib.blake2b(digest_size=8)
        for c in self.chunks:
            h.update(c.chunk_id.encode("ascii", "ignore"))
        h.update(str(self.manifest.get("embed_model", "")).encode())
        return h.hexdigest()

    def attach_vector_backend(self, collection: str | None = None) -> str:
        """Resolve ``SETU_VECTOR_BACKEND`` and mirror vectors into it if needed.

        The Qdrant collection name carries the index fingerprint. The v1 code reused one
        name and skipped the upsert when the point count matched, so a rebuilt index of
        the same size silently kept serving the previous build's vectors. With the
        fingerprint in the name, a different build can never read stale points.
        """
        want = settings.vector_backend.lower()
        self.backend_error = None
        if want == "flat":
            self.vector_store = self.flat
            return self.flat.name
        try:
            from .qdrant_store import QdrantStore, qdrant_available

            if want == "qdrant" or (want == "auto" and qdrant_available()):
                name = f"{collection or settings.qdrant_collection}_{self.fingerprint()}"
                store = QdrantStore(self.dim, collection=name)
                store.ensure_payload_indexes()
                if store.size() != len(self.chunks):
                    store.upsert(
                        list(range(len(self.chunks))),
                        self.vectors,
                        [
                            {"layer": c.layer.value, "lang": c.lang, "doc_id": c.doc_id}
                            for c in self.chunks
                        ],
                    )
                self.vector_store = store
                return store.name
        except Exception as exc:  # noqa: BLE001 - flat store is a correct fallback
            # Recorded, not swallowed: /health reports why Qdrant is not in use.
            self.backend_error = f"{type(exc).__name__}: {exc}"
        self.vector_store = self.flat
        return self.flat.name

    # ==================================================================== access
    def record(self, row: int) -> ChunkRecord:
        return self.chunks[row]

    def row_of(self, chunk_id: str) -> int | None:
        return self.by_chunk_id.get(chunk_id)

    def mask_for(self, layer: Layer) -> np.ndarray | None:
        return self.layer_masks.get(layer)

    def sentence_vectors(self, rec: ChunkRecord) -> np.ndarray:
        """Sentence vectors covered by a chunk (for late-interaction scoring)."""
        doc = self.docs.get(rec.doc_id)
        if doc is None or doc.sent_count == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        lo, hi = rec.sent_span
        lo = max(0, min(lo, doc.sent_count))
        hi = max(lo, min(hi or doc.sent_count, doc.sent_count))
        if hi <= lo:
            lo, hi = 0, doc.sent_count
        return self.sent_vectors[doc.sent_offset + lo : doc.sent_offset + hi]

    def sentences_of(self, rec: ChunkRecord) -> list[tuple[Sentence, np.ndarray]]:
        """Sentences of a chunk paired with their vectors, in document order."""
        doc = self.docs.get(rec.doc_id)
        if doc is None:
            return []
        lo, hi = rec.sent_span
        lo = max(0, min(lo, doc.sent_count))
        hi = max(lo, min(hi or doc.sent_count, doc.sent_count))
        if hi <= lo:
            lo, hi = 0, doc.sent_count
        vecs = self.sent_vectors[doc.sent_offset + lo : doc.sent_offset + hi]
        sents = doc.sentences[lo:hi]
        if len(vecs) != len(sents):
            return [(s, np.zeros(self.dim, dtype=np.float32)) for s in sents]
        return list(zip(sents, vecs, strict=True))

    def parent_of(self, rec: ChunkRecord) -> ChunkRecord | None:
        if not rec.parent_id:
            return None
        row = self.by_chunk_id.get(rec.parent_id)
        return self.chunks[row] if row is not None else None

    def rows_of_doc(self, doc_id: str) -> list[int]:
        """Every chunk row belonging to one document, across all lattice layers."""
        return list(self.by_doc.get(doc_id, ()))

    # ------------------------------------------------------- parallel passages
    #
    # MS MARCO-XI is a parallel corpus: ingest writes each passage once per language under
    # ``{query_id}:{passage_rank}:{lang}``, so the *same* passage in another language is a
    # dictionary lookup away. That is what makes "answer in the language you asked in"
    # achievable without a translation model - retrieval can land wherever it likes, and the
    # answerer can then read the same evidence in the user's own language.
    #
    # Two levels, because the corpus is only near-parallel: ingest drops passages shorter
    # than a minimum per language, so a given rank can be missing in some of them.
    def parallel_doc_id(self, doc_id: str, lang: str) -> str | None:
        """The same passage in ``lang``, or None if this corpus does not have it."""
        parts = doc_id.rsplit(":", 2)
        if len(parts) != 3:
            return None
        group, rank, cur = parts
        if cur == lang:
            return doc_id
        sibling = f"{group}:{rank}:{lang}"
        return sibling if sibling in self.docs else None

    def group_doc_ids(self, doc_id: str, lang: str) -> list[str]:
        """Sibling passages of the same source question, in ``lang``.

        The fallback for when the exact passage rank was not indexed in that language. Every
        passage in a group was retrieved for the same original question, so they are all
        plausible evidence - the span scorer still decides whether any of them is usable.
        """
        parts = doc_id.rsplit(":", 2)
        if len(parts) != 3:
            return []
        if self._by_group_lang is None:
            table: dict[tuple[str, str], list[str]] = {}
            for did, rec in self.docs.items():
                bits = did.rsplit(":", 2)
                if len(bits) == 3:
                    table.setdefault((bits[0], rec.lang), []).append(did)
            self._by_group_lang = table
        return list(self._by_group_lang.get((parts[0], lang), ()))

    def languages(self) -> list[str]:
        return sorted(self._lang_counts())

    def stats(self) -> dict[str, Any]:
        return {
            **self.manifest,
            "vector_backend_error": self.backend_error,
            "vector_store": self.vector_store.stats(),
            "bm25": self.bm25.stats(),
            "avg_chunk_chars": round(
                float(np.mean([len(c.text) for c in self.chunks])) if self.chunks else 0.0, 1
            ),
        }

    # =============================================================== persistence
    def save(self, directory: Path | None = None) -> Path:
        d = Path(directory or settings.index_dir)
        d.mkdir(parents=True, exist_ok=True)

        np.save(d / "vectors.npy", self.vectors)
        np.save(d / "sent_vectors.npy", self.sent_vectors)
        np.save(d / "cohesion.npy", self.cohesion)

        with (d / "chunks.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.chunks:
                fh.write(json.dumps(rec.to_json(), ensure_ascii=False) + "\n")

        with (d / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for rec in self.docs.values():
                fh.write(
                    json.dumps(
                        {
                            "doc_id": rec.doc_id,
                            "text": rec.text,
                            "lang": rec.lang,
                            "meta": rec.meta,
                            "sentences": [[s.start, s.end] for s in rec.sentences],
                            "sent_offset": rec.sent_offset,
                            "sent_count": rec.sent_count,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        self.bm25.save(d)
        (d / "manifest.json").write_text(
            json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return d

    @classmethod
    def load(cls, directory: Path | None = None) -> CorpusIndex:
        d = Path(directory or settings.index_dir)
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        built_with = manifest.get("embed_model")
        if built_with and built_with != settings.embed_model:
            # Vectors from another model live in a different space: every query would
            # silently retrieve garbage. Refuse instead and say how to fix it.
            raise ValueError(
                f"index at {d} was built with '{built_with}' but SETU_EMBED_MODEL is "
                f"'{settings.embed_model}'. Rebuild the index or restore the setting."
            )
        idx = cls(dim=manifest.get("dim", settings.embed_dim))
        idx.manifest = manifest

        idx.vectors = np.load(d / "vectors.npy")
        idx.sent_vectors = np.load(d / "sent_vectors.npy")
        idx.cohesion = np.load(d / "cohesion.npy")
        idx.dim = int(idx.vectors.shape[1])

        with (d / "docs.jsonl").open("r", encoding="utf-8") as fh:
            for line in fh:
                o = json.loads(line)
                text = o["text"]
                sents = [Sentence(text[a:b], a, b) for a, b in o["sentences"]]
                idx.docs[o["doc_id"]] = DocRecord(
                    doc_id=o["doc_id"], text=text, lang=o["lang"], meta=o["meta"],
                    sentences=sents, sent_offset=o["sent_offset"], sent_count=o["sent_count"],
                )

        with (d / "chunks.jsonl").open("r", encoding="utf-8") as fh:
            for row, line in enumerate(fh):
                rec = ChunkRecord.from_json(row, json.loads(line))
                idx.chunks.append(rec)
                idx.by_chunk_id[rec.chunk_id] = row
                idx.by_doc.setdefault(rec.doc_id, []).append(row)

        idx._build_masks()
        idx.bm25 = BM25Index.load(d)
        idx.flat = FlatStore(idx.dim)
        idx.flat.set_matrix(idx.vectors)
        idx.vector_store = idx.flat
        return idx

    @staticmethod
    def exists(directory: Path | None = None) -> bool:
        d = Path(directory or settings.index_dir)
        return all((d / f).exists() for f in ("manifest.json", "chunks.jsonl", "vectors.npy"))


__all__ = ["ChunkRecord", "CorpusIndex", "DocRecord", "LAYERS"]
