"""Index build: documents -> chunk lattice -> vectors + BM25 -> on-disk bundle."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from ..chunking.base import Document
from ..chunking.lattice import ChunkLattice
from ..config import settings
from ..embed import get_encoder
from ..index.corpus import CorpusIndex


def build_index(
    docs: Sequence[Document],
    *,
    out_dir: Path | None = None,
    batch: int = 256,
    log: Callable[[str], None] = print,
) -> tuple[CorpusIndex, dict[str, Any]]:
    if not docs:
        raise ValueError("no documents to index")

    out_dir = Path(out_dir or settings.index_dir)
    encoder = get_encoder()

    log(f"  encoder      {encoder.model_name}")
    t_warm = encoder.warm()
    log(f"  backend      {encoder.backend} (warm in {t_warm:.0f} ms, dim {encoder.dim})")

    lattice = ChunkLattice(encoder)
    t0 = time.perf_counter()
    groups: list[list] = []
    done = 0
    for chunks in lattice.build(docs, batch=batch):
        groups.append(chunks)
        done += 1
        if done % 250 == 0:
            log(f"  lattice      {done}/{len(docs)} docs -> {lattice.stats.chunks} chunks")
    t_lattice = time.perf_counter() - t0
    log(f"  lattice      {len(docs)} docs -> {lattice.stats.chunks} chunks in {t_lattice:.1f}s")
    for layer, count in sorted(lattice.stats.per_layer.items()):
        avg = lattice.stats.avg_chars.get(layer, 0.0)
        ov = lattice.stats.avg_overlap.get(layer, 0.0)
        extra = f", avg overlap {ov * 100:.0f}%" if ov > 0.001 else ""
        log(f"    {layer:<9} {count:>7} chunks  (avg {avg:.0f} chars{extra})")

    t0 = time.perf_counter()
    total_chunks = lattice.stats.chunks

    def on_embed(done: int) -> None:
        elapsed = time.perf_counter() - t0
        rate = done / max(elapsed, 0.001)
        eta = (total_chunks - done) / max(rate, 1e-6)
        log(f"  embedding    {done}/{total_chunks} chunks  "
            f"({rate:.0f}/s, eta {eta:.0f}s)")

    index = CorpusIndex().build(docs, groups, encoder, progress=on_embed)
    t_embed = time.perf_counter() - t0
    log(f"  embedded     {len(index.chunks)} chunks in {t_embed:.1f}s "
        f"({len(index.chunks) / max(t_embed, 0.001):.0f} chunks/s)")
    log(f"  bm25         {index.bm25.stats()['vocabulary']} terms, "
        f"{index.bm25.stats()['postings']} postings")

    index.save(out_dir)
    log(f"  saved        {out_dir}")

    report = {
        "documents": len(docs),
        "chunks": len(index.chunks),
        "sentences": int(index.sent_vectors.shape[0]),
        "lattice": lattice.stats.to_dict(),
        "lattice_seconds": round(t_lattice, 2),
        "embed_seconds": round(t_embed, 2),
        "index_dir": str(out_dir),
        **index.manifest,
    }
    return index, report


def load_or_build(docs_path: Path | None = None, *, log: Callable[[str], None] = print) -> CorpusIndex:
    """Load a saved index, else build one from the corpus JSONL, else from the sample."""
    if CorpusIndex.exists():
        log(f"loading index from {settings.index_dir}")
        return CorpusIndex.load()

    from .msmarco_xi import read_corpus

    path = Path(docs_path) if docs_path else settings.corpus_file
    if not path.exists():
        path = settings.sample_corpus_file
        log(f"no ingested corpus found - falling back to sample: {path}")
    if not path.exists():
        raise FileNotFoundError(
            "No corpus available. Run:  python -m setu.cli ingest --langs hi,ta --rows 300"
        )
    docs = read_corpus(path)
    log(f"building index from {len(docs)} documents in {path}")
    index, _ = build_index(docs, log=log)
    return index


__all__ = ["build_index", "load_or_build"]
