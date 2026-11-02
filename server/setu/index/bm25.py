"""Sparse retrieval: BM25 over a real inverted index.

Why not ``rank_bm25``
---------------------
The popular pure-Python implementation scores *every* document for *every* query
by looping in Python. On a 60k-chunk lattice that is roughly 40-80 ms - a third
of the entire latency budget spent on the cheap half of a hybrid retriever.

This implementation stores the corpus as a CSC sparse matrix of precomputed BM25
term weights. A query touches only the posting lists of its own terms, so cost
scales with query length instead of corpus size: a 6-token query is a handful of
sparse column slices plus one ``np.add.at``, which measures in the low
single-digit milliseconds on a 60k x 90k matrix.

Precomputing the weights is valid because the BM25 document-side factor

    w(t, d) = idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * |d| / avgdl))

depends only on the document, never on the query. The query side of standard
BM25 (with default ``k3 -> inf``) is just the term count, which we apply as a
multiplier at query time.

Lexical matching earns its place in a multilingual RAG system: dense encoders
routinely miss exact identifiers - drug names, model numbers, transliterated
proper nouns, years - which is precisely what BM25 is good at.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..text import tokenize


class BM25Index:
    """Precomputed-weight BM25 with posting-list-only query cost."""

    def __init__(self, k1: float = 1.35, b: float = 0.72) -> None:
        self.k1 = k1
        self.b = b
        self.vocab: dict[str, int] = {}
        self.matrix: sp.csc_matrix | None = None  # (n_docs, n_terms) BM25 weights
        self.doc_len: np.ndarray = np.zeros(0, dtype=np.float32)
        self.avgdl: float = 0.0
        self.n_docs: int = 0

    # ---------------------------------------------------------------- building
    def fit(self, texts: Sequence[str]) -> BM25Index:
        self.n_docs = len(texts)
        if self.n_docs == 0:
            self.matrix = sp.csc_matrix((0, 0), dtype=np.float32)
            return self

        rows: list[int] = []
        cols: list[int] = []
        tfs: list[float] = []
        doc_len = np.zeros(self.n_docs, dtype=np.float32)
        df = Counter()

        for i, text in enumerate(texts):
            toks = tokenize(text)
            doc_len[i] = len(toks)
            counts = Counter(toks)
            for term, tf in counts.items():
                tid = self.vocab.get(term)
                if tid is None:
                    tid = len(self.vocab)
                    self.vocab[term] = tid
                rows.append(i)
                cols.append(tid)
                tfs.append(float(tf))
                df[tid] += 1

        n_terms = len(self.vocab)
        self.doc_len = doc_len
        self.avgdl = float(doc_len.mean()) if self.n_docs else 0.0

        tf_arr = np.asarray(tfs, dtype=np.float32)
        row_arr = np.asarray(rows, dtype=np.int32)
        col_arr = np.asarray(cols, dtype=np.int32)

        # Lucene-style probabilistic idf, guaranteed positive.
        df_arr = np.zeros(n_terms, dtype=np.float32)
        for tid, count in df.items():
            df_arr[tid] = count
        idf = np.log(1.0 + (self.n_docs - df_arr + 0.5) / (df_arr + 0.5)).astype(np.float32)

        denom_norm = (1.0 - self.b) + self.b * (doc_len[row_arr] / (self.avgdl or 1.0))
        weights = idf[col_arr] * (tf_arr * (self.k1 + 1.0)) / (tf_arr + self.k1 * denom_norm)

        coo = sp.coo_matrix(
            (weights.astype(np.float32), (row_arr, col_arr)),
            shape=(self.n_docs, n_terms),
            dtype=np.float32,
        )
        self.matrix = coo.tocsc()
        return self

    # ----------------------------------------------------------------- querying
    def search(
        self,
        query: str,
        k: int = 40,
        *,
        candidate_mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(indices, scores)`` for the top ``k`` documents.

        ``candidate_mask`` is an optional boolean array over documents used to
        restrict scoring to one lattice layer or language without a second index.
        """
        if self.matrix is None or self.n_docs == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        counts = Counter(t for t in tokenize(query) if t in self.vocab)
        if not counts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        scores = np.zeros(self.n_docs, dtype=np.float32)
        indptr, indices, data = self.matrix.indptr, self.matrix.indices, self.matrix.data
        for term, qtf in counts.items():
            tid = self.vocab[term]
            lo, hi = indptr[tid], indptr[tid + 1]
            if lo == hi:
                continue
            np.add.at(scores, indices[lo:hi], data[lo:hi] * float(qtf))

        if candidate_mask is not None:
            scores = np.where(candidate_mask, scores, 0.0)

        nz = np.flatnonzero(scores)
        if nz.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        if nz.size > k:
            part = np.argpartition(-scores[nz], k - 1)[:k]
            nz = nz[part]
        order = np.argsort(-scores[nz])
        top = nz[order]
        return top.astype(np.int64), scores[top]

    # ------------------------------------------------------------ persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        assert self.matrix is not None
        sp.save_npz(directory / "bm25_matrix.npz", self.matrix.tocsr())
        np.save(directory / "bm25_doclen.npy", self.doc_len)
        (directory / "bm25_meta.json").write_text(
            json.dumps(
                {"k1": self.k1, "b": self.b, "avgdl": self.avgdl,
                 "n_docs": self.n_docs, "vocab": self.vocab},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        meta = json.loads((directory / "bm25_meta.json").read_text(encoding="utf-8"))
        idx = cls(k1=meta["k1"], b=meta["b"])
        idx.vocab = meta["vocab"]
        idx.avgdl = meta["avgdl"]
        idx.n_docs = meta["n_docs"]
        idx.doc_len = np.load(directory / "bm25_doclen.npy")
        idx.matrix = sp.load_npz(directory / "bm25_matrix.npz").tocsc()
        return idx

    def stats(self) -> dict[str, object]:
        nnz = int(self.matrix.nnz) if self.matrix is not None else 0
        return {
            "documents": self.n_docs,
            "vocabulary": len(self.vocab),
            "postings": nnz,
            "avg_doc_len": round(self.avgdl, 2),
            "k1": self.k1,
            "b": self.b,
        }


__all__ = ["BM25Index"]
