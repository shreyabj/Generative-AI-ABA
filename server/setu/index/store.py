"""Vector store abstraction + the in-process flat backend.

SETU talks to its vector database through one narrow protocol so the backend is a
deployment decision, not an architectural one:

``QdrantStore``  A real vector database (HNSW, payload filtering, persistence).
                 Used when a Qdrant URL or on-disk path is configured. This is
                 the production path and what ``docker compose up`` gives you.

``FlatStore``    Exact cosine search as a single dense matmul, in-process, zero
                 network hops. At lattice scale (tens of thousands of 384-dim
                 vectors) a brute-force matmul is ~1-3 ms, which is *faster* than
                 an HNSW round trip over localhost and has no recall loss at all.
                 It is the default for the demo and the benchmark because it makes
                 latency numbers reproducible rather than dependent on a sidecar.

Both support layer- and language-filtered search, which the router relies on.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class VectorStore(Protocol):
    name: str

    def upsert(self, ids: Sequence[int], vectors: np.ndarray) -> None: ...

    def search(
        self, query: np.ndarray, k: int, *, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def size(self) -> int: ...

    def stats(self) -> dict[str, Any]: ...


class FlatStore:
    """Exact cosine over an L2-normalised float32 matrix."""

    name = "flat-exact"

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._vectors: np.ndarray = np.zeros((0, dim), dtype=np.float32)

    # ---------------------------------------------------------------- mutation
    def set_matrix(self, vectors: np.ndarray) -> None:
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.dim = int(self._vectors.shape[1]) if self._vectors.size else self.dim

    def upsert(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        needed = (max(ids) + 1) if len(ids) else 0
        if self._vectors.shape[0] < needed:
            pad = np.zeros((needed - self._vectors.shape[0], self.dim), dtype=np.float32)
            self._vectors = np.vstack([self._vectors, pad]) if self._vectors.size else pad
        self._vectors[np.asarray(ids, dtype=np.int64)] = vectors

    # ------------------------------------------------------------------ search
    def search(
        self, query: np.ndarray, k: int, *, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._vectors.shape[0] == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(-1)

        if mask is not None:
            cand = np.flatnonzero(mask)
            if cand.size == 0:
                return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
            sims = self._vectors[cand] @ q
            k = min(k, cand.size)
            part = np.argpartition(-sims, k - 1)[:k] if k < cand.size else np.arange(cand.size)
            order = part[np.argsort(-sims[part])]
            return cand[order].astype(np.int64), sims[order].astype(np.float32)

        sims = self._vectors @ q
        n = sims.shape[0]
        k = min(k, n)
        part = np.argpartition(-sims, k - 1)[:k] if k < n else np.arange(n)
        order = part[np.argsort(-sims[part])]
        return order.astype(np.int64), sims[order].astype(np.float32)

    def vectors_for(self, idx: np.ndarray) -> np.ndarray:
        return self._vectors[idx]

    def size(self) -> int:
        return int(self._vectors.shape[0])

    def stats(self) -> dict[str, Any]:
        mb = self._vectors.nbytes / (1024 * 1024)
        return {
            "backend": self.name,
            "vectors": self.size(),
            "dim": self.dim,
            "memory_mb": round(mb, 2),
            "recall": 1.0,
            "note": "exact search - no ANN approximation",
        }

    # -------------------------------------------------------------- persistence
    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self._vectors)

    @classmethod
    def load(cls, directory: Path) -> FlatStore:
        vecs = np.load(directory / "vectors.npy")
        store = cls(dim=int(vecs.shape[1]) if vecs.size else 384)
        store.set_matrix(vecs)
        return store


__all__ = ["FlatStore", "VectorStore"]
