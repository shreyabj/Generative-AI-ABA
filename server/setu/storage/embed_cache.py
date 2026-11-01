"""A persistent, content-addressed embedding cache in front of the encoder.

Collections are rebuilt whenever a document is added or removed (see
:mod:`setu.knowledge.service` for why). That is only cheap if unchanged text is never
re-embedded, so every sentence and chunk vector is stored under
``(model, sha256(text))``. A rebuild then embeds only text it has never seen; on a
collection where one document changed, that is a few percent of the work.

The key includes the model name, so switching ``SETU_EMBED_MODEL`` can never serve a
vector from the wrong space.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..embed import Encoder, get_encoder
from .repos import EmbeddingCacheRepo


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CachedEncoder:
    """Duck-types :class:`setu.embed.Encoder` for the ingest path."""

    def __init__(self, encoder: Encoder | None = None, repo: EmbeddingCacheRepo | None = None) -> None:
        self.inner = encoder or get_encoder()
        self.repo = repo or EmbeddingCacheRepo()
        self.hits = 0
        self.misses = 0

    # --- pass-through surface used by the lattice and CorpusIndex ---------------
    @property
    def model_name(self) -> str:
        return self.inner.model_name

    @property
    def backend(self) -> str:
        return self.inner.backend

    @property
    def dim(self) -> int:
        return self.inner.dim

    def warm(self) -> float:
        return self.inner.warm()

    def encode_query(self, text: str) -> np.ndarray:
        return self.inner.encode_query(text)

    # --- cached batch encode -----------------------------------------------------
    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.inner.dim), dtype=np.float32)
        hashes = [text_hash(t) for t in texts]
        found = self.repo.get_many(self.model_name, list(dict.fromkeys(hashes)))
        out: np.ndarray | None = None
        missing_idx: list[int] = []
        rows: dict[int, np.ndarray] = {}
        for i, h in enumerate(hashes):
            blob = found.get(h)
            if blob is None:
                missing_idx.append(i)
            else:
                rows[i] = np.frombuffer(blob, dtype=np.float32)
        self.hits += n - len(missing_idx)
        self.misses += len(missing_idx)

        fresh: np.ndarray | None = None
        if missing_idx:
            uniq: dict[str, int] = {}
            order: list[str] = []
            for i in missing_idx:
                if hashes[i] not in uniq:
                    uniq[hashes[i]] = len(order)
                    order.append(texts[i])
            fresh = self.inner.encode(order, batch_size=batch_size)
            self.repo.put_many(
                self.model_name, int(fresh.shape[1]),
                [(h, fresh[j].astype(np.float32).tobytes()) for h, j in uniq.items()],
            )
            for i in missing_idx:
                rows[i] = fresh[uniq[hashes[i]]]

        dim = int(next(iter(rows.values())).shape[0])
        out = np.zeros((n, dim), dtype=np.float32)
        for i, v in rows.items():
            out[i] = v
        return out

    def stats(self) -> dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses}


__all__ = ["CachedEncoder", "text_hash"]
