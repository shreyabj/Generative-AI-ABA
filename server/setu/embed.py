"""Embedding encoder.

Design constraints, in priority order:

1. **Multilingual.** The corpus is 14 Indic languages plus English and a Hindi
   question must retrieve an English passage, so the encoder has to share a
   vector space across scripts. ``paraphrase-multilingual-MiniLM-L12-v2`` does.
2. **Fast on CPU.** A 200 ms budget cannot absorb a PyTorch import, let alone a
   large model. We run the ONNX graph through ``fastembed``, which keeps a single
   short query at roughly 10-25 ms on a laptop core.
3. **Warm.** First inference on any ONNX graph pays an allocation cost. The
   encoder is warmed at startup so the first real user query is not the one that
   eats it.
4. **Cached.** Repeat and near-repeat queries (very common in a voice UI, where
   partial transcripts arrive incrementally) hit an LRU instead of the model.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Iterable, Sequence

import numpy as np

from .config import settings

_L2_EPS = 1e-12


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        n = float(np.linalg.norm(mat))
        return mat / (n + _L2_EPS)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + _L2_EPS)


class _LRU:
    def __init__(self, cap: int) -> None:
        self.cap = cap
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> np.ndarray | None:
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key: str, val: np.ndarray) -> None:
        with self._lock:
            self._d[key] = val
            self._d.move_to_end(key)
            while len(self._d) > self.cap:
                self._d.popitem(last=False)

    @property
    def stats(self) -> dict[str, float | int]:
        total = self.hits + self.misses
        return {
            "size": len(self._d),
            "capacity": self.cap,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }


class Encoder:
    """Lazily-loaded, thread-safe, cached sentence encoder."""

    def __init__(self, model_name: str | None = None, dim: int | None = None) -> None:
        self.model_name = model_name or settings.embed_model
        self.dim = dim or settings.embed_dim
        self._model = None
        self._backend = "uninitialised"
        self._lock = threading.Lock()
        self._cache = _LRU(settings.embed_cache_size)
        self._warm = False

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding  # type: ignore

                kwargs = {"model_name": self.model_name}
                if settings.embed_threads:
                    kwargs["threads"] = settings.embed_threads
                self._model = TextEmbedding(**kwargs)
                self._backend = "fastembed-onnx"
                return
            except Exception as fe:  # noqa: BLE001
                fe_err = fe
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore

                self._model = SentenceTransformer(self.model_name)
                self._backend = "sentence-transformers"
                return
            except Exception as st_err:  # noqa: BLE001
                raise RuntimeError(
                    "No embedding backend available. Install fastembed (preferred) or "
                    f"sentence-transformers.\n  fastembed: {fe_err}\n  sentence-transformers: {st_err}"
                ) from st_err

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def ready(self) -> bool:
        return self._model is not None and self._warm

    def warm(self) -> float:
        """Load + run a throwaway batch. Returns milliseconds spent."""
        import time

        t0 = time.perf_counter()
        self._load()
        self._raw_encode(["सेतु warm up", "setu warm up", "செட்டு warm up"])
        self._warm = True
        return (time.perf_counter() - t0) * 1000.0

    # ----------------------------------------------------------------- encoding
    def _raw_encode(self, texts: Sequence[str]) -> np.ndarray:
        self._load()
        assert self._model is not None
        if self._backend == "fastembed-onnx":
            # One internal batch per call: this method is only ever handed a group that
            # `encode()` has already length-bucketed, and `parallel` is deliberately left
            # unset - on Windows fastembed's parallel path spawns processes per call,
            # which is catastrophically slower than it sounds.
            vecs = list(self._model.embed(list(texts), batch_size=len(texts)))
            arr = np.asarray(vecs, dtype=np.float32)
        else:
            arr = np.asarray(
                self._model.encode(list(texts), batch_size=len(texts), show_progress_bar=False),
                dtype=np.float32,
            )
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.dim:
            self.dim = int(arr.shape[1])
        return l2_normalize(arr)

    def encode(self, texts: Sequence[str], *, batch_size: int | None = None) -> np.ndarray:
        """Encode many texts (ingest path).

        **Length-bucketed batching.** A transformer batch is padded to its longest member,
        so a batch mixing a 20-token sentence with a 400-token passage does 400 tokens of
        work for every row. The chunk lattice makes this especially bad, because it
        deliberately emits nodes at four different scales. Measured on this corpus, naive
        batching at ``batch_size=256`` ran at 42 texts/s while encoding the same texts one
        at a time ran at 175 texts/s - batching was *slower per item*, purely from padding.

        Sorting by length before batching makes each batch homogeneous, so padding
        approaches zero. The original order is restored via the inverse permutation, which
        matters: silently returning vectors in sorted order would misalign every chunk in
        the index against its text. There is a test for exactly that.
        """
        n = len(texts)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if n == 1:
            return self._raw_encode(texts)

        bs = batch_size or settings.embed_batch
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.zeros((n, self.dim), dtype=np.float32)
        first = True

        for start in range(0, n, bs):
            idx = order[start : start + bs]
            block = self._raw_encode([texts[i] for i in idx])
            if first and block.shape[1] != out.shape[1]:
                # The model reported a different dimension than configured.
                out = np.zeros((n, block.shape[1]), dtype=np.float32)
            first = False
            out[idx] = block
        return out

    def encode_query(self, text: str) -> np.ndarray:
        """Encode one query (serve path - LRU cached, returns a 1-D vector)."""
        key = hashlib.blake2b(text.strip().casefold().encode("utf-8"), digest_size=16).hexdigest()
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        vec = self._raw_encode([text])[0]
        self._cache.put(key, vec)
        return vec

    def encode_iter(self, texts: Iterable[str], batch_size: int | None = None) -> Iterable[np.ndarray]:
        buf: list[str] = []
        bs = batch_size or settings.embed_batch
        for t in texts:
            buf.append(t)
            if len(buf) >= bs:
                yield self._raw_encode(buf)
                buf = []
        if buf:
            yield self._raw_encode(buf)

    # -------------------------------------------------------------------- stats
    def stats(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "backend": self._backend,
            "dim": self.dim,
            "warm": self._warm,
            "cache": self._cache.stats,
        }


_encoder: Encoder | None = None
_enc_lock = threading.Lock()


def get_encoder() -> Encoder:
    global _encoder
    if _encoder is None:
        with _enc_lock:
            if _encoder is None:
                _encoder = Encoder()
    return _encoder


def cosine_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Cosine similarity for pre-normalised inputs (a plain matmul)."""
    q = queries if queries.ndim == 2 else queries.reshape(1, -1)
    return q @ docs.T


__all__ = ["Encoder", "cosine_matrix", "get_encoder", "l2_normalize"]
