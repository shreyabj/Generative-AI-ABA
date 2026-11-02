"""Qdrant vector-database backend.

Selected when ``SETU_VECTOR_BACKEND=qdrant`` (or ``auto`` with a reachable URL /
configured on-disk path). Layer and language filters are pushed down into Qdrant
as payload conditions with keyword indexes, so a router decision like "search the
atomic layer only, Hindi and English" becomes a server-side filter rather than
over-fetching and discarding client-side.

The store speaks the same integer-id space as the rest of SETU: point id ``i`` is
row ``i`` of the chunk table, so callers never need to translate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..config import settings


class QdrantStore:
    name = "qdrant"

    def __init__(self, dim: int, collection: str | None = None) -> None:
        from qdrant_client import QdrantClient  # imported lazily
        from qdrant_client.models import Distance, VectorParams

        self.dim = dim
        self.collection = collection or settings.qdrant_collection

        if settings.qdrant_url:
            self.client = QdrantClient(url=settings.qdrant_url, timeout=5.0)
            self.mode = f"server:{settings.qdrant_url}"
        elif settings.qdrant_path:
            self.client = QdrantClient(path=settings.qdrant_path)
            self.mode = f"embedded:{settings.qdrant_path}"
        else:
            self.client = QdrantClient(location=":memory:")
            self.mode = "embedded::memory:"

        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        self._payloads: list[dict[str, Any]] = []
        self._indexed_fields: set[str] = set()

    # ---------------------------------------------------------------- mutation
    def ensure_payload_indexes(self, fields: Sequence[str] = ("layer", "lang", "doc_id")) -> None:
        from qdrant_client.models import PayloadSchemaType

        for f in fields:
            if f in self._indexed_fields:
                continue
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=f,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            except Exception:  # noqa: BLE001 - already exists / local mode no-op
                pass
            self._indexed_fields.add(f)

    def upsert(
        self,
        ids: Sequence[int],
        vectors: np.ndarray,
        payloads: Sequence[dict[str, Any]] | None = None,
        *,
        batch: int = 512,
    ) -> None:
        from qdrant_client.models import PointStruct

        vectors = np.asarray(vectors, dtype=np.float32)
        payloads = list(payloads or [{} for _ in ids])
        ids = list(ids)
        for i in range(0, len(ids), batch):
            pts = [
                PointStruct(id=int(pid), vector=vec.tolist(), payload=pl)
                for pid, vec, pl in zip(
                    ids[i : i + batch], vectors[i : i + batch], payloads[i : i + batch], strict=True
                )
            ]
            self.client.upsert(collection_name=self.collection, points=pts, wait=True)

    # ------------------------------------------------------------------ search
    def search_filtered(
        self,
        query: np.ndarray,
        k: int,
        *,
        layers: Sequence[str] | None = None,
        langs: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        from qdrant_client.models import FieldCondition, Filter, MatchAny

        must = []
        if layers:
            must.append(FieldCondition(key="layer", match=MatchAny(any=list(layers))))
        if langs:
            must.append(FieldCondition(key="lang", match=MatchAny(any=list(langs))))
        flt = Filter(must=must) if must else None

        res = self.client.query_points(
            collection_name=self.collection,
            query=np.asarray(query, dtype=np.float32).reshape(-1).tolist(),
            limit=k,
            query_filter=flt,
            with_payload=False,
            search_params=None,
        ).points
        if not res:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        return (
            np.asarray([int(p.id) for p in res], dtype=np.int64),
            np.asarray([float(p.score) for p in res], dtype=np.float32),
        )

    def search(
        self, query: np.ndarray, k: int, *, mask: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Protocol-compatible search. A boolean ``mask`` is applied post-hoc.

        Prefer :meth:`search_filtered`, which pushes the predicate server-side.
        Mask mode over-fetches 4x to compensate for filtered-out points.
        """
        if mask is None:
            return self.search_filtered(query, k)
        idx, scores = self.search_filtered(query, max(k * 4, k + 16))
        if idx.size == 0:
            return idx, scores
        keep = mask[idx]
        return idx[keep][:k], scores[keep][:k]

    def size(self) -> int:
        try:
            return int(self.client.count(self.collection, exact=True).count)
        except Exception:  # noqa: BLE001
            return 0

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "mode": self.mode,
            "collection": self.collection,
            "vectors": self.size(),
            "dim": self.dim,
            "distance": "cosine",
            "indexed_payload_fields": sorted(self._indexed_fields),
        }


def qdrant_available() -> bool:
    """True when a Qdrant server responds, or an embedded path is configured."""
    if settings.qdrant_path:
        return True
    if not settings.qdrant_url:
        return False
    try:
        import httpx

        r = httpx.get(f"{settings.qdrant_url.rstrip('/')}/readyz", timeout=1.2)
        return r.status_code < 500
    except Exception:  # noqa: BLE001
        return False


__all__ = ["QdrantStore", "qdrant_available"]
