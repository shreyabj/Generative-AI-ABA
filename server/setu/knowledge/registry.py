"""In-memory registry of loaded collection indexes.

Each knowledge collection owns one :class:`~setu.index.corpus.CorpusIndex` plus the
engine and extractive answerer bound to it. A rebuild constructs a complete new index
off to the side and then swaps the registry entry in one assignment, so a query that is
already running keeps the index it started with and never sees a half-built one.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

from ..answer.extractive import ExtractiveAnswerer
from ..guardrails.groundedness import GroundednessGate
from ..index.corpus import CorpusIndex
from ..retrieval.engine import RetrievalEngine

BUILTIN_ID = "col_msmarco_sample"
BUILTIN_NAME = "MS MARCO-XI sample"


@dataclass
class LoadedCollection:
    id: str
    name: str
    kind: str
    index: CorpusIndex
    version: int = 0
    engine: RetrievalEngine = field(init=False)
    reflex: ExtractiveAnswerer = field(init=False)
    groundedness: GroundednessGate = field(init=False)

    def __post_init__(self) -> None:
        self.engine = RetrievalEngine(self.index)
        self.reflex = ExtractiveAnswerer(self.index)
        self.groundedness = GroundednessGate(self.index)

    @property
    def dated(self) -> bool:
        """Whether documents carry publication dates (relaxes the temporal-deixis gate)."""
        return self.kind != "builtin"

    def summary(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "kind": self.kind, "version": self.version,
                "chunks": len(self.index.chunks), "sections": len(self.index.docs)}


class CollectionRegistry:
    def __init__(self) -> None:
        self._items: dict[str, LoadedCollection] = {}
        self._lock = threading.Lock()

    def put(self, item: LoadedCollection) -> None:
        with self._lock:
            self._items[item.id] = item

    def get(self, cid: str) -> LoadedCollection | None:
        return self._items.get(cid)

    def drop(self, cid: str) -> None:
        with self._lock:
            self._items.pop(cid, None)

    def ids(self) -> list[str]:
        return list(self._items)

    def items(self) -> list[LoadedCollection]:
        return list(self._items.values())


__all__ = ["BUILTIN_ID", "BUILTIN_NAME", "CollectionRegistry", "LoadedCollection"]
