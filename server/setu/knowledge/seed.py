"""Seed the demonstration collections.

1. **India Business & Economy 2026** - real public documents listed in
   ``data/sources/business.json``, each fetched live through the SSRF-safe importer
   (robots.txt honoured). Sources that refuse automated access are listed in the
   manifest's ``excluded`` block with the reason, not worked around.
2. **Red-team fixtures (synthetic)** - small documents that exist only to attack the
   pipeline (embedded injection, contradictory sources, a misleading passage). Every one
   is flagged ``synthetic`` and says so in its own text.

Nothing here generates content and presents it as a real report: if a live source cannot
be fetched it simply is not in the collection, and the seed report says which.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import settings
from ..evaluation import redteam
from .service import KnowledgeError, KnowledgeService


def _ensure(ks: KnowledgeService, name: str, description: str, kind: str = "user") -> dict[str, Any]:
    col = ks.collections.by_name(name)
    if col is None:
        col = ks.create_collection(name, description, kind=kind)
    return col


def seed_business(ks: KnowledgeService, manifest: Path | None = None) -> dict[str, Any]:
    path = manifest or (settings.sources_dir / "business.json")
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    col = _ensure(ks, m["collection"], m["description"])
    added, skipped = [], []
    for src in m["sources"]:
        try:
            doc = ks.add_url(col["id"], src["url"], extra_meta={
                "publisher": src.get("publisher"), "licence": src.get("licence"),
                "source_kind": src.get("kind"), "manifest_key": src.get("key"),
            })
            (skipped if doc.get("duplicate") else added).append(src["key"])
        except KnowledgeError as exc:
            skipped.append(f"{src['key']}: {exc}")
    return {"collection_id": col["id"], "queued": added, "already_present": skipped,
            "excluded_by_policy": m.get("excluded", [])}


def seed_redteam(ks: KnowledgeService) -> dict[str, Any]:
    col = _ensure(ks, redteam.FIXTURE_COLLECTION,
                  "SYNTHETIC test fixtures for the red-team suite: an embedded prompt injection, two "
                  "contradictory sources and a misleading passage. Not real companies.", kind="synthetic")
    added = []
    for f in redteam.FIXTURES:
        out = ks.add_text(col["id"], title=f["title"], text=f["text"], filename=f["filename"],
                          meta={"synthetic": True, "publisher": "SETU test fixtures (synthetic)"})
        if not out.get("duplicate"):
            added.append(f["filename"])
    return {"collection_id": col["id"], "added": added}


def curated_items(ks: KnowledgeService, collection_id: str) -> list[dict[str, Any]]:
    """Business evaluation items. Relevant documents are referenced by manifest key."""
    by_key: dict[str, str] = {}
    for d in ks.documents.all_in(collection_id):
        key = (d.get("meta") or {}).get("manifest_key")
        if key and d["status"] == "completed":
            by_key[key] = d["id"]
    spec = json.loads((settings.sources_dir / "business_eval.json").read_text(encoding="utf-8"))
    items = []
    for it in spec["items"]:
        rel = [by_key[k] for k in it["relevant"] if k in by_key]
        if not rel:
            continue
        items.append({"question": it["question"], "lang": it.get("lang", "en"), "relevant_doc_ids": rel,
                      "reference_answer": it.get("reference"), "label_source": "curated",
                      "tags": it.get("tags", [])})
    return items


__all__ = ["curated_items", "seed_business", "seed_redteam"]
