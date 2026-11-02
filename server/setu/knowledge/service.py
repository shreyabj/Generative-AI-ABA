"""Knowledge collections: ingestion jobs, per-collection indexes, atomic rebuilds.

Pipeline for every document::

    upload / URL -> validate -> store raw -> extract -> clean -> metadata -> segment
                 -> chunk lattice -> embed (cached) -> vector + BM25 index -> persist -> ready

Why one index per collection, rebuilt on change
-----------------------------------------------
``CorpusIndex`` is row-positional: the vector matrix, BM25 matrix, layer masks and the
sentence table all assume one contiguous build, and BM25's IDF is a global statistic.
Making it incremental would mean rewriting the most latency-sensitive code in the
project. Instead each collection gets its own index, rebuilt whenever its documents
change. The embedding cache (``setu.storage.embed_cache``) makes that cheap: unchanged
text is never re-embedded, so a rebuild costs chunking plus a BM25 fit. Collections are
physically isolated as a side effect - a query against collection A cannot touch B's
rows. The trade-off, documented in docs/KNOWLEDGE.md, is that rebuild time grows with
collection size; past roughly 50k chunks a segmented index is the next step.

Jobs run on one background worker thread, so ingestion never blocks request handling,
and rebuilds of the same collection are serialised. Jobs left running by a restart are
marked failed on boot rather than silently half-applied.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..chunking.lattice import ChunkLattice
from ..config import settings
from ..embed import get_encoder
from ..index.corpus import CorpusIndex
from ..storage.db import utcnow
from ..storage.embed_cache import CachedEncoder, text_hash
from ..storage.repos import ChunkRepo, CollectionRepo, DocumentRepo, JobRepo
from .extract import Extracted, ExtractionError, extract, sniff_kind
from .fetch import FetchError, safe_fetch, validate_url
from .injection import scan
from .registry import BUILTIN_ID, BUILTIN_NAME, CollectionRegistry, LoadedCollection
from .segment import segment, to_documents

log = logging.getLogger("setu.knowledge")

STATES = ("uploaded", "extracting", "chunking", "embedding", "indexing", "completed", "failed")
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._ -]+")


class KnowledgeError(ValueError):
    """User-facing validation error (maps to HTTP 400/404/409)."""

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = status


def safe_filename(name: str) -> str:
    base = Path(name or "upload").name
    base = _SAFE_NAME.sub("_", base).strip(" .") or "upload"
    return base[:180]


class KnowledgeService:
    def __init__(self, registry: CollectionRegistry, *, root: Path | None = None) -> None:
        self.registry = registry
        self.root = Path(root or settings.collections_dir)
        self.collections = CollectionRepo()
        self.documents = DocumentRepo()
        self.chunks = ChunkRepo()
        self.jobs = JobRepo()
        self._queue: queue.Queue[str] = queue.Queue()
        self._pending: set[str] = set()
        self._pending_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_rebuild: dict[str, dict[str, Any]] = {}

    # ================================================================== paths
    def _dir(self, cid: str) -> Path:
        return self.root / cid

    def _raw_path(self, cid: str, did: str) -> Path:
        return self._dir(cid) / "raw" / did

    def _extracted_path(self, cid: str, did: str) -> Path:
        return self._dir(cid) / "extracted" / f"{did}.json"

    def _index_dir(self, cid: str, version: int) -> Path:
        return self._dir(cid) / f"index_v{version}"

    # =================================================================== boot
    def register_builtin(self, index: CorpusIndex) -> None:
        """Expose the v1 MS MARCO sample index as a read-only collection."""
        row = self.collections.get(BUILTIN_ID)
        n_docs = len(index.docs)
        if row is None:
            self.collections.create(
                cid=BUILTIN_ID, name=BUILTIN_NAME, kind="builtin", embed_model=settings.embed_model,
                description="The bundled MS MARCO-XI multilingual passage sample (read-only). "
                            "Short web passages in 10 Indian languages + English.",
            )
        self.collections.update(BUILTIN_ID, status="ready", doc_count=n_docs,
                                chunk_count=len(index.chunks), embed_model=settings.embed_model)
        self.registry.put(LoadedCollection(BUILTIN_ID, BUILTIN_NAME, "builtin", index))

    def boot(self) -> dict[str, Any]:
        failed = self.jobs.fail_interrupted()
        loaded, errors = [], {}
        for col in self.collections.list():
            if col["kind"] == "builtin":
                continue
            d = self._index_dir(col["id"], col["index_version"])
            if col["status"] in ("ready", "rebuilding") and CorpusIndex.exists(d):
                try:
                    idx = CorpusIndex.load(d)
                    idx.attach_vector_backend(f"setu_{col['id']}")
                    self.registry.put(LoadedCollection(col["id"], col["name"], col["kind"], idx,
                                                       version=col["index_version"]))
                    self.collections.update(col["id"], status="ready", error=None)
                    loaded.append(col["id"])
                except Exception as exc:  # noqa: BLE001
                    errors[col["id"]] = f"{type(exc).__name__}: {exc}"
                    self.collections.update(col["id"], status="error", error=errors[col["id"]])
            elif col["status"] == "rebuilding":
                self.collections.update(col["id"], status="error", error="rebuild interrupted - rebuild again")
        # Documents caught mid-flight by a restart.
        for col in self.collections.list():
            for doc in self.documents.all_in(col["id"], ["uploaded", "extracting", "chunking", "embedding", "indexing"]):
                self.documents.update(doc["id"], status="failed", error="interrupted by server restart - re-index to retry")
        self.start_worker()
        return {"loaded": loaded, "errors": errors, "interrupted_jobs": failed}

    def start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="setu-ingest", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        self._queue.put("")

    # ============================================================ collections
    def create_collection(self, name: str, description: str = "", *, kind: str = "user") -> dict[str, Any]:
        name = name.strip()
        if not (1 <= len(name) <= 120):
            raise KnowledgeError("collection name must be 1-120 characters")
        if self.collections.by_name(name):
            raise KnowledgeError(f"a collection named '{name}' already exists", 409)
        col = self.collections.create(name=name, description=description.strip()[:2000], kind=kind,
                                      embed_model=settings.embed_model)
        self._dir(col["id"]).mkdir(parents=True, exist_ok=True)
        return col

    def get_collection(self, cid: str) -> dict[str, Any]:
        col = self.collections.get(cid)
        if col is None:
            raise KnowledgeError("collection not found", 404)
        return col

    def update_collection(self, cid: str, *, name: str | None = None, description: str | None = None) -> dict[str, Any]:
        col = self.get_collection(cid)
        if col["kind"] == "builtin":
            raise KnowledgeError("the built-in collection is read-only", 403)
        values: dict[str, Any] = {}
        if name is not None:
            name = name.strip()
            if not (1 <= len(name) <= 120):
                raise KnowledgeError("collection name must be 1-120 characters")
            other = self.collections.by_name(name)
            if other and other["id"] != cid:
                raise KnowledgeError(f"a collection named '{name}' already exists", 409)
            values["name"] = name
        if description is not None:
            values["description"] = description.strip()[:2000]
        out = self.collections.update(cid, **values) if values else col
        loaded = self.registry.get(cid)
        if loaded and "name" in values:
            loaded.name = values["name"]
        return out  # type: ignore[return-value]

    def delete_collection(self, cid: str) -> None:
        col = self.get_collection(cid)
        if col["kind"] == "builtin":
            raise KnowledgeError("the built-in collection cannot be deleted", 403)
        self.registry.drop(cid)
        self.collections.delete(cid)
        shutil.rmtree(self._dir(cid), ignore_errors=True)

    # ============================================================== documents
    def add_upload(self, cid: str, filename: str, data: bytes, content_type: str | None = None) -> dict[str, Any]:
        col = self.get_collection(cid)
        if col["kind"] == "builtin":
            raise KnowledgeError("documents cannot be added to the built-in collection", 403)
        limit = int(settings.max_upload_mb * 1024 * 1024)
        if len(data) > limit:
            raise KnowledgeError(f"'{filename}' is larger than the {settings.max_upload_mb:.0f} MB limit", 413)
        if not data:
            raise KnowledgeError(f"'{filename}' is empty")
        fname = safe_filename(filename)
        try:
            kind = sniff_kind(data, fname)
        except ExtractionError as exc:
            raise KnowledgeError(f"'{fname}': {exc}", 415) from exc
        digest = hashlib.sha256(data).hexdigest()
        dup = self.documents.find_hash(cid, digest)
        if dup:
            return {**dup, "duplicate": True}
        doc = self.documents.create(
            collection_id=cid, source_type="upload", filename=fname, title=fname,
            mime_type=content_type or kind, byte_size=len(data), content_hash=digest, status="uploaded",
            meta={"kind": kind},
        )
        raw = self._raw_path(cid, doc["id"])
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(data)
        job = self.jobs.create(kind="ingest", collection_id=cid, document_id=doc["id"])
        self._enqueue(cid)
        return {**doc, "job_id": job["id"], "duplicate": False}

    def add_url(self, cid: str, url: str, *, extra_meta: dict[str, Any] | None = None) -> dict[str, Any]:
        col = self.get_collection(cid)
        if col["kind"] == "builtin":
            raise KnowledgeError("documents cannot be added to the built-in collection", 403)
        try:
            validate_url(url)
        except FetchError as exc:
            raise KnowledgeError(str(exc)) from exc
        url = url.strip()
        digest = "url:" + hashlib.sha256(url.encode()).hexdigest()[:58]
        dup = self.documents.find_hash(cid, digest)
        if dup and dup["status"] != "failed":
            return {**dup, "duplicate": True}
        if dup:
            self.documents.delete(dup["id"])
        doc = self.documents.create(
            collection_id=cid, source_type="url", source_url=url, domain=urlsplit(url).hostname,
            title=url, content_hash=digest, status="uploaded", meta=dict(extra_meta or {}),
            synthetic=bool((extra_meta or {}).get("synthetic")),
        )
        job = self.jobs.create(kind="ingest", collection_id=cid, document_id=doc["id"])
        self._enqueue(cid)
        return {**doc, "job_id": job["id"], "duplicate": False}

    def add_text(self, cid: str, *, title: str, text: str, filename: str, meta: dict[str, Any]) -> dict[str, Any]:
        """Add an in-memory text document (used for labelled synthetic demonstration data)."""
        data = text.encode("utf-8")
        out = self.add_upload(cid, filename, data, "text/markdown")
        if not out.get("duplicate"):
            self.documents.update(out["id"], title=title, synthetic=bool(meta.get("synthetic")),
                                  meta={**(self.documents.get(out["id"]) or {}).get("meta", {}), **meta})
        return out

    def get_document(self, did: str) -> dict[str, Any]:
        doc = self.documents.get(did)
        if doc is None:
            raise KnowledgeError("document not found", 404)
        return doc

    def delete_document(self, did: str) -> dict[str, Any]:
        doc = self.get_document(did)
        cid = doc["collection_id"]
        if (self.collections.get(cid) or {}).get("kind") == "builtin":
            raise KnowledgeError("the built-in collection is read-only", 403)
        self.documents.delete(did)
        for p in (self._raw_path(cid, did), self._extracted_path(cid, did)):
            p.unlink(missing_ok=True)
        job = self.jobs.create(kind="rebuild", collection_id=cid, state="queued",
                               payload={"reason": f"document {did} deleted"})
        self._enqueue(cid)
        return {"deleted": did, "rebuild_job_id": job["id"]}

    def reindex_document(self, did: str) -> dict[str, Any]:
        doc = self.get_document(did)
        cid = doc["collection_id"]
        if doc["source_type"] == "upload" and not self._raw_path(cid, did).exists():
            raise KnowledgeError("original file is missing; upload it again", 409)
        self.documents.update(did, status="uploaded", error=None)
        self._extracted_path(cid, did).unlink(missing_ok=True)
        job = self.jobs.create(kind="reindex", collection_id=cid, document_id=did)
        self._enqueue(cid)
        return job

    def rebuild(self, cid: str) -> dict[str, Any]:
        col = self.get_collection(cid)
        if col["kind"] == "builtin":
            raise KnowledgeError("the built-in collection is rebuilt with `setu.cli build --sample`", 403)
        job = self.jobs.create(kind="rebuild", collection_id=cid, state="queued", payload={"reason": "manual"})
        self._enqueue(cid)
        return job

    def document_chunks(self, did: str, *, layer: str | None, limit: int, offset: int) -> dict[str, Any]:
        self.get_document(did)
        rows, total = self.chunks.list_for_document(did, layer=layer, limit=limit, offset=offset)
        return {"chunks": rows, "total": total, "limit": limit, "offset": offset}

    # ================================================================= worker
    def _enqueue(self, cid: str) -> None:
        with self._pending_lock:
            if cid in self._pending:
                return
            self._pending.add(cid)
        self._queue.put(cid)

    def wait_idle(self, timeout: float = 300.0) -> bool:
        """Block until no collection work is queued or running (tests, CLI seeding)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._pending_lock:
                if not self._pending and self._queue.empty() and not self._busy:
                    return True
            time.sleep(0.1)
        return False

    _busy = False

    def _run(self) -> None:
        while not self._stop.is_set():
            cid = self._queue.get()
            if not cid:
                continue
            # Leave the pending set *before* processing, so work submitted while this
            # batch runs re-queues the collection instead of being dropped.
            with self._pending_lock:
                self._pending.discard(cid)
                self._busy = True
            try:
                self.process_collection(cid)
            except Exception as exc:  # noqa: BLE001 - the worker must survive any job
                log.exception("collection %s processing failed", cid)
                self.collections.update(cid, status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                with self._pending_lock:
                    self._busy = False

    def process_collection(self, cid: str) -> None:
        if self.collections.get(cid) is None:
            return
        jobs = list(reversed(self.jobs.list(collection_id=cid, states=("uploaded", "queued"), limit=500)))
        if not jobs:
            return
        now = utcnow()
        for j in jobs:
            self.jobs.update(j["id"], state="extracting" if j["document_id"] else "indexing",
                             started_at=now, progress=0.05)

        # ---- extract each new / re-indexed document --------------------------
        ok_jobs: list[dict[str, Any]] = []
        for j in jobs:
            did = j["document_id"]
            if not did:
                ok_jobs.append(j)
                continue
            try:
                self.documents.update(did, status="extracting", error=None)
                self._extract_document(cid, did)
                self.documents.update(did, status="chunking")
                self.jobs.update(j["id"], state="chunking", progress=0.3, message="text extracted")
                ok_jobs.append(j)
            except (ExtractionError, FetchError, KnowledgeError) as exc:
                self._fail(j, did, str(exc))
            except Exception as exc:  # noqa: BLE001
                log.exception("extraction failed for %s", did)
                self._fail(j, did, f"{type(exc).__name__}: {exc}")

        if not ok_jobs:
            return

        # ---- one rebuild covers every job in this batch -----------------------
        def stage(state: str, progress: float, msg: str) -> None:
            for j in ok_jobs:
                self.jobs.update(j["id"], state=state, progress=progress, message=msg)
                if j["document_id"]:
                    self.documents.update(j["document_id"], status=state)

        try:
            report = self.rebuild_index(cid, on_stage=stage)
        except Exception as exc:  # noqa: BLE001
            log.exception("rebuild failed for %s", cid)
            for j in ok_jobs:
                self._fail(j, j["document_id"], f"indexing failed: {type(exc).__name__}: {exc}")
            self.collections.update(cid, status="error", error=f"{type(exc).__name__}: {exc}")
            return
        done = utcnow()
        for j in ok_jobs:
            self.jobs.update(j["id"], state="completed", progress=1.0, finished_at=done,
                             message=f"indexed - collection now {report['chunks']} chunks",
                             payload={**(j.get("payload") or {}), "rebuild": report})

    def _fail(self, job: dict[str, Any], did: str | None, error: str) -> None:
        self.jobs.update(job["id"], state="failed", error=error[:2000], finished_at=utcnow())
        if did:
            self.documents.update(did, status="failed", error=error[:2000])

    # -------------------------------------------------------------- extraction
    def _extract_document(self, cid: str, did: str) -> Extracted:
        doc = self.get_document(did)
        if doc["source_type"] == "url":
            res = safe_fetch(doc["source_url"])
            ex = extract(res.content, Path(urlsplit(res.final_url).path).name or "page.html",
                         url=res.final_url, content_type=res.content_type)
            raw = self._raw_path(cid, did)
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(res.content)
            values: dict[str, Any] = {
                "retrieved_at": utcnow(), "mime_type": res.content_type, "byte_size": len(res.content),
                "domain": urlsplit(res.final_url).hostname,
                "meta": {**(doc.get("meta") or {}), "final_url": res.final_url, "redirects": res.redirects,
                         "site": ex.meta.get("site"), "author": ex.meta.get("author")},
            }
        else:
            data = self._raw_path(cid, did).read_bytes()
            ex = extract(data, doc["filename"] or "upload.txt")
            values = {"meta": {**(doc.get("meta") or {})}}
        meta_title = (doc.get("meta") or {}).get("title")
        title = (meta_title or ex.title or doc.get("filename") or doc.get("source_url") or did).strip()[:400]
        published = (doc.get("meta") or {}).get("published_at") or ex.meta.get("published_at")
        sections = segment(ex)
        from ..text import detect_lang

        values.update(
            title=title, page_count=ex.page_count, char_count=len(ex.text), section_count=len(sections),
            lang=detect_lang(ex.text[:4000]), published_at=str(published)[:40] if published else None,
        )
        values["meta"]["warnings"] = ex.warnings
        values["meta"]["kind"] = ex.kind
        self.documents.update(did, **values)
        path = self._extracted_path(cid, did)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "text": ex.text, "title": title, "kind": ex.kind, "pages": ex.pages,
            "headings": ex.headings, "meta": ex.meta, "warnings": ex.warnings,
        }, ensure_ascii=False), encoding="utf-8")
        return ex

    def _load_extracted(self, cid: str, did: str) -> Extracted | None:
        path = self._extracted_path(cid, did)
        if not path.exists():
            return None
        o = json.loads(path.read_text(encoding="utf-8"))
        return Extracted(text=o["text"], title=o.get("title", ""), kind=o.get("kind", "text"),
                         pages=[tuple(p) for p in o.get("pages", [])],  # type: ignore[misc]
                         headings=[tuple(h) for h in o.get("headings", [])],  # type: ignore[misc]
                         meta=o.get("meta", {}), warnings=o.get("warnings", []))

    # ----------------------------------------------------------------- rebuild
    def rebuild_index(self, cid: str, *, on_stage=None) -> dict[str, Any]:  # noqa: ANN001
        col = self.get_collection(cid)
        on_stage = on_stage or (lambda *a: None)
        t0 = time.perf_counter()
        self.collections.update(cid, status="rebuilding", error=None)

        live = self.documents.all_in(cid, ["completed", "chunking", "embedding", "indexing"])
        lattice_docs = []
        per_doc_sections: dict[str, int] = {}
        for d in live:
            ex = self._load_extracted(cid, d["id"])
            if ex is None:
                self.documents.update(d["id"], status="failed", error="extracted text missing - re-index")
                continue
            secs = segment(ex)
            docs = to_documents(ex, secs, document_id=d["id"], collection_id=cid, base_meta={
                "doc_title": d.get("title"), "filename": d.get("filename"), "source_url": d.get("source_url"),
                "source_type": d.get("source_type"), "published_at": d.get("published_at"),
                "synthetic": bool(d.get("synthetic")), "domain": d.get("domain"),
            })
            per_doc_sections[d["id"]] = len(docs)
            lattice_docs.extend(docs)

        if not lattice_docs:
            self.registry.drop(cid)
            self.chunks.replace_collection(cid, [])
            self.collections.update(cid, status="empty", doc_count=0, chunk_count=0)
            self._cleanup_versions(cid, keep=None)
            return {"documents": 0, "sections": 0, "chunks": 0, "seconds": 0.0}

        on_stage("chunking", 0.4, f"segmenting {len(live)} documents into {len(lattice_docs)} sections")
        enc = CachedEncoder(get_encoder())
        lattice = ChunkLattice(enc)
        groups = list(lattice.build(lattice_docs, batch=128))
        on_stage("embedding", 0.6, f"embedding {lattice.stats.chunks} chunks")
        index = CorpusIndex().build(lattice_docs, groups, enc)
        on_stage("indexing", 0.85, "writing vector and BM25 indexes")

        version = int(col["index_version"]) + 1
        out_dir = self._index_dir(cid, version)
        index.save(out_dir)
        index.attach_vector_backend(f"setu_{cid}")

        # ---- chunk metadata of record ---------------------------------------
        rows = []
        flags_per_doc: dict[str, int] = {}
        counts_per_doc: dict[str, int] = {}
        model = settings.embed_model
        created = utcnow()
        for i, rec in enumerate(index.chunks):
            dmeta = index.docs[rec.doc_id].meta
            did = dmeta["document_id"]
            pages = [p for p in (dmeta.get("page_map") or []) if p[2] > rec.char_start and p[1] < rec.char_end]
            labels = scan(rec.text)
            if labels:
                flags_per_doc[did] = flags_per_doc.get(did, 0) + 1
                rec.meta["injection_flags"] = labels
            counts_per_doc[did] = counts_per_doc.get(did, 0) + 1
            rows.append({
                "id": rec.chunk_id, "collection_id": cid, "document_id": did, "section_id": rec.doc_id,
                "layer": rec.layer.value, "chunk_index": int(rec.meta.get("ordinal", i)),
                "page_start": min(p[0] for p in pages) if pages else dmeta.get("page_start"),
                "page_end": max(p[0] for p in pages) if pages else dmeta.get("page_end"),
                "section_heading": dmeta.get("section"), "char_start": rec.char_start,
                "char_end": rec.char_end, "parent_id": rec.parent_id, "lang": rec.lang,
                "text": rec.text, "content_hash": text_hash(rec.text), "embed_model": model,
                "injection_flag": bool(labels), "created_at": created,
            })
        # Chunk ids are content-derived; identical text in two documents would collide.
        seen: set[str] = set()
        unique_rows = []
        for r in rows:
            if r["id"] in seen:
                continue
            seen.add(r["id"])
            unique_rows.append(r)
        self.chunks.replace_collection(cid, unique_rows)
        if any(flags_per_doc.values()):
            index.save(out_dir)  # persist injection flags in chunks.jsonl too

        for d in live:
            if d["id"] in per_doc_sections:
                self.documents.update(d["id"], status="completed", chunk_count=counts_per_doc.get(d["id"], 0),
                                      injection_flags=flags_per_doc.get(d["id"], 0), error=None)

        self.registry.put(LoadedCollection(cid, col["name"], col["kind"], index, version=version))
        self.collections.update(cid, status="ready", index_version=version, error=None,
                                doc_count=len(per_doc_sections), chunk_count=len(index.chunks))
        self._cleanup_versions(cid, keep=version)
        report = {
            "documents": len(per_doc_sections), "sections": len(lattice_docs), "chunks": len(index.chunks),
            "per_layer": index.manifest.get("per_layer"), "seconds": round(time.perf_counter() - t0, 2),
            "embedding_cache": enc.stats(), "injection_flagged_chunks": sum(flags_per_doc.values()),
            "version": version,
        }
        self.last_rebuild[cid] = report
        return report

    def _cleanup_versions(self, cid: str, keep: int | None) -> None:
        d = self._dir(cid)
        if not d.exists():
            return
        for p in d.glob("index_v*"):
            if keep is None or p.name != f"index_v{keep}":
                shutil.rmtree(p, ignore_errors=True)

    # ================================================================== stats
    def overview(self) -> dict[str, Any]:
        cols = self.collections.list()
        return {
            "collections": len(cols),
            "documents": sum(int(c["doc_count"] or 0) for c in cols),
            "chunks": sum(int(c["chunk_count"] or 0) for c in cols),
            "failed_documents": self.documents.count(status="failed"),
            "active_jobs": len(self.jobs.list(states=("uploaded", "queued", "extracting", "chunking",
                                                      "embedding", "indexing"))),
        }


__all__ = ["KnowledgeError", "KnowledgeService", "STATES", "safe_filename"]
