"""Repository layer over :mod:`setu.storage.db`.

Every SQL statement in SETU lives here. Services and routes speak in dicts and ids, so
moving to PostgreSQL never touches them. Each method opens its own short transaction;
nothing holds a connection across an await.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import and_, delete, func, insert, select, update
from sqlalchemy.engine import Engine

from . import db
from .db import new_id, row_dict, utcnow


class _Repo:
    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine

    @property
    def engine(self) -> Engine:
        return self._engine or db.get_engine()


# ================================================================== collections
class CollectionRepo(_Repo):
    def create(self, *, name: str, description: str = "", kind: str = "user",
               embed_model: str, cid: str | None = None, settings: dict | None = None) -> dict[str, Any]:
        cid = cid or new_id("col_")
        with self.engine.begin() as c:
            c.execute(insert(db.collections).values(
                id=cid, name=name, description=description, kind=kind, embed_model=embed_model,
                status="empty", settings=settings or {}, created_at=utcnow(), updated_at=utcnow(),
            ))
        return self.get(cid)  # type: ignore[return-value]

    def get(self, cid: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.collections).where(db.collections.c.id == cid)).first()
        return row_dict(r) if r else None

    def by_name(self, name: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.collections).where(db.collections.c.name == name)).first()
        return row_dict(r) if r else None

    def list(self) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.collections).order_by(db.collections.c.created_at)).fetchall()
        return [row_dict(r) for r in rows]

    def update(self, cid: str, **values: Any) -> dict[str, Any] | None:
        values["updated_at"] = utcnow()
        with self.engine.begin() as c:
            c.execute(update(db.collections).where(db.collections.c.id == cid).values(**values))
        return self.get(cid)

    def delete(self, cid: str) -> None:
        with self.engine.begin() as c:
            c.execute(delete(db.chunks).where(db.chunks.c.collection_id == cid))
            c.execute(delete(db.documents).where(db.documents.c.collection_id == cid))
            c.execute(delete(db.collections).where(db.collections.c.id == cid))


# ==================================================================== documents
class DocumentRepo(_Repo):
    def create(self, **values: Any) -> dict[str, Any]:
        values.setdefault("id", new_id("doc_"))
        values.setdefault("created_at", utcnow())
        values["updated_at"] = utcnow()
        with self.engine.begin() as c:
            c.execute(insert(db.documents).values(**values))
        return self.get(values["id"])  # type: ignore[return-value]

    def get(self, did: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.documents).where(db.documents.c.id == did)).first()
        return row_dict(r) if r else None

    def find_hash(self, cid: str, content_hash: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.documents).where(and_(
                db.documents.c.collection_id == cid, db.documents.c.content_hash == content_hash,
            ))).first()
        return row_dict(r) if r else None

    def list(self, cid: str | None = None, *, status: str | None = None, q: str | None = None,
             limit: int = 50, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        conds = []
        if cid:
            conds.append(db.documents.c.collection_id == cid)
        if status:
            conds.append(db.documents.c.status == status)
        if q:
            like = f"%{q}%"
            conds.append(db.documents.c.title.ilike(like) | db.documents.c.filename.ilike(like)
                         | db.documents.c.source_url.ilike(like))
        where = and_(*conds) if conds else None
        stmt = select(db.documents)
        cnt = select(func.count()).select_from(db.documents)
        if where is not None:
            stmt, cnt = stmt.where(where), cnt.where(where)
        stmt = stmt.order_by(db.documents.c.created_at.desc()).limit(limit).offset(offset)
        with self.engine.connect() as c:
            rows = c.execute(stmt).fetchall()
            total = int(c.execute(cnt).scalar() or 0)
        return [row_dict(r) for r in rows], total

    def all_in(self, cid: str, statuses: Sequence[str] | None = None) -> list[dict[str, Any]]:
        stmt = select(db.documents).where(db.documents.c.collection_id == cid)
        if statuses:
            stmt = stmt.where(db.documents.c.status.in_(list(statuses)))
        with self.engine.connect() as c:
            rows = c.execute(stmt.order_by(db.documents.c.created_at)).fetchall()
        return [row_dict(r) for r in rows]

    def update(self, did: str, **values: Any) -> None:
        values["updated_at"] = utcnow()
        with self.engine.begin() as c:
            c.execute(update(db.documents).where(db.documents.c.id == did).values(**values))

    def delete(self, did: str) -> None:
        with self.engine.begin() as c:
            c.execute(delete(db.chunks).where(db.chunks.c.document_id == did))
            c.execute(delete(db.documents).where(db.documents.c.id == did))

    def count(self, cid: str | None = None, status: str | None = None) -> int:
        stmt = select(func.count()).select_from(db.documents)
        if cid:
            stmt = stmt.where(db.documents.c.collection_id == cid)
        if status:
            stmt = stmt.where(db.documents.c.status == status)
        with self.engine.connect() as c:
            return int(c.execute(stmt).scalar() or 0)

    def recent(self, limit: int = 8) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.documents).order_by(db.documents.c.created_at.desc()).limit(limit)).fetchall()
        return [row_dict(r) for r in rows]


# ======================================================================= chunks
class ChunkRepo(_Repo):
    def replace_collection(self, cid: str, rows: Iterable[dict[str, Any]]) -> int:
        rows = list(rows)
        with self.engine.begin() as c:
            c.execute(delete(db.chunks).where(db.chunks.c.collection_id == cid))
            for i in range(0, len(rows), 500):
                c.execute(insert(db.chunks), rows[i:i + 500])
        return len(rows)

    def list_for_document(self, did: str, *, layer: str | None = None, limit: int = 50,
                          offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        conds = [db.chunks.c.document_id == did]
        if layer:
            conds.append(db.chunks.c.layer == layer)
        where = and_(*conds)
        with self.engine.connect() as c:
            rows = c.execute(select(db.chunks).where(where).order_by(
                db.chunks.c.section_id, db.chunks.c.layer, db.chunks.c.chunk_index,
            ).limit(limit).offset(offset)).fetchall()
            total = int(c.execute(select(func.count()).select_from(db.chunks).where(where)).scalar() or 0)
        return [row_dict(r) for r in rows], total

    def counts_by_document(self, cid: str) -> dict[str, int]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.chunks.c.document_id, func.count()).where(
                db.chunks.c.collection_id == cid).group_by(db.chunks.c.document_id)).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def layer_counts(self, cid: str | None = None) -> dict[str, int]:
        stmt = select(db.chunks.c.layer, func.count()).group_by(db.chunks.c.layer)
        if cid:
            stmt = stmt.where(db.chunks.c.collection_id == cid)
        with self.engine.connect() as c:
            return {r[0]: int(r[1]) for r in c.execute(stmt).fetchall()}


# ========================================================================= jobs
class JobRepo(_Repo):
    def create(self, *, kind: str, collection_id: str | None = None, document_id: str | None = None,
               payload: dict | None = None, state: str = "uploaded") -> dict[str, Any]:
        jid = new_id("job_")
        with self.engine.begin() as c:
            c.execute(insert(db.jobs).values(
                id=jid, kind=kind, collection_id=collection_id, document_id=document_id,
                state=state, progress=0.0, message="", payload=payload or {}, created_at=utcnow(),
            ))
        return self.get(jid)  # type: ignore[return-value]

    def get(self, jid: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.jobs).where(db.jobs.c.id == jid)).first()
        return row_dict(r) if r else None

    def update(self, jid: str, **values: Any) -> None:
        with self.engine.begin() as c:
            c.execute(update(db.jobs).where(db.jobs.c.id == jid).values(**values))

    def list(self, *, collection_id: str | None = None, states: Sequence[str] | None = None,
             limit: int = 50) -> list[dict[str, Any]]:
        stmt = select(db.jobs)
        if collection_id:
            stmt = stmt.where(db.jobs.c.collection_id == collection_id)
        if states:
            stmt = stmt.where(db.jobs.c.state.in_(list(states)))
        with self.engine.connect() as c:
            rows = c.execute(stmt.order_by(db.jobs.c.created_at.desc()).limit(limit)).fetchall()
        return [row_dict(r) for r in rows]

    def fail_interrupted(self) -> int:
        """Jobs left mid-flight by a crash or restart are marked failed, never resumed silently."""
        live = ("uploaded", "extracting", "chunking", "embedding", "indexing", "queued", "running")
        with self.engine.begin() as c:
            res = c.execute(update(db.jobs).where(db.jobs.c.state.in_(live)).values(
                state="failed", error="interrupted by server restart", finished_at=utcnow()))
        return int(res.rowcount or 0)


# ============================================================ embedding cache
class EmbeddingCacheRepo(_Repo):
    def get_many(self, model: str, hashes: Sequence[str]) -> dict[str, bytes]:
        out: dict[str, bytes] = {}
        with self.engine.connect() as c:
            for i in range(0, len(hashes), 900):
                part = list(hashes[i:i + 900])
                rows = c.execute(select(db.embedding_cache.c.text_hash, db.embedding_cache.c.vector).where(
                    and_(db.embedding_cache.c.model == model, db.embedding_cache.c.text_hash.in_(part))
                )).fetchall()
                out.update({r[0]: r[1] for r in rows})
        return out

    def put_many(self, model: str, dim: int, items: Sequence[tuple[str, bytes]]) -> None:
        if not items:
            return
        with self.engine.begin() as c:
            existing = set()
            for i in range(0, len(items), 900):
                part = [h for h, _ in items[i:i + 900]]
                existing |= {r[0] for r in c.execute(select(db.embedding_cache.c.text_hash).where(
                    and_(db.embedding_cache.c.model == model, db.embedding_cache.c.text_hash.in_(part))))}
            fresh = [{"model": model, "text_hash": h, "dim": dim, "vector": v}
                     for h, v in items if h not in existing]
            seen: set[str] = set()
            fresh = [f for f in fresh if not (f["text_hash"] in seen or seen.add(f["text_hash"]))]
            for i in range(0, len(fresh), 500):
                c.execute(insert(db.embedding_cache), fresh[i:i + 500])

    def count(self) -> int:
        with self.engine.connect() as c:
            return int(c.execute(select(func.count()).select_from(db.embedding_cache)).scalar() or 0)


# ================================================================ conversations
class ConversationRepo(_Repo):
    def create(self, *, session_id: str, title: str = "New research",
               collection_ids: list[str] | None = None) -> dict[str, Any]:
        cid = new_id("conv_")
        with self.engine.begin() as c:
            c.execute(insert(db.conversations).values(
                id=cid, session_id=session_id, title=title, collection_ids=collection_ids or [],
                created_at=utcnow(), updated_at=utcnow(),
            ))
        return self.get(cid)  # type: ignore[return-value]

    def get(self, cid: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.conversations).where(db.conversations.c.id == cid)).first()
        return row_dict(r) if r else None

    def list(self, session_id: str, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.conversations).where(db.conversations.c.session_id == session_id)
                             .order_by(db.conversations.c.updated_at.desc()).limit(limit).offset(offset)).fetchall()
            counts = dict(c.execute(select(db.messages.c.conversation_id, func.count()).group_by(
                db.messages.c.conversation_id)).fetchall())
        out = []
        for r in rows:
            d = row_dict(r)
            d["message_count"] = int(counts.get(d["id"], 0))
            out.append(d)
        return out

    def recent(self, limit: int = 6) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.conversations).order_by(db.conversations.c.updated_at.desc())
                             .limit(limit)).fetchall()
        return [row_dict(r) for r in rows]

    def count(self) -> int:
        with self.engine.connect() as c:
            return int(c.execute(select(func.count()).select_from(db.conversations)).scalar() or 0)

    def update(self, cid: str, **values: Any) -> None:
        values["updated_at"] = utcnow()
        with self.engine.begin() as c:
            c.execute(update(db.conversations).where(db.conversations.c.id == cid).values(**values))

    def delete(self, cid: str) -> None:
        with self.engine.begin() as c:
            c.execute(delete(db.messages).where(db.messages.c.conversation_id == cid))
            c.execute(delete(db.conversations).where(db.conversations.c.id == cid))

    def add_message(self, conversation_id: str, *, role: str, content: str, **values: Any) -> dict[str, Any]:
        mid = new_id("msg_")
        with self.engine.begin() as c:
            c.execute(insert(db.messages).values(
                id=mid, conversation_id=conversation_id, role=role, content=content,
                created_at=utcnow(), **values,
            ))
            c.execute(update(db.conversations).where(db.conversations.c.id == conversation_id)
                      .values(updated_at=utcnow()))
        return {"id": mid}

    def messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.messages).where(db.messages.c.conversation_id == conversation_id)
                             .order_by(db.messages.c.created_at).limit(limit)).fetchall()
        return [row_dict(r) for r in rows]


# ======================================================================= traces
class TraceRepo(_Repo):
    def save(self, *, turn_id: str, session_id: str | None, conversation_id: str | None, kind: str,
             query: str, status: str, style: str | None, pipeline_ms: float, result: dict[str, Any],
             retain: int = 2000) -> None:
        with self.engine.begin() as c:
            c.execute(delete(db.traces).where(db.traces.c.turn_id == turn_id))
            c.execute(insert(db.traces).values(
                turn_id=turn_id, session_id=session_id, conversation_id=conversation_id, kind=kind,
                query=query, status=status, style=style, pipeline_ms=pipeline_ms, result=result,
                created_at=utcnow(),
            ))
            total = int(c.execute(select(func.count()).select_from(db.traces)).scalar() or 0)
            if total > retain:
                cutoff = c.execute(select(db.traces.c.created_at).order_by(db.traces.c.created_at.desc())
                                   .offset(retain).limit(1)).scalar()
                if cutoff is not None:
                    c.execute(delete(db.traces).where(db.traces.c.created_at <= cutoff))

    def get(self, turn_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.traces).where(db.traces.c.turn_id == turn_id)).first()
        return row_dict(r) if r else None

    def list(self, *, session_id: str | None = None, limit: int = 30, offset: int = 0) -> list[dict[str, Any]]:
        cols = [db.traces.c.turn_id, db.traces.c.session_id, db.traces.c.conversation_id, db.traces.c.kind,
                db.traces.c.query, db.traces.c.status, db.traces.c.style, db.traces.c.pipeline_ms,
                db.traces.c.created_at]
        stmt = select(*cols)
        if session_id:
            stmt = stmt.where(db.traces.c.session_id == session_id)
        with self.engine.connect() as c:
            rows = c.execute(stmt.order_by(db.traces.c.created_at.desc()).limit(limit).offset(offset)).fetchall()
        return [row_dict(r) for r in rows]

    def stats(self) -> dict[str, Any]:
        with self.engine.connect() as c:
            total = int(c.execute(select(func.count()).select_from(db.traces)).scalar() or 0)
            by_status = dict(c.execute(select(db.traces.c.status, func.count()).group_by(db.traces.c.status)).fetchall())
            by_style = dict(c.execute(select(db.traces.c.style, func.count()).group_by(db.traces.c.style)).fetchall())
        return {"total": total, "by_status": by_status, "by_style": {str(k): v for k, v in by_style.items()}}


# =================================================================== evaluation
class EvalRepo(_Repo):
    def upsert_dataset(self, *, name: str, description: str, collection_ids: list[str],
                       label_source: str, items: list[dict[str, Any]]) -> dict[str, Any]:
        with self.engine.begin() as c:
            r = c.execute(select(db.eval_datasets).where(db.eval_datasets.c.name == name)).first()
            if r:
                did = r._mapping["id"]
                c.execute(update(db.eval_datasets).where(db.eval_datasets.c.id == did).values(
                    description=description, collection_ids=collection_ids, label_source=label_source))
                c.execute(delete(db.eval_items).where(db.eval_items.c.dataset_id == did))
            else:
                did = new_id("ds_")
                c.execute(insert(db.eval_datasets).values(
                    id=did, name=name, description=description, collection_ids=collection_ids,
                    label_source=label_source, created_at=utcnow()))
            for it in items:
                c.execute(insert(db.eval_items).values(id=new_id("it_"), dataset_id=did, **it))
        return self.dataset(did)  # type: ignore[return-value]

    def dataset(self, did: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.eval_datasets).where(db.eval_datasets.c.id == did)).first()
            if not r:
                return None
            d = row_dict(r)
            d["item_count"] = int(c.execute(select(func.count()).select_from(db.eval_items)
                                            .where(db.eval_items.c.dataset_id == did)).scalar() or 0)
        return d

    def datasets(self) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            ids = [r[0] for r in c.execute(select(db.eval_datasets.c.id).order_by(db.eval_datasets.c.created_at))]
        return [d for d in (self.dataset(i) for i in ids) if d]

    def items(self, did: str) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.eval_items).where(db.eval_items.c.dataset_id == did)).fetchall()
        return [row_dict(r) for r in rows]

    def create_run(self, *, dataset_id: str, name: str, config: dict[str, Any]) -> dict[str, Any]:
        rid = new_id("run_")
        with self.engine.begin() as c:
            c.execute(insert(db.eval_runs).values(id=rid, dataset_id=dataset_id, name=name, config=config,
                                                  status="queued", created_at=utcnow()))
        return self.run(rid)  # type: ignore[return-value]

    def run(self, rid: str) -> dict[str, Any] | None:
        with self.engine.connect() as c:
            r = c.execute(select(db.eval_runs).where(db.eval_runs.c.id == rid)).first()
        return row_dict(r) if r else None

    def runs(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.eval_runs).order_by(db.eval_runs.c.created_at.desc()).limit(limit)).fetchall()
        return [row_dict(r) for r in rows]

    def update_run(self, rid: str, **values: Any) -> None:
        with self.engine.begin() as c:
            c.execute(update(db.eval_runs).where(db.eval_runs.c.id == rid).values(**values))

    def add_results(self, rid: str, rows: list[dict[str, Any]]) -> None:
        with self.engine.begin() as c:
            for r in rows:
                c.execute(insert(db.eval_results).values(id=new_id("res_"), run_id=rid, **r))

    def results(self, rid: str) -> list[dict[str, Any]]:
        with self.engine.connect() as c:
            rows = c.execute(select(db.eval_results).where(db.eval_results.c.run_id == rid)).fetchall()
        return [row_dict(r) for r in rows]

    def delete_run(self, rid: str) -> None:
        with self.engine.begin() as c:
            c.execute(delete(db.eval_results).where(db.eval_results.c.run_id == rid))
            c.execute(delete(db.eval_runs).where(db.eval_runs.c.id == rid))


__all__ = [
    "ChunkRepo", "CollectionRepo", "ConversationRepo", "DocumentRepo", "EmbeddingCacheRepo",
    "EvalRepo", "JobRepo", "TraceRepo",
]
