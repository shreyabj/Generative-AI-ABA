"""Metadata database (SETU 2.0).

SQLite by default, through SQLAlchemy Core so the same tables run on PostgreSQL by
changing ``SETU_DATABASE_URL``. Vectors are *not* stored here: they live in each
collection's index files (or Qdrant). This database is the record of what exists -
collections, documents, chunk metadata, ingestion jobs, conversations, traces and
evaluation runs - and the embedding cache that makes collection rebuilds cheap.

Schema changes are applied by :func:`init_db`: ``create_all`` creates missing tables,
and ``schema_version`` records the version so a future migration step has something to
compare against. There is exactly one version so far.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine

from ..config import settings

SCHEMA_VERSION = 1
metadata = MetaData()


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"


schema_version = Table(
    "schema_version", metadata,
    Column("version", Integer, primary_key=True),
    Column("applied_at", DateTime(timezone=True), default=utcnow),
)

collections = Table(
    "collections", metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("description", Text, default=""),
    Column("kind", String(20), nullable=False, default="user"),  # builtin | user | synthetic
    Column("embed_model", String(200), nullable=False),
    Column("index_version", Integer, nullable=False, default=0),
    Column("status", String(20), nullable=False, default="empty"),  # empty|ready|rebuilding|error
    Column("error", Text),
    Column("doc_count", Integer, nullable=False, default=0),
    Column("chunk_count", Integer, nullable=False, default=0),
    Column("settings", JSON, default=dict),
    Column("created_at", DateTime(timezone=True), default=utcnow),
    Column("updated_at", DateTime(timezone=True), default=utcnow, onupdate=utcnow),
)

documents = Table(
    "documents", metadata,
    Column("id", String(40), primary_key=True),
    Column("collection_id", String(40), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False),
    Column("source_type", String(20), nullable=False),  # upload | url | builtin
    Column("filename", String(400)),
    Column("source_url", Text),
    Column("domain", String(255)),
    Column("title", Text),
    Column("mime_type", String(120)),
    Column("byte_size", Integer, default=0),
    Column("content_hash", String(64), nullable=False),
    Column("lang", String(8)),
    Column("page_count", Integer),
    Column("char_count", Integer, default=0),
    Column("section_count", Integer, default=0),
    Column("chunk_count", Integer, default=0),
    Column("published_at", String(40)),
    Column("retrieved_at", DateTime(timezone=True)),
    Column("status", String(20), nullable=False, default="uploaded"),
    Column("error", Text),
    Column("synthetic", Boolean, default=False),
    Column("injection_flags", Integer, default=0),
    Column("meta", JSON, default=dict),
    Column("created_at", DateTime(timezone=True), default=utcnow),
    Column("updated_at", DateTime(timezone=True), default=utcnow, onupdate=utcnow),
    UniqueConstraint("collection_id", "content_hash", name="uq_doc_hash"),
)

chunks = Table(
    "chunks", metadata,
    Column("id", String(40), primary_key=True),  # lattice chunk_id, unique per collection
    Column("collection_id", String(40), ForeignKey("collections.id", ondelete="CASCADE"), nullable=False),
    Column("document_id", String(40), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
    Column("section_id", String(80), nullable=False),
    Column("layer", String(12), nullable=False),
    Column("chunk_index", Integer, nullable=False),
    Column("page_start", Integer),
    Column("page_end", Integer),
    Column("section_heading", Text),
    Column("char_start", Integer),
    Column("char_end", Integer),
    Column("parent_id", String(40)),
    Column("lang", String(8)),
    Column("text", Text, nullable=False),
    Column("content_hash", String(64), nullable=False),
    Column("embed_model", String(200), nullable=False),
    Column("injection_flag", Boolean, default=False),
    Column("created_at", DateTime(timezone=True), default=utcnow),
)
Index("ix_chunks_doc", chunks.c.collection_id, chunks.c.document_id)

jobs = Table(
    "jobs", metadata,
    Column("id", String(40), primary_key=True),
    Column("kind", String(20), nullable=False),  # ingest | reindex | rebuild | eval
    Column("collection_id", String(40)),
    Column("document_id", String(40)),
    Column("state", String(20), nullable=False, default="uploaded"),
    Column("progress", Float, default=0.0),
    Column("message", Text, default=""),
    Column("error", Text),
    Column("payload", JSON, default=dict),
    Column("created_at", DateTime(timezone=True), default=utcnow),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
)
Index("ix_jobs_state", jobs.c.state)

embedding_cache = Table(
    "embedding_cache", metadata,
    Column("model", String(200), primary_key=True),
    Column("text_hash", String(64), primary_key=True),
    Column("dim", Integer, nullable=False),
    Column("vector", LargeBinary, nullable=False),
)

conversations = Table(
    "conversations", metadata,
    Column("id", String(40), primary_key=True),
    Column("session_id", String(80), nullable=False),
    Column("title", Text, default="New research"),
    Column("collection_ids", JSON, default=list),
    Column("created_at", DateTime(timezone=True), default=utcnow),
    Column("updated_at", DateTime(timezone=True), default=utcnow, onupdate=utcnow),
)
Index("ix_conv_session", conversations.c.session_id)

messages = Table(
    "messages", metadata,
    Column("id", String(40), primary_key=True),
    Column("conversation_id", String(40), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
    Column("role", String(12), nullable=False),  # user | assistant
    Column("content", Text, nullable=False),
    Column("resolved_query", Text),
    Column("style", String(20)),
    Column("lang", String(8)),
    Column("turn_id", String(40)),
    Column("payload", JSON, default=dict),  # full research result for assistant turns
    Column("created_at", DateTime(timezone=True), default=utcnow),
)
Index("ix_msg_conv", messages.c.conversation_id, messages.c.created_at)

traces = Table(
    "traces", metadata,
    Column("turn_id", String(40), primary_key=True),
    Column("session_id", String(80)),
    Column("conversation_id", String(40)),
    Column("kind", String(20), default="research"),
    Column("query", Text),
    Column("status", String(20)),
    Column("style", String(20)),
    Column("pipeline_ms", Float),
    Column("result", JSON),
    Column("created_at", DateTime(timezone=True), default=utcnow),
)
Index("ix_traces_session", traces.c.session_id, traces.c.created_at)

eval_datasets = Table(
    "eval_datasets", metadata,
    Column("id", String(40), primary_key=True),
    Column("name", String(200), nullable=False, unique=True),
    Column("description", Text, default=""),
    Column("collection_ids", JSON, default=list),
    Column("label_source", String(40), default="curated"),
    Column("created_at", DateTime(timezone=True), default=utcnow),
)

eval_items = Table(
    "eval_items", metadata,
    Column("id", String(40), primary_key=True),
    Column("dataset_id", String(40), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False),
    Column("question", Text, nullable=False),
    Column("lang", String(8)),
    Column("relevant_doc_ids", JSON, default=list),
    Column("reference_answer", Text),
    Column("label_source", String(40), default="curated"),
    Column("reviewed_by", String(80)),
    Column("tags", JSON, default=list),
)

eval_runs = Table(
    "eval_runs", metadata,
    Column("id", String(40), primary_key=True),
    Column("dataset_id", String(40), ForeignKey("eval_datasets.id", ondelete="CASCADE"), nullable=False),
    Column("name", String(200)),
    Column("config", JSON, default=dict),
    Column("status", String(20), default="queued"),
    Column("error", Text),
    Column("metrics", JSON, default=dict),
    Column("started_at", DateTime(timezone=True)),
    Column("finished_at", DateTime(timezone=True)),
    Column("created_at", DateTime(timezone=True), default=utcnow),
)

eval_results = Table(
    "eval_results", metadata,
    Column("id", String(40), primary_key=True),
    Column("run_id", String(40), ForeignKey("eval_runs.id", ondelete="CASCADE"), nullable=False),
    Column("item_id", String(40)),
    Column("question", Text),
    Column("metrics", JSON, default=dict),
    Column("judge", String(20), default="automated"),
)
Index("ix_eval_results_run", eval_results.c.run_id)


# ======================================================================= engine
_engine: Engine | None = None
_lock = threading.Lock()


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def make_engine(url: str) -> Engine:
    kwargs: dict[str, Any] = {
        "future": True,
        "json_serializer": lambda o: json.dumps(o, ensure_ascii=False, default=_json_default),
    }
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
    eng = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(eng, "connect")
        def _pragmas(dbapi_conn, _rec):  # noqa: ANN001
            cur = dbapi_conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.execute("PRAGMA synchronous=NORMAL")
            cur.close()
    return eng


def init_db(url: str | None = None) -> Engine:
    eng = make_engine(url or settings.db_url)
    metadata.create_all(eng)
    with eng.begin() as conn:
        have = conn.execute(schema_version.select()).fetchall()
        if not have:
            conn.execute(schema_version.insert().values(version=SCHEMA_VERSION))
    return eng


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                _engine = init_db()
    return _engine


def set_engine(eng: Engine | None) -> None:
    """Swap the process-wide engine (tests use an isolated database)."""
    global _engine
    _engine = eng


def row_dict(row: Any) -> dict[str, Any]:
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            if v.tzinfo is None:
                v = v.replace(tzinfo=timezone.utc)
            d[k] = v.isoformat()
    return d


__all__ = [
    "SCHEMA_VERSION", "chunks", "collections", "conversations", "documents", "embedding_cache",
    "eval_datasets", "eval_items", "eval_results", "eval_runs", "get_engine", "init_db", "jobs",
    "messages", "metadata", "new_id", "row_dict", "set_engine", "traces", "utcnow",
]
