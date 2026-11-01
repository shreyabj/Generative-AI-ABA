"""Ingest ai4bharat/MSMARCO-XI.

Dataset shape (per row)
-----------------------
``query``                the question, machine-translated into the target language
``Eng_Query``            the original English question
``Answer`` / ``Eng_Answer``   the gold answer in both languages
``query_type``           MS MARCO's own taxonomy: DESCRIPTION / NUMERIC / ENTITY / ...
``passages.English_passages``     10 candidate passages, English
``passages.Translated_passages``  the same 10, translated
``passages.is_selected``          human label: which passage actually answers the query

Two things we exploit
---------------------
**Bilingual pairing.** English and translated passages are stored as *separate documents
sharing a query id*. Because the encoder is multilingual, a Hindi question can retrieve
the English passage and vice versa - genuine cross-lingual retrieval, and the answer
composer then prefers spans in the asker's language (see ``W_LANG`` in
:mod:`setu.answer.extractive`).

**Free ground truth.** ``is_selected`` is a human relevance label, so we get a real
retrieval evaluation set with no annotation work: recall@k against the selected passage,
and answer overlap against the gold ``Answer``. ``scripts/evaluate.py`` uses both. Most
RAG demos have no way to tell whether their retriever is any good; this one does.

Download strategy
-----------------
The full dataset is ~55 GB, so we only ever touch the validation shards (~460 MB per
language, ~6.4 GB for all fourteen, versus tens of GB for train).

We measured the obvious cheaper options before settling on a plain download:

* The Hub dataset-viewer ``/rows`` API returns HTTP 500 for this dataset - no server-side
  pagination available.
* Streaming through ``HfFileSystem`` does not help either, because each shard is written as
  a **single row group** of ~98k rows. ``read_row_group`` and ``iter_batches`` both have to
  pull the entire column chunk before decoding the first row: fetching 3 rows measured
  216 s, and 64 rows measured 250 s. Column projection does not rescue it, since the two
  passage columns are 445 MB of the 460 MB.

So ingest downloads the shard once through ``hf_hub_download`` - resumable, checksum-verified
and cached under ``HF_HOME``, so re-ingesting at a different row count is instant - then reads
it from local disk where row-group access actually is cheap. ``--rows`` controls how much of
the shard is turned into documents, not how much is transferred.

For a zero-download start, ``data/sample/corpus.jsonl`` ships with the repo and
``python -m setu.cli build --sample`` indexes it in a few seconds.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..chunking.base import Document
from ..text import XI_LANGS, normalize

REPO_ID = "ai4bharat/MSMARCO-XI"
# File stems differ slightly from the README table; these are the actual repo paths.
VAL_FILES: dict[str, str] = {
    "as": "asmval", "bn": "benval", "gu": "gujval", "hi": "hinval", "kn": "kanval",
    "ml": "malval", "mr": "marval", "ne": "nepval", "or": "orival", "pa": "panval",
    "sa": "sanval", "ta": "tamval", "te": "telval", "ur": "urdval",
}
TRAIN_FILES: dict[str, str] = {
    "as": "asmtrain", "bn": "bentrain", "gu": "gujtrain", "hi": "hintrain", "kn": "kantrain",
    "ml": "maltrain", "mr": "martrain", "ne": "neptrain", "or": "oritrain", "pa": "pantrain",
    "sa": "santrain", "ta": "tamtrain", "te": "teltrain", "ur": "urdtrain",
}

MIN_PASSAGE_CHARS = 90


@dataclass(slots=True)
class EvalQuery:
    """One gold record. Bilingual on both sides, so the same question can be evaluated
    in the Indic language and in English against the same labelled passages."""

    query: str
    lang: str
    query_id: int
    query_type: str
    gold_answer: str
    gold_doc_ids: list[str]
    eng_query: str = ""
    gold_answer_en: str = ""


def _hf_path(lang: str, split: str) -> str:
    table = VAL_FILES if split == "validation" else TRAIN_FILES
    stem = table.get(lang)
    if stem is None:
        raise ValueError(f"language '{lang}' not in MS MARCO-XI (have: {sorted(table)})")
    return f"datasets/{REPO_ID}/{split}/{stem}.parquet"


NEEDED_COLUMNS = ["query", "Eng_Query", "Answer", "Eng_Answer", "query_id", "query_type", "passages"]


def shard_path(lang: str, *, split: str = "validation") -> Path:
    """Download (or reuse from cache) the Parquet shard for one language."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=_hf_path(lang, split).split(f"{REPO_ID}/", 1)[1],
        )
    )


def iter_rows(
    lang: str, *, split: str = "validation", limit: int = 500, local_path: Path | None = None
) -> Iterator[dict[str, Any]]:
    """Yield up to ``limit`` rows, reading batches from the local Parquet shard."""
    import pyarrow.parquet as pq

    path = Path(local_path) if local_path else shard_path(lang, split=split)
    pf = pq.ParquetFile(path)
    yielded = 0
    for batch in pf.iter_batches(batch_size=256, columns=NEEDED_COLUMNS):
        for row in batch.to_pylist():
            yield row
            yielded += 1
            if yielded >= limit:
                return


def _passage_lists(row: dict[str, Any]) -> tuple[list[str], list[str], list[int]]:
    p = row.get("passages") or {}
    if isinstance(p, list):  # some readers surface a list-of-struct layout
        eng = [str(x.get("English_passages", "")) for x in p]
        tr = [str(x.get("Translated_passages", "")) for x in p]
        sel = [int(x.get("is_selected", 0) or 0) for x in p]
        return eng, tr, sel
    return (
        [str(x) for x in (p.get("English_passages") or [])],
        [str(x) for x in (p.get("Translated_passages") or [])],
        [int(x or 0) for x in (p.get("is_selected") or [])],
    )


def row_to_documents(
    row: dict[str, Any], lang: str, *, include_english: bool = True
) -> tuple[list[Document], EvalQuery | None]:
    eng_passages, tr_passages, is_selected = _passage_lists(row)
    if not (eng_passages or tr_passages):
        return [], None

    qid = int(row.get("query_id") or 0)
    qtype = str(row.get("query_type") or "UNKNOWN")
    query_local = normalize(str(row.get("query") or ""), casefold=False)
    query_eng = normalize(str(row.get("Eng_Query") or ""), casefold=False).lstrip(")").strip()
    answer_local = normalize(str(row.get("Answer") or ""), casefold=False)
    answer_eng = normalize(str(row.get("Eng_Answer") or ""), casefold=False)

    total = max(len(eng_passages), len(tr_passages))
    docs: list[Document] = []
    gold: list[str] = []

    for i in range(total):
        sel = is_selected[i] if i < len(is_selected) else 0
        for code, texts, q in (
            ("en", eng_passages, query_eng),
            (lang, tr_passages, query_local),
        ):
            if code == "en" and not include_english:
                continue
            if i >= len(texts):
                continue
            body = normalize(texts[i], casefold=False)
            if len(body) < MIN_PASSAGE_CHARS:
                continue
            doc_id = f"{qid}:{i}:{code}"
            docs.append(
                Document(
                    doc_id=doc_id,
                    text=body,
                    lang=code,
                    meta={
                        "query_id": qid,
                        "query": q,
                        "query_type": qtype,
                        "is_selected": bool(sel),
                        "passage_rank": i,
                        "passage_count": total,
                        "source": "msmarco-xi",
                        "split_lang": lang,
                    },
                )
            )
            if sel:
                gold.append(doc_id)

    ev: EvalQuery | None = None
    if query_local and answer_local and answer_local.lower() not in {"no answer present.", "none"}:
        ev = EvalQuery(
            query=query_local, lang=lang, query_id=qid, query_type=qtype,
            gold_answer=answer_local, gold_doc_ids=gold, eng_query=query_eng,
            gold_answer_en=answer_eng,
        )
    return docs, ev


def load_xi(
    langs: Sequence[str],
    *,
    split: str = "validation",
    rows_per_lang: int = 300,
    include_english: bool = True,
    on_progress: Any | None = None,
    on_shard: Any | None = None,
) -> tuple[list[Document], list[EvalQuery]]:
    docs: dict[str, Document] = {}
    evals: list[EvalQuery] = []
    for lang in langs:
        if lang not in XI_LANGS.values():
            raise ValueError(f"'{lang}' is not an MS MARCO-XI language")
        if on_shard:
            on_shard(lang)
        path = shard_path(lang, split=split)
        if on_shard:
            on_shard(lang, path)
        count = 0
        for row in iter_rows(lang, split=split, limit=rows_per_lang, local_path=path):
            new_docs, ev = row_to_documents(row, lang, include_english=include_english)
            for d in new_docs:
                docs.setdefault(d.doc_id, d)
            if ev:
                evals.append(ev)
            count += 1
            if on_progress and count % 50 == 0:
                on_progress(lang, count, len(docs))
        if on_progress:
            on_progress(lang, count, len(docs))
    return list(docs.values()), evals


# --------------------------------------------------------------------- JSONL I/O
def write_corpus(docs: Iterable[Document], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for d in docs:
            fh.write(
                json.dumps(
                    {"doc_id": d.doc_id, "text": d.text, "lang": d.lang, "meta": d.meta},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def read_corpus(path: Path) -> list[Document]:
    out: list[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(
                Document(doc_id=o["doc_id"], text=o["text"], lang=o.get("lang", "en"), meta=o.get("meta") or {})
            )
    return out


def write_queries(evals: Iterable[EvalQuery], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for e in evals:
            fh.write(
                json.dumps(
                    {
                        "query": e.query, "lang": e.lang, "query_id": e.query_id,
                        "query_type": e.query_type, "gold_answer": e.gold_answer,
                        "gold_doc_ids": e.gold_doc_ids, "eng_query": e.eng_query,
                        "gold_answer_en": e.gold_answer_en,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def read_queries(path: Path) -> list[EvalQuery]:
    out: list[EvalQuery] = []
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out.append(
                EvalQuery(
                    query=o["query"], lang=o.get("lang", "en"), query_id=o.get("query_id", 0),
                    query_type=o.get("query_type", "UNKNOWN"), gold_answer=o.get("gold_answer", ""),
                    gold_doc_ids=o.get("gold_doc_ids") or [], eng_query=o.get("eng_query", ""),
                    gold_answer_en=o.get("gold_answer_en", ""),
                )
            )
    return out


__all__ = [
    "REPO_ID",
    "TRAIN_FILES",
    "VAL_FILES",
    "EvalQuery",
    "iter_rows",
    "load_xi",
    "read_corpus",
    "read_queries",
    "row_to_documents",
    "shard_path",
    "write_corpus",
    "write_queries",
]
