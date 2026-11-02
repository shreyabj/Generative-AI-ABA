"""Shared helpers for the measurement scripts (bench / evaluate / redteam / ablate).

Recreated in SETU 2.0: the original was excluded from the distributed archive by the
``_*.py`` scratch rule in .gitignore (now whitelisted).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))


def hr(title: str) -> None:
    print(f"\n{title}\n" + "-" * max(len(title), 60))


def load_pipeline():
    """``(pipeline, index, warmup_report)`` - the same boot path the server uses."""
    from setu.ingest.build import load_or_build
    from setu.pipeline import SetuPipeline

    index = load_or_build(log=lambda *_: None)
    pipe = SetuPipeline(index)
    warm = pipe.warm()
    return pipe, index, warm


def load_gold():
    """Gold queries: data/corpus/queries.jsonl when ingested, else derived from the index.

    The bundled sample ships no queries.jsonl, so the fallback builds one EvalQuery per
    (source question, language) whose group contains a human ``is_selected`` passage.
    Those have no gold answer string, so answer-recall is reported as 0 for them.
    """
    from setu.ingest.msmarco_xi import EvalQuery, read_queries

    path = ROOT / "data" / "corpus" / "queries.jsonl"
    rows = read_queries(path) if path.exists() else []
    if rows:
        return rows
    from setu.index.corpus import CorpusIndex

    idx = CorpusIndex.load()
    groups: dict[int, dict] = {}
    for did, rec in idx.docs.items():
        qid = rec.meta.get("query_id")
        if qid is None:
            continue
        g = groups.setdefault(int(qid), {"gold": [], "q": {}, "type": rec.meta.get("query_type") or ""})
        if rec.meta.get("is_selected"):
            g["gold"].append(did)
        if rec.meta.get("query"):
            g["q"].setdefault(rec.lang, str(rec.meta["query"]))
    out = []
    for qid, g in sorted(groups.items()):
        if not g["gold"]:
            continue
        for lang, q in sorted(g["q"].items()):
            out.append(EvalQuery(query=q, lang=lang, query_id=qid, query_type=g["type"], gold_answer="",
                                 gold_doc_ids=sorted(g["gold"])))
    return out
