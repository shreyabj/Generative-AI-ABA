"""SETU command line.

    python -m setu.cli ingest --langs hi,ta --rows 400   download + index MS MARCO-XI
    python -m setu.cli build  --sample                   index the bundled sample corpus
    python -m setu.cli serve                             run the API + UI
    python -m setu.cli ask "how many calories in a banana"
    python -m setu.cli stats                             index and lattice summary
    python -m setu.cli sample --n 420                    write the offline sample corpus

Measurement scripts live alongside, so they can be run in CI:

    python scripts/bench.py    --n 300      P50 / P70 / P100 latency report
    python scripts/evaluate.py             retrieval recall + answer overlap vs gold labels
    python scripts/redteam.py              guardrail suite
    python scripts/ablate.py               chunking strategy ablation
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .config import settings
from .text import LANG_NAMES

BANNER = r"""
   ___ ___ _____ _   _
  / __| __|_   _| | | |    Speech -> Evidence -> Truth -> Utterance
  \__ \ _|  | | | |_| |    sub-200ms multilingual voice RAG
  |___/___| |_|  \___/     MS MARCO-XI
"""


def _log(msg: str = "") -> None:
    print(msg, flush=True)


# ==================================================================== commands
def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingest.build import build_index
    from .ingest.msmarco_xi import load_xi, write_corpus, write_queries

    langs = [s.strip() for s in args.langs.split(",") if s.strip()]
    _log(BANNER)
    _log(f"ingest  languages : {', '.join(f'{c} ({LANG_NAMES.get(c, c)})' for c in langs)}")
    _log(f"        rows/lang : {args.rows}   split: {args.split}")
    _log(f"        english   : {'paired' if not args.no_english else 'excluded'}")
    _log("")
    _log("Each language shard is ~460 MB and is cached by huggingface_hub after the first run.")
    _log("")

    def on_shard(lang: str, path: Path | None = None) -> None:
        if path is None:
            _log(f"  [{lang}] resolving shard (downloading if not cached)...")
        else:
            _log(f"  [{lang}] shard ready: {path.stat().st_size / 1e6:.0f} MB")

    def on_progress(lang: str, rows: int, docs: int) -> None:
        _log(f"  [{lang}] {rows} rows -> {docs} documents")

    docs, evals = load_xi(
        langs, split=args.split, rows_per_lang=args.rows,
        include_english=not args.no_english, on_progress=on_progress, on_shard=on_shard,
    )
    if not docs:
        _log("no documents produced - check the language codes")
        return 1

    n = write_corpus(docs, settings.corpus_file)
    q = write_queries(evals, settings.data_dir / "corpus" / "queries.jsonl")
    _log("")
    _log(f"corpus  {n} documents -> {settings.corpus_file}")
    _log(f"eval    {q} gold queries -> {settings.data_dir / 'corpus' / 'queries.jsonl'}")
    _log("")
    _log("building index")
    _, report = build_index(docs, log=lambda m: _log(m))
    _log("")
    _log(json.dumps({k: report[k] for k in ("documents", "chunks", "sentences", "per_layer")}, indent=2))
    _log("")
    _log("ready:  python -m setu.cli serve")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .ingest.build import build_index
    from .ingest.msmarco_xi import read_corpus

    path = (
        settings.sample_corpus_file if args.sample
        else (Path(args.corpus) if args.corpus else settings.corpus_file)
    )
    if not path.exists():
        _log(f"corpus not found: {path}")
        _log("run:  python -m setu.cli ingest --langs hi,ta --rows 400")
        return 1
    docs = read_corpus(path)
    _log(BANNER)
    _log(f"build   {len(docs)} documents from {path}")
    _, report = build_index(docs, log=lambda m: _log(m))
    _log("")
    _log(f"done: {report['chunks']} chunks across {len(report['per_layer'])} lattice layers")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    _log(BANNER)
    _log(f"serving on http://{args.host}:{args.port}   (UI + /api/docs)")
    if not settings.stt_enabled:
        _log("note: ELEVENLABS_API_KEY not set - the UI will use browser speech recognition")
    uvicorn.run(
        "setu.app:app", host=args.host, port=args.port,
        reload=args.reload, log_level=settings.log_level,
    )
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    from .harness.contracts import AnswerMode, AskRequest
    from .ingest.build import load_or_build
    from .pipeline import SetuPipeline

    index = load_or_build(log=lambda m: None)
    pipe = SetuPipeline(index)
    warm = pipe.warm()

    async def run() -> int:
        result = await pipe.ask(
            AskRequest(text=args.query, lang=args.lang, mode=AnswerMode(args.mode), top_k=args.top_k)
        )
        if args.json:
            _log(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0

        _log("")
        _log(f"  Q  {result.query}")
        _log("")
        _log(f"  A  {result.answer or '(no answer)'}")
        _log("")
        _log(f"     status {result.status.value}   trust {result.trust_score:.1f}/100   "
             f"path {result.path}   lang {result.answer_lang}")
        _log(f"     pipeline {result.pipeline_ms:.1f} ms   "
             f"SLO {'MET' if result.slo_met else 'BREACHED'} ({settings.budget.total_ms:.0f} ms)")
        if result.route:
            layers = " ".join(f"{lp.layer.value}:{lp.k}" for lp in result.route.layers)
            _log(f"     route  {result.route.shape.value}  [{layers}]  -> {result.route.reason}")
        _log("")
        _log("     stages")
        for s in result.spans:
            if s.duration_ms < 0.05 and s.status == "ok":
                continue
            flag = "" if s.status == "ok" else f"  ({s.status})"
            _log(f"       {s.duration_ms:7.2f} ms  {'  ' * s.depth}{s.name}{flag}")
        _log("")
        _log("     guardrails")
        for v in result.guards.verdicts:
            mark = {"allow": "ok  ", "flag": "flag", "block": "BLOK"}[v.decision.value]
            _log(f"       [{mark}] {v.gate:<24} {v.score:>6.3f}  {v.reason}")
        if result.citations:
            _log("")
            _log("     evidence")
            for c in result.citations:
                _log(f"       [{c.marker}] {c.layer.value:<9} {c.doc_id:<20} {c.score:.3f}")
                _log(f"           {c.snippet[:150]}")
        if result.warnings:
            _log("")
            for w in result.warnings:
                _log(f"     ! {w}")
        _log("")
        return 0

    if args.verbose:
        _log(f"(warmup {warm['warm_ms']:.0f} ms, {index.stats()['n_chunks']} chunks)")
    return asyncio.run(run())


def cmd_stats(_: argparse.Namespace) -> int:
    from .index.corpus import CorpusIndex

    if not CorpusIndex.exists():
        _log("no index found - run `python -m setu.cli build --sample`")
        return 1
    index = CorpusIndex.load()
    _log(json.dumps(index.stats(), indent=2, ensure_ascii=False))
    return 0


def cmd_sample(args: argparse.Namespace) -> int:
    """Write a small offline corpus so the repo runs with no download."""
    from .ingest.msmarco_xi import read_corpus, write_corpus

    src = Path(args.source) if args.source else settings.corpus_file
    if not src.exists():
        _log(f"source corpus not found: {src}")
        return 1
    docs = read_corpus(src)
    # Keep whole query groups together so gold labels stay meaningful.
    by_query: dict[str, list] = {}
    for d in docs:
        by_query.setdefault(str(d.meta.get("query_id")), []).append(d)
    picked: list = []
    for group in by_query.values():
        if len(picked) >= args.n:
            break
        picked.extend(group)
    n = write_corpus(picked[: args.n], settings.sample_corpus_file)
    _log(f"wrote {n} documents to {settings.sample_corpus_file}")
    return 0


# ====================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="setu", description="SETU - sub-200ms multilingual voice RAG")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", help="download MS MARCO-XI shards, build the corpus and index")
    ing.add_argument("--langs", default="hi,ta", help="comma-separated ISO-639-1 codes (default: hi,ta)")
    ing.add_argument("--rows", type=int, default=400, help="rows per language (default: 400)")
    ing.add_argument("--split", default="validation", choices=["validation", "train"])
    ing.add_argument("--no-english", action="store_true", help="skip the paired English passages")
    ing.set_defaults(fn=cmd_ingest)

    b = sub.add_parser("build", help="build the index from an existing corpus JSONL")
    b.add_argument("--corpus", help="path to a corpus JSONL")
    b.add_argument("--sample", action="store_true", help="use the bundled sample corpus")
    b.set_defaults(fn=cmd_build)

    s = sub.add_parser("serve", help="run the API and UI")
    s.add_argument("--host", default=settings.host)
    s.add_argument("--port", type=int, default=settings.port)
    s.add_argument("--reload", action="store_true")
    s.set_defaults(fn=cmd_serve)

    a = sub.add_parser("ask", help="ask one question from the terminal")
    a.add_argument("query")
    a.add_argument("--lang", default=None)
    a.add_argument("--mode", default="reflex", choices=["reflex", "deliberate", "auto"])
    a.add_argument("--top-k", type=int, default=None, dest="top_k")
    a.add_argument("--json", action="store_true")
    a.add_argument("--verbose", action="store_true")
    a.set_defaults(fn=cmd_ask)

    st = sub.add_parser("stats", help="print index and lattice statistics")
    st.set_defaults(fn=cmd_stats)

    sm = sub.add_parser("sample", help="write a small offline sample corpus")
    sm.add_argument("--n", type=int, default=420)
    sm.add_argument("--source", default=None)
    sm.set_defaults(fn=cmd_sample)

    sd = sub.add_parser("seed", help="queue the demo collections on a running server (SETU 2.0)")
    sd.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of the running server")
    sd.add_argument("--no-business", action="store_true")
    sd.add_argument("--no-redteam", action="store_true")
    sd.add_argument("--wait", type=float, default=900.0, help="seconds to wait for ingestion to finish")
    sd.set_defaults(fn=cmd_seed)

    return p


def cmd_seed(args: argparse.Namespace) -> int:
    """Seed via the running server's API so its in-memory registry picks up the collections."""
    import time

    import httpx

    headers = {"X-SETU-Session": "cli-seed-session"}
    if settings.api_token:
        headers["Authorization"] = f"Bearer {settings.api_token}"
    c = httpx.Client(base_url=args.api.rstrip("/") + "/api/v1", timeout=60, headers=headers)
    try:
        r = c.post("/admin/seed", json={"business": not args.no_business, "redteam": not args.no_redteam})
    except httpx.HTTPError as exc:
        print(f"could not reach {args.api}: {exc}. Start the server first:  python -m setu.cli serve")
        return 2
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2, ensure_ascii=False))
    deadline = time.time() + args.wait
    while time.time() < deadline:
        active = c.get("/jobs", params={"active": True}).json()["jobs"]
        if not active:
            break
        print(f"  waiting: {len(active)} ingestion job(s) running ...")
        time.sleep(4)
    for col in c.get("/collections").json()["collections"]:
        print(f"  {col['name']:<40} {col['status']:<10} docs={col['doc_count']:<4} chunks={col['chunk_count']}")
    failed = [j for j in c.get("/jobs", params={"limit": 50}).json()["jobs"] if j["state"] == "failed"]
    for j in failed[:10]:
        print(f"  FAILED job {j['id']}: {j['error']}")
    if not args.no_business:
        r = c.post("/eval/datasets/business-curated")
        print("  curated eval dataset:", r.json().get("item_count", r.text[:200]))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.fn(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
