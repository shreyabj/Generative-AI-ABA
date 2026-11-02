import { useEffect, useState } from "react";
import ClassicApp from "../App";
import { fmt, req } from "./client";
import { Badge, Bars, Card, Json, Stat, State, statusTone, useData } from "./ui";

const PAGES = [
  ["dashboard", "Overview"], ["research", "Research"], ["knowledge", "Knowledge Base"],
  ["explorer", "RAG Explorer"], ["eval", "Evaluation Lab"], ["voice", "Voice"], ["settings", "Settings"],
] as const;
type Page = (typeof PAGES)[number][0];

function useHash(): [string, (h: string) => void] {
  const [h, setH] = useState(location.hash.slice(1) || "dashboard");
  useEffect(() => {
    const f = () => setH(location.hash.slice(1) || "dashboard");
    addEventListener("hashchange", f);
    return () => removeEventListener("hashchange", f);
  }, []);
  return [h, (x) => (location.hash = x)];
}

export default function App2() {
  const [hash, go] = useHash();
  const [page, arg] = hash.split("/") as [Page, string?];
  if (page === "voice") return <><Nav page={page} go={go} /><div className="v2-voice"><ClassicApp /></div></>;
  return (
    <div className="v2">
      <Nav page={page} go={go} />
      <main className="v2-main">
        {page === "dashboard" && <Dashboard go={go} />}
        {page === "research" && <Research convId={arg} go={go} />}
        {page === "knowledge" && <Knowledge />}
        {page === "explorer" && <Explorer turnId={arg} go={go} />}
        {page === "eval" && <EvalLab />}
        {page === "settings" && <Settings />}
      </main>
    </div>
  );
}

function Nav({ page, go }: { page: string; go: (h: string) => void }) {
  return (
    <nav className="v2-nav">
      <div className="v2-brand">SETU <small>2.0</small></div>
      {PAGES.map(([k, l]) => (
        <button key={k} className={page === k ? "on" : ""} onClick={() => go(k)}>{l}</button>
      ))}
    </nav>
  );
}

/* ================================================================ dashboard */
function Dashboard({ go }: { go: (h: string) => void }) {
  const { data: d, error, loading } = useData(() => req("/dashboard/summary"), [], 8000);
  return (
    <State loading={loading && !d} error={error}>
      {d && (
        <>
          <h1>Overview</h1>
          <div className="v2-grid4">
            <Stat label="Documents" value={d.knowledge.documents} hint={`${d.knowledge.failed_documents} failed`} />
            <Stat label="Indexed chunks" value={d.knowledge.chunks.toLocaleString()} />
            <Stat label="Collections" value={d.knowledge.collections} hint={`${d.knowledge.active_jobs} active jobs`} />
            <Stat label="Research turns" value={d.research.total} hint={`${d.research.conversations} conversations`} />
          </div>
          <div className="v2-grid2">
            <Card title="Collections">
              <table className="v2-table"><thead><tr><th>Name</th><th>Status</th><th>Docs</th><th>Chunks</th></tr></thead>
                <tbody>{d.collections.map((c: any) => (
                  <tr key={c.id}><td>{c.name}</td><td><Badge tone={statusTone(c.status)}>{c.status}</Badge></td><td>{c.doc_count}</td><td>{c.chunk_count}</td></tr>))}</tbody></table>
            </Card>
            <Card title="System">
              <dl className="v2-dl">
                <dt>Embedding model</dt><dd>{d.embedding.model} ({d.embedding.dim}-d, {d.embedding.backend})</dd>
                <dt>Embedding cache</dt><dd>{d.embedding.cache_vectors} vectors</dd>
                <dt>Vector store</dt><dd>{d.vector_store?.backend}</dd>
                <dt>LLM</dt><dd>{d.llm.enabled ? d.llm.model : "not configured"} · breaker {d.llm.breaker.state}</dd>
                <dt>Speech</dt><dd>{d.speech.enabled ? "ElevenLabs" : "browser fallback"}</dd>
                <dt>Ingest worker</dt><dd>{d.health.worker ? "running" : "stopped"} · DB {d.health.database}</dd>
              </dl>
            </Card>
            <Card title="Live retrieval latency (v1 /ask)">
              {d.retrieval.live.count ? <dl className="v2-dl">
                <dt>Turns</dt><dd>{d.retrieval.live.count}</dd>
                <dt>P50 / P95</dt><dd>{fmt.ms(d.retrieval.live.pipeline_ms?.p50)} / {fmt.ms(d.retrieval.live.pipeline_ms?.p95)}</dd>
                <dt>SLO compliance</dt><dd>{fmt.pct(d.retrieval.live.slo_compliance)}</dd></dl> : <p className="v2-muted">No /ask turns since startup.</p>}
              {d.retrieval.latest_eval && <p className="v2-muted">Latest eval: {d.retrieval.latest_eval.name} — hybrid recall@5 {fmt.num(d.retrieval.latest_eval.metrics?.presets?.hybrid?.["recall@5"])}</p>}
            </Card>
            <Card title="Recent research (this browser)">
              <State empty={!d.recent_traces.length && "No research yet."}>
                <ul className="v2-list">{d.recent_traces.map((t: any) => (
                  <li key={t.turn_id} onClick={() => go(`explorer/${t.turn_id}`)}><Badge tone={statusTone(t.status)}>{t.status}</Badge> {t.query} <span className="v2-muted">{t.style} · {fmt.ms(t.pipeline_ms)}</span></li>))}</ul>
              </State>
            </Card>
            <Card title="Recent uploads">
              <ul className="v2-list">{d.recent_documents.map((x: any) => (
                <li key={x.id}><Badge tone={statusTone(x.status)}>{x.status}</Badge> {x.title?.slice(0, 70)} <span className="v2-muted">{x.chunk_count} chunks</span></li>))}</ul>
            </Card>
            <Card title="Recent jobs">
              <ul className="v2-list">{d.recent_jobs.map((j: any) => (
                <li key={j.id}><Badge tone={statusTone(j.state)}>{j.state}</Badge> {j.kind} <span className="v2-muted">{j.error || j.message}</span></li>))}</ul>
            </Card>
          </div>
        </>
      )}
    </State>
  );
}

/* ================================================================= research */
const STYLES = ["quick", "detailed", "comparative", "critical", "executive"];
const FRAMEWORKS = ["", "swot", "pestle", "porter", "financial", "risk", "competitor", "recommendations", "scenario"];

function Research({ convId, go }: { convId?: string; go: (h: string) => void }) {
  const convs = useData(() => req("/conversations"), [convId]);
  const cols = useData(() => req("/collections"), []);
  const conv = useData(() => (convId ? req(`/conversations/${convId}`) : Promise.resolve(null)), [convId]);
  const [q, setQ] = useState("");
  const [style, setStyle] = useState("detailed");
  const [fw, setFw] = useState("");
  const [sel, setSel] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [lang, setLang] = useState("");
  const [listening, setListening] = useState(false);

  useEffect(() => {
    if (!sel.length && cols.data) setSel(cols.data.collections.filter((c: any) => c.kind === "user" && c.loaded).map((c: any) => c.id));
  }, [cols.data]);

  const ask = async (text = q) => {
    if (!text.trim()) return;
    setBusy(true); setErr(null);
    try {
      const r = await req("/research/ask", { body: { question: text, style, framework: fw || null, collection_ids: sel, conversation_id: convId || null, lang: lang || null } });
      setQ("");
      if (!convId) go(`research/${r.conversation_id}`); else { conv.reload(); convs.reload(); }
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  };
  const voice = () => {
    const C: any = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!C) { setErr("Speech recognition is not supported in this browser (use Chrome or Edge)."); return; }
    const r = new C(); r.lang = lang ? `${lang}-IN` : "en-IN"; r.interimResults = true;
    r.onresult = (e: any) => setQ(Array.from(e.results).map((x: any) => x[0].transcript).join(" "));
    r.onend = () => setListening(false); setListening(true); r.start();
  };
  const msgs = conv.data?.messages ?? [];
  return (
    <div className="v2-research">
      <aside className="v2-side">
        <button className="v2-btn" onClick={() => go("research")}>+ New research</button>
        <State loading={convs.loading && !convs.data} error={convs.error} empty={convs.data && !convs.data.conversations.length && "No conversations yet."}>
          <ul className="v2-list">{convs.data?.conversations.map((c: any) => (
            <li key={c.id} className={c.id === convId ? "on" : ""} onClick={() => go(`research/${c.id}`)}>
              {c.title}<span className="v2-muted"> · {c.message_count / 2 | 0}</span>
              <button className="v2-x" title="Delete" onClick={async (e) => { e.stopPropagation(); await req(`/conversations/${c.id}`, { method: "DELETE" }); if (c.id === convId) go("research"); else convs.reload(); }}>×</button>
            </li>))}</ul>
        </State>
      </aside>
      <section className="v2-chat">
        <div className="v2-msgs">
          {!convId && <div className="v2-empty"><h1>Research workspace</h1><p>Ask across your collections. Every claim is checked against the passage it cites.</p></div>}
          {msgs.map((m: any) => m.role === "user"
            ? <div key={m.id} className="v2-q">{m.content}</div>
            : <Answer key={m.id} r={m.payload} onFollow={(t) => ask(t)} go={go} />)}
          {busy && <div className="v2-state"><span className="v2-spin" /> Retrieving, generating and verifying…</div>}
        </div>
        {err && <div className="v2-state err">{err}</div>}
        <div className="v2-ask">
          <div className="v2-row">
            <select value={style} onChange={(e) => setStyle(e.target.value)} aria-label="Answer mode">{STYLES.map((s) => <option key={s}>{s}</option>)}</select>
            <select value={fw} onChange={(e) => setFw(e.target.value)} aria-label="Business framework">{FRAMEWORKS.map((s) => <option key={s} value={s}>{s || "no framework"}</option>)}</select>
            <select value={lang} onChange={(e) => setLang(e.target.value)} aria-label="Language">{["", "en", "hi", "kn", "ta", "te", "bn", "mr", "gu", "ml", "pa"].map((l) => <option key={l} value={l}>{l || "auto language"}</option>)}</select>
            <div className="v2-chips">{cols.data?.collections.filter((c: any) => c.loaded).map((c: any) => (
              <label key={c.id} className={sel.includes(c.id) ? "on" : ""}><input type="checkbox" checked={sel.includes(c.id)} onChange={() => setSel(sel.includes(c.id) ? sel.filter((x) => x !== c.id) : [...sel, c.id])} />{c.name}</label>))}</div>
          </div>
          <div className="v2-row">
            <textarea value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask a research question…" onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); void ask(); } }} />
            <button className={`v2-btn ${listening ? "rec" : ""}`} onClick={voice} title="Voice input">{listening ? "● Listening" : "🎙"}</button>
            <button className="v2-btn primary" disabled={busy || !q.trim()} onClick={() => ask()}>Ask</button>
          </div>
        </div>
      </section>
    </div>
  );
}

function Answer({ r, onFollow, go }: { r: any; onFollow: (t: string) => void; go: (h: string) => void }) {
  const [open, setOpen] = useState<number | null>(null);
  if (!r) return null;
  const copy = () => navigator.clipboard?.writeText(r.answer_markdown || r.summary || "");
  const exportJson = () => { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(r, null, 2)], { type: "application/json" })); a.download = `setu-${r.turn_id}.json`; a.click(); };
  const speak = () => { const u = new SpeechSynthesisUtterance((r.summary || r.sections?.[0]?.claims?.map((c: any) => c.text).join(" ") || "").replace(/\[\d+\]/g, "")); u.lang = `${r.lang}-IN`; speechSynthesis.cancel(); speechSynthesis.speak(u); };
  const cite = (ms: number[]) => ms.map((m) => <button key={m} className="v2-cite" onClick={() => setOpen(m)}>{m}</button>);
  return (
    <div className="v2-a">
      <div className="v2-a-head">
        <Badge tone={statusTone(r.status)}>{r.status}</Badge> <Badge>{r.style}{r.framework ? ` · ${r.framework}` : ""}</Badge> <Badge>{r.path}</Badge>
        <span className="v2-muted">trust {fmt.num(r.trust_score, 0)} · groundedness {fmt.num(r.groundedness, 2)} · {fmt.ms(r.timings?.total_ms)}</span>
        <span className="v2-grow" />
        <button className="v2-link" onClick={copy}>Copy</button><button className="v2-link" onClick={exportJson}>Export</button><button className="v2-link" onClick={speak}>Speak</button><button className="v2-link" onClick={() => go(`explorer/${r.turn_id}`)}>Pipeline →</button>
      </div>
      {r.resolved_query !== r.question && <p className="v2-muted">Resolved as ({r.rewrite_method}): <i>{r.resolved_query}</i></p>}
      {r.summary && <p className="v2-summary">{r.summary}</p>}
      {r.sections?.map((s: any, i: number) => (
        <div key={i}>{s.heading && <h4>{s.heading}</h4>}
          <ul>{s.claims.map((c: any, j: number) => <li key={j}>{c.text} {cite(c.citations)} {c.kind !== "fact" && <Badge tone="info">{c.kind}</Badge>}</li>)}</ul></div>))}
      {r.table && <table className="v2-table"><thead><tr>{r.table.columns.map((c: string) => <th key={c}>{c}</th>)}</tr></thead><tbody>{r.table.rows.map((row: string[], i: number) => <tr key={i}>{row.map((x, j) => <td key={j}>{x}</td>)}</tr>)}</tbody></table>}
      {r.calculations?.length > 0 && <Card title="Calculations (computed by SETU, not the model)">{r.calculations.map((c: any, i: number) => <p key={i}>{c.label}: <code>{c.expression} = {c.result}{c.unit}</code> {!c.verified_inputs && <Badge tone="warn">input unverified</Badge>}</p>)}</Card>}
      {r.contradictions?.length > 0 && <Card title="⚖ Sources disagree">{r.contradictions.map((c: any, i: number) => (
        <div key={i}><b>{c.topic}</b> <Badge>{c.detector}</Badge>{c.statements.map((s: any) => <blockquote key={s.marker}>[{s.marker}] {s.quote} <span className="v2-muted">{s.title} {s.published_at}</span></blockquote>)}<p className="v2-muted">{c.explanation}</p></div>))}</Card>}
      {r.missing_information && <p className="v2-warn">{r.missing_information}</p>}
      {r.removed_claims?.length > 0 && <details><summary>{r.removed_claims.length} unsupported claim(s) removed</summary><ul>{r.removed_claims.map((c: any, i: number) => <li key={i}><s>{c.text}</s> — {c.removed_reason}</li>)}</ul></details>}
      {r.warnings?.map((w: string) => <p key={w} className="v2-muted">⚠ {w}</p>)}
      <details open={open != null}><summary>Evidence ({r.evidence?.length})</summary>
        {r.evidence?.map((e: any) => (
          <div key={e.marker} className={`v2-ev ${open === e.marker ? "on" : ""} ${e.cited ? "cited" : ""}`}>
            <b>[{e.marker}]</b> {e.source_url ? <a href={e.source_url} target="_blank" rel="noreferrer">{e.title}</a> : e.title}
            {e.page_start && <span className="v2-muted"> p.{e.page_start}</span>} {e.section && <span className="v2-muted"> · {e.section}</span>}
            {e.synthetic && <Badge tone="warn">synthetic</Badge>} {e.injection_flags?.length > 0 && <Badge tone="bad">injection text</Badge>}
            <span className="v2-muted"> · {e.layer} · score {fmt.num(e.score)}</span>
            <p>{e.text}</p></div>))}
      </details>
      {r.follow_ups?.length > 0 && <div className="v2-follow">{r.follow_ups.map((f: string) => <button key={f} onClick={() => onFollow(f)}>{f}</button>)}</div>}
    </div>
  );
}

/* ================================================================ knowledge */
function Knowledge() {
  const cols = useData(() => req("/collections"), [], 5000);
  const [cid, setCid] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const active = cid ?? cols.data?.collections.find((c: any) => c.kind === "user")?.id ?? cols.data?.collections[0]?.id;
  const run = async (f: () => Promise<unknown>) => { setErr(null); try { await f(); cols.reload(); } catch (e) { setErr(String(e)); } };
  return (
    <>
      <h1>Knowledge Base</h1>
      {err && <div className="v2-state err">{err}</div>}
      <div className="v2-grid-side">
        <Card title="Collections">
          <State loading={cols.loading && !cols.data} error={cols.error}>
            <ul className="v2-list">{cols.data?.collections.map((c: any) => (
              <li key={c.id} className={c.id === active ? "on" : ""} onClick={() => setCid(c.id)}>
                {c.name} <Badge tone={statusTone(c.status)}>{c.status}</Badge><div className="v2-muted">{c.doc_count} docs · {c.chunk_count} chunks</div></li>))}</ul>
          </State>
          <div className="v2-row"><input value={name} onChange={(e) => setName(e.target.value)} placeholder="New collection name" />
            <button className="v2-btn" disabled={!name.trim()} onClick={() => run(async () => { const c = await req("/collections", { body: { name } }); setName(""); setCid(c.id); })}>Create</button></div>
        </Card>
        {active && <CollectionView key={active} cid={active} onChange={cols.reload} setErr={setErr} />}
      </div>
    </>
  );
}

function CollectionView({ cid, onChange, setErr }: { cid: string; onChange: () => void; setErr: (e: string | null) => void }) {
  const col = useData(() => req(`/collections/${cid}`), [cid], 4000);
  const [page, setPage] = useState(0);
  const docs = useData(() => req(`/collections/${cid}/documents?limit=20&offset=${page * 20}`), [cid, page], 4000);
  const [urls, setUrls] = useState("");
  const [doc, setDoc] = useState<string | null>(null);
  const ro = col.data?.kind === "builtin";
  const act = async (f: () => Promise<unknown>) => { setErr(null); try { await f(); col.reload(); docs.reload(); onChange(); } catch (e) { setErr(String(e)); } };
  return (
    <div>
      <State loading={col.loading && !col.data} error={col.error}>
        {col.data && <Card title={col.data.name} extra={!ro && <span>
          <button className="v2-link" onClick={() => { const n = prompt("Rename collection", col.data.name); if (n) void act(() => req(`/collections/${cid}`, { method: "PATCH", body: { name: n } })); }}>Rename</button>
          <button className="v2-link" onClick={() => act(() => req(`/collections/${cid}/rebuild`, { method: "POST", body: {} }))}>Rebuild index</button>
          <button className="v2-link bad" onClick={() => confirm("Delete this collection and all its documents?") && act(() => req(`/collections/${cid}`, { method: "DELETE" }))}>Delete</button></span>}>
          <p className="v2-muted">{col.data.description}</p>
          <p>Status <Badge tone={statusTone(col.data.status)}>{col.data.status}</Badge> · {col.data.doc_count} documents · {col.data.chunk_count} chunks · layers {Object.entries(col.data.layers || {}).map(([k, v]) => `${k} ${v}`).join(", ")}</p>
          {col.data.error && <p className="v2-warn">{col.data.error}</p>}
          {!ro && <div className="v2-row">
            <label className="v2-btn">Upload files<input hidden type="file" multiple accept=".pdf,.docx,.txt,.md,.csv" onChange={(e) => { const fd = new FormData(); Array.from(e.target.files || []).forEach((f) => fd.append("files", f)); void act(async () => { const r = await req(`/collections/${cid}/documents`, { form: fd }); if (r.rejected.length) setErr(r.rejected.map((x: any) => x.error).join("; ")); }); e.target.value = ""; }} /></label>
            <input value={urls} onChange={(e) => setUrls(e.target.value)} placeholder="https://… (space-separated URLs)" />
            <button className="v2-btn" disabled={!urls.trim()} onClick={() => act(async () => { const r = await req(`/collections/${cid}/urls`, { body: { urls: urls.split(/\s+/).filter(Boolean) } }); setUrls(""); if (r.rejected.length) setErr(r.rejected.map((x: any) => `${x.url}: ${x.error}`).join("; ")); })}>Import URLs</button>
          </div>}
        </Card>}
      </State>
      <Card title="Documents">
        <State loading={docs.loading && !docs.data} error={docs.error} empty={docs.data && !docs.data.documents.length && "No documents yet."}>
          <table className="v2-table"><thead><tr><th>Title</th><th>Status</th><th>Chunks</th><th>Pages</th><th>Published</th><th /></tr></thead>
            <tbody>{docs.data?.documents.map((d: any) => (
              <tr key={d.id}><td><button className="v2-link" onClick={() => setDoc(d.id)}>{(d.title || d.id).slice(0, 80)}</button>{d.synthetic && <Badge tone="warn">synthetic</Badge>}{d.injection_flags > 0 && <Badge tone="bad">{d.injection_flags} flagged</Badge>}{d.error && <div className="v2-warn">{d.error}</div>}</td>
                <td><Badge tone={statusTone(d.status)}>{d.status}</Badge></td><td>{d.chunk_count}</td><td>{d.page_count ?? "–"}</td><td>{d.published_at ?? "–"}</td>
                <td>{!ro && <><button className="v2-link" onClick={() => act(() => req(`/documents/${d.id}/reindex`, { method: "POST", body: {} }))}>Re-index</button><button className="v2-link bad" onClick={() => confirm("Delete document?") && act(() => req(`/documents/${d.id}`, { method: "DELETE" }))}>Delete</button></>}</td></tr>))}</tbody></table>
          <div className="v2-row"><button className="v2-btn" disabled={!page} onClick={() => setPage(page - 1)}>‹</button><span className="v2-muted">{page * 20 + 1}–{Math.min((page + 1) * 20, docs.data?.total ?? 0)} of {docs.data?.total}</span><button className="v2-btn" disabled={(page + 1) * 20 >= (docs.data?.total ?? 0)} onClick={() => setPage(page + 1)}>›</button></div>
        </State>
      </Card>
      {doc && !ro && <DocChunks did={doc} close={() => setDoc(null)} />}
    </div>
  );
}

function DocChunks({ did, close }: { did: string; close: () => void }) {
  const [layer, setLayer] = useState("");
  const [off, setOff] = useState(0);
  const d = useData(() => Promise.all([req(`/documents/${did}`), req(`/documents/${did}/chunks?limit=20&offset=${off}${layer ? `&layer=${layer}` : ""}`)]), [did, layer, off]);
  return (
    <Card title="Document chunks" extra={<button className="v2-link" onClick={close}>Close</button>}>
      <State loading={d.loading && !d.data} error={d.error}>{d.data && <>
        <p><b>{d.data[0].title}</b> {d.data[0].source_url && <a href={d.data[0].source_url} target="_blank" rel="noreferrer">source</a>}<br /><span className="v2-muted">{d.data[0].mime_type} · {d.data[0].char_count} chars · {d.data[0].section_count} sections · lang {d.data[0].lang} · hash {d.data[0].content_hash?.slice(0, 12)}</span></p>
        {d.data[0].meta?.warnings?.map((w: string) => <p key={w} className="v2-muted">⚠ {w}</p>)}
        <select value={layer} onChange={(e) => { setLayer(e.target.value); setOff(0); }}>{["", "atomic", "semantic", "window", "document"].map((l) => <option key={l} value={l}>{l || "all layers"}</option>)}</select>
        {d.data[1].chunks.map((c: any) => <div key={c.id} className="v2-ev"><span className="v2-muted">{c.layer} · #{c.chunk_index} · p.{c.page_start ?? "–"} · {c.section_heading ?? ""} · {c.id}</span>{c.injection_flag && <Badge tone="bad">injection text</Badge>}<p>{c.text}</p></div>)}
        <div className="v2-row"><button className="v2-btn" disabled={!off} onClick={() => setOff(off - 20)}>‹</button><span className="v2-muted">{off + 1}–{Math.min(off + 20, d.data[1].total)} of {d.data[1].total}</span><button className="v2-btn" disabled={off + 20 >= d.data[1].total} onClick={() => setOff(off + 20)}>›</button></div>
      </>}</State>
    </Card>
  );
}

/* ================================================================= explorer */
function Explorer({ turnId, go }: { turnId?: string; go: (h: string) => void }) {
  const list = useData(() => req("/traces?limit=40"), []);
  const t = useData(() => (turnId ? req(`/traces/${turnId}`) : Promise.resolve(null)), [turnId]);
  const [sel, setSel] = useState<string | null>(null);
  const r = t.data?.result;
  const stages = r?.stages ?? [];
  const max = Math.max(1, ...stages.map((s: any) => s.started_ms + s.duration_ms));
  const st = stages.find((s: any) => s.key === sel);
  return (
    <div className="v2-grid-side">
      <Card title="Recent turns">
        <State loading={list.loading && !list.data} error={list.error} empty={list.data && !list.data.traces.length && "Ask something in Research first."}>
          <ul className="v2-list">{list.data?.traces.map((x: any) => <li key={x.turn_id} className={x.turn_id === turnId ? "on" : ""} onClick={() => go(`explorer/${x.turn_id}`)}><Badge tone={statusTone(x.status)}>{x.status}</Badge> {x.query?.slice(0, 60)}<div className="v2-muted">{x.style} · {fmt.ms(x.pipeline_ms)}</div></li>)}</ul>
        </State>
      </Card>
      <div>
        <h1>RAG Explorer</h1>
        {!turnId && <p className="v2-muted">Pick a turn to see every pipeline stage as it actually executed.</p>}
        <State loading={!!turnId && t.loading} error={t.error}>{r && <>
          <p><b>{r.question}</b> → <Badge tone={statusTone(r.status)}>{r.status}</Badge> {r.path} · {fmt.ms(r.timings?.total_ms)}</p>
          <div className="v2-flow">{stages.map((s: any) => (
            <button key={s.key + s.started_ms} className={`v2-stage ${statusTone(s.status)} ${sel === s.key ? "on" : ""}`} onClick={() => setSel(s.key)}>
              <span>{s.name}</span><small>{s.status} · {fmt.ms(s.duration_ms)}</small>
              <i style={{ marginLeft: `${(s.started_ms / max) * 100}%`, width: `${Math.max(0.5, (s.duration_ms / max) * 100)}%` }} /></button>))}</div>
          {st && <Card title={st.name}>{st.error && <p className="v2-warn">{st.error}</p>}
            {st.key.startsWith("retrieve.") || st.key === "fusion" || st.key === "rerank" ? <RetrievalTable st={st} /> : null}
            <details open={!["retrieve.dense", "retrieve.sparse", "fusion", "rerank"].includes(st.key)}><summary>Input / output</summary><Json value={{ input: st.input, output: st.output }} /></details></Card>}
          <Card title="Final context & scores"><table className="v2-table"><thead><tr><th>#</th><th>Source</th><th>Layer</th><th>Dense</th><th>BM25</th><th>Fused (rank)</th><th>Rerank</th><th>Cited</th></tr></thead>
            <tbody>{r.evidence.map((e: any) => <tr key={e.marker}><td>{e.marker}</td><td title={e.text}>{(e.title || e.doc_id).slice(0, 50)}</td><td>{e.layer}</td><td>{fmt.num(e.dense_score)}</td><td>{fmt.num(e.sparse_score, 2)}</td><td>{fmt.num(e.fused_score)} ({e.rank_fused ?? "–"})</td><td>{fmt.num(e.rerank_score)}</td><td>{e.cited ? "✓" : ""}</td></tr>)}</tbody></table></Card>
        </>}</State>
      </div>
    </div>
  );
}

function RetrievalTable({ st }: { st: any }) {
  const runs = st.output?.per_run ?? [];
  return <>{runs.map((run: any, i: number) => {
    let rows: any[] = run.result || [];
    if (st.key === "retrieve.dense") rows = rows.flatMap((l: any) => l.results.map((x: any) => ({ ...x, layer: l.layer })));
    return <div key={i}><p className="v2-muted">{run.collection} · “{run.query}”</p>
      <table className="v2-table"><thead><tr><th>Rank</th><th>Chunk</th><th>Layer</th>{st.key === "fusion" ? <><th>RRF</th><th>Blend</th><th>Score</th></> : st.key === "rerank" ? <><th>maxsim</th><th>lexical</th><th>Score</th><th>Fused rank</th></> : <th>Score</th>}<th>Preview</th></tr></thead>
        <tbody>{rows.slice(0, 12).map((x: any, j: number) => <tr key={j}><td>{x.rank}</td><td>{x.chunk_id}</td><td>{x.layer}</td>
          {st.key === "fusion" ? <><td>{fmt.num(x.rrf, 4)}</td><td>{fmt.num(x.blend)}</td><td>{fmt.num(x.score)}</td></> : st.key === "rerank" ? <><td>{fmt.num(x.maxsim)}</td><td>{fmt.num(x.lexical)}</td><td>{fmt.num(x.score)}</td><td>{x.fused_rank}</td></> : <td>{fmt.num(x.score)}</td>}
          <td className="v2-muted">{x.preview}</td></tr>)}</tbody></table></div>;
  })}</>;
}

/* ================================================================= eval lab */
function EvalLab() {
  const ds = useData(() => req("/eval/datasets"), []);
  const runs = useData(() => req("/eval/runs"), [], 4000);
  const rt = useData(() => req("/eval/redteam"), []);
  const reports = useData(() => req("/eval/reports"), []);
  const [presets, setPresets] = useState(["hybrid", "dense-only", "sparse-only", "hybrid-no-rerank"]);
  const [answers, setAnswers] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [rid, setRid] = useState<string | null>(null);
  const run = runs.data?.runs.find((r: any) => r.id === rid) ?? runs.data?.runs.find((r: any) => r.status === "completed");
  const P = run?.metrics?.presets ?? {};
  const go = async (f: () => Promise<unknown>) => { setBusy(true); setErr(null); try { await f(); } catch (e) { setErr(String(e)); } finally { setBusy(false); runs.reload(); rt.reload(); } };
  return (
    <>
      <h1>Evaluation Lab</h1>
      <p className="v2-muted">All numbers below come from stored runs. Retrieval metrics are document-level. Faithfulness, citation precision and answer relevance are automated heuristics, not human judgements.</p>
      {err && <div className="v2-state err">{err}</div>}
      <div className="v2-grid2">
        <Card title="Start a run">
          <State loading={ds.loading} error={ds.error}>
            {ds.data?.datasets.map((d: any) => <div key={d.id} className="v2-row"><b>{d.name}</b> <Badge>{d.label_source}</Badge> <span className="v2-muted">{d.item_count} items</span><span className="v2-grow" />
              <button className="v2-btn" disabled={busy} onClick={() => go(() => req("/eval/runs", { body: { dataset_id: d.id, presets, with_answers: answers, limit: answers ? 20 : undefined } }))}>Run</button></div>)}
            <div className="v2-chips">{["hybrid", "dense-only", "sparse-only", "hybrid-no-rerank", "hybrid-forced-rerank", "hybrid-no-expansion", "hybrid+prf"].map((p) => <label key={p} className={presets.includes(p) ? "on" : ""}><input type="checkbox" checked={presets.includes(p)} onChange={() => setPresets(presets.includes(p) ? presets.filter((x) => x !== p) : [...presets, p])} />{p}</label>)}</div>
            <label><input type="checkbox" checked={answers} onChange={(e) => setAnswers(e.target.checked)} /> also score answers (quick style, first 20 items)</label>
          </State>
        </Card>
        <Card title="Runs">
          <ul className="v2-list">{runs.data?.runs.map((r: any) => <li key={r.id} className={r.id === run?.id ? "on" : ""} onClick={() => setRid(r.id)}><Badge tone={statusTone(r.status)}>{r.status}</Badge> {r.name}<div className="v2-muted">{fmt.date(r.created_at)} {r.error}</div></li>)}</ul>
        </Card>
      </div>
      {run && <Card title={`Results — ${run.name}`} extra={<span><a className="v2-link" href={`/api/v1/eval/runs/${run.id}/export?format=json`}>JSON</a> <a className="v2-link" href={`/api/v1/eval/runs/${run.id}/export?format=csv`}>CSV</a></span>}>
        <table className="v2-table"><thead><tr><th>Preset</th>{["recall@1", "recall@5", "recall@10", "precision@5", "mrr@10", "ndcg@10"].map((k) => <th key={k}>{k}</th>)}<th>P50 ms</th><th>P95 ms</th></tr></thead>
          <tbody>{Object.entries(P).map(([k, v]: any) => <tr key={k}><td>{k}</td>{["recall@1", "recall@5", "recall@10", "precision@5", "mrr@10", "ndcg@10"].map((m) => <td key={m}>{fmt.num(v[m])}</td>)}<td>{fmt.num(v.retrieval_ms_p50, 1)}</td><td>{fmt.num(v.retrieval_ms_p95, 1)}</td></tr>)}</tbody></table>
        <div className="v2-grid2">
          <div><h4>Recall@5 by configuration</h4><Bars rows={Object.entries(P).map(([k, v]: any) => ({ label: k, value: v["recall@5"] }))} max={1} /></div>
          <div><h4>MRR@10 by configuration</h4><Bars rows={Object.entries(P).map(([k, v]: any) => ({ label: k, value: v["mrr@10"] }))} max={1} /></div>
        </div>
        {run.metrics?.answers && <><h4>Answer quality (automated)</h4><dl className="v2-dl">{["faithfulness", "citation_precision", "context_relevance", "answer_relevance", "reference_recall", "answered_rate", "abstain_wrong", "end_to_end_ms_p50"].map((k) => <><dt key={k}>{k}</dt><dd>{fmt.num(run.metrics.answers[k])}</dd></>)}</dl></>}
      </Card>}
      <div className="v2-grid2">
        <Card title="Red team (research pipeline)" extra={<button className="v2-btn" disabled={busy} onClick={() => go(() => req("/eval/redteam", { body: {} }))}>Run red team</button>}>
          <State loading={rt.loading} error={rt.error} empty={rt.data && !rt.data.cases && "Not run yet."}>
            <p>{rt.data?.passed}/{rt.data?.cases} passed</p>
            <table className="v2-table"><tbody>{rt.data?.results?.map((x: any) => <tr key={x.name}><td>{x.passed ? "✓" : "✗"}</td><td>{x.family}</td><td>{x.expect}</td><td><Badge tone={statusTone(x.status)}>{x.status}</Badge></td></tr>)}</tbody></table>
          </State>
        </Card>
        <Card title="Committed v1 reports (scripts/)">
          <State loading={reports.loading} error={reports.error}>{reports.data && <dl className="v2-dl">
            {reports.data.latency && <><dt>Latency P50 / P100</dt><dd>{fmt.ms(reports.data.latency.pipeline_ms?.p50)} / {fmt.ms(reports.data.latency.pipeline_ms?.p100)} ({reports.data.latency.samples} turns)</dd></>}
            {reports.data.quality && <><dt>v1 quality recall@5 / MRR</dt><dd>{fmt.num(reports.data.quality.recall?.["@5"])} / {fmt.num(reports.data.quality["mrr@10"])} ({reports.data.quality.queries} q)</dd></>}
            {reports.data.redteam && <><dt>v1 red team</dt><dd>{reports.data.redteam.passed}/{reports.data.redteam.cases}</dd></>}
          </dl>}</State>
        </Card>
      </div>
    </>
  );
}

/* ================================================================= settings */
function Settings() {
  const s = useData(() => req("/settings"), []);
  const [tok, setTok] = useState(localStorage.getItem("setu.token") || "");
  return (
    <>
      <h1>Settings & diagnostics</h1>
      <Card title="API token (stored in this browser only)">
        <div className="v2-row"><input type="password" value={tok} onChange={(e) => setTok(e.target.value)} placeholder="only needed when SETU_API_TOKEN is set" /><button className="v2-btn" onClick={() => { tok ? localStorage.setItem("setu.token", tok) : localStorage.removeItem("setu.token"); }}>Save</button></div>
        <p className="v2-muted">Server requires a token: {String(s.data?.security?.api_token_required ?? "–")}. Configuration is read-only here; change it with SETU_* environment variables.</p>
      </Card>
      <State loading={s.loading} error={s.error}>{s.data && <div className="v2-grid2">
        {["embedding", "chunking", "retrieval", "guardrails", "generation", "speech", "vector_store", "storage", "limits", "diagnostics"].map((k) => <Card key={k} title={k}><Json value={s.data[k]} /></Card>)}
      </div>}</State>
    </>
  );
}
