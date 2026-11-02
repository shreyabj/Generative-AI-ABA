import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { langNative } from "../languages";
import type { QuestionBank, SampleQuery } from "../types";
import { Icon, Icons } from "./ui";

/**
 * Browse every question this corpus was built from.
 *
 * The library is a slice of MS MARCO-XI, not the web, so "ask me anything" is a promise it
 * cannot keep. This panel is the honest version of that promise: here is the actual set of
 * questions behind the passages, in every language they were indexed in, and clicking one
 * asks it in that language.
 *
 * Questions with no evidence in this slice are shown rather than hidden, marked "no answer
 * here". They are the ones MS MARCO itself labels as having no answer present, and seeing
 * them is what makes an abstention legible instead of looking like a failure.
 */
export function QuestionBrowser({
  onPick,
  onClose,
}: {
  onPick: (sample: SampleQuery) => void;
  onClose: () => void;
}) {
  const [bank, setBank] = useState<QuestionBank | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [lang, setLang] = useState<string>("");
  const [query, setQuery] = useState("");
  const [answerableOnly, setAnswerableOnly] = useState(true);
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let alive = true;
    api
      .questions({ limit: 400 })
      .then((b) => alive && setBank(b))
      .catch((e) => alive && setError(e instanceof Error ? e.message : "Could not load questions."));
    const t = setTimeout(() => search.current?.focus(), 40);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  // Filtering client-side: the whole bank is a few hundred rows, so a round trip per
  // keystroke would add latency and tell the user nothing new.
  const rows = useMemo(() => {
    const all = bank?.questions ?? [];
    const needle = query.trim().toLowerCase();
    return all.filter(
      (r) =>
        (!lang || r.lang === lang) &&
        (!answerableOnly || r.answerable) &&
        (!needle || r.query.toLowerCase().includes(needle)),
    );
  }, [bank, lang, query, answerableOnly]);

  const langs = bank?.languages ?? [];

  return (
    <div className="ovl qb-ovl" role="dialog" aria-modal="true" aria-labelledby="qb-title">
      <div className="ovl-card qb-card">
        <button className="ovl-x" onClick={onClose} aria-label="Close">
          <Icon path={Icons.cross} size={16} />
        </button>

        <h2 id="qb-title" className="qb-title">
          Questions in the library
        </h2>
        <p className="qb-sub">
          The corpus was built from these questions, in every language it was indexed in. Pick one
          and it will be asked in that language.
        </p>

        <div className="qb-controls">
          <div className="qb-search">
            <input
              ref={search}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search questions…"
              aria-label="Search questions"
              spellCheck={false}
            />
          </div>
          <label className="qb-toggle">
            <input
              type="checkbox"
              checked={answerableOnly}
              onChange={(e) => setAnswerableOnly(e.target.checked)}
            />
            Only ones with an answer here
          </label>
        </div>

        <div className="qb-langs" role="tablist" aria-label="Filter by language">
          <button
            role="tab"
            aria-selected={lang === ""}
            className={`qb-chip ${lang === "" ? "on" : ""}`}
            onClick={() => setLang("")}
          >
            All
          </button>
          {langs.map((l) => (
            <button
              key={l.code}
              role="tab"
              aria-selected={lang === l.code}
              className={`qb-chip ${lang === l.code ? "on" : ""}`}
              onClick={() => setLang(l.code)}
              title={`${l.name} — ${l.answerable} answerable`}
            >
              {langNative(l.code)}
            </button>
          ))}
        </div>

        <div className="qb-list">
          {error && <p className="qb-empty">{error}</p>}
          {!error && !bank && <p className="qb-empty">Loading…</p>}
          {bank && rows.length === 0 && (
            <p className="qb-empty">
              Nothing matches. Try clearing the filters, or turn off “only ones with an answer”.
            </p>
          )}
          {rows.map((r, i) => (
            <button
              className="qb-row"
              key={`${r.lang}-${r.query_id ?? i}`}
              onClick={() => onPick(r)}
              title={`Ask in ${r.lang_name}`}
            >
              <span className="qb-row-lang">{langNative(r.lang)}</span>
              <span className="qb-row-q" lang={r.lang}>
                {r.query}
              </span>
              {r.answerable ? (
                <span className="qb-badge ok" title="The corpus has evidence for this one">
                  <Icon path={Icons.check} size={11} />
                  answerable
                </span>
              ) : (
                <span className="qb-badge no" title="MS MARCO marks this one as having no answer present">
                  no answer here
                </span>
              )}
            </button>
          ))}
        </div>

        <div className="qb-foot">
          <span>
            {rows.length} of {bank?.questions.length ?? 0} shown
          </span>
          <span className="dimmer">Esc to close</span>
        </div>
      </div>
    </div>
  );
}
