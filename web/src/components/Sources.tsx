import { langNative } from "../languages";
import type { TurnResult } from "../types";
import { Icon, Icons } from "./ui";

/**
 * The passages behind the answer.
 *
 * Collapsed by default — most people just want the answer — but one click away, because
 * "here is exactly what I read" is the whole reason to trust the thing. Retrieval scores,
 * layer names and chunk ids are deliberately *not* here; they explained the ranking, not the
 * answer, and they made the panel look like a debugger.
 */
export function Sources({
  turn,
  activeCite,
  onCite,
  defaultOpen,
}: {
  turn: TurnResult;
  activeCite: number | null;
  onCite: (m: number | null) => void;
  defaultOpen?: boolean;
}) {
  const cited = new Map(turn.citations.map((c) => [c.chunk_id, c.marker]));
  const hits = turn.evidence.filter((h) => cited.has(h.chunk_id));
  const rest = turn.evidence.filter((h) => !cited.has(h.chunk_id));
  const shown = hits.length > 0 ? [...hits, ...rest.slice(0, 2)] : turn.evidence.slice(0, 3);
  if (shown.length === 0) return null;

  return (
    <details className="card rise d2" open={defaultOpen}>
      <summary className="src-head">
        <Icon path={Icons.book} size={15} style={{ color: "var(--ink-3)" }} />
        <span className="src-title">
          {hits.length > 0
            ? `Where this came from · ${hits.length} quoted`
            : `Closest passages · ${shown.length}`}
        </span>
        <Icon path={Icons.chevron} size={16} className="src-caret" />
      </summary>

      <div className="src-list">
        {shown.map((h) => {
          const marker = cited.get(h.chunk_id);
          const on = marker !== undefined && marker === activeCite;
          return (
            <article
              className={`src ${on ? "on" : ""}`}
              key={h.chunk_id}
              onMouseEnter={() => marker !== undefined && onCite(marker)}
              onMouseLeave={() => onCite(null)}
            >
              <header className="src-bar">
                {marker !== undefined ? (
                  <span className="src-num">{marker}</span>
                ) : (
                  <span className="src-num ghost">–</span>
                )}
                <span className="tiny dim">
                  {marker !== undefined ? "quoted in the answer" : "read, but not quoted"}
                </span>
                <div className="spacer" />
                <span className="chip">{langNative(h.lang)}</span>
                {h.meta.is_selected === true && (
                  <span
                    className="chip chip-lime"
                    title="A human reviewer marked this passage as the one that answers this question"
                  >
                    <Icon path={Icons.check} size={11} />
                    verified
                  </span>
                )}
              </header>
              <div className="src-text">{h.text}</div>
            </article>
          );
        })}
      </div>
    </details>
  );
}
