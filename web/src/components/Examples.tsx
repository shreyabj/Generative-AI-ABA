import { langNative } from "../languages";
import type { SampleQuery } from "../types";
import { Icon, Icons } from "./ui";

/**
 * One question per indexed language, in that language, ready to click.
 *
 * These are real questions from the corpus, and the server only offers ones it has evidence
 * for - a suggested example that comes back "I don't know" reads as a broken product even
 * when the abstention is correct. The language label is the point of the row: it shows a
 * Kannada speaker a Kannada question before they have typed anything, and clicking it sets
 * the language as well as asking, so the answer comes back in that language too.
 */
export function Examples({
  samples,
  busy,
  onAsk,
  onExplore,
}: {
  samples: SampleQuery[];
  busy: boolean;
  onAsk: (sample: SampleQuery) => void;
  onExplore: () => void;
}) {
  if (samples.length === 0) return null;

  return (
    <div className="ex">
      <div className="ex-head">
        <span className="ex-title">Try one, in any language</span>
        <button className="ex-more" onClick={onExplore} disabled={busy}>
          <Icon path={Icons.book} size={13} />
          Explore more questions
        </button>
      </div>

      <div className="ex-grid">
        {samples.map((s, i) => (
          <button
            className="ex-card"
            key={`${s.lang}-${s.query_id ?? i}`}
            onClick={() => onAsk(s)}
            disabled={busy}
            title={`Ask this in ${s.lang_name}`}
            style={{ animationDelay: `${Math.min(i * 45, 400)}ms` }}
          >
            <span className="ex-lang">
              <b>{langNative(s.lang)}</b>
              <span>{s.lang_name}</span>
            </span>
            <span className="ex-q" lang={s.lang}>
              {s.query}
            </span>
            <Icon path={Icons.chevron} size={14} className="ex-go" />
          </button>
        ))}
      </div>
    </div>
  );
}
