import { LANGUAGES } from "../languages";
import type { AppConfig } from "../types";

/**
 * Shown until the first question is asked. Says what the thing does and, importantly, shows
 * the languages in their own scripts — that communicates "this is for me" to a Kannada or
 * Bengali speaker far faster than the word "multilingual" does.
 */
export function Hero({ config }: { config: AppConfig | null }) {
  const indexed = new Set((config?.languages ?? []).map((l) => l.code));
  const shown = LANGUAGES.filter((l) => indexed.has(l.code));

  return (
    <div className="hero rise">
      <h1>
        Ask anything, out loud.
        <br />
        Get an answer you can <em>check</em>.
      </h1>
      <p>
        Speak or type in your own language. SETU searches a library of real web passages, quotes the
        exact sentences it used, and tells you when it doesn&rsquo;t know.
      </p>
      {shown.length > 0 && (
        <div className="hero-langs">
          {shown.map((l) => (
            <span className="hero-lang" key={l.code} title={l.name}>
              {l.native}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
