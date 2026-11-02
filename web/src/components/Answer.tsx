import { useCallback, useEffect, useRef, useState } from "react";
import { langNative } from "../languages";
import type { TurnResult } from "../types";
import { Icon, Icons } from "./ui";

interface Props {
  turn: TurnResult;
  activeCite: number | null;
  onCite: (m: number | null) => void;
  onSpeak: () => void;
  speaking: boolean;
  voiceAvailable: boolean;
}

/** Plain-language explanation for each way an answer can be withheld. */
const WHY: Record<string, string> = {
  "retrieval.topicality":
    "Nothing in the library was close enough to your question, so guessing would have been the only option.",
  "output.groundedness":
    "It found related passages, but none of them actually supported an answer — so it stopped rather than filling the gap.",
  "output.trust-floor":
    "It could put an answer together, but not one it was confident enough to show you.",
  "input.injection": "That looked like an attempt to change SETU's own instructions.",
  "input.safety": "That's a request SETU won't help with.",
  "input.sanity": "That didn't look like a question it could work with.",
};

function Body({
  text,
  active,
  onCite,
}: {
  text: string;
  active: number | null;
  onCite: (m: number | null) => void;
}) {
  const parts = text.split(/(\[\d+\])/g);
  return (
    <p className="ans-body">
      {parts.map((part, i) => {
        const m = /^\[(\d+)\]$/.exec(part);
        if (!m) return <span key={i}>{part}</span>;
        const marker = Number(m[1]);
        return (
          <button
            key={i}
            className={`cite ${active === marker ? "cite-on" : ""}`}
            onMouseEnter={() => onCite(marker)}
            onMouseLeave={() => onCite(null)}
            onClick={() => onCite(active === marker ? null : marker)}
            title={`From source ${marker} — hover to highlight it below`}
          >
            {marker}
          </button>
        );
      })}
    </p>
  );
}

/**
 * Copy the answer as prose.
 *
 * The `[1]` markers are an interface affordance - they are only meaningful next to the source
 * list they point at - so they come out, the same way they do before the text is spoken. What
 * lands on the clipboard is the sentence a person actually wants to paste.
 */
function useCopy(text: string) {
  const [copied, setCopied] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (timer.current) window.clearTimeout(timer.current);
    },
    [],
  );

  const copy = useCallback(async () => {
    const clean = text
      .replace(/\[\d+\]/g, "")
      .replace(/\s+([.,;:!?।॥])/g, "$1")
      .replace(/\s+/g, " ")
      .trim();
    if (!clean) return;

    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(clean);
      } else {
        // The async Clipboard API needs a secure context, which a plain-http LAN address is
        // not. Falling back keeps the button working when the demo is opened from a phone.
        const ta = document.createElement("textarea");
        ta.value = clean;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }
      setCopied(true);
      if (timer.current) window.clearTimeout(timer.current);
      timer.current = window.setTimeout(() => setCopied(false), 1800);
    } catch {
      /* Nothing useful to say: the button simply does not confirm. */
    }
  }, [text]);

  return { copied, copy };
}

export function Answer({ turn, activeCite, onCite, onSpeak, speaking, voiceAvailable }: Props) {
  const { copied, copy } = useCopy(turn.answer);
  const answered = turn.status === "answered" || turn.status === "degraded";
  const refused = turn.status === "refused";
  const why = turn.guards.blocked_by ? WHY[turn.guards.blocked_by] : null;
  const sources = new Set(turn.citations.map((c) => c.doc_id)).size;
  const strong = turn.trust_score >= 70;

  return (
    <div className="card ans rise">
      <div className="ans-top">
        {answered ? (
          <span className="chip chip-lime">
            <Icon path={Icons.check} size={12} />
            Answer
          </span>
        ) : refused ? (
          <span className="chip chip-coral">
            <Icon path={Icons.cross} size={12} />
            Declined
          </span>
        ) : (
          <span className="chip chip-amber">
            <Icon path={Icons.info} size={12} />
            Not in the library
          </span>
        )}

        {answered && <span className="chip">{langNative(turn.answer_lang)}</span>}

        <div className="spacer" />

        {answered && (
          <button
            className="btn btn-sm"
            onClick={onSpeak}
            disabled={!voiceAvailable}
            title={
              voiceAvailable
                ? `Read aloud in ${langNative(turn.answer_lang)}`
                : "No speech voice available on this device"
            }
          >
            <Icon path={speaking ? Icons.stop : Icons.speaker} size={13} />
            {speaking ? "Stop" : "Listen"}
          </button>
        )}

        {answered && (
          <button
            className={`btn btn-sm btn-icon ans-copy ${copied ? "done" : ""}`}
            onClick={() => void copy()}
            title={copied ? "Copied to clipboard" : "Copy the answer"}
            aria-label={copied ? "Answer copied to clipboard" : "Copy the answer"}
          >
            <Icon path={copied ? Icons.check : Icons.copy} size={14} />
          </button>
        )}

        <span className="sr-only" role="status" aria-live="polite">
          {copied ? "Answer copied to clipboard" : ""}
        </span>
      </div>

      {turn.query && <p className="ans-q">You asked: {turn.query}</p>}

      {answered ? (
        <Body text={turn.answer} active={activeCite} onCite={onCite} />
      ) : (
        <p className="ans-body soft">{turn.answer}</p>
      )}

      {!answered && why && (
        <div className="ans-why">
          <Icon path={Icons.info} size={15} />
          <span>{why}</span>
        </div>
      )}

      {answered && (
        <div className="verdict">
          <span className={`verdict-badge ${strong ? "verdict-good" : "verdict-mid"}`}>
            <Icon path={strong ? Icons.check : Icons.warn} size={13} />
            {strong ? "Checked against its sources" : "Partly supported — worth a look"}
          </span>
          <span className="verdict-meta">
            {sources} source{sources === 1 ? "" : "s"} · answered in {turn.pipeline_ms.toFixed(0)} ms
          </span>
        </div>
      )}
    </div>
  );
}
