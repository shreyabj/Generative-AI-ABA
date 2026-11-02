import { useEffect, useRef, useState } from "react";
import { LANG_BY_CODE } from "../languages";
import { LanguagePicker } from "./LanguagePicker";
import { Icon, Icons } from "./ui";

interface Props {
  busy: boolean;
  recording: boolean;
  level: number;
  interim: string;
  recordingMs: number;
  micSupported: boolean;
  micError: string | null;
  corpusLangs: string[];
  lang: string;
  onLang: (v: string) => void;
  onAsk: (text: string) => void;
  onToggleMic: () => void;
  hasNativeVoice: (lang: string) => boolean;
}

/** Relative heights for the level bars, so the middle of the group is tallest. */
const BAR_SCALE = [0.45, 0.75, 1, 0.75, 0.45];

/**
 * The one thing this page asks you to do.
 *
 * A single big microphone button and a single text field, centred, with the language
 * control directly underneath. The button label always says what will happen next
 * ("Tap to talk" → "Listening… tap to finish") rather than relying on an icon change,
 * because a first-time visitor should never have to guess whether it is recording.
 */
export function AskPanel({
  busy,
  recording,
  level,
  interim,
  recordingMs,
  micSupported,
  micError,
  corpusLangs,
  lang,
  onLang,
  onAsk,
  onToggleMic,
  hasNativeVoice,
}: Props) {
  const [text, setText] = useState("");
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing =
        document.activeElement instanceof HTMLInputElement ||
        document.activeElement instanceof HTMLTextAreaElement;
      if (e.key === "/" && !typing) {
        e.preventDefault();
        input.current?.focus();
      }
      if (e.code === "Space" && !typing && !e.repeat) {
        e.preventDefault();
        onToggleMic();
      }
      if (e.key === "Escape") input.current?.blur();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onToggleMic]);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    const q = text.trim();
    if (!q || busy) return;
    onAsk(q);
    setText("");
  };

  const orbClass = recording ? "orb orb-live" : busy ? "orb orb-busy" : "orb orb-idle";
  const inset = 22 - level * 18;
  const current = lang ? LANG_BY_CODE[lang] : null;
  const placeholder = current
    ? `Ask in ${current.native}…`
    : "Ask anything — type here or tap the microphone";

  return (
    <div className="card card-overflow card-ask rise">
      <div className="ask">
        <button
          className={orbClass}
          onClick={onToggleMic}
          disabled={!micSupported || busy}
          aria-label={recording ? "Stop recording" : "Start recording"}
          aria-pressed={recording}
        >
          <span className="orb-ring" />
          <span className="orb-ring r2" />
          <span className="orb-wave" />
          <span className="orb-wave w2" />
          {recording && (
            <span className="orb-level" style={{ inset: `${inset}px`, opacity: 0.3 + level * 0.7 }} />
          )}
          <span className="orb-core">
            {busy && <span className="orb-spin" />}
            {recording ? (
              // Real level bars, driven by the analyser node - so what you see is the audio
              // actually being captured, not a canned animation.
              <span className="orb-bars" aria-hidden="true">
                {BAR_SCALE.map((k, i) => (
                  <i key={i} style={{ height: `${Math.max(4, level * 34 * k + 4)}px` }} />
                ))}
              </span>
            ) : (
              <Icon path={Icons.mic} size={27} className="orb-glyph" />
            )}
          </span>
        </button>

        <div className={`orb-cap ${recording ? "rec" : ""}`}>
          {recording ? (
            `Listening… ${(recordingMs / 1000).toFixed(1)}s — tap to finish`
          ) : busy ? (
            <span className="row" style={{ justifyContent: "center" }}>
              <span className="thinking" aria-hidden="true">
                <i />
                <i />
                <i />
              </span>
              Finding an answer…
            </span>
          ) : micSupported ? (
            "Tap to talk"
          ) : (
            "Microphone unavailable — type below"
          )}
        </div>

        <form className="ask-form" onSubmit={submit}>
          <div className="ask-input">
            <input
              ref={input}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder={placeholder}
              aria-label="Your question"
              spellCheck={false}
              autoComplete="off"
            />
            <button
              className="ask-send"
              type="submit"
              disabled={!text.trim() || busy}
              aria-label="Ask"
              title="Ask"
            >
              <Icon path={Icons.send} size={17} />
            </button>
          </div>

          <div className="ask-row">
            <LanguagePicker
              value={lang}
              onChange={onLang}
              corpusLangs={corpusLangs}
              disabled={busy}
              hasNativeVoice={hasNativeVoice}
            />
            <span className="hint">
              <kbd>Space</kbd> talk · <kbd>/</kbd> type
            </span>
            {micError && <span className="chip chip-coral">{micError}</span>}
          </div>

          {recording && (
            <div className="transcript fade-in">
              <b>hearing you</b>
              {interim || <span className="dimmer">keep talking…</span>}
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
