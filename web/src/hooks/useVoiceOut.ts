import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api";
import { bcp47 } from "../languages";

export type VoiceEngine = "elevenlabs" | "browser" | "none";

export interface VoiceOutState {
  speaking: boolean;
  available: boolean;
  browserVoices: number;
  /** Language codes for which a genuinely native local voice is installed. */
  nativeLangs: string[];
  lastEngine: VoiceEngine;
  error: string | null;
}

/**
 * Speaking the answer, in the right voice for the language.
 *
 * Two engines, and which one is *better* depends on the language:
 *
 * - **A native OS voice**, when the device has one for that language. A real Kannada voice
 *   reading Kannada beats any cross-lingual model, because the pronunciation and prosody are
 *   native rather than approximated.
 * - **ElevenLabs**, otherwise. The server picks a per-language voice and pins pronunciation
 *   where the API supports it (see `VOICE_BY_LANG` in setu/speech/elevenlabs.py).
 *
 * So the rule is: prefer a native local voice for the answer's language; fall back to
 * ElevenLabs; fall back to any local voice. This is what fixes the "why is Kannada being read
 * with an English accent" problem — previously *one* English voice read all ten languages.
 *
 * Everything works with no API key at all, because the browser path needs none.
 */
export function useVoiceOut(elevenLabsEnabled: boolean) {
  const [state, setState] = useState<VoiceOutState>({
    speaking: false,
    available: false,
    browserVoices: 0,
    nativeLangs: [],
    lastEngine: "none",
    error: null,
  });

  const audio = useRef<HTMLAudioElement | null>(null);
  const voices = useRef<SpeechSynthesisVoice[]>([]);
  const synthOk = typeof window !== "undefined" && "speechSynthesis" in window;

  // Identifies the current playback attempt.
  //
  // The engines below are a fallback chain: if ElevenLabs fails, the browser voice takes
  // over. Stopping playback looks exactly like a failure from inside that chain - pausing an
  // <audio> element rejects its pending play() with AbortError, and speechSynthesis.cancel()
  // fires the utterance's onerror - so pressing Stop used to silence one engine and hand the
  // answer straight to the other. The user heard a second voice start.
  //
  // So every attempt carries a token. stop() invalidates it, and the chain checks the token
  // before deciding a failure is worth falling back from: a cancellation ends the attempt, a
  // genuine error still tries the next engine.
  const runId = useRef(0);

  // Voice lists load asynchronously in Chromium and are empty on first read.
  useEffect(() => {
    if (!synthOk) {
      setState((s) => ({ ...s, available: elevenLabsEnabled }));
      return;
    }
    const load = () => {
      voices.current = window.speechSynthesis.getVoices();
      const langs = Array.from(
        new Set(voices.current.map((v) => v.lang.toLowerCase().split("-")[0])),
      ).sort();
      setState((s) => ({
        ...s,
        browserVoices: voices.current.length,
        nativeLangs: langs,
        available: elevenLabsEnabled || voices.current.length > 0,
      }));
    };
    load();
    window.speechSynthesis.addEventListener("voiceschanged", load);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", load);
  }, [elevenLabsEnabled, synthOk]);

  const stop = useCallback(() => {
    // Invalidate first: the pause() below rejects the in-flight promise, and the chain reads
    // this token to know the rejection was us and not a broken engine.
    runId.current += 1;
    const el = audio.current;
    audio.current = null;
    if (el) {
      el.pause();
      el.removeAttribute("src");
      if (el.dataset.url) {
        URL.revokeObjectURL(el.dataset.url);
        delete el.dataset.url;
      }
    }
    if (synthOk) window.speechSynthesis.cancel();
    setState((s) => (s.speaking ? { ...s, speaking: false } : s));
  }, [synthOk]);

  /** A voice whose own language matches — not a fallback. */
  const nativeVoice = useCallback((lang: string): SpeechSynthesisVoice | null => {
    const want = bcp47(lang).toLowerCase();
    const base = want.split("-")[0];
    const list = voices.current;
    return (
      list.find((v) => v.lang.toLowerCase() === want) ??
      list.find((v) => v.lang.toLowerCase().startsWith(base)) ??
      null
    );
  }, []);

  const anyVoice = useCallback(
    (lang: string) => nativeVoice(lang) ?? voices.current[0] ?? null,
    [nativeVoice],
  );

  const speakWithBrowser = useCallback(
    (text: string, lang: string) =>
      new Promise<void>((resolve, reject) => {
        if (!synthOk) return reject(new Error("this browser has no speech synthesis"));
        window.speechSynthesis.cancel();
        const u = new SpeechSynthesisUtterance(text);
        const v = anyVoice(lang);
        if (v) u.voice = v;
        u.lang = v?.lang ?? bcp47(lang);
        u.rate = 1.0;
        u.onend = () => resolve();
        u.onerror = (e) => reject(new Error(e.error || "speech synthesis failed"));
        window.speechSynthesis.speak(u);
      }),
    [anyVoice, synthOk],
  );

  const playBlob = useCallback(
    (blob: Blob) =>
      new Promise<void>((resolve, reject) => {
        const url = URL.createObjectURL(blob);
        const el = new Audio(url);
        el.dataset.url = url;
        audio.current = el;
        const done = (fn: () => void) => {
          URL.revokeObjectURL(url);
          delete el.dataset.url;
          if (audio.current === el) audio.current = null;
          fn();
        };
        el.onended = () => done(resolve);
        el.onerror = () => done(() => reject(new Error("could not play the audio")));
        void el.play().catch((e) => done(() => reject(e)));
      }),
    [],
  );

  const speak = useCallback(
    async (text: string, lang = "en", preloaded?: Blob): Promise<VoiceEngine> => {
      const clean = text
        .replace(/\[\d+\]/g, "")
        .replace(/\s+/g, " ")
        .trim();
      if (!clean) return "none";
      stop(); // also bumps runId, superseding any attempt already in flight
      const id = runId.current;
      const current = () => runId.current === id;
      setState((s) => ({ ...s, speaking: true, error: null }));

      const hasNative = nativeVoice(lang) !== null;
      const order: VoiceEngine[] = preloaded
        ? ["elevenlabs", "browser"]
        : hasNative
          ? ["browser", "elevenlabs"] // a real local voice for this language wins
          : ["elevenlabs", "browser"];

      let lastErr: unknown = null;
      for (const engine of order) {
        if (!current()) return "none";
        try {
          if (engine === "elevenlabs") {
            if (!elevenLabsEnabled) continue;
            const blob = preloaded ?? (await api.tts(clean, lang));
            // The fetch above is awaited, so Stop may have landed while it was in flight.
            if (!current()) return "none";
            await playBlob(blob);
          } else {
            if (!synthOk || voices.current.length === 0) continue;
            await speakWithBrowser(clean, lang);
          }
          if (!current()) return "none";
          setState((s) => ({ ...s, speaking: false, lastEngine: engine }));
          return engine;
        } catch (e) {
          // Stopped on purpose, or superseded by a newer answer. Not a failure, and
          // emphatically not a reason to start the other engine on the same text.
          if (!current()) return "none";
          lastErr = e;
        }
      }

      if (!current()) return "none";
      setState((s) => ({
        ...s,
        speaking: false,
        lastEngine: "none",
        error: lastErr instanceof Error ? lastErr.message : "voice output failed",
      }));
      return "none";
    },
    [elevenLabsEnabled, nativeVoice, playBlob, speakWithBrowser, stop, synthOk],
  );

  /** Does this device have a real voice for `lang`? Used to badge the language picker. */
  const hasNativeVoice = useCallback((lang: string) => nativeVoice(lang) !== null, [nativeVoice]);

  useEffect(() => stop, [stop]);

  return { ...state, speak, stop, hasNativeVoice };
}
