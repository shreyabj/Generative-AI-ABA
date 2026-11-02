import { useCallback, useEffect, useRef, useState } from "react";

/* The Web Speech API is still vendor-prefixed and absent from lib.dom.d.ts. */
type SpeechRecognitionLike = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  start: () => void;
  stop: () => void;
  abort: () => void;
  onresult: ((e: any) => void) | null;
  onerror: ((e: any) => void) | null;
  onend: (() => void) | null;
};

function getRecognitionCtor(): (new () => SpeechRecognitionLike) | null {
  const w = window as unknown as Record<string, unknown>;
  return (w.SpeechRecognition ?? w.webkitSpeechRecognition) as (new () => SpeechRecognitionLike) | null;
}

const BCP47: Record<string, string> = {
  en: "en-IN", hi: "hi-IN", bn: "bn-IN", ta: "ta-IN", te: "te-IN", mr: "mr-IN",
  gu: "gu-IN", kn: "kn-IN", ml: "ml-IN", pa: "pa-IN", or: "or-IN", as: "as-IN",
  ne: "ne-NP", sa: "hi-IN", ur: "ur-IN",
};

/**
 * Interim transcripts from the browser, used purely to drive speculative prefetch.
 *
 * This is not the transcription of record - ElevenLabs Scribe is, and it runs on the
 * blob after recording stops. This hook exists because the browser can produce partial
 * text *while the user is still speaking*, which is exactly when the retrieval side is
 * otherwise sitting idle. Feeding those partials to `/api/v1/prefetch` means retrieval
 * for the final query is frequently already complete before the audio upload even starts.
 *
 * Chromium-only, and entirely optional: without it the pipeline runs identically, just
 * without the head start.
 */
export function useLiveSpeech(onPartial: (text: string) => void) {
  const Ctor = getRecognitionCtor();
  const [supported] = useState(!!Ctor);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const rec = useRef<SpeechRecognitionLike | null>(null);
  const cb = useRef(onPartial);
  const lastSent = useRef("");
  cb.current = onPartial;

  const stop = useCallback(() => {
    try {
      rec.current?.stop();
    } catch {
      /* already stopped */
    }
    rec.current = null;
    setListening(false);
  }, []);

  useEffect(() => stop, [stop]);

  const start = useCallback(
    (lang = "en") => {
      if (!Ctor) return;
      stop();
      const r = new Ctor();
      r.lang = BCP47[lang] ?? "en-IN";
      r.continuous = true;
      r.interimResults = true;
      r.maxAlternatives = 1;
      r.onresult = (e: any) => {
        let text = "";
        for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
        text = text.trim();
        setInterim(text);
        // Only escalate to the server on meaningful growth, so we prefetch a handful
        // of times per utterance rather than on every keystroke-sized update.
        if (text.length >= 10 && text.length - lastSent.current.length >= 6) {
          lastSent.current = text;
          cb.current(text);
        }
      };
      r.onerror = () => setListening(false);
      r.onend = () => setListening(false);
      rec.current = r;
      lastSent.current = "";
      setInterim("");
      try {
        r.start();
        setListening(true);
      } catch {
        setListening(false);
      }
    },
    [Ctor, stop],
  );

  return { supported, listening, interim, start, stop };
}
