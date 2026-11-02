import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, StreamChannel } from "./api";
import { Answer } from "./components/Answer";
import { AskPanel } from "./components/AskPanel";
import { Examples } from "./components/Examples";
import { Header } from "./components/Header";
import { Hero } from "./components/Hero";
import { Notice } from "./components/Notice";
import { QuestionBrowser } from "./components/QuestionBrowser";
import { Sources } from "./components/Sources";
import { Welcome } from "./components/Welcome";
import { Icon, Icons } from "./components/ui";
import { useLiveSpeech } from "./hooks/useLiveSpeech";
import { useMic } from "./hooks/useMic";
import { useTheme } from "./hooks/useTheme";
import { useVoiceOut } from "./hooks/useVoiceOut";
import type { AppConfig, HealthReport, SampleQuery, TurnResult } from "./types";
import { LANG_BY_CODE } from "./languages";

const SEEN_KEY = "setu.welcomed.v2";

export default function App() {
  const [health, setHealth] = useState<HealthReport | null>(null);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [samples, setSamples] = useState<SampleQuery[]>([]);

  const [turn, setTurn] = useState<TurnResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [lang, setLang] = useState("");
  const [readAloud, setReadAloud] = useState(false);
  const [activeCite, setActiveCite] = useState<number | null>(null);
  const [showWelcome, setShowWelcome] = useState(false);
  const [showQuestions, setShowQuestions] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { theme, toggle: toggleTheme } = useTheme();
  const voice = useVoiceOut(Boolean(config?.tts.enabled));
  const answerRef = useRef<HTMLDivElement>(null);

  /* ------------------------------------------------------------ boot polling */
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const h = await api.health();
        if (!alive) return;
        setHealth(h);
        if (h.status === "ready") {
          const [c, s] = await Promise.all([api.config(), api.samples(12)]);
          if (!alive) return;
          setConfig(c);
          setSamples(s.samples);
          if (!localStorage.getItem(SEEN_KEY)) setShowWelcome(true);
          return;
        }
        setTimeout(poll, 1200);
      } catch {
        if (!alive) return;
        setTimeout(poll, 2200);
      }
    };
    void poll();
    return () => {
      alive = false;
    };
  }, []);

  /* -------------------------------------------------- speculative prefetch */
  const channel = useMemo(() => new StreamChannel(() => {}), []);
  useEffect(() => () => channel.close(), [channel]);

  const speech = useLiveSpeech((partial) => {
    channel.send({ type: "partial", text: partial, lang: lang || null });
  });

  /* ------------------------------------------------------------------ speak */
  const speakAnswer = useCallback(
    async (t: TurnResult, preloaded?: Blob) => {
      if (voice.speaking) {
        voice.stop();
        return;
      }
      await voice.speak(t.answer, t.answer_lang, preloaded);
    },
    [voice],
  );

  const show = useCallback(
    (t: TurnResult) => {
      setTurn(t);
      setActiveCite(null);
      setError(null);
      requestAnimationFrame(() =>
        answerRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }),
      );
    },
    [],
  );

  /* -------------------------------------------------------------- ask (text) */
  //
  // `langOverride` exists because clicking a suggested question has to set the picker *and*
  // ask in one gesture. Relying on setLang followed by ask() sent the previous language:
  // the state update had not applied yet, so a Kannada example was asked as whatever was
  // selected before it, and the answer came back in that language.
  const ask = useCallback(
    async (text: string, langOverride?: string | null) => {
      if (busy) return;
      setBusy(true);
      voice.stop();
      const use = langOverride === undefined ? lang : langOverride;
      try {
        const t = await api.ask({ text, lang: use || null, mode: "reflex" });
        show(t);
        if (readAloud && t.answer && t.status !== "refused") void speakAnswer(t);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Something went wrong. Please try again.");
      } finally {
        setBusy(false);
      }
    },
    [busy, lang, readAloud, show, speakAnswer, voice],
  );

  /** Ask a suggested question in its own language. */
  const askSample = useCallback(
    (s: SampleQuery) => {
      const code = LANG_BY_CODE[s.lang] ? s.lang : "";
      setLang(code);
      setShowQuestions(false);
      void ask(s.query, code);
    },
    [ask],
  );

  /* ------------------------------------------------------------- ask (voice) */
  const onRecorded = useCallback(
    async (blob: Blob, _mime: string, _ms: number) => {
      setBusy(true);
      try {
        if (config?.stt.enabled) {
          const res = await api.voice(blob, { lang: lang || null, mode: "reflex", speak: readAloud });
          const t = res.turn;
          show(t);
          if (readAloud && t.answer && t.status !== "refused") {
            let pre: Blob | undefined;
            if (res.audio_base64) {
              const bin = atob(res.audio_base64);
              const bytes = new Uint8Array(bin.length);
              for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
              pre = new Blob([bytes], { type: res.audio_mime ?? "audio/mpeg" });
            }
            void speakAnswer(t, pre);
          }
        } else {
          const text = speech.interim.trim();
          if (!text) {
            setError("I didn't catch that. Try again, or type your question.");
            return;
          }
          const t = await api.ask({ text, lang: lang || null, mode: "reflex" });
          show(t);
          if (readAloud && t.answer && t.status !== "refused") void speakAnswer(t);
        }
      } catch (e) {
        setError(e instanceof Error ? e.message : "Couldn't process that recording.");
      } finally {
        setBusy(false);
      }
    },
    [config?.stt.enabled, lang, readAloud, show, speakAnswer, speech.interim],
  );

  const mic = useMic(onRecorded);

  const toggleMic = useCallback(() => {
    if (busy) return;
    if (mic.recording) {
      mic.stop();
      speech.stop();
    } else {
      voice.stop();
      setError(null);
      void mic.start();
      if (speech.supported) speech.start(lang || "en");
    }
  }, [busy, lang, mic, speech, voice]);

  const dismissWelcome = useCallback(() => {
    localStorage.setItem(SEEN_KEY, "1");
    setShowWelcome(false);
  }, []);

  /* --------------------------------------------------------------- boot view */
  if (health?.status !== "ready") {
    return (
      <>
        <Ambient />
        <div className="shell">
          <div className="boot">
            <div>
              <div className="brand-name" style={{ fontSize: 28, letterSpacing: "0.2em" }}>
                SETU
              </div>
              <div className="brand-sub" style={{ marginTop: 6 }}>
                answers with sources
              </div>
            </div>
            {health?.status === "error" ? (
              <>
                <p className="dim" style={{ maxWidth: 520 }}>
                  The search library couldn&rsquo;t be loaded.
                </p>
                <pre>
                  {health.error}
                  {"\n\n"}Build one first:{"\n"}
                  {"  python -m setu.cli build --sample"}
                </pre>
              </>
            ) : (
              <>
                <div className="boot-bar">
                  <i />
                </div>
                <p className="dim tiny">Getting the library ready…</p>
              </>
            )}
          </div>
        </div>
      </>
    );
  }

  const corpusLangs = (config?.languages ?? []).map((l) => l.code);

  return (
    <>
      <Ambient />
      {showWelcome && <Welcome onClose={dismissWelcome} />}
      {showQuestions && (
        <QuestionBrowser onPick={askSample} onClose={() => setShowQuestions(false)} />
      )}

      <div className="shell">
        <Header
          config={config}
          health={health}
          readAloud={readAloud}
          onReadAloud={(v) => {
            setReadAloud(v);
            if (!v) voice.stop();
            else if (turn?.answer && turn.status !== "refused") void speakAnswer(turn);
          }}
          voice={voice}
          theme={theme}
          onTheme={toggleTheme}
          onHelp={() => setShowWelcome(true)}
        />

        <main className="main">
          {!turn && <Hero config={config} />}

          <Notice
            config={config}
            voice={voice}
            micSupported={mic.supported}
            browserSttSupported={speech.supported}
          />

          <AskPanel
            busy={busy}
            recording={mic.recording}
            level={mic.level}
            interim={speech.interim}
            recordingMs={mic.durationMs}
            micSupported={mic.supported}
            micError={mic.error}
            corpusLangs={corpusLangs}
            lang={lang}
            onLang={setLang}
            onAsk={ask}
            onToggleMic={toggleMic}
            hasNativeVoice={voice.hasNativeVoice}
          />

          {!turn && (
            <Examples
              samples={samples}
              busy={busy}
              onAsk={askSample}
              onExplore={() => setShowQuestions(true)}
            />
          )}

          {error && (
            <div className="notice fade-in">
              <Icon path={Icons.warn} size={15} />
              <span>{error}</span>
              <button className="notice-x" onClick={() => setError(null)} aria-label="Dismiss">
                <Icon path={Icons.cross} size={14} />
              </button>
            </div>
          )}

          <div ref={answerRef} style={{ display: "contents" }}>
            {turn && (
              <>
                <Answer
                  turn={turn}
                  activeCite={activeCite}
                  onCite={setActiveCite}
                  onSpeak={() => void speakAnswer(turn)}
                  speaking={voice.speaking}
                  voiceAvailable={voice.available}
                />
                <Sources turn={turn} activeCite={activeCite} onCite={setActiveCite} />
              </>
            )}
          </div>

          {turn && (
            <div className="samples" style={{ marginTop: 4 }}>
              <button className="sample" onClick={() => setTurn(null)} disabled={busy}>
                <span className="sample-lang">new</span>
                <span className="sample-q">Ask something else</span>
              </button>
              <button className="sample" onClick={() => setShowQuestions(true)} disabled={busy}>
                <span className="sample-lang">
                  <Icon path={Icons.book} size={12} />
                </span>
                <span className="sample-q">Explore more questions</span>
              </button>
            </div>
          )}
        </main>

        <footer className="ftr">
          <span>Library: MS MARCO-XI · {config?.languages.length} languages</span>
          <button onClick={() => setShowWelcome(true)}>How it works</button>
          <a href="/api/docs" target="_blank" rel="noreferrer">
            API
          </a>
          <span>#RAGInGoa</span>
        </footer>
      </div>
    </>
  );
}

function Ambient() {
  return (
    <div className="ambient" aria-hidden="true">
      <div className="ambient-grid" />
      <div className="ambient-grain" />
    </div>
  );
}
