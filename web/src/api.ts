import type {
  AppConfig,
  CorpusStats,
  HealthReport,
  MetricsReport,
  QuestionBank,
  SampleQuery,
  TranscriptResult,
  TurnResult,
} from "./types";

const BASE = "/api/v1";

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* keep statusText */
    }
    throw new Error(`${res.status} ${detail}`);
  }
  return (await res.json()) as T;
}

export const api = {
  health: () => json<HealthReport>("/health"),
  config: () => json<AppConfig>("/config"),
  corpus: () => json<CorpusStats>("/corpus"),
  metrics: () => json<MetricsReport>("/metrics"),
  resetMetrics: () => json<{ status: string }>("/metrics", { method: "DELETE" }),
  tools: () => json<{ count: number; manifest: unknown[] }>("/tools"),
  samples: (limit = 12) =>
    json<{ samples: SampleQuery[]; languages: string[] }>(`/samples?limit=${limit}`),

  /** Every question the corpus was built from, for the explore panel. */
  questions: (opts: { lang?: string | null; q?: string; answerableOnly?: boolean; limit?: number } = {}) => {
    const p = new URLSearchParams();
    if (opts.lang) p.set("lang", opts.lang);
    if (opts.q) p.set("q", opts.q);
    if (opts.answerableOnly) p.set("answerable_only", "true");
    p.set("limit", String(opts.limit ?? 200));
    return json<QuestionBank>(`/questions?${p.toString()}`);
  },

  ask: (body: { text: string; lang?: string | null; mode?: string; top_k?: number }) =>
    json<TurnResult>("/ask", { method: "POST", body: JSON.stringify(body) }),

  prefetch: (text: string, lang?: string | null) =>
    json<{ prefetched: boolean; cached?: boolean; ms?: number; hits?: number }>("/prefetch", {
      method: "POST",
      body: JSON.stringify({ text, lang }),
    }),

  /** Full voice turn: STT on the server, then the pipeline. */
  async voice(
    blob: Blob,
    opts: { lang?: string | null; mode?: string; speak?: boolean } = {},
  ): Promise<{ turn: TurnResult; audio_base64?: string; audio_mime?: string }> {
    const form = new FormData();
    form.append("audio", blob, "turn.webm");
    if (opts.lang) form.append("lang", opts.lang);
    form.append("mode", opts.mode ?? "reflex");
    form.append("speak", String(Boolean(opts.speak)));
    const res = await fetch(`${BASE}/voice`, { method: "POST", body: form });
    if (!res.ok) {
      let detail = res.statusText;
      try {
        const b = await res.json();
        detail = typeof b.detail === "string" ? b.detail : JSON.stringify(b.detail);
      } catch {
        /* noop */
      }
      throw new Error(detail);
    }
    return await res.json();
  },

  async stt(blob: Blob, lang?: string | null): Promise<TranscriptResult> {
    const form = new FormData();
    form.append("audio", blob, "turn.webm");
    if (lang) form.append("lang", lang);
    const res = await fetch(`${BASE}/stt`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return await res.json();
  },

  /** `lang` picks the voice, the model and the pronunciation pin on the server. */
  async tts(text: string, lang?: string): Promise<Blob> {
    const res = await fetch(`${BASE}/tts`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, lang }),
    });
    if (!res.ok) throw new Error(`text-to-speech failed: ${res.status}`);
    return await res.blob();
  },
};

/** Persistent websocket used for speculative prefetch during speech. */
export class StreamChannel {
  private ws: WebSocket | null = null;
  private queue: string[] = [];

  constructor(private onMessage: (msg: Record<string, unknown>) => void) {}

  connect(): void {
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) return;
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}${BASE}/stream`);
    ws.onopen = () => {
      this.queue.forEach((m) => ws.send(m));
      this.queue = [];
    };
    ws.onmessage = (e) => {
      try {
        this.onMessage(JSON.parse(e.data));
      } catch {
        /* ignore malformed frame */
      }
    };
    ws.onclose = () => {
      this.ws = null;
    };
    ws.onerror = () => {
      this.ws = null;
    };
    this.ws = ws;
  }

  send(msg: Record<string, unknown>): void {
    const raw = JSON.stringify(msg);
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(raw);
    else {
      this.queue.push(raw);
      if (this.queue.length > 8) this.queue.shift();
      this.connect();
    }
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }
}
