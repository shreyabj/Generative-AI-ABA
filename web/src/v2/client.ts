// SETU 2.0 API client. Every request carries the browser's session id (conversation and
// trace ownership) and, if the user saved one in Settings, the API token.
const BASE = "/api/v1";

export function sessionId(): string {
  let s = localStorage.getItem("setu.session");
  if (!s) {
    s = (crypto.randomUUID?.() ?? Math.random().toString(36).slice(2) + Date.now().toString(36)).replace(/[^A-Za-z0-9_-]/g, "");
    localStorage.setItem("setu.session", s);
  }
  return s;
}

function headers(extra: Record<string, string> = {}): Record<string, string> {
  const h: Record<string, string> = { "X-SETU-Session": sessionId(), ...extra };
  const t = localStorage.getItem("setu.token");
  if (t) h["Authorization"] = `Bearer ${t}`;
  return h;
}

export async function req<T = any>(path: string, opts: { method?: string; body?: unknown; form?: FormData } = {}): Promise<T> {
  const init: RequestInit = { method: opts.method ?? (opts.body || opts.form ? "POST" : "GET"), headers: headers() };
  if (opts.form) init.body = opts.form;
  else if (opts.body !== undefined) {
    init.body = JSON.stringify(opts.body);
    (init.headers as Record<string, string>)["Content-Type"] = "application/json";
  }
  const r = await fetch(BASE + path, init);
  if (!r.ok) {
    let msg = `${r.status}`;
    try {
      const j = await r.json();
      msg += " " + (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
    } catch { /* not json */ }
    throw new Error(msg);
  }
  const ct = r.headers.get("content-type") || "";
  return (ct.includes("json") ? r.json() : r.text()) as Promise<T>;
}

export const fmt = {
  ms: (v?: number | null) => (v == null ? "–" : v < 1 ? `${v.toFixed(2)} ms` : v < 1000 ? `${v.toFixed(1)} ms` : `${(v / 1000).toFixed(2)} s`),
  pct: (v?: number | null) => (v == null ? "–" : `${(v * 100).toFixed(1)}%`),
  num: (v?: number | null, d = 3) => (v == null ? "–" : Number(v).toFixed(d)),
  date: (s?: string | null) => (s ? new Date(s).toLocaleString() : "–"),
};
