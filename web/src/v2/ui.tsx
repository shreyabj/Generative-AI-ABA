import { ReactNode, useCallback, useEffect, useRef, useState } from "react";

/** Data hook with loading / error / reload and optional polling. */
export function useData<T>(fn: () => Promise<T>, deps: unknown[] = [], pollMs = 0) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const fnRef = useRef(fn);
  fnRef.current = fn;
  const load = useCallback(async () => {
    try {
      setError(null);
      setData(await fnRef.current());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);
  useEffect(() => {
    setLoading(true);
    void load();
    if (!pollMs) return;
    const t = setInterval(load, pollMs);
    return () => clearInterval(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
  return { data, error, loading, reload: load, setData };
}

export function State({ loading, error, empty, children }: { loading?: boolean; error?: string | null; empty?: string | false; children?: ReactNode }) {
  if (error) return <div className="v2-state err" role="alert">⚠ {error}</div>;
  if (loading) return <div className="v2-state"><span className="v2-spin" /> Loading…</div>;
  if (empty) return <div className="v2-state">{empty}</div>;
  return <>{children}</>;
}

export function Card({ title, extra, children, className = "" }: { title?: ReactNode; extra?: ReactNode; children: ReactNode; className?: string }) {
  return (
    <section className={`v2-card ${className}`}>
      {(title || extra) && <header><h3>{title}</h3>{extra}</header>}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: ReactNode }) {
  return (
    <div className="v2-stat">
      <div className="v2-stat-v">{value}</div>
      <div className="v2-stat-l">{label}</div>
      {hint && <div className="v2-stat-h">{hint}</div>}
    </div>
  );
}

export function Badge({ tone = "", children }: { tone?: string; children: ReactNode }) {
  return <span className={`v2-badge ${tone}`}>{children}</span>;
}

export function statusTone(s?: string) {
  if (!s) return "";
  if (["ready", "completed", "answered", "ok", "allow"].includes(s)) return "good";
  if (["failed", "error", "refused", "blocked", "block"].includes(s)) return "bad";
  if (["partial", "flagged", "flag", "abstained", "skipped"].includes(s)) return "warn";
  return "info";
}

/** Horizontal bar chart in plain SVG (no chart library). */
export function Bars({ rows, max, fmt = (v: number) => v.toFixed(3) }: { rows: { label: string; value: number; tone?: string }[]; max?: number; fmt?: (v: number) => string }) {
  const m = max ?? Math.max(1e-9, ...rows.map((r) => r.value));
  return (
    <div className="v2-bars">
      {rows.map((r) => (
        <div key={r.label} className="v2-bar">
          <span className="v2-bar-l" title={r.label}>{r.label}</span>
          <span className="v2-bar-t"><span className={`v2-bar-f ${r.tone ?? ""}`} style={{ width: `${Math.max(1, (r.value / m) * 100)}%` }} /></span>
          <span className="v2-bar-v">{fmt(r.value)}</span>
        </div>
      ))}
    </div>
  );
}

export function Json({ value }: { value: unknown }) {
  return <pre className="v2-json">{JSON.stringify(value, null, 2)}</pre>;
}
