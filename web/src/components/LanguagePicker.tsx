import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { LANGUAGES, type Lang } from "../languages";
import { Icon, Icons } from "./ui";

/** Gap between the trigger and the popup, and the minimum margin from the viewport edge. */
const GAP = 8;
const EDGE = 12;
const POP_W = 330;
const POP_MAX_H = 408;
const CHROME_H = 56; /* the search field above the list */

interface Placement {
  left: number;
  width: number;
  listMax: number;
  up: boolean;
  top?: number;
  bottom?: number;
}

/**
 * Language picker.
 *
 * A native `<select>` was unusable: browsers render the popup with the OS theme, so options
 * were near-invisible. This is a custom listbox — styled, keyboard navigable, searchable, and
 * showing every language in its own script, because a Kannada speaker scans for "ಕನ್ನಡ".
 *
 * Two groups, because coverage genuinely differs and pretending otherwise misleads:
 * languages with indexed passages answer in that language; the rest still work (the search is
 * cross-lingual) but answer in English. A small speaker badge marks languages this device has
 * a real voice for, so you know before you press play whether it will sound right.
 *
 * The list is rendered into `document.body` through a portal rather than next to the trigger.
 * Absolute positioning was not enough: the ask card animates in with `.rise`, whose filled
 * transform/opacity animation makes it a permanent stacking context with `z-index: auto`. Any
 * z-index on the popup was therefore scoped *inside* that card, so the sample buttons and the
 * answer cards that follow it in the DOM painted over the list. A portal escapes the stacking
 * context (and any future clipping ancestor) instead of fighting it.
 */
export function LanguagePicker({
  value,
  onChange,
  corpusLangs,
  disabled,
  hasNativeVoice,
}: {
  value: string;
  onChange: (code: string) => void;
  corpusLangs: string[];
  disabled?: boolean;
  hasNativeVoice?: (lang: string) => boolean;
}) {
  const [open, setOpen] = useState(false);
  const [place, setPlace] = useState<Placement | null>(null);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);

  const inCorpus = useMemo(() => new Set(corpusLangs), [corpusLangs]);

  const { indexed, other } = useMemo(() => {
    const q = query.trim().toLowerCase();
    const match = (l: Lang) =>
      !q || l.name.toLowerCase().includes(q) || l.native.includes(query.trim()) || l.code.includes(q);
    const hits = LANGUAGES.filter(match);
    return {
      indexed: hits.filter((l) => inCorpus.has(l.code)),
      other: hits.filter((l) => !inCorpus.has(l.code)),
    };
  }, [inCorpus, query]);

  const flat = useMemo(() => [...indexed, ...other], [indexed, other]);

  /**
   * Position against the trigger in viewport coordinates. Opens upwards when there is more
   * room above, because the picker sits near the bottom of the ask card and a downward-only
   * list would run past the fold on short viewports.
   */
  const measure = useCallback(() => {
    const r = trigger.current?.getBoundingClientRect();
    if (!r) return;
    const width = Math.min(POP_W, window.innerWidth - EDGE * 2);
    const below = window.innerHeight - r.bottom - GAP - EDGE;
    const above = r.top - GAP - EDGE;
    const up = below < 260 && above > below;
    const room = Math.min(POP_MAX_H, Math.max(up ? above : below, 160));
    const left = Math.min(
      Math.max(EDGE, r.left + r.width / 2 - width / 2),
      Math.max(EDGE, window.innerWidth - width - EDGE),
    );
    setPlace({
      left,
      width,
      up,
      listMax: Math.max(120, room - CHROME_H),
      ...(up ? { bottom: window.innerHeight - r.top + GAP } : { top: r.bottom + GAP }),
    });
  }, []);

  useLayoutEffect(() => {
    if (open) measure();
  }, [open, measure]);

  useEffect(() => {
    if (!open) return;

    const onDoc = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!root.current?.contains(t) && !pop.current?.contains(t)) setOpen(false);
    };
    // `true` so an ancestor scroll container also re-positions, not just the window.
    const onMove = () => measure();

    document.addEventListener("mousedown", onDoc);
    window.addEventListener("resize", onMove);
    window.addEventListener("scroll", onMove, true);
    const t = setTimeout(() => search.current?.focus(), 30);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      window.removeEventListener("resize", onMove);
      window.removeEventListener("scroll", onMove, true);
      clearTimeout(t);
    };
  }, [open, measure]);

  useEffect(() => setActive(0), [query]);

  const close = useCallback(() => {
    setOpen(false);
    setQuery("");
    trigger.current?.focus();
  }, []);

  const choose = (code: string) => {
    onChange(code);
    setOpen(false);
    setQuery("");
  };

  // Portal children still bubble through the React tree, so this handler covers the popup too.
  const onKey = (e: React.KeyboardEvent) => {
    if (!open) return;
    if (e.key === "Escape") {
      e.preventDefault();
      return close();
    }
    if (e.key === "Tab") return close();
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => Math.min(a + 1, flat.length));
    }
    if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => Math.max(a - 1, 0));
    }
    if (e.key === "Enter") {
      e.preventDefault();
      if (active === 0) choose("");
      else if (flat[active - 1]) choose(flat[active - 1].code);
    }
  };

  const current = LANGUAGES.find((l) => l.code === value);

  const Row = ({ l, idx }: { l: Lang; idx: number }) => (
    <button
      type="button"
      role="option"
      aria-selected={value === l.code}
      className={`lp-opt ${value === l.code ? "sel" : ""} ${active === idx ? "act" : ""}`}
      onClick={() => choose(l.code)}
      onMouseEnter={() => setActive(idx)}
    >
      <span className="lp-native">{l.native}</span>
      <span className="lp-name">{l.name}</span>
      {hasNativeVoice?.(l.code) ? (
        <span className="lp-voice" title={`This device has a ${l.name} voice installed`}>
          <Icon path={Icons.speaker} size={11} />
        </span>
      ) : (
        <span />
      )}
    </button>
  );

  const popup = place && (
    <div
      ref={pop}
      className={`lp-pop ${place.up ? "up" : ""}`}
      role="listbox"
      aria-label="Ask in which language"
      style={{
        left: place.left,
        width: place.width,
        top: place.top,
        bottom: place.bottom,
        ["--lp-list-max" as string]: `${place.listMax}px`,
      }}
    >
      <div className="lp-search">
        <input
          ref={search}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search 15 languages…"
          aria-label="Search languages"
        />
      </div>

      <div className="lp-list">
        <button
          type="button"
          role="option"
          aria-selected={value === ""}
          className={`lp-opt ${value === "" ? "sel" : ""} ${active === 0 ? "act" : ""}`}
          onClick={() => choose("")}
          onMouseEnter={() => setActive(0)}
        >
          <span className="lp-native">Any language</span>
          <span className="lp-name">detected automatically</span>
          <span />
        </button>

        {indexed.length > 0 && (
          <>
            <div className="lp-group">
              Full support
              <span className="lp-group-hint">answers come back in this language</span>
            </div>
            {indexed.map((l, i) => (
              <Row key={l.code} l={l} idx={i + 1} />
            ))}
          </>
        )}

        {other.length > 0 && (
          <>
            <div className="lp-group">
              Ask anyway
              <span className="lp-group-hint">understood, but answers arrive in English for now</span>
            </div>
            {other.map((l, i) => (
              <Row key={l.code} l={l} idx={indexed.length + 1 + i} />
            ))}
          </>
        )}

        {flat.length === 0 && <p className="lp-none">Nothing matches “{query}”.</p>}
      </div>
    </div>
  );

  return (
    <div className="lp" ref={root} onKeyDown={onKey}>
      <button
        ref={trigger}
        type="button"
        className="lp-btn"
        onClick={() => setOpen((o) => !o)}
        disabled={disabled}
        aria-haspopup="listbox"
        aria-expanded={open}
        title="Which language you're asking in"
      >
        <Icon path={Icons.globe} size={14} />
        <span className="lp-btn-v">{current ? current.native : "Any language"}</span>
        <Icon path={Icons.chevron} size={13} />
      </button>

      {open && popup && createPortal(popup, document.body)}
    </div>
  );
}
