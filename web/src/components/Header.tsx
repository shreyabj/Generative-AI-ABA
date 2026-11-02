import type { Theme } from "../hooks/useTheme";
import type { VoiceOutState } from "../hooks/useVoiceOut";
import type { AppConfig, HealthReport } from "../types";
import { Icon, Icons } from "./ui";

interface Props {
  config: AppConfig | null;
  health: HealthReport | null;
  readAloud: boolean;
  onReadAloud: (v: boolean) => void;
  voice: VoiceOutState;
  theme: Theme;
  onTheme: () => void;
  onHelp: () => void;
}

/** The SETU mark: a bridge arc with a signal crossing it. */
function Mark() {
  return (
    <svg className="brand-mark" viewBox="0 0 40 40" role="img" aria-label="SETU">
      <path className="pier" d="M7 31V20M20 31v-6M33 31V20" />
      <path className="arc" d="M5 30C5 16 11.6 9 20 9s15 7 15 21" />
      <path className="pier" d="M3 31h34" />
      <circle className="trav" cx="20" cy="9" r="3.1" />
    </svg>
  );
}

export function Header({
  config,
  health,
  readAloud,
  onReadAloud,
  voice,
  theme,
  onTheme,
  onHelp,
}: Props) {
  const langCount = config?.languages.length ?? 0;

  return (
    <header className="hdr">
      <div className="brand">
        <Mark />
        <div>
          <div className="brand-name">SETU</div>
          <div className="brand-sub">
            {langCount > 0 ? `${langCount} languages · answers with sources` : "answers with sources"}
          </div>
        </div>
      </div>

      <div className="spacer" />

      <button
        className={`btn btn-sm ${readAloud ? "btn-on" : ""}`}
        onClick={() => onReadAloud(!readAloud)}
        disabled={!voice.available}
        aria-pressed={readAloud}
        title={
          voice.available
            ? readAloud
              ? "Answers are read aloud. Click to stop."
              : "Read every answer aloud, in the language it is written in."
            : "No speech voices are available on this device"
        }
      >
        <Icon path={readAloud && voice.available ? Icons.speaker : Icons.mute} size={14} />
        <span className="hide-sm">{voice.speaking ? "Speaking…" : "Read aloud"}</span>
      </button>

      <button
        className="btn btn-sm btn-icon"
        onClick={onTheme}
        title={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
        aria-label={theme === "light" ? "Switch to dark mode" : "Switch to light mode"}
      >
        <Icon path={theme === "light" ? Icons.moon : Icons.sun} size={15} />
      </button>

      <button
        className="btn btn-sm btn-icon"
        onClick={onHelp}
        title="What is this?"
        aria-label="What is this?"
      >
        <Icon path={Icons.help} size={15} />
      </button>

      {health?.status !== "ready" && (
        <span className="chip chip-amber">
          <span className="dot dot-warn" />
          starting
        </span>
      )}
    </header>
  );
}
