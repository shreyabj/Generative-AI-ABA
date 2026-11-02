import { useState } from "react";
import type { VoiceOutState } from "../hooks/useVoiceOut";
import type { AppConfig } from "../types";
import { Icon, Icons } from "./ui";

/**
 * One dismissible line, and only when something is genuinely limited.
 *
 * This replaced a five-item status strip. That strip was accurate and nobody wanted it: a
 * first-time visitor does not need a report on which inference providers are configured. The
 * only thing worth interrupting them for is a capability that is missing *and* fixable — and
 * even then, once.
 */
export function Notice({
  config,
  voice,
  micSupported,
  browserSttSupported,
}: {
  config: AppConfig | null;
  voice: VoiceOutState;
  micSupported: boolean;
  browserSttSupported: boolean;
}) {
  const [hidden, setHidden] = useState(false);
  if (!config || hidden) return null;

  let msg: string | null = null;

  if (!micSupported) {
    msg =
      "Your browser won't give this page a microphone. You can still type your question — and note that browsers only allow the mic over HTTPS.";
  } else if (!config.stt.enabled && !browserSttSupported) {
    msg =
      "Voice input isn't available here: this browser can't transcribe speech and no ElevenLabs key is set. Typing works fine.";
  } else if (!voice.available) {
    msg =
      "This device has no speech voices installed, so answers can't be read aloud. Everything else works.";
  }

  if (!msg) return null;

  return (
    <div className="notice fade-in">
      <Icon path={Icons.info} size={15} />
      <span>{msg}</span>
      <button className="notice-x" onClick={() => setHidden(true)} aria-label="Dismiss">
        <Icon path={Icons.cross} size={14} />
      </button>
    </div>
  );
}
