import { Icon, Icons } from "./ui";

/** First-run explainer. Three steps, no jargon, shown once. */
export function Welcome({ onClose }: { onClose: () => void }) {
  return (
    <div className="ovl" role="dialog" aria-modal="true" aria-labelledby="w-title">
      <div className="ovl-card">
        <button className="ovl-x" onClick={onClose} aria-label="Close">
          <Icon path={Icons.cross} size={16} />
        </button>

        <svg viewBox="0 0 40 40" aria-hidden="true" className="brand-mark" style={{ width: 44, height: 44 }}>
          <path className="pier" d="M7 31V20M20 31v-6M33 31V20" />
          <path className="arc" d="M5 30C5 16 11.6 9 20 9s15 7 15 21" />
          <path className="pier" d="M3 31h34" />
          <circle className="trav" cx="20" cy="9" r="3.1" />
        </svg>

        <h2 id="w-title" className="ovl-title">
          Answers you can actually check
        </h2>
        <p className="ovl-sub">
          SETU searches a fixed library of real web passages in 10 Indian languages. It quotes the
          sentences it used — and says so when it doesn&rsquo;t know.
        </p>

        <div className="ovl-steps">
          <div className="ovl-step">
            <span className="ovl-num">1</span>
            <div>
              <b>Ask in your language</b>
              <p>
                Tap the microphone or just type. Hindi, Kannada, Tamil, Telugu, Bengali, Marathi,
                Malayalam, Gujarati, Punjabi or English.
              </p>
            </div>
          </div>
          <div className="ovl-step">
            <span className="ovl-num">2</span>
            <div>
              <b>Listen, if you like</b>
              <p>
                Turn on <b>Read aloud</b> and answers are spoken back in the same language, using a
                voice that suits it.
              </p>
            </div>
          </div>
          <div className="ovl-step">
            <span className="ovl-num">3</span>
            <div>
              <b>Check the sources</b>
              <p>
                The little numbers in an answer are citations. Open “Where this came from” to read
                the original passages yourself.
              </p>
            </div>
          </div>
        </div>

        <button className="ovl-go" onClick={onClose}>
          Start asking
          <Icon path={Icons.send} size={15} />
        </button>
      </div>
    </div>
  );
}
