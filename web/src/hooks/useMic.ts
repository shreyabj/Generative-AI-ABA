import { useCallback, useEffect, useRef, useState } from "react";

export interface MicState {
  recording: boolean;
  level: number;
  error: string | null;
  supported: boolean;
  durationMs: number;
}

/**
 * Microphone capture with a live amplitude signal.
 *
 * MediaRecorder gives us the encoded blob for Scribe; a parallel AnalyserNode on the
 * same stream gives us an RMS level for the orb visualiser. Both read one stream, so
 * the waveform is genuinely the audio being sent rather than a decorative animation.
 */
export function useMic(onStop: (blob: Blob, mime: string, durationMs: number) => void) {
  const [state, setState] = useState<MicState>({
    recording: false,
    level: 0,
    error: null,
    supported: typeof navigator !== "undefined" && !!navigator.mediaDevices?.getUserMedia,
    durationMs: 0,
  });

  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);
  const audioCtx = useRef<AudioContext | null>(null);
  const analyser = useRef<AnalyserNode | null>(null);
  const raf = useRef<number>(0);
  const startedAt = useRef<number>(0);
  const cb = useRef(onStop);
  cb.current = onStop;

  const teardown = useCallback(() => {
    cancelAnimationFrame(raf.current);
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    analyser.current = null;
    if (audioCtx.current && audioCtx.current.state !== "closed") void audioCtx.current.close();
    audioCtx.current = null;
    recorder.current = null;
  }, []);

  useEffect(() => teardown, [teardown]);

  const meter = useCallback(() => {
    const node = analyser.current;
    if (!node) return;
    const buf = new Uint8Array(node.fftSize);
    const tick = () => {
      if (!analyser.current) return;
      analyser.current.getByteTimeDomainData(buf);
      let sum = 0;
      for (let i = 0; i < buf.length; i++) {
        const v = (buf[i] - 128) / 128;
        sum += v * v;
      }
      const rms = Math.sqrt(sum / buf.length);
      setState((s) => ({
        ...s,
        // Perceptual curve: quiet speech should still visibly move the orb.
        level: Math.min(1, Math.pow(rms * 3.4, 0.7)),
        durationMs: performance.now() - startedAt.current,
      }));
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
  }, []);

  const start = useCallback(async () => {
    if (state.recording) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
      stream.current = s;

      const ctx = new AudioContext();
      audioCtx.current = ctx;
      const src = ctx.createMediaStreamSource(s);
      const node = ctx.createAnalyser();
      node.fftSize = 1024;
      node.smoothingTimeConstant = 0.7;
      src.connect(node);
      analyser.current = node;

      const mime = ["audio/webm;codecs=opus", "audio/webm", "audio/mp4"].find((m) =>
        MediaRecorder.isTypeSupported(m),
      );
      const rec = new MediaRecorder(s, mime ? { mimeType: mime } : undefined);
      chunks.current = [];
      rec.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.current.push(e.data);
      };
      rec.onstop = () => {
        const type = rec.mimeType || "audio/webm";
        const blob = new Blob(chunks.current, { type });
        const ms = performance.now() - startedAt.current;
        teardown();
        setState((p) => ({ ...p, recording: false, level: 0, durationMs: ms }));
        if (blob.size > 1200) cb.current(blob, type, ms);
      };
      recorder.current = rec;
      startedAt.current = performance.now();
      rec.start(220);
      setState((p) => ({ ...p, recording: true, error: null, durationMs: 0 }));
      meter();
    } catch (e) {
      setState((p) => ({
        ...p,
        recording: false,
        error: e instanceof Error ? e.message : "microphone unavailable",
      }));
    }
  }, [meter, state.recording, teardown]);

  const stop = useCallback(() => {
    if (recorder.current?.state === "recording") recorder.current.stop();
    else {
      teardown();
      setState((p) => ({ ...p, recording: false, level: 0 }));
    }
  }, [teardown]);

  return { ...state, start, stop };
}
