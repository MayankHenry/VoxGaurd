import { useEffect, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";
const MAX_SEC = 30; // the training pipeline accepts 1-30s clips
const BARS = 160;

/* ---------- audio helpers ---------- */
const peaksOf = (pcm, n = BARS) => {
  const step = Math.max(1, Math.floor(pcm.length / n));
  return Array.from({ length: n }, (_, i) => {
    let m = 0;
    for (let j = i * step; j < Math.min(pcm.length, (i + 1) * step); j++) m = Math.max(m, Math.abs(pcm[j]));
    return m;
  });
};

// Decode anything the browser can play, then re-encode as 16 kHz mono WAV (same format as the training data).
async function toWav16k(blob) {
  const ctx = new AudioContext();
  const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
  ctx.close();
  const off = new OfflineAudioContext(1, Math.max(1, Math.ceil(buf.duration * 16000)), 16000);
  const src = off.createBufferSource();
  src.buffer = buf; src.connect(off.destination); src.start();
  const pcm = (await off.startRendering()).getChannelData(0);
  const v = new DataView(new ArrayBuffer(44 + pcm.length * 2));
  const s = (o, t) => [...t].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  s(0, "RIFF"); v.setUint32(4, 36 + pcm.length * 2, true); s(8, "WAVEfmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, 16000, true); v.setUint32(28, 32000, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, "data"); v.setUint32(40, pcm.length * 2, true);
  pcm.forEach((x, i) => v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x)) * 0x7fff, true));
  return { wav: new Blob([v], { type: "audio/wav" }), peaks: peaksOf(pcm), duration: buf.duration };
}

function drawBars(canvas, peaks, color) {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = canvas;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr); g.clearRect(0, 0, w, h);
  const slot = w / BARS;
  g.fillStyle = color;
  for (let i = 0; i < BARS; i++) {
    const p = peaks[i] ?? 0;
    const bh = Math.max(3, Math.min(1, p * 1.6) * h * 0.9);
    g.globalAlpha = peaks[i] === undefined ? 0.18 : 1;
    g.beginPath(); g.roundRect(i * slot + slot * 0.2, (h - bh) / 2, slot * 0.6, bh, 3); g.fill();
  }
}

/* ---------- app ---------- */
export default function App() {
  const [online, setOnline] = useState(null);
  const [phase, setPhase] = useState("idle"); // idle | recording | ready | analyzing | done | error
  const [clip, setClip] = useState(null); // { wav, peaks, duration, name }
  const [result, setResult] = useState(null);
  const [msg, setMsg] = useState("");
  const [secs, setSecs] = useState(0);
  const [drag, setDrag] = useState(false);

  const canvas = useRef(null);
  const live = useRef([]);
  const rec = useRef({});

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setOnline(r.ok)).catch(() => setOnline(false));
  }, []);

  const verdictFake = result && /fake|synth|spoof|ai/i.test(result.label);
  const color = phase === "recording" ? "#ff5d8f" : result ? (verdictFake ? "#ff5d8f" : "#5eead4") : "#8fa2ff";

  useEffect(() => {
    if (canvas.current && phase !== "recording") drawBars(canvas.current, clip?.peaks ?? [], color);
  }, [clip, color, phase]);

  async function load(blob, name) {
    setResult(null); setMsg("");
    try {
      const c = await toWav16k(blob);
      if (c.duration < 1 || c.duration > MAX_SEC + 1) throw new Error(`Clip is ${c.duration.toFixed(1)}s. Use 1 to ${MAX_SEC} seconds.`);
      setClip({ ...c, name }); setPhase("ready");
    } catch (e) {
      setClip(null); setPhase("error"); setMsg(e.message || "Could not read this audio file.");
    }
  }

  async function startRec() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const ctx = new AudioContext();
      const an = ctx.createAnalyser(); an.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(an);
      const mr = new MediaRecorder(stream), chunks = [];
      mr.ondataavailable = (e) => chunks.push(e.data);
      mr.onstop = () => {
        cancelAnimationFrame(rec.current.raf); clearInterval(rec.current.timer);
        stream.getTracks().forEach((t) => t.stop()); ctx.close();
        load(new Blob(chunks, { type: mr.mimeType }), "recording.wav");
      };
      live.current = []; setSecs(0); setResult(null); setClip(null); setPhase("recording");
      const buf = new Uint8Array(an.fftSize);
      const tick = () => {
        an.getByteTimeDomainData(buf);
        live.current.push(Math.max(...buf.map((b) => Math.abs(b - 128))) / 128);
        drawBars(canvas.current, live.current.slice(-BARS), "#ff5d8f");
        rec.current.raf = requestAnimationFrame(tick);
      };
      tick();
      const t0 = Date.now();
      rec.current.timer = setInterval(() => {
        const s = Math.floor((Date.now() - t0) / 1000); setSecs(s);
        if (s >= MAX_SEC) mr.stop();
      }, 250);
      rec.current.mr = mr; mr.start();
    } catch {
      setPhase("error"); setMsg("Microphone access was blocked. Allow it in your browser, or upload a file instead.");
    }
  }

  async function analyze() {
    setPhase("analyzing"); setMsg("");
    try {
      const fd = new FormData();
      fd.append("file", clip.wav, (clip.name || "clip").replace(/\.\w+$/, "") + ".wav");
      const up = await fetch(`${API}/upload`, { method: "POST", body: fd });
      if (!up.ok) throw new Error((await up.json().catch(() => ({}))).detail || "Upload failed.");
      const { sample_id } = await up.json();
      // Expected contract (not in the backend yet): POST /predict/{sample_id} -> { label: "real" | "fake", confidence: 0..1 }
      const pr = await fetch(`${API}/predict/${sample_id}`, { method: "POST" });
      if (pr.status === 404) {
        setPhase("ready");
        setMsg("Clip uploaded, but the backend has no /predict endpoint yet, so there is no verdict to show.");
        return;
      }
      if (!pr.ok) throw new Error("The model could not analyse this clip.");
      setResult(await pr.json()); setPhase("done");
    } catch (e) {
      setPhase("error"); setMsg(e.message?.includes("fetch") ? `Cannot reach the API at ${API}. Is the backend running?` : e.message);
    }
  }

  const busy = phase === "recording" || phase === "analyzing";
  const pct = result ? Math.round(result.confidence * 100) : 0;

  return (
    <main className="mx-auto max-w-5xl px-5 py-8 sm:py-12">
      <header className="flex items-center justify-between">
        <h1 className="font-display text-2xl font-bold tracking-tight">VoxGuard</h1>
        <span className="flex items-center gap-2 text-sm text-dim" role="status">
          <i className={`h-2.5 w-2.5 rounded-full ${online ? "bg-human" : online === false ? "bg-synth" : "bg-dim"}`} />
          {online ? "API connected" : online === false ? "API offline" : "Checking API"}
        </span>
      </header>

      <section className="mt-12 max-w-2xl">
        <h2 className="font-display text-5xl font-bold leading-[1.02] tracking-tight sm:text-6xl">Is that voice a real person?</h2>
        <p className="mt-5 max-w-xl text-lg text-dim">
          Record a few seconds or drop in a call recording. VoxGuard looks at pitch, timbre and spectral patterns for signs of an AI-generated or cloned voice.
        </p>
      </section>

      <section
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); const f = e.dataTransfer.files[0]; if (f && !busy) load(f, f.name); }}
        className={`mt-10 rounded-3xl border bg-panel/70 p-5 backdrop-blur sm:p-8 ${drag ? "border-human" : "border-line"}`}
      >
        <canvas ref={canvas} className="h-40 w-full sm:h-56" aria-label="Audio waveform" />

        <div className="mt-6 flex flex-wrap items-center gap-3">
          {phase === "recording" ? (
            <button onClick={() => rec.current.mr.stop()} className="rec flex h-14 items-center gap-3 rounded-full bg-synth px-7 font-semibold text-ink">
              <span className="h-3 w-3 rounded-sm bg-ink" /> Stop · {secs}s
            </button>
          ) : (
            <button disabled={busy} onClick={startRec} className="flex h-14 items-center gap-3 rounded-full bg-mist px-7 font-semibold text-ink transition hover:bg-white disabled:opacity-40">
              <span className="h-3 w-3 rounded-full bg-synth" /> Record
            </button>
          )}
          <label className={`flex h-14 cursor-pointer items-center rounded-full border border-line px-6 font-medium transition hover:border-dim ${busy ? "pointer-events-none opacity-40" : ""}`}>
            Upload audio
            <input type="file" accept=".wav,.mp3,.m4a,.flac" hidden onChange={(e) => e.target.files[0] && load(e.target.files[0], e.target.files[0].name)} />
          </label>
          <button
            disabled={!clip || busy}
            onClick={analyze}
            className="ml-auto h-14 rounded-full bg-human px-8 font-semibold text-ink transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-30"
          >
            {phase === "analyzing" ? "Analysing…" : "Check this voice"}
          </button>
        </div>
        <p className="mt-4 text-sm text-dim">
          {clip ? `${clip.name} · ${clip.duration.toFixed(1)}s, converted to 16 kHz mono` : `WAV, MP3, M4A or FLAC, 1 to ${MAX_SEC} seconds. Dropping a file here works too.`}
        </p>
        {msg && <p role="alert" className="mt-3 rounded-xl border border-line bg-ink/60 px-4 py-3 text-sm">{msg}</p>}
      </section>

      {result && (
        <section className="mt-6 rounded-3xl border p-6 sm:p-8" style={{ borderColor: color, background: `${color}14` }}>
          <p className="font-display text-3xl font-bold sm:text-4xl" style={{ color }}>
            {verdictFake ? "This sounds AI-generated" : "This sounds like a real person"}
          </p>
          <div className="mt-5 h-3 overflow-hidden rounded-full bg-ink/70" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
            <div className="h-full rounded-full transition-all duration-700" style={{ width: `${pct}%`, background: color }} />
          </div>
          <p className="mt-3 text-dim">Model confidence {pct}%. Treat this as a signal, not proof, especially on noisy phone-call audio.</p>
        </section>
      )}
    </main>
  );
}
