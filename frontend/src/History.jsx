const fake = (v) => /fake|synth|spoof|ai/i.test(v.label);

export default function History({ items, onClear, onDetect }) {
  if (!items.length)
    return (
      <section className="max-w-xl">
        <h2 className="font-display text-4xl font-bold tracking-tight">Nothing checked yet</h2>
        <p className="mt-4 text-lg text-dim">Clips you upload will show up here. They are stored only in this browser.</p>
        <button onClick={onDetect} className="mt-6 h-12 rounded-full bg-mist px-6 font-semibold text-ink hover:bg-white">Check a voice</button>
      </section>
    );

  return (
    <section>
      <div className="flex items-end justify-between gap-4">
        <h2 className="font-display text-4xl font-bold tracking-tight">History</h2>
        <button onClick={onClear} className="text-sm text-dim underline underline-offset-4 hover:text-mist">Clear list</button>
      </div>
      <ul className="mt-6 divide-y divide-line overflow-hidden rounded-3xl border border-line bg-panel/70">
        {items.map((it) => (
          <li key={it.id} className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
            <div className="min-w-0">
              <p className="truncate font-medium">{it.name}</p>
              <p className="text-sm text-dim">{new Date(it.at).toLocaleString()} · {it.duration.toFixed(1)}s</p>
            </div>
            {it.verdict ? (
              <span className="rounded-full px-4 py-1.5 text-sm font-semibold text-ink" style={{ background: fake(it.verdict) ? "#ff5d8f" : "#5eead4" }}>
                {fake(it.verdict) ? "AI-generated" : "Real"} · {Math.round(it.verdict.confidence * 100)}%
              </span>
            ) : (
              <span className="rounded-full border border-line px-4 py-1.5 text-sm text-dim">Uploaded, no verdict yet</span>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
