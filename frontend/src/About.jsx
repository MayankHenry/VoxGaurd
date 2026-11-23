const STEPS = [
  ["Capture", "You record in the browser or upload a call recording."],
  ["Normalise", "The clip is converted to 16 kHz mono WAV, the same format the model is trained on."],
  ["Extract features", "The backend measures how the voice sounds, not what it says."],
  ["Classify", "A model trained on real and synthetic speech scores the feature vector."],
  ["Verdict", "You get a label and a confidence value."],
];
const FEATURES = [
  ["MFCCs", "A compact description of timbre. Cloned voices often shape it slightly differently."],
  ["Spectral flatness", "Synthetic speech tends to be flatter and less noisy than a real throat and room."],
  ["Pitch stability", "Real voices wobble naturally. Many TTS voices hold pitch unnaturally smoothly."],
  ["Centroid and bandwidth", "Where the energy sits in the spectrum. High frequencies often differ in cloned audio."],
];

export default function About() {
  return (
    <section className="grid gap-12 lg:grid-cols-2">
      <div>
        <h2 className="font-display text-4xl font-bold tracking-tight">How VoxGuard decides</h2>
        <ol className="mt-6 space-y-4">
          {STEPS.map(([t, d], i) => (
            <li key={t} className="flex gap-4">
              <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full border border-line font-display text-sm font-bold text-human">{i + 1}</span>
              <p><b className="font-semibold">{t}.</b> <span className="text-dim">{d}</span></p>
            </li>
          ))}
        </ol>
      </div>
      <div>
        <h3 className="font-display text-2xl font-bold">What it listens for</h3>
        <dl className="mt-5 space-y-3">
          {FEATURES.map(([t, d]) => (
            <div key={t} className="rounded-2xl border border-line bg-panel/70 px-5 py-4">
              <dt className="font-semibold">{t}</dt>
              <dd className="mt-1 text-dim">{d}</dd>
            </div>
          ))}
        </dl>
        <p className="mt-6 text-sm text-dim">
          The model is trained on a small dataset, so treat a result as a signal rather than proof, especially on noisy phone audio.
        </p>
      </div>
    </section>
  );
}
