import { useEffect, useState } from "react";
import { API } from "./api.js";
import Detect from "./Detect.jsx";
import History from "./History.jsx";
import About from "./About.jsx";

const TABS = [["detect", "Detect"], ["history", "History"], ["about", "How it works"]];
const KEY = "voxguard-history";

const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };

export default function App() {
  const [tab, setTab] = useState("detect");
  const [online, setOnline] = useState(null);
  const [items, setItems] = useState(load);

  useEffect(() => {
    fetch(`${API}/health`).then((r) => setOnline(r.ok)).catch(() => setOnline(false));
  }, []);
  useEffect(() => { try { localStorage.setItem(KEY, JSON.stringify(items)); } catch { /* storage unavailable */ } }, [items]);

  // upsert by id, newest first
  const save = (e) => setItems((p) => [e, ...p.filter((x) => x.id !== e.id)].slice(0, 50));

  return (
    <div className="mx-auto max-w-5xl px-5 py-6 sm:py-10">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="font-display text-2xl font-bold tracking-tight">VoxGuard</h1>
        <nav className="flex gap-1 rounded-full border border-line bg-panel/60 p-1" aria-label="Sections">
          {TABS.map(([id, label]) => (
            <button key={id} onClick={() => setTab(id)} aria-current={tab === id ? "page" : undefined}
              className={`rounded-full px-4 py-2 text-sm font-medium transition ${tab === id ? "bg-mist text-ink" : "text-dim hover:text-mist"}`}>
              {label}{id === "history" && items.length > 0 ? ` (${items.length})` : ""}
            </button>
          ))}
        </nav>
        <span className="flex items-center gap-2 text-sm text-dim" role="status">
          <i className={`h-2.5 w-2.5 rounded-full ${online ? "bg-human" : online === false ? "bg-synth" : "bg-dim"}`} />
          {online ? "API connected" : online === false ? "API offline" : "Checking API"}
        </span>
      </header>

      <main className="mt-12">
        {tab === "detect" && <Detect onSaved={save} />}
        {tab === "history" && <History items={items} onClear={() => setItems([])} onDetect={() => setTab("detect")} />}
        {tab === "about" && <About />}
      </main>

      <footer className="mt-16 border-t border-line pt-6 text-sm text-dim">
        VoxGuard · mini-project, GLA University. Results are a signal, not proof.
      </footer>
    </div>
  );
}
