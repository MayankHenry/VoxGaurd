"""Tiny SQLite log of predictions. Stores results only - never the audio."""
import sqlite3
from datetime import datetime, timezone
from .config import DB_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    filename TEXT,
    duration_sec REAL,
    verdict TEXT NOT NULL,
    label TEXT NOT NULL,
    synthetic_probability REAL NOT NULL,
    latency_ms REAL
)
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(_SCHEMA)
    return c


def log_prediction(filename, duration_sec, verdict, label, synthetic_probability, latency_ms) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO predictions (created_at, filename, duration_sec, verdict, label, synthetic_probability, latency_ms) "
            "VALUES (?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), filename, duration_sec, verdict, label,
             synthetic_probability, latency_ms),
        )
        return cur.lastrowid


def recent_predictions(limit: int = 20) -> list[dict]:
    with _conn() as c:
        rows = c.execute("SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def latency_stats() -> dict:
    with _conn() as c:
        rows = [r[0] for r in c.execute("SELECT latency_ms FROM predictions WHERE latency_ms IS NOT NULL")]
    if not rows:
        return {"count": 0}
    rows.sort()
    p95 = rows[min(len(rows) - 1, int(0.95 * len(rows)))]
    return {"count": len(rows), "mean_ms": round(sum(rows) / len(rows), 1), "p95_ms": round(p95, 1)}
