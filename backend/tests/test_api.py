import joblib
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from app import config, db, main
from app.features import FEATURE_NAMES
from app.inference import ModelService


@pytest.fixture
def client(tmp_path, monkeypatch):
    """App wired to a throw-away TEST model (random data) - tests API behaviour, not accuracy."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, len(FEATURE_NAMES)))
    y = np.array(["real"] * 30 + ["synthetic"] * 30)
    sc = StandardScaler().fit(X)
    rf = RandomForestClassifier(n_estimators=20, random_state=0).fit(sc.transform(X), y)
    mp = tmp_path / "m.joblib"
    joblib.dump({"model": rf, "scaler": sc, "feature_cols": FEATURE_NAMES,
                 "feature_idx": list(range(len(FEATURE_NAMES))), "version": "test"}, mp)
    monkeypatch.setattr(main, "service", ModelService(mp, tmp_path / "none.json"))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    monkeypatch.setattr(config, "UPLOAD_DIR", tmp_path / "uploads")
    (tmp_path / "uploads").mkdir()
    return TestClient(main.app)


def _wav(path, dur=2.0, silent=False):
    t = np.linspace(0, dur, int(16000 * dur), endpoint=False)
    sig = np.zeros_like(t) if silent else 0.3 * np.sin(2 * np.pi * 150 * t) + 0.01 * np.random.randn(len(t))
    sf.write(path, sig, 16000)
    return path


def _post(client, path, url="/predict"):
    with open(path, "rb") as f:
        return client.post(url, files={"file": (path.name, f, "audio/wav")})


def test_health_reports_model_ready(client):
    r = client.get("/health").json()
    assert r["status"] == "ok" and r["model_ready"] is True


def test_predict_happy_path(client, tmp_path):
    r = _post(client, _wav(tmp_path / "a.wav"))
    assert r.status_code == 200
    d = r.json()
    assert d["verdict"] in {"likely_real", "likely_synthetic", "uncertain"}
    assert abs(sum(d["probabilities"].values()) - 1) < 1e-3
    assert len(d["top_signals"]) == 3 and d["latency_ms"] > 0


def test_predict_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "service", ModelService(tmp_path / "missing.joblib", tmp_path / "x.json"))
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    c = TestClient(main.app)
    assert c.get("/health").json()["model_ready"] is False
    assert _post(c, _wav(tmp_path / "a.wav")).status_code == 503


def test_rejects_bad_extension(client, tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("hello")
    assert _post(client, p).status_code == 400


def test_rejects_too_short(client, tmp_path):
    assert _post(client, _wav(tmp_path / "s.wav", dur=0.3)).status_code == 400


def test_rejects_too_long(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_DURATION_SEC", 1.5)
    assert _post(client, _wav(tmp_path / "l.wav", dur=2.0)).status_code == 400


def test_silent_audio_422(client, tmp_path):
    assert _post(client, _wav(tmp_path / "z.wav", silent=True)).status_code == 422


def test_oversize_413(client, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_UPLOAD_BYTES", 1000)
    assert _post(client, _wav(tmp_path / "big.wav", dur=2.0)).status_code == 413


def test_upload_then_predict_by_id(client, tmp_path):
    up = _post(client, _wav(tmp_path / "u.wav"), "/upload").json()
    assert client.post(f"/predict/{up['sample_id']}").status_code == 200


def test_unknown_and_malicious_sample_ids_404(client):
    assert client.post("/predict/00000000-0000-0000-0000-000000000000").status_code == 404
    assert client.post("/predict/..%2F..%2Fetc%2Fpasswd").status_code == 404


def test_history_and_stats_record_predictions(client, tmp_path):
    _post(client, _wav(tmp_path / "h.wav"))
    items = client.get("/history").json()["items"]
    assert len(items) == 1 and "filename" in items[0]
    assert client.get("/stats").json()["count"] == 1


def test_predict_does_not_retain_audio(client, tmp_path):
    before = list(config.UPLOAD_DIR.iterdir())
    _post(client, _wav(tmp_path / "p.wav"))
    assert list(config.UPLOAD_DIR.iterdir()) == before
