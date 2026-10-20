"""Central configuration (override via environment variables / .env)."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BACKEND_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
REAL_DIR = RAW_DIR / "real"
SYNTH_DIR = RAW_DIR / "synthetic"
UPLOAD_DIR = RAW_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_PATH = Path(os.getenv("VOXGUARD_MODEL_PATH", MODEL_DIR / "voxguard_model.joblib"))
METRICS_PATH = Path(os.getenv("VOXGUARD_METRICS_PATH", MODEL_DIR / "model_metrics.json"))
DB_PATH = Path(os.getenv("VOXGUARD_DB_PATH", DATA_DIR / "voxguard.db"))

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
MAX_UPLOAD_BYTES = int(os.getenv("VOXGUARD_MAX_UPLOAD_MB", "10")) * 1024 * 1024
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
UNCERTAIN_BAND = float(os.getenv("VOXGUARD_UNCERTAIN_BAND", "0.65"))  # max-prob below this => "uncertain"
CORS_ORIGINS = os.getenv("VOXGUARD_CORS_ORIGINS", "*").split(",")

for d in (RAW_DIR, REAL_DIR, SYNTH_DIR, UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
