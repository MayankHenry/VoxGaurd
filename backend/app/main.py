"""
VoxGuard API.

Endpoints
  GET  /health                 liveness + whether a trained model is loaded
  GET  /model/info             model version, selection + evaluation report, limitations
  POST /predict                upload audio -> verdict (audio deleted right after analysis)
  POST /upload                 store audio, returns sample_id
  POST /predict/{sample_id}    analyse a previously uploaded sample (404 if unknown)
  GET  /history                recent predictions (results only, never audio)
  GET  /stats                  latency statistics (for the p95 NFR)

Error codes: 400 bad input (type / duration), 404 unknown sample, 413 file too large,
422 undecodable or silent audio, 503 model not trained yet.
"""
import shutil
import tempfile
import uuid
from pathlib import Path

import librosa
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from . import config, db
from .inference import ModelNotReady, service

APP_VERSION = "0.5.0"

app = FastAPI(title="VoxGuard API", version=APP_VERSION,
              description="Detects whether an audio clip is human speech or AI-generated/synthetic.")
app.add_middleware(CORSMiddleware, allow_origins=config.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])


# ---------- helpers ----------
def _save_upload(file: UploadFile, dest_dir: Path, stem: str) -> Path:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in config.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type '{ext}'. Allowed: {sorted(config.ALLOWED_EXTENSIONS)}")
    dest = dest_dir / f"{stem}{ext}"
    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > config.MAX_UPLOAD_BYTES:
                out.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(413, f"File too large (max {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
            out.write(chunk)
    return dest


def _analyse(path: Path, display_name: str) -> dict:
    try:
        duration = librosa.get_duration(path=str(path))
    except Exception:
        raise HTTPException(422, "Could not decode audio file.")
    if duration < config.MIN_DURATION_SEC:
        raise HTTPException(400, f"Audio too short ({duration:.1f}s). Minimum is {config.MIN_DURATION_SEC}s.")
    if duration > config.MAX_DURATION_SEC:
        raise HTTPException(400, f"Audio too long ({duration:.1f}s). Maximum is {config.MAX_DURATION_SEC}s.")
    try:
        result = service.predict_file(str(path))
    except ModelNotReady as e:
        raise HTTPException(503, str(e))
    except RuntimeError as e:
        raise HTTPException(422, str(e))
    result["filename"] = display_name
    result["id"] = db.log_prediction(display_name, result["duration_sec"], result["verdict"], result["label"],
                                     result["synthetic_probability"], result["latency_ms"])
    return result


# ---------- routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION, "model_ready": service.ready}


@app.get("/model/info")
def model_info():
    return service.info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """One-shot analysis. The uploaded audio is deleted immediately after analysis."""
    with tempfile.TemporaryDirectory() as td:
        path = _save_upload(file, Path(td), "clip")
        return _analyse(path, file.filename or path.name)


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    sample_id = str(uuid.uuid4())
    dest = _save_upload(file, config.UPLOAD_DIR, sample_id)
    return {"sample_id": sample_id, "filename": file.filename, "status": "received", "stored_as": dest.name}


@app.post("/predict/{sample_id}")
def predict_stored(sample_id: str):
    try:
        uuid.UUID(sample_id)  # also blocks path traversal
    except ValueError:
        raise HTTPException(404, "Unknown sample_id.")
    matches = list(config.UPLOAD_DIR.glob(f"{sample_id}.*"))
    if not matches:
        raise HTTPException(404, "Unknown sample_id.")
    return _analyse(matches[0], matches[0].name)


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)):
    return {"items": db.recent_predictions(limit)}


@app.get("/stats")
def stats():
    return db.latency_stats()
