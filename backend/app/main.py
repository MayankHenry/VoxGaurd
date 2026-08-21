"""
VoxGuard backend entrypoint.

Day 1 scope: FastAPI skeleton with a health check route and an audio
upload stub route. Feature extraction and model inference are wired
in on later days once the dataset pipeline (see app/features.py) is
validated.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import uuid
from pathlib import Path

APP_VERSION = "0.1.0-day1"
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}

app = FastAPI(title="VoxGuard API", version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten before deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    """
    Day 1: accept an audio file and store it under data/raw/uploads.
    Returns a sample_id used by later pipeline stages (feature
    extraction, inference) once those are implemented.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    sample_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{sample_id}{ext}"

    with dest.open("wb") as out_file:
        shutil.copyfileobj(file.file, out_file)

    return {
        "sample_id": sample_id,
        "filename": file.filename,
        "stored_path": str(dest),
        "status": "received",
        "note": "Feature extraction + classification not yet wired up (Day 2+).",
    }
