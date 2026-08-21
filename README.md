# VoxGuard — AI-Generated Voice / Deepfake Detector for Audio Calls

Mini-project (Team: Mayank + Radhika Gupta, 3rd Year B.Tech CSE AI/ML, GLA University).

Detects whether an audio clip is authentic human speech or AI-generated/
voice-cloned speech, using spectral and prosodic features rather than
treating it as a repurposed video-deepfake problem.

## Build Log

### Day 1 — Project Scaffolding + Feature Extraction Pipeline ✅

**Goal:** Get the foundational audio feature-extraction pipeline working
and tested *before* touching any model or dataset code, so later days
build on something verified.

**Done:**
- Repo structure set up (`backend/app`, `backend/data`, `backend/models`,
  `backend/tests`, `frontend`)
- FastAPI skeleton (`app/main.py`) with:
  - `GET /health` — verified working
  - `POST /upload` — accepts audio file, validates extension
    (`.wav/.mp3/.m4a/.flac`), stores under `data/raw/uploads/`,
    returns a `sample_id` — verified working via FastAPI TestClient
- Feature extraction module (`app/features.py`) extracting:
  - MFCCs (13 coefficients, mean + std)
  - Spectral flatness, centroid, bandwidth
  - Pitch (f0) mean + std via `librosa.pyin`
  - Packaged into a flat feature vector for classifier input (Day 4+)
- Unit tests (`tests/test_features.py`) using a synthetically generated
  test tone (since real ASVspoof samples aren't downloaded yet — that's
  Day 2). Tests confirm:
  - Extraction runs end-to-end without errors
  - No NaNs in output
  - Correctly raises on silent/empty audio
  - Feature vector shape is exactly as expected (31-dim)
- `requirements.txt` pinned for reproducibility

**Verified today (not just written — actually run):**
- `python3 tests/test_features.py` → both tests pass
- FastAPI `/health` and `/upload` routes tested via TestClient → both return 200

**Known limitation carried into Day 2:** no real dataset yet — today's
tests use a synthetic sine-wave tone purely to validate the *pipeline*,
not model accuracy. Day 2 is dataset acquisition (ASVspoof subset +
generating our own TTS samples) so we have real real-vs-synthetic pairs
to extract features from.

### Day 2 — (not started)
Planned: acquire ASVspoof dataset subset, generate synthetic voice
samples with 1-2 open TTS/voice-cloning tools, build the labeled
dataset folder structure (`data/raw/real/`, `data/raw/synthetic/`).

## Tech Stack
- Backend: FastAPI, librosa/torchaudio for audio processing
- ML: MFCC/spectral features → CNN on spectrograms or XGBoost on
  extracted features (decision point: Day 4-5, after seeing real data)
- Frontend: React (not started — Day 10+ per proposal timeline)
- Dataset: ASVspoof + self-generated TTS/voice-clone samples
