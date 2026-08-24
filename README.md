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

### Day 2 — Dataset Pipeline (Synthetic Class + Real-Class Sourcing Plan) ✅

**Goal:** Get labeled audio data flowing through the Day 1 feature
extractor into a manifest that Day 4-5 model training can consume.

**Constraint hit and handled:** this sandbox's network is restricted
to a small domain allowlist (GitHub, PyPI, Ubuntu mirrors) — it
cannot reach ASVspoof/VCTK/Kaggle hosts directly. Rather than fake
this, the synthetic side was built fully offline and the real side
was made an explicit, documented next step for a machine with normal
internet access.

**Done:**
- Installed `espeak-ng` (offline TTS engine, via apt — no external
  dataset host needed)
- `scripts/generate_synthetic_samples.py` — generates AI-voice
  samples with varied voice/pitch/speaking-rate combinations across
  10 different sentences (scam-call-style phrasing, matching the
  project's real-world framing). **Run and verified: 10 synthetic
  `.wav` files generated successfully.**
- `data/raw/real/README.md` — concrete instructions for sourcing the
  real-speech class: ASVspoof2019 or VCTK download links, or
  recording team members' own voices as a fast unblock option
- `scripts/build_manifest.py` — scans `data/raw/real/` +
  `data/raw/synthetic/`, runs each file through Day 1's
  `extract_features()`, writes `data/processed/manifest.csv`.
  **Run and verified:** correctly detected 0 real / 10 synthetic
  samples, printed a clear warning about the missing real class, and
  still wrote a valid 10-row manifest with real feature values (not
  NaNs) — confirming Day 1 → Day 2 integration works end to end.

**Known limitation carried into Day 3:** manifest currently has
synthetic-only data (0 real samples). **Action item for the team:**
before Day 4 (model training), at least one of you needs to either
record ~30-40 short voice samples, or download a small ASVspoof/VCTK
subset on a personal machine and drop the files into
`backend/data/raw/real/` — see the README there for exact steps.
Re-run `python scripts/build_manifest.py` afterward to regenerate the
manifest with both classes.

### Day 3 — (not started)

## Tech Stack
- Backend: FastAPI, librosa/torchaudio for audio processing
- ML: MFCC/spectral features → CNN on spectrograms or XGBoost on
  extracted features (decision point: Day 4-5, after seeing real data)
- Frontend: React (not started — Day 10+ per proposal timeline)
- Dataset: ASVspoof + self-generated TTS/voice-clone samples
