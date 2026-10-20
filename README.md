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

### Day 3 — Training Harness + EDA (Pipeline Verified, Not Yet Trained) ✅

**Goal:** Build the model training/evaluation code so it's ready the
moment real-speech samples land — without producing any fake or
misleading accuracy numbers in the meantime.

**Done:**
- `scripts/explore_features.py` — EDA on the manifest: class balance,
  zero-variance feature check, NaN check, saves a feature-distribution
  plot. **Run and verified:** confirmed no NaNs and no dead features
  in the current 10-sample synthetic set; correctly printed a warning
  that class-separation analysis isn't possible with one class only.
- `scripts/train_model.py` — train/test split (stratified), feature
  scaling, RandomForest baseline, full evaluation report (accuracy,
  F1, confusion matrix), model saved via joblib. Includes an explicit
  `validate_dataset()` guard that **refuses to train** if fewer than
  2 classes or fewer than 5 samples/class are present, rather than
  silently returning a meaningless metric.

**Verified today:**
- Ran `train_model.py` against the real (synthetic-only) manifest →
  correctly refused with a clear explanation and exit code 1, instead
  of reporting a fake "100% accuracy."
- Separately exercised the actual training code path (split → scale →
  fit → evaluate → save) against a throwaway fabricated two-class
  dataset, purely to confirm the mechanics work — got a sane ~40%
  accuracy on what's essentially noise, which is exactly what should
  happen and confirms the model isn't trivially overfitting to some
  pipeline artifact. That throwaway model/data was deleted — nothing
  fake is committed to the repo.

**Still blocked on the same thing as Day 2:** real-class samples.
**Action item unchanged:** add ~30-40 real speech samples to
`backend/data/raw/real/` (record or download — see the README there),
then run `build_manifest.py` followed by `train_model.py` for the
first real training run.

### Day 4 — Real-Sample Intake Pipeline (Verified with Simulated Messy Data) ✅

**Context:** team has started recording real speech but hasn't
organized it yet — raw phone/laptop recordings typically come in
inconsistent formats and sample rates, and can include accidental
silent or too-short clips.

**Goal:** build the intake step that turns "pile of raw recordings"
into clean, validated, consistently-formatted files in
`data/raw/real/`, so dropping in messy recordings doesn't quietly
poison the dataset (e.g. the classifier learning "44kHz vs 16kHz" as
a fake signal instead of an actual voice-cloning artifact).

**Done:**
- `scripts/prepare_real_samples.py` — takes a staging folder of raw
  recordings (any of wav/mp3/m4a/flac/ogg), validates each one
  (loads correctly, not silent, 1-30s duration), resamples everything
  to 16kHz mono (matching the synthetic samples' format so sample
  rate can't leak as a spurious signal), and copies accepted files
  into `data/raw/real/` with a systematic `real_<speaker>_NNN.wav`
  naming scheme.

**Verified today (with simulated messy data, since real recordings
weren't uploaded to this environment):**
- Built a deliberately messy test batch: a 44.1kHz recording, an
  8kHz recording, a silent file, and a 0.3s accidental clip
- Ran the intake script → correctly **accepted** both real recordings
  and resampled them to a verified 16kHz, correctly **rejected** the
  silent and too-short files with clear reasons
- Rebuilt the manifest with the 2 accepted samples → correctly
  detected 2 real + 10 synthetic
- Ran `train_model.py` → correctly refused again, this time for a
  different reason ("real: 2, below minimum 5") — confirming the
  Day 3 safety guard catches *both* the single-class case and the
  too-few-samples-per-class case
- Deleted the simulated test recordings afterward — they were
  synthetic stand-ins for testing the script, not real human speech,
  so nothing fake is in the repo

**Action item unchanged, now with a tool ready for it:** record
~30-40 real clips (5-10 sentences each, ideally from more than one
speaker — Mayank + Radhika together gets there faster), drop the raw
files anywhere, then run:
```
python scripts/prepare_real_samples.py --input <your_raw_recordings_folder> --speaker mayank
python scripts/build_manifest.py
python scripts/train_model.py
```
That last command will be the **first real trained model** once
there are 5+ real samples.

### Day 5 — (not started)

## Tech Stack
- Backend: FastAPI, librosa/torchaudio for audio processing
- ML: MFCC/spectral features → CNN on spectrograms or XGBoost on
  extracted features (decision point: Day 4-5, after seeing real data)
- Frontend: React (not started — Day 10+ per proposal timeline)
- Dataset: ASVspoof + self-generated TTS/voice-clone samples