# VoxGuard backend

FastAPI service + ML pipeline that classifies an audio clip as human or AI-generated speech.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/run_pipeline.py --train-only            # trains from the committed data/processed/manifest.csv (~1-2 min)
uvicorn app.main:app --reload                          # http://localhost:8000/docs
python -m pytest -q                                    # 29 tests
```
Everything in one container: `docker compose up --build` from the repo root.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /health | liveness + `model_ready` |
| GET | /model/info | model version, metrics, limitations |
| POST | /predict | analyse an uploaded clip (audio deleted right after) |
| POST | /upload, /predict/{sample_id} | two-step flow used by the frontend; stored clip deleted after analysis, stale ones purged after 1 h |
| GET | /history, /stats | recent results (never audio), latency statistics |

Errors: 400 bad type/duration, 404 unknown sample, 413 too large, 422 undecodable/silent, 503 no model.

## Pipeline
`fetch_dev_real_samples.py` / `fetch_extra_real_speakers.py` -> `generate_synthetic_samples.py` ->
`build_manifest.py --augment` -> `train_model.py` (grouped CV, LOSO, LOEO; speaker/engine-balanced weights;
refuses to train on unusable data) -> `models/voxguard_model.joblib` + `model_metrics.json`.
More data: `import_asvspoof.py`, `generate_neural_samples.py`; final test: `evaluate_external.py`.
Results and caveats: see `EXPERIMENTS.md` and the `limitations` list inside `model_metrics.json`.

## Design notes
- Audio is read at 8 kHz (telephone band) and peak-normalised so sample rate / loudness cannot leak the label.
- Same telephone degradation (band-pass 300-3400 Hz, G.711 mu-law, noise) is applied to both classes.
- Verdict `uncertain` when max probability < 0.65 (`VOXGUARD_UNCERTAIN_BAND`).
- Env vars: `VOXGUARD_CORS_ORIGINS`, `VOXGUARD_MAX_UPLOAD_MB`, `VOXGUARD_DB_PATH`, `VOXGUARD_UPLOAD_TTL_SEC`.
