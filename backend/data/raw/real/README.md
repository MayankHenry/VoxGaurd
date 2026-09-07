# Real (human) speech samples

This sandbox environment's network is restricted to a small domain
allowlist (GitHub, PyPI, Ubuntu package mirrors) and cannot reach
dataset-hosting sites like Zenodo, OpenSLR, or Kaggle. So the "real"
class has to be sourced on a normal machine (your laptop) and dropped
into this folder.

## Option A — ASVspoof2019 (recommended, matches the proposal)
The standard benchmark dataset for this exact problem (real vs.
spoofed/synthetic speech).

1. Go to https://datashare.ed.ac.uk/handle/10283/3336 (ASVspoof2019)
2. Download the **LA (Logical Access)** partition — it's large, so
   grab just the `train` subset's *bonafide* (real) samples to start
3. Copy a subset (50-100 files is plenty for a first training pass)
   of the bonafide `.flac`/`.wav` files into this `data/raw/real/`
   folder

## Option B — VCTK Corpus (simpler, smaller download)
Multi-speaker English speech corpus, easier to get a manageable
subset from.

1. https://datashare.ed.ac.uk/handle/10283/3443
2. Download a few speakers' folders (each has ~400 short clips)
3. Copy ~50-100 `.wav` files into `data/raw/real/`

## Option C — Record your own (fastest, no download)
If you want to unblock Day 3 (feature extraction on real data)
immediately without waiting on a large download:

1. Each team member records ~15-20 short sentences (5-10 seconds
   each) on your phone or laptop mic — reading the same sentence
   pool used in `scripts/generate_synthetic_samples.py` keeps text
   content comparable across real vs. synthetic
2. Export as `.wav` and drop into this folder
3. This gives ~30-40 real samples quickly; supplement with Option A
   or B later for a more robust dataset before final training

## Naming convention
Keep filenames descriptive, e.g. `real_mayank_001.wav`,
`real_asvspoof_LA_0023.wav` — the manifest builder
(`scripts/build_manifest.py`) doesn't require a specific format, it
just scans this folder and labels everything in it "real".

## Once files are here
Run:
```
python scripts/build_manifest.py
```
This scans both `data/raw/real/` and `data/raw/synthetic/`, extracts
features via `app/features.py`, and writes `data/processed/manifest.csv`
— the input for Day 4-5 model training.
