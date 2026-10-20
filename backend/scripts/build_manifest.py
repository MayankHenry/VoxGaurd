"""
Build data/processed/manifest.csv: one row per audio file = metadata + feature vector.

Filename conventions (parsed to get generator + utterance group):
  real_<speaker>_<uttid>.wav          -> label real,      family human
  synth_<family>_<variant>_<uttid>.wav -> label synthetic, family <family>
`group` = utterance id; training uses it so the same sentence never appears in both
train and test folds (prevents leakage).

Usage:  python scripts/build_manifest.py
"""
import sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.features import extract_features, FEATURE_NAMES

EXTS = {".wav", ".flac", ".mp3"}


def parse(path: Path, label: str):
    stem = path.stem
    if label == "synthetic" and stem.startswith("synth_"):
        parts = stem.split("_", 3)
        if len(parts) == 4:
            return parts[1], parts[2], parts[3]
    if label == "real" and stem.startswith("real_"):
        parts = stem.split("_", 2)
        if len(parts) == 3:
            return "human", parts[1], parts[2]
    return ("human" if label == "real" else "unknown"), "", stem


def one(path: Path, label: str):
    try:
        feats = extract_features(str(path))
    except RuntimeError as e:
        return None, f"{path.name}: {e}"
    family, variant, group = parse(path, label)
    row = {"filename": path.name, "label": label, "family": family, "variant": variant, "group": group}
    row.update(dict(zip(FEATURE_NAMES, feats.to_vector().tolist())))
    row["duration_sec"] = feats.duration_sec
    return row, None


def main():
    jobs = [(p, "real") for p in sorted(config.REAL_DIR.iterdir()) if p.suffix.lower() in EXTS]
    jobs += [(p, "synthetic") for p in sorted(config.SYNTH_DIR.iterdir()) if p.suffix.lower() in EXTS]
    n_real = sum(1 for _, l in jobs if l == "real")
    print(f"Found {n_real} real, {len(jobs) - n_real} synthetic files. Extracting features ...")
    if n_real == 0:
        print("WARNING: no real samples - see scripts/fetch_dev_real_samples.py / data/raw/real/README.md")

    results = Parallel(n_jobs=-1)(delayed(one)(p, l) for p, l in jobs)
    rows = [r for r, _ in results if r]
    for _, err in results:
        if err:
            print("SKIPPED", err)
    if not rows:
        print("Nothing processed; manifest not written.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    df.to_csv(config.MANIFEST_PATH, index=False)
    print(f"Manifest written: {config.MANIFEST_PATH}")
    print(df.groupby(["label", "family"]).size().to_string())


if __name__ == "__main__":
    main()
