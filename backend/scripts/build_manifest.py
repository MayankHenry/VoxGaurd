"""
Build data/processed/manifest.csv: one row per (audio file x augmentation variant) = metadata + features.

Filename conventions (parsed to get generator + utterance group):
  real_<speaker>_<uttid>.wav           -> label real,      family human
  synth_<family>_<variant>_<uttid>.wav -> label synthetic, family <family>
`group` = utterance id; training uses it so the same sentence never appears in both train and
test folds. `aug` = none | codec | noisy (telephone-channel variants, see app/augment.py);
all variants of a clip share variant/group so they always land in the same fold.

Usage:
  python scripts/build_manifest.py              # clean features only
  python scripts/build_manifest.py --augment    # + G.711/bandpass ("codec") and codec+noise ("noisy") variants
"""
import argparse
import sys
from pathlib import Path
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.augment import VARIANTS, apply_variant
from app.features import FEATURE_NAMES, FEATURE_SR, features_from_array, load_audio

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


def one(path: Path, label: str, variants):
    try:
        y = load_audio(str(path))
    except RuntimeError as e:
        return [], f"{path.name}: {e}"
    family, variant, group = parse(path, label)
    rows = []
    for v in variants:
        try:
            feats = features_from_array(apply_variant(y, v, FEATURE_SR, key=path.name), FEATURE_SR, source=path.name)
        except RuntimeError as e:
            return rows, f"{path.name}[{v}]: {e}"
        row = {"filename": path.name, "label": label, "family": family, "variant": variant,
               "group": group, "aug": v}
        row.update(dict(zip(FEATURE_NAMES, feats.to_vector().tolist())))
        row["duration_sec"] = feats.duration_sec
        rows.append(row)
    return rows, None


def collect(real_dir=None, synth_dir=None):
    real_dir, synth_dir = real_dir or config.REAL_DIR, synth_dir or config.SYNTH_DIR
    jobs = [(p, "real") for p in sorted(real_dir.iterdir()) if p.suffix.lower() in EXTS]
    jobs += [(p, "synthetic") for p in sorted(synth_dir.iterdir()) if p.suffix.lower() in EXTS]
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--augment", action="store_true", help="add codec + noisy telephone-channel variants")
    args = ap.parse_args()
    variants = list(VARIANTS) if args.augment else ["none"]

    jobs = collect()
    n_real = sum(1 for _, l in jobs if l == "real")
    print(f"Found {n_real} real, {len(jobs) - n_real} synthetic files; variants={variants}. Extracting features ...")
    if n_real == 0:
        print("WARNING: no real samples - see scripts/fetch_dev_real_samples.py / data/raw/real/README.md")

    results = Parallel(n_jobs=-1)(delayed(one)(p, l, variants) for p, l in jobs)
    rows = [r for rs, _ in results for r in rs]
    for _, err in results:
        if err:
            print("SKIPPED", err)
    if not rows:
        print("Nothing processed; manifest not written.")
        sys.exit(1)
    df = pd.DataFrame(rows)
    df.to_csv(config.MANIFEST_PATH, index=False)
    print(f"Manifest written: {config.MANIFEST_PATH}  ({len(df)} rows)")
    print(df.groupby(["label", "family", "aug"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
