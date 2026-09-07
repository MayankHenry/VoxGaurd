"""
Day 2: Build the labeled feature manifest.

Scans data/raw/real/ and data/raw/synthetic/, runs each file through
the Day 1 feature extraction pipeline (app/features.py), and writes
a flat CSV where each row is one audio sample's feature vector plus
its label. This CSV is the direct input for Day 4-5 model training —
keeping feature extraction and model training decoupled means we can
re-run this any time new samples are dropped into either folder
without touching training code.

Usage:
    python scripts/build_manifest.py
"""
import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.features import extract_features

BACKEND_DIR = Path(__file__).resolve().parent.parent
REAL_DIR = BACKEND_DIR / "data" / "raw" / "real"
SYNTHETIC_DIR = BACKEND_DIR / "data" / "raw" / "synthetic"
OUTPUT_CSV = BACKEND_DIR / "data" / "processed" / "manifest.csv"

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3"}


def collect_files(directory: Path, label: str):
    if not directory.exists():
        return []
    return [(f, label) for f in sorted(directory.iterdir()) if f.suffix.lower() in AUDIO_EXTENSIONS]


def build_manifest():
    real_files = collect_files(REAL_DIR, "real")
    synthetic_files = collect_files(SYNTHETIC_DIR, "synthetic")

    print(f"Found {len(real_files)} real samples, {len(synthetic_files)} synthetic samples.")

    if len(real_files) == 0:
        print(
            "\nWARNING: no real samples found in data/raw/real/. "
            "See data/raw/real/README.md for how to add them. "
            "Proceeding with synthetic-only manifest for now — "
            "this is NOT enough to train a real classifier, only to "
            "validate the pipeline end to end."
        )

    all_files = real_files + synthetic_files
    rows = []
    failed = []

    for file_path, label in all_files:
        try:
            features = extract_features(str(file_path))
            row = {"filename": file_path.name, "label": label}
            row.update({f"mfcc_mean_{i}": v for i, v in enumerate(features.mfcc_mean)})
            row.update({f"mfcc_std_{i}": v for i, v in enumerate(features.mfcc_std)})
            row["spectral_flatness_mean"] = features.spectral_flatness_mean
            row["spectral_centroid_mean"] = features.spectral_centroid_mean
            row["spectral_bandwidth_mean"] = features.spectral_bandwidth_mean
            row["pitch_mean"] = features.pitch_mean
            row["pitch_std"] = features.pitch_std
            row["duration_sec"] = features.duration_sec
            rows.append(row)
        except RuntimeError as e:
            failed.append((file_path.name, str(e)))
            print(f"SKIPPED {file_path.name}: {e}")

    if not rows:
        print("No samples successfully processed. Manifest not written.")
        return

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with OUTPUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nManifest written: {OUTPUT_CSV}")
    print(f"  Total samples: {len(rows)}  (real: {len(real_files) - len([x for x in failed if 'real' in x[0]])}, "
          f"synthetic: {len(synthetic_files) - len([x for x in failed if 'synth' in x[0]])})")
    if failed:
        print(f"  Failed/skipped: {len(failed)}")


if __name__ == "__main__":
    build_manifest()
