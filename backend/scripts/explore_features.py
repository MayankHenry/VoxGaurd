"""
Day 3: Exploratory data analysis on the feature manifest.

Before writing any training code, sanity-check what's actually in
data/processed/manifest.csv — class balance, feature ranges, any
degenerate (zero-variance) features that would be useless to a
classifier. This matters more than usual here because the manifest
is currently synthetic-only (see Day 2 log) and it's easy to draw
false conclusions from single-class data if we're not careful.

Usage:
    python scripts/explore_features.py
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless — no display in this environment
import matplotlib.pyplot as plt

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "data" / "processed" / "manifest.csv"
OUTPUT_DIR = BACKEND_DIR / "data" / "processed"


def main():
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run scripts/build_manifest.py first.")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    print(f"Loaded manifest: {len(df)} samples, {len(df.columns)} columns\n")

    print("Class balance:")
    counts = df["label"].value_counts()
    print(counts.to_string())
    if counts.shape[0] < 2:
        print(
            "\n*** Only one class present. This is expected per the Day 2 log — "
            "no real-speech samples have been added to data/raw/real/ yet. "
            "EDA below still runs on the synthetic class alone to check for "
            "feature pipeline issues (NaNs, zero-variance columns), but class-"
            "separation analysis is not possible until real samples are added. ***\n"
        )

    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    stats = df[feature_cols].describe().T[["mean", "std", "min", "max"]]

    zero_variance = stats[stats["std"] < 1e-8]
    if not zero_variance.empty:
        print(f"WARNING: {len(zero_variance)} zero-variance feature(s) found (uninformative for a classifier):")
        print(zero_variance.index.tolist())
    else:
        print("No zero-variance features — all extracted features show variation across samples.")

    nan_counts = df[feature_cols].isna().sum()
    total_nans = nan_counts.sum()
    if total_nans > 0:
        print(f"\nWARNING: {total_nans} NaN values found across features:")
        print(nan_counts[nan_counts > 0].to_string())
    else:
        print("No NaNs in any feature column.")

    # Save a quick distribution plot for a few key features, per class if possible
    key_features = ["pitch_mean", "pitch_std", "spectral_flatness_mean", "spectral_centroid_mean"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feat in zip(axes.flat, key_features):
        for label, group in df.groupby("label"):
            ax.hist(group[feat], alpha=0.6, label=label, bins=10)
        ax.set_title(feat)
        ax.legend()
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "feature_distributions.png"
    fig.savefig(plot_path)
    print(f"\nSaved feature distribution plot: {plot_path}")


if __name__ == "__main__":
    main()
