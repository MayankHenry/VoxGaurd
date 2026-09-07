"""
Day 3: Baseline classifier training harness.

This builds the train/evaluate pipeline itself — NOT a trained model
with real accuracy numbers, since the manifest is still synthetic-
only (see Day 2 log). Running this today against single-class data
is deliberate: it proves the harness correctly detects and refuses to
train on an unusable dataset, instead of silently producing a
meaningless "100% accuracy" that would be actively misleading in a
report.

Once real-speech samples are added to data/raw/real/ and the manifest
is rebuilt (scripts/build_manifest.py), this same script will train
for real — no changes needed.

Usage:
    python scripts/train_model.py
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib

BACKEND_DIR = Path(__file__).resolve().parent.parent
MANIFEST_PATH = BACKEND_DIR / "data" / "processed" / "manifest.csv"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_SAMPLES_PER_CLASS = 5  # arbitrary floor for a meaningful test split


def load_manifest():
    if not MANIFEST_PATH.exists():
        print(f"No manifest found at {MANIFEST_PATH}. Run scripts/build_manifest.py first.")
        sys.exit(1)
    return pd.read_csv(MANIFEST_PATH)


def validate_dataset(df: pd.DataFrame):
    """
    Refuse to train on data that can't produce a meaningful result,
    rather than silently returning a misleading metric.
    """
    class_counts = df["label"].value_counts()

    if len(class_counts) < 2:
        print(
            "\nCANNOT TRAIN: only one class present in the manifest "
            f"({class_counts.to_dict()}).\n"
            "This is expected right now — see Day 2/3 log: real-speech "
            "samples haven't been added to data/raw/real/ yet.\n"
            "Add real samples (see data/raw/real/README.md), re-run "
            "scripts/build_manifest.py, then re-run this script.\n"
        )
        return False

    under_threshold = class_counts[class_counts < MIN_SAMPLES_PER_CLASS]
    if not under_threshold.empty:
        print(
            f"\nCANNOT TRAIN RELIABLY: class(es) below the minimum "
            f"{MIN_SAMPLES_PER_CLASS} samples needed for a meaningful "
            f"train/test split: {under_threshold.to_dict()}\n"
            "Add more samples to the underrepresented class before training.\n"
        )
        return False

    return True


def train(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in ("filename", "label")]
    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced")
    clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label="synthetic", average="binary")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)

    print(f"\nTest set size: {len(y_test)}")
    print(f"Accuracy: {acc:.3f}")
    print(f"F1 (synthetic class): {f1:.3f}")
    print(f"Confusion matrix (labels={list(clf.classes_)}):\n{cm}")
    print(f"\n{classification_report(y_test, y_pred)}")

    joblib.dump({"model": clf, "scaler": scaler, "feature_cols": feature_cols}, MODEL_DIR / "baseline_rf.joblib")
    print(f"Model saved: {MODEL_DIR / 'baseline_rf.joblib'}")


if __name__ == "__main__":
    df = load_manifest()
    if validate_dataset(df):
        train(df)
    else:
        sys.exit(1)
