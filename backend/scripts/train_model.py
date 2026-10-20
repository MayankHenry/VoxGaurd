"""
Train + honestly evaluate the VoxGuard classifier.

Why the protocol looks like this
  A first version evaluated with grouped CV on (LJSpeech vs TTS) and scored ~100%, but the
  same model called 14/15 UNSEEN real speakers "synthetic": it had learned the speaker/channel,
  not "human vs synthetic". So model selection now uses LEAVE-ONE-SPEAKER-OUT (LOSO):

  * LOSO  - hold out each real speaker (and a disjoint slice of synthetic utterances),
            train on the rest, test on the held-out speaker. Score = mean(recall_real per
            speaker, recall_synthetic). THIS drives model/feature selection.
  * LOEO  - leave-one-engine-out over the TTS families (generalisation to unseen generators).
            Score = mean over held-out engines of 0.5*(recall_synthetic + recall_real).
  Selection score = 0.5*LOSO + 0.5*LOEO, so a model must generalise on BOTH axes.
  * grouped CV - reported for reference only; optimistic (same speakers in train and test).

  Training uses speaker/engine-balanced sample weights so one voice (LJSpeech) can't dominate.

Safety guard: refuses to train on <2 classes or too few samples/groups.

Usage:  python scripts/train_model.py
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.features import FEATURE_NAMES
from app.models import XGBStringClassifier

MIN_SAMPLES_PER_CLASS = 5
MIN_GROUPS = 5
N_FOLDS = 5
SEED = 42
MIN_SPEAKER_CLIPS = 3   # real "speakers" with fewer clips are pooled into one LOSO fold

FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "no_mfcc_mean": [n for n in FEATURE_NAMES if not n.startswith("mfcc_mean_")],  # drops channel-heavy cepstral means
}

LIMITATIONS = [
    "Real class is only ~5 distinct speakers (LJSpeech + a few test-sample voices); add ASVspoof2019 bonafide (many speakers) and your own recordings before claiming real-world accuracy.",
    "Synthetic class comes from classical TTS engines (espeak-ng, Flite, Festival), not neural voice cloning; performance on modern cloners (ElevenLabs, XTTS, OpenVoice) is NOT measured here.",
    "All audio is band-limited to 8 kHz (telephone band) during feature extraction.",
    "Text content is matched to the real clips for only 8/32 LJSpeech utterances (others use similar-style sentences).",
    "Small dataset: treat metrics as pipeline validation, not a benchmark.",
]


def candidates():
    return {
        "random_forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1),
        "xgboost": lambda: XGBStringClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.9,
                                               eval_metric="logloss", random_state=SEED, n_jobs=1),
        "logistic_regression": lambda: LogisticRegression(max_iter=3000, C=0.5),
    }


def validate_dataset(df: pd.DataFrame) -> bool:
    counts = df["label"].value_counts()
    if len(counts) < 2:
        print(f"\nCANNOT TRAIN: only one class present ({counts.to_dict()}). "
              "Add real samples (scripts/fetch_dev_real_samples.py or data/raw/real/README.md), "
              "re-run build_manifest.py.\n")
        return False
    small = counts[counts < MIN_SAMPLES_PER_CLASS]
    if not small.empty:
        print(f"\nCANNOT TRAIN RELIABLY: classes below {MIN_SAMPLES_PER_CLASS} samples: {small.to_dict()}\n")
        return False
    if df["group"].nunique() < MIN_GROUPS:
        print(f"\nCANNOT TRAIN RELIABLY: need at least {MIN_GROUPS} distinct utterance groups.\n")
        return False
    return True


def sample_weights(sub: pd.DataFrame) -> np.ndarray:
    """Each class gets equal total weight; inside a class every speaker (real) / engine family
    (synthetic) gets equal total weight, so no single voice or engine dominates."""
    w = np.zeros(len(sub))
    lab, key = sub["label"].values, np.where(sub["label"] == "real", sub["variant"], sub["family"])
    classes = [c for c in ("real", "synthetic") if (lab == c).any()]
    for c in classes:
        idx = np.where(lab == c)[0]
        keys, counts = np.unique(key[idx], return_counts=True)
        per_key = dict(zip(keys, counts))
        for i in idx:
            w[i] = 1.0 / per_key[key[i]] / len(keys)
        w[idx] *= 1.0 / len(classes) / w[idx].sum()
    return w * len(sub)


def _fit_predict(make_model, Xtr, ytr, wtr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = make_model()
    m.fit(sc.transform(Xtr), ytr, sample_weight=wtr)
    proba = m.predict_proba(sc.transform(Xte))
    return proba[:, list(m.classes_).index("synthetic")]


def _rate(mask_true, pred_is_true):
    return round(float(np.mean(pred_is_true[mask_true])), 4) if mask_true.any() else None


def _metrics(y_true, p_synth):
    y_pred = np.where(p_synth >= 0.5, "synthetic", "real")
    out = {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "recall_real": _rate(y_true == "real", y_pred == "real"),
        "recall_synthetic": _rate(y_true == "synthetic", y_pred == "synthetic"),
        "n_test": int(len(y_true)),
    }
    if len(set(y_true)) == 2:
        out["f1_synthetic"] = round(float(f1_score(y_true, y_pred, pos_label="synthetic")), 4)
        out["roc_auc"] = round(float(roc_auc_score(y_true == "synthetic", p_synth)), 4)
    out["confusion_matrix(real,synthetic)"] = confusion_matrix(y_true, y_pred, labels=["real", "synthetic"]).tolist()
    return out


def grouped_cv(df, X, y, make_model):
    groups = df["group"].values
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, df["group"].nunique())).split(X, y, groups):
        p[te] = _fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te])
    return _metrics(y, p)


def speaker_folds(df):
    real = df[df.label == "real"]
    counts = real["variant"].value_counts()
    spk = np.where(real["variant"].map(counts) >= MIN_SPEAKER_CLIPS, real["variant"], "other_speakers")
    mapping = dict(zip(real.index, spk))
    names = sorted(set(spk))
    synth_groups = sorted(df[df.label == "synthetic"]["group"].unique())
    folds = {}
    for i, name in enumerate(names):
        held_groups = set(synth_groups[i::len(names)])           # disjoint synthetic slice per fold
        te_real = [ix for ix, s in mapping.items() if s == name]
        te_syn = df.index[(df.label == "synthetic") & df["group"].isin(held_groups)].tolist()
        te = np.array(sorted(te_real + te_syn))
        tr = np.array(sorted(set(df.index) - set(te_real) - set(te_syn)))
        folds[name] = (tr, te)
    return folds


def leave_one_speaker_out(df, X, y, make_model):
    per, pooled_p, pooled_y = {}, [], []
    for name, (tr, te) in speaker_folds(df).items():
        p = _fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te])
        m = _metrics(y[te], p)
        per[name] = {"recall_real": m["recall_real"], "recall_synthetic": m["recall_synthetic"],
                     "n_real": int((y[te] == "real").sum())}
        pooled_p.append(p); pooled_y.append(y[te])
    rr = [v["recall_real"] for v in per.values() if v["recall_real"] is not None]
    pooled = _metrics(np.concatenate(pooled_y), np.concatenate(pooled_p))
    macro_real = float(np.mean(rr))
    score = 0.5 * (macro_real + pooled["recall_synthetic"])
    return {"score": round(score, 4), "macro_recall_real": round(macro_real, 4),
            "recall_synthetic": pooled["recall_synthetic"], "per_speaker": per}


def leave_one_engine_out(df, X, y, make_model):
    fam, groups = df["family"].values, df["group"].values
    families = [f for f in sorted(set(fam)) if f != "human"]
    if len(families) < 2:
        return {"skipped": "need at least two synthetic families"}
    res = {}
    for held in families:
        P, Y = [], []
        for tr, te in GroupKFold(n_splits=min(N_FOLDS, df["group"].nunique())).split(X, y, groups):
            tr = tr[fam[tr] != held]
            te = te[(fam[te] == held) | (fam[te] == "human")]
            if len(te) == 0 or len(set(y[tr])) < 2:
                continue
            P.append(_fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te])); Y.append(y[te])
        if P:
            res[f"unseen_{held}"] = _metrics(np.concatenate(Y), np.concatenate(P))
    return res


def loeo_score(loeo: dict) -> float:
    vals = [0.5 * (v["recall_synthetic"] + v["recall_real"]) for v in loeo.values() if "recall_synthetic" in v]
    return float(np.mean(vals)) if vals else 0.0


def main():
    if not config.MANIFEST_PATH.exists():
        print(f"No manifest at {config.MANIFEST_PATH}. Run scripts/build_manifest.py first.")
        sys.exit(1)
    df = pd.read_csv(config.MANIFEST_PATH).fillna({"variant": ""}).reset_index(drop=True)
    if not validate_dataset(df):
        sys.exit(1)
    y = df["label"].values
    print(f"Dataset: {df['label'].value_counts().to_dict()} | real sources: "
          f"{df[df.label=='real']['variant'].value_counts().to_dict()}\n")

    report, best = {}, (None, None, -1.0)
    for fs_name, cols in FEATURE_SETS.items():
        X = df[cols].values.astype(np.float64)
        for name, mk in candidates().items():
            loso = leave_one_speaker_out(df, X, y, mk)
            loeo = leave_one_engine_out(df, X, y, mk)
            cv = grouped_cv(df, X, y, mk)
            combined = round(0.5 * loso["score"] + 0.5 * loeo_score(loeo), 4)
            report[f"{name}|{fs_name}"] = {"selection_score": combined, "loso": loso,
                                           "loeo": loeo, "grouped_cv_optimistic": cv}
            print(f"{name:20s} {fs_name:13s} score={combined:.3f} | LOSO={loso['score']:.3f} "
                  f"(real={loso['macro_recall_real']:.2f}) LOEO={loeo_score(loeo):.3f} "
                  f"| in-dist CV={cv['balanced_accuracy']:.3f}")
            if combined > best[2]:
                best = (name, fs_name, combined)

    name, fs_name, score = best
    cols = FEATURE_SETS[fs_name]
    X = df[cols].values.astype(np.float64)
    print(f"\nSelected: {name} / {fs_name}  (combined score {score:.3f})")
    for k, v in report[f"{name}|{fs_name}"]["loso"]["per_speaker"].items():
        print(f"   held-out speaker {k:15s} recall_real={v['recall_real']}  (n_real={v['n_real']})")
    loeo = report[f"{name}|{fs_name}"]["loeo"]
    for k, v in loeo.items():
        if "recall_synthetic" in v:
            print(f"   {k:18s} detected_synth={v['recall_synthetic']}  real_ok={v['recall_real']}")

    sc = StandardScaler().fit(X)
    model = candidates()[name]()
    model.fit(sc.transform(X), y, sample_weight=sample_weights(df))
    idx = [FEATURE_NAMES.index(c) for c in cols]
    version = f"{datetime.now(timezone.utc):%Y%m%d}-{name}-{fs_name}-n{len(df)}"
    joblib.dump({"model": model, "scaler": sc, "feature_cols": cols, "feature_idx": idx, "version": version},
                config.MODEL_PATH)

    metrics = {
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected": {"model": name, "feature_set": fs_name, "selection_score": score},
        "dataset": {"n_samples": int(len(df)), "by_label": df["label"].value_counts().to_dict(),
                    "by_family": df["family"].value_counts().to_dict(),
                    "real_sources": df[df.label == "real"]["variant"].value_counts().to_dict()},
        "selection_metric": "0.5*LOSO + 0.5*LOEO (see script docstring)",
        "all_candidates": report, "leave_one_engine_out": loeo, "limitations": LIMITATIONS,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model  -> {config.MODEL_PATH}\nSaved report -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
