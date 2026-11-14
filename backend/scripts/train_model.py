"""
Train + honestly evaluate the VoxGuard classifier.

Evaluation protocol (why it looks like this)
  A first version evaluated with grouped CV on (LJSpeech vs TTS) and scored ~100%, but the same
  model called 14/15 UNSEEN real speakers "synthetic": it had learned the speaker/channel, not
  "human vs synthetic". So model selection uses two generalisation axes:

  * LOSO - leave-speakers-out: hold out real speakers (and a disjoint slice of synthetic
           utterances), train on the rest. Score = mean(macro recall_real, recall_synthetic).
  * LOEO - leave-one-engine-out: hold out one TTS/attack family. Score = mean over families of
           0.5*(recall_synthetic + recall_real).
  Selection score = 0.5*LOSO + 0.5*LOEO. Grouped CV is reported for reference only (optimistic).

  Every metric is reported for "all" test rows and split into "clean" and "degraded" (telephone
  codec / noise variants, see app/augment.py). EER (ASVspoof's metric) is included.
  Training uses speaker/engine-balanced sample weights so no single voice or engine dominates.

Safety guard: refuses to train on <2 classes or too few samples/groups.

Usage:
  python scripts/train_model.py
  python scripts/train_model.py --models xgboost,logistic_regression          # faster
  python scripts/train_model.py --train-variants none --no-save               # ablation: clean-only training
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.features import FEATURE_NAMES
from app.metrics import binary_metrics as _metrics
from app.models import XGBStringClassifier

MIN_SAMPLES_PER_CLASS = 5
MIN_GROUPS = 5
N_FOLDS = 5
MAX_SPEAKER_FOLDS = 8
SEED = 42
MIN_SPEAKER_CLIPS = 3   # real "speakers" with fewer base clips are pooled into one fold

FEATURE_SETS = {
    "all": FEATURE_NAMES,
    "no_mfcc_mean": [n for n in FEATURE_NAMES if not n.startswith("mfcc_mean_")],
}

LIMITATIONS = [
    "Real class has few distinct speakers unless ASVspoof2019 bonafide / own recordings are added (scripts/import_asvspoof.py).",
    "Synthetic class is classical TTS (espeak-ng, Flite, Festival) unless neural / ASVspoof attacks are added; performance on modern cloners (ElevenLabs, XTTS, OpenVoice) is NOT measured otherwise.",
    "All audio is band-limited to 8 kHz (telephone band) during feature extraction; 'degraded' rows simulate G.711 mu-law + 300-3400 Hz band-pass (+ noise), not a real telephony network.",
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


def _by_condition(y, p, aug):
    """Metrics for all rows, and separately for clean vs degraded (telephone-channel) rows."""
    out = {"all": _metrics(y, p)}
    clean, deg = aug == "none", aug != "none"
    if clean.any() and deg.any():
        out["clean"], out["degraded"] = _metrics(y[clean], p[clean]), _metrics(y[deg], p[deg])
    return out


def grouped_cv(df, X, y, make_model, train_ok):
    groups, aug = df["group"].values, df["aug"].values
    p = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=min(N_FOLDS, df["group"].nunique())).split(X, y, groups):
        tr = tr[train_ok[tr]]
        p[te] = _fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te])
    return _by_condition(y, p, aug)


def speaker_folds(df):
    """Group real speakers into at most MAX_SPEAKER_FOLDS folds (singletons pooled together)."""
    real = df[df.label == "real"]
    base_counts = real.drop_duplicates("filename")["variant"].value_counts()
    spk = real["variant"].where(real["variant"].map(base_counts) >= MIN_SPEAKER_CLIPS, "other_speakers")
    names = sorted(spk.unique())
    k = min(len(names), MAX_SPEAKER_FOLDS)
    synth_groups = sorted(df[df.label == "synthetic"]["group"].unique())
    folds = {}
    for i in range(k):
        members = names[i::k]
        te_real = real.index[spk.isin(members).values].tolist()
        held = set(synth_groups[i::k])                      # disjoint synthetic slice per fold
        te_syn = df.index[(df.label == "synthetic") & df["group"].isin(held)].tolist()
        te = np.array(sorted(te_real + te_syn))
        tr = np.array(sorted(set(df.index) - set(te)))
        folds["+".join(members) if len(members) <= 2 else f"{members[0]}+{len(members) - 1}more"] = (tr, te)
    return folds


def leave_one_speaker_out(df, X, y, make_model, train_ok):
    aug = df["aug"].values
    per, data = {}, []
    for name, (tr, te) in speaker_folds(df).items():
        tr = tr[train_ok[tr]]
        p = _fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te])
        m = _metrics(y[te], p)
        per[name] = {"recall_real": m["recall_real"], "recall_synthetic": m["recall_synthetic"],
                     "n_real_rows": int((y[te] == "real").sum())}
        data.append((y[te], p, aug[te]))
    Y, P, A = (np.concatenate(z) for z in zip(*data))
    cond = _by_condition(Y, P, A)
    macro = {}
    for cname, mask_fn in (("all", lambda a: np.ones(len(a), bool)), ("clean", lambda a: a == "none"),
                           ("degraded", lambda a: a != "none")):
        vals = []
        for y_f, p_f, a_f in data:
            m = mask_fn(a_f) & (y_f == "real")
            if m.any():
                vals.append(float(np.mean(p_f[m] < 0.5)))
        if vals:
            macro[cname] = round(float(np.mean(vals)), 4)
    score = 0.5 * (macro["all"] + cond["all"]["recall_synthetic"])
    return {"score": round(score, 4), "macro_recall_real": macro, "pooled": cond, "per_fold": per}


def leave_one_engine_out(df, X, y, make_model, train_ok):
    fam, groups, aug = df["family"].values, df["group"].values, df["aug"].values
    families = [f for f in sorted(set(fam)) if f != "human"]
    if len(families) < 2:
        return {"skipped": "need at least two synthetic families"}
    res = {}
    for held in families:
        P, Y, A = [], [], []
        for tr, te in GroupKFold(n_splits=min(N_FOLDS, df["group"].nunique())).split(X, y, groups):
            tr = tr[(fam[tr] != held) & train_ok[tr]]
            te = te[(fam[te] == held) | (fam[te] == "human")]
            if len(te) == 0 or len(set(y[tr])) < 2:
                continue
            P.append(_fit_predict(make_model, X[tr], y[tr], sample_weights(df.iloc[tr]), X[te]))
            Y.append(y[te]); A.append(aug[te])
        if P:
            res[f"unseen_{held}"] = _by_condition(np.concatenate(Y), np.concatenate(P), np.concatenate(A))
    return res


def loeo_score(loeo: dict) -> float:
    vals = [0.5 * (v["all"]["recall_synthetic"] + v["all"]["recall_real"])
            for v in loeo.values() if isinstance(v, dict) and "all" in v]
    return float(np.mean(vals)) if vals else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(candidates()), help="comma-separated subset of candidates")
    ap.add_argument("--train-variants", default="all",
                    help="augmentation variants used for TRAINING (e.g. 'none' = clean only). "
                         "Test folds always contain every variant, so robustness is measured either way.")
    ap.add_argument("--no-save", action="store_true", help="evaluate only; do not overwrite the saved model")
    args = ap.parse_args()
    chosen = {k: v for k, v in candidates().items() if k in args.models.split(",")}

    if not config.MANIFEST_PATH.exists():
        print(f"No manifest at {config.MANIFEST_PATH}. Run scripts/build_manifest.py first.")
        sys.exit(1)
    df = pd.read_csv(config.MANIFEST_PATH).fillna({"variant": ""})
    if "aug" not in df.columns:
        df["aug"] = "none"
    df = df.reset_index(drop=True)
    if not validate_dataset(df):
        sys.exit(1)
    y = df["label"].values
    train_ok = (np.ones(len(df), bool) if args.train_variants == "all"
                else df["aug"].isin(args.train_variants.split(",")).values)
    real_sources = df[df.label == "real"].drop_duplicates("filename")["variant"].value_counts().to_dict()
    print(f"Rows: {len(df)} | labels {df['label'].value_counts().to_dict()} | variants {df['aug'].value_counts().to_dict()}")
    print(f"Real sources (clips): {real_sources} | training on variants: {args.train_variants}\n")

    report, best = {}, (None, None, -1.0)
    for fs_name, cols in FEATURE_SETS.items():
        X = df[cols].values.astype(np.float64)
        for name, mk in chosen.items():
            loso = leave_one_speaker_out(df, X, y, mk, train_ok)
            loeo = leave_one_engine_out(df, X, y, mk, train_ok)
            cv = grouped_cv(df, X, y, mk, train_ok)
            combined = round(0.5 * loso["score"] + 0.5 * loeo_score(loeo), 4)
            report[f"{name}|{fs_name}"] = {"selection_score": combined, "loso": loso, "loeo": loeo,
                                           "grouped_cv_optimistic": cv}
            print(f"{name:20s} {fs_name:13s} score={combined:.3f} | LOSO={loso['score']:.3f} "
                  f"(real={loso['macro_recall_real']['all']:.2f}) LOEO={loeo_score(loeo):.3f} "
                  f"| in-dist CV={cv['all']['balanced_accuracy']:.3f}")
            if combined > best[2]:
                best = (name, fs_name, combined)

    name, fs_name, score = best
    cols = FEATURE_SETS[fs_name]
    X = df[cols].values.astype(np.float64)
    sel = report[f"{name}|{fs_name}"]
    print(f"\nSelected: {name} / {fs_name}  (combined score {score:.3f})")
    for k, v in sel["loso"]["per_fold"].items():
        print(f"   held-out {k:22s} recall_real={v['recall_real']}  (rows={v['n_real_rows']})")
    print(f"   macro recall_real over held-out speakers: {sel['loso']['macro_recall_real']}")
    for k, v in sel["loeo"].items():
        if isinstance(v, dict) and "all" in v:
            line = f"   {k:22s} detected_synth={v['all']['recall_synthetic']} real_ok={v['all']['recall_real']} EER={v['all'].get('eer')}"
            if "degraded" in v:
                line += f" | degraded: detected={v['degraded']['recall_synthetic']} real_ok={v['degraded']['recall_real']}"
            print(line)

    if args.no_save:
        print("\n--no-save: model/report not written.")
        return
    sc = StandardScaler().fit(X[train_ok])
    model = candidates()[name]()
    model.fit(sc.transform(X[train_ok]), y[train_ok], sample_weight=sample_weights(df[train_ok]))
    idx = [FEATURE_NAMES.index(c) for c in cols]
    version = f"{datetime.now(timezone.utc):%Y%m%d}-{name}-{fs_name}-n{len(df)}"
    joblib.dump({"model": model, "scaler": sc, "feature_cols": cols, "feature_idx": idx, "version": version},
                config.MODEL_PATH)
    metrics = {
        "version": version, "trained_at": datetime.now(timezone.utc).isoformat(),
        "selected": {"model": name, "feature_set": fs_name, "selection_score": score},
        "dataset": {"n_rows": int(len(df)), "n_clips": int(df["filename"].nunique()),
                    "by_label": df["label"].value_counts().to_dict(),
                    "by_family": df.drop_duplicates("filename")["family"].value_counts().to_dict(),
                    "by_variant": df["aug"].value_counts().to_dict(), "trained_on_variants": args.train_variants,
                    "real_sources_clips": real_sources},
        "selection_metric": "0.5*LOSO + 0.5*LOEO (see script docstring)",
        "all_candidates": report, "leave_one_engine_out": sel["loeo"], "limitations": LIMITATIONS,
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model  -> {config.MODEL_PATH}\nSaved report -> {config.METRICS_PATH}")


if __name__ == "__main__":
    main()
