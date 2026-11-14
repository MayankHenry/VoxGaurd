"""
Evaluate the saved model on audio it has NEVER seen (the honest final test).

    python scripts/evaluate_external.py --real data/holdout/real --fake data/holdout/fake
    python scripts/evaluate_external.py --real my_voice_clips --fake cloned_clips --variants none,codec,noisy

Reports balanced accuracy, recall on each class, ROC-AUC and EER (ASVspoof's headline metric,
lower is better) for every requested channel variant ('codec'/'noisy' simulate a phone call).
Typical use: train on the ASVspoof *train* partition (attacks A01-A06), then point this at the
*eval* partition (attacks A07-A19, unseen) imported with `import_asvspoof.py --role holdout`.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config
from app.augment import VARIANTS, apply_variant
from app.features import FEATURE_SR, features_from_array, load_audio
from app.metrics import binary_metrics

EXTS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def _vectors(path: Path, variants):
    try:
        y = load_audio(str(path))
        return {v: features_from_array(apply_variant(y, v, FEATURE_SR, key=path.name), FEATURE_SR,
                                       source=path.name).to_vector() for v in variants}
    except RuntimeError as e:
        return str(e)


def evaluate_dirs(real_dir, fake_dir, variants=("none",), model_path=None):
    bundle = joblib.load(model_path or config.MODEL_PATH)
    idx = bundle.get("feature_idx")
    files = [(p, "real") for p in sorted(Path(real_dir).iterdir()) if p.suffix.lower() in EXTS]
    files += [(p, "synthetic") for p in sorted(Path(fake_dir).iterdir()) if p.suffix.lower() in EXTS]
    if not files:
        raise SystemExit("No audio files found in the given folders.")
    vecs = Parallel(n_jobs=-1)(delayed(_vectors)(p, variants) for p, _ in files)
    skipped = [(p.name, v) for (p, _), v in zip(files, vecs) if isinstance(v, str)]
    out = {"n_real": sum(1 for (_, l), v in zip(files, vecs) if l == "real" and not isinstance(v, str)),
           "n_fake": sum(1 for (_, l), v in zip(files, vecs) if l == "synthetic" and not isinstance(v, str)),
           "skipped": skipped, "model_version": bundle.get("version"), "by_variant": {}}
    classes = list(bundle["model"].classes_)
    pos = classes.index("synthetic")
    for var in variants:
        y, X = [], []
        for (_, label), v in zip(files, vecs):
            if isinstance(v, str):
                continue
            vec = v[var]
            X.append(vec[idx] if idx is not None else vec)
            y.append(label)
        proba = bundle["model"].predict_proba(bundle["scaler"].transform(np.array(X)))[:, pos]
        out["by_variant"][var] = binary_metrics(np.array(y), proba)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real", required=True, help="folder of real (human) clips")
    ap.add_argument("--fake", required=True, help="folder of fake / AI-generated clips")
    ap.add_argument("--variants", default="none", help=f"comma list from {VARIANTS}")
    ap.add_argument("--out", help="optional path to write the JSON report")
    args = ap.parse_args()
    res = evaluate_dirs(args.real, args.fake, tuple(args.variants.split(",")))
    print(f"model {res['model_version']} | real={res['n_real']} fake={res['n_fake']} skipped={len(res['skipped'])}")
    for var, m in res["by_variant"].items():
        print(f"[{var:6s}] balanced_acc={m['balanced_accuracy']}  recall_real={m['recall_real']}  "
              f"recall_fake={m['recall_synthetic']}  AUC={m.get('roc_auc')}  EER={m.get('eer')}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print("report ->", args.out)


if __name__ == "__main__":
    main()
