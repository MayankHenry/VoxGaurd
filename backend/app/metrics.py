"""Shared evaluation metrics (used by training and by evaluate_external)."""
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve


def eer(y_true, p_synth):
    """Equal Error Rate - the standard ASVspoof metric (lower is better)."""
    y = np.asarray(y_true) == "synthetic"
    if y.all() or (~y).all():
        return None
    fpr, tpr, _ = roc_curve(y, p_synth)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return round(float((fpr[i] + fnr[i]) / 2), 4)


def _rate(mask, hit):
    return round(float(np.mean(hit[mask])), 4) if mask.any() else None


def binary_metrics(y_true, p_synth, threshold=0.5):
    y_true = np.asarray(y_true)
    y_pred = np.where(np.asarray(p_synth) >= threshold, "synthetic", "real")
    out = {
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "recall_real": _rate(y_true == "real", y_pred == "real"),
        "recall_synthetic": _rate(y_true == "synthetic", y_pred == "synthetic"),
        "n_test": int(len(y_true)),
    }
    if len(set(y_true)) == 2:
        out["f1_synthetic"] = round(float(f1_score(y_true, y_pred, pos_label="synthetic")), 4)
        out["roc_auc"] = round(float(roc_auc_score(y_true == "synthetic", p_synth)), 4)
        out["eer"] = eer(y_true, p_synth)
    out["confusion_matrix(real,synthetic)"] = confusion_matrix(y_true, y_pred, labels=["real", "synthetic"]).tolist()
    return out
