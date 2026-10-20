"""Model loading + prediction. Kept separate from the web layer so it is easy to test."""
import json
import time
import joblib
import numpy as np
from . import config
from .features import extract_features, FEATURE_NAMES


class ModelNotReady(RuntimeError):
    """Raised when no trained model artifact exists yet."""


class ModelService:
    def __init__(self, model_path=None, metrics_path=None):
        self.model_path = model_path or config.MODEL_PATH
        self.metrics_path = metrics_path or config.METRICS_PATH
        self._bundle = None

    @property
    def ready(self) -> bool:
        return self._bundle is not None or self.model_path.exists()

    def load(self):
        if self._bundle is None:
            if not self.model_path.exists():
                raise ModelNotReady(
                    f"No trained model at {self.model_path}. Run `python scripts/train_model.py` first."
                )
            self._bundle = joblib.load(self.model_path)
        return self._bundle

    def info(self) -> dict:
        info = {"ready": self.ready, "model_path": str(self.model_path)}
        if self.metrics_path.exists():
            info["metrics"] = json.loads(self.metrics_path.read_text())
        return info

    def predict_file(self, path: str) -> dict:
        b = self.load()
        t0 = time.perf_counter()
        feats = extract_features(path)
        x = feats.to_vector()
        x = (x[b["feature_idx"]] if b.get("feature_idx") is not None else x).reshape(1, -1)
        xs = b["scaler"].transform(x)
        model = b["model"]
        proba = model.predict_proba(xs)[0]
        classes = list(model.classes_)
        probs = {c: float(p) for c, p in zip(classes, proba)}
        p_synth = probs.get("synthetic", 0.0)
        top = int(np.argmax(proba))
        label, confidence = classes[top], float(proba[top])

        if confidence < config.UNCERTAIN_BAND:
            verdict = "uncertain"
        else:
            verdict = "likely_synthetic" if label == "synthetic" else "likely_real"

        return {
            "verdict": verdict,
            "label": label,
            "confidence": round(confidence, 4),
            "synthetic_probability": round(p_synth, 4),
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "top_signals": self._explain(b, xs[0], x[0]),
            "duration_sec": round(feats.duration_sec, 2),
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            "model_version": b.get("version", "unknown"),
        }

    @staticmethod
    def _explain(bundle, z, raw, k=3):
        """Top-k features by (global importance x |z-score|). Plain-language hints for the UI."""
        imp = getattr(bundle["model"], "feature_importances_", None)
        if imp is None:
            return []
        score = imp * np.abs(z)
        names = bundle.get("feature_cols", FEATURE_NAMES)
        out = []
        for i in np.argsort(score)[::-1][:k]:
            out.append({
                "feature": names[i],
                "value": round(float(raw[i]), 4),
                "z_score": round(float(z[i]), 2),
                "direction": "above typical" if z[i] > 0 else "below typical",
            })
        return out


service = ModelService()
