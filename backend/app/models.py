"""Small estimator wrappers (must live in an importable module so joblib can unpickle them)."""
import numpy as np
from xgboost import XGBClassifier


class XGBStringClassifier:
    """Composition wrapper: XGBoost that accepts string labels such as 'real'/'synthetic'."""

    def __init__(self, **params):
        self.params = params
        self._clf = XGBClassifier(**params)
        self.classes_ = None

    def fit(self, X, y, sample_weight=None):
        self.classes_ = np.array(sorted(set(y)))
        self._clf.fit(X, np.searchsorted(self.classes_, y), sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        return self._clf.predict_proba(X)

    def predict(self, X):
        return self.classes_[self._clf.predict(X).astype(int)]

    @property
    def feature_importances_(self):
        return self._clf.feature_importances_
