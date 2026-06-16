from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_logistic_pipeline(random_state: int = 42, class_weight="balanced") -> Pipeline:
    """Create the primary calibrated-logistic baseline pipeline used in the manuscript."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(max_iter=2000, random_state=random_state, class_weight=class_weight)),
    ])


def safe_calibrated(model, X, y, method: str = "sigmoid", inner_cv: int = 3):
    """Fit a calibrated classifier when both classes are present; otherwise fit the base model."""
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return model.fit(X, y), False
    try:
        cal = CalibratedClassifierCV(estimator=model, method=method, cv=inner_cv)
        cal.fit(X, y)
        return cal, True
    except Exception:
        return model.fit(X, y), False


def safe_predict_proba(model, X):
    """Return positive-class probabilities, falling back to min-max-scaled decision scores."""
    try:
        p = model.predict_proba(X)
        pos = list(model.classes_).index(1)
        return p[:, pos]
    except Exception:
        s = model.decision_function(X)
        return (s - s.min()) / (s.max() - s.min() + 1e-9)
