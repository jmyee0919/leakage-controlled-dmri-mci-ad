from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def calibration_slope_intercept(y_true, y_prob, eps: float = 1e-6):
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    y = np.asarray(y_true, dtype=float)
    logit = np.log(p / (1 - p))
    X = np.vstack([np.ones_like(logit), logit]).T
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(beta[1]), float(beta[0])


def metrics_all(y_true, y_prob):
    y = np.asarray(y_true)
    p = np.asarray(y_prob, dtype=float)
    ok = ~np.isnan(p)
    y = y[ok]
    p = p[ok]
    if len(np.unique(y)) < 2:
        return {"roc_auc": np.nan, "pr_auc": np.nan, "brier": np.nan, "cal_slope": np.nan, "cal_intercept": np.nan}
    slope, intercept = calibration_slope_intercept(y, p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
