from __future__ import annotations

import numpy as np
import pandas as pd


def decision_curve(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Compute simple decision-curve net benefit values for a binary prediction model."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob, dtype=float)
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    rows = []
    n = len(y)
    prevalence = y.mean() if n else np.nan
    for t in thresholds:
        pred = p >= t
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        nb = tp / n - fp / n * (t / (1 - t))
        treat_all = prevalence - (1 - prevalence) * (t / (1 - t))
        rows.append({"threshold": float(t), "net_benefit": float(nb), "treat_all": float(treat_all), "treat_none": 0.0})
    return pd.DataFrame(rows)
