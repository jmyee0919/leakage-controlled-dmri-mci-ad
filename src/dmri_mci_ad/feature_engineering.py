from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_relative_change(baseline, followup, months, eps: float = 1e-6):
    """Compute annualized relative MRI change: (followup - baseline) / (|baseline| + eps) * 12/months."""
    b = np.asarray(baseline, dtype=float)
    f = np.asarray(followup, dtype=float)
    m = np.asarray(months, dtype=float)
    return (f - b) / (np.abs(b) + eps) * (12.0 / m)


def winsorize_series(x, lower_q: float = 0.01, upper_q: float = 0.99):
    """Winsorize a numeric array/series at the requested quantiles."""
    s = pd.to_numeric(pd.Series(x), errors="coerce")
    lo, hi = s.quantile([lower_q, upper_q])
    return s.clip(lo, hi)


def add_missing_indicators(df: pd.DataFrame, columns: list[str], suffix: str = "_isna") -> pd.DataFrame:
    """Add binary missingness indicators and fill missing values with 0 for DeltaMRI columns."""
    out = df.copy()
    for col in columns:
        out[f"{col}{suffix}"] = out[col].isna().astype(int) if col in out else 1
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        else:
            out[col] = 0.0
    return out
