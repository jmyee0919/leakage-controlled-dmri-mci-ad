import re
import math
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression


# =========================================================
# 설정
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "A_generated"
OUT_DIR = PROJECT_ROOT / "A_results_batch"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.30
BOOTSTRAP_B = 500
CALIB_METHOD = "sigmoid"


# =========================================================
# 유틸
# =========================================================
def ensure_X(df, feats):
    return df.reindex(columns=feats).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

def build_pipeline():
    steps = [
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("scaler", StandardScaler(with_mean=True)),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ]
    return Pipeline(steps)

def safe_calibrated(pipe, X, y, method="sigmoid", inner_cv=3):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return pipe.fit(X, y), False
    try:
        cal = CalibratedClassifierCV(estimator=pipe, method=method, cv=inner_cv)
        cal.fit(X, y)
        return cal, True
    except Exception:
        return pipe.fit(X, y), False

def safe_predict_proba(model, X):
    X = np.asarray(X, float)
    try:
        p = model.predict_proba(X)
        pos = list(model.classes_).index(1)
        return p[:, pos]
    except Exception:
        s = model.decision_function(X)
        s = (s - s.min()) / (s.max() - s.min() + 1e-9)
        return s

def metrics_all(y_true, y_prob):
    ok = ~np.isnan(y_prob)
    y = np.asarray(y_true)[ok]
    p = np.asarray(y_prob)[ok]

    if len(np.unique(y)) < 2:
        return dict.fromkeys(
            ["roc_auc", "pr_auc", "brier", "cal_slope", "cal_intercept"],
            np.nan
        )

    rocA = roc_auc_score(y, p)
    prA = average_precision_score(y, p)
    brier = brier_score_loss(y, p)

    eps = 1e-6
    z = np.clip(p, eps, 1 - eps)
    x = np.log(z / (1 - z))
    Xmat = np.vstack([np.ones_like(x), x]).T
    beta = np.linalg.lstsq(Xmat, y, rcond=None)[0]

    return {
        "roc_auc": float(rocA),
        "pr_auc": float(prA),
        "brier": float(brier),
        "cal_intercept": float(beta[0]),
        "cal_slope": float(beta[1]),
    }

def group_bootstrap_metrics(df, y, p, groups, B=500, seed=42):
    rng = np.random.default_rng(seed)
    ids = pd.Series(groups).astype(str).unique()
    mets = []

    for _ in range(B):
        samp = rng.choice(ids, len(ids), replace=True)
        mask = pd.Series(groups).isin(samp).values
        mets.append(metrics_all(np.asarray(y)[mask], np.asarray(p)[mask]))

    out = {}
    keys = ["roc_auc", "pr_auc", "brier", "cal_slope", "cal_intercept"]

    for k in keys:
        arr = np.array([m[k] for m in mets if not np.isnan(m[k])], dtype=float)
        if len(arr) == 0:
            out[k] = (np.nan, (np.nan, np.nan))
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[k] = (float(np.nanmean(arr)), (float(lo), float(hi)))
    return out


# =========================================================
# Correct paired DeLong
# =========================================================
def _compute_midrank(x):
    x = np.asarray(x)
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(N, dtype=float)
    out[J] = T
    return out

def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    k = predictions_sorted_transposed.shape[0]

    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)

    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)

    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n

    return aucs, delongcov

def delong_2sample_test(y_true, p1, p2):
    y_true = np.asarray(y_true).astype(int)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    mask = ~(np.isnan(y_true) | np.isnan(p1) | np.isnan(p2))
    y_true = y_true[mask]
    p1 = p1[mask]
    p2 = p2[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan, np.nan

    order = np.argsort(-y_true)  # positives first
    y_true = y_true[order]
    p1 = p1[order]
    p2 = p2[order]

    label_1_count = int(np.sum(y_true))
    preds = np.vstack([p1, p2])

    aucs, cov = _fast_delong(preds, label_1_count)
    diff = aucs[1] - aucs[0]

    if np.ndim(cov) == 0:
        var = float(cov)
    else:
        var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])

    if var <= 0:
        return float(diff), np.nan, np.nan

    z = diff / np.sqrt(var)
    p = math.erfc(abs(z) / np.sqrt(2.0))
    return float(diff), float(z), float(p)


# =========================================================
# 데이터 빌더 (A-version synthetic)
# =========================================================
def build_table_A(snapshot_path, delta_path):
    snap = pd.read_csv(snapshot_path)
    delt = pd.read_csv(delta_path)

    snap.columns = [c.lower() for c in snap.columns]
    delt.columns = [c.lower() for c in delt.columns]

    for df in [snap, delt]:
        df["subjectid"] = df["subjectid"].astype(str)

    snap = snap.drop_duplicates(subset=["subjectid"]).copy()
    delt = delt.drop_duplicates(subset=["subjectid"]).copy()

    if "interval_months" not in delt.columns:
        raise ValueError(f"{delta_path.name} must contain interval_months")

    delt["interval_months"] = (
        pd.to_numeric(delt["interval_months"], errors="coerce")
        .replace(0, np.nan)
        .fillna(0.1)
    )

    # annualized relative changes
    delt["st29sv_relchange_emp"] = (
        ((delt["st29sv_follow"] - delt["st29sv_base"]) / delt["st29sv_base"])
        / (delt["interval_months"] / 12)
    )
    delt["st37sv_relchange_emp"] = (
        ((delt["st37sv_follow"] - delt["st37sv_base"]) / delt["st37sv_base"])
        / (delt["interval_months"] / 12)
    )
    delt["st149sv_relchange_emp"] = (
        ((delt["st149sv_follow"] - delt["st149sv_base"]) / delt["st149sv_base"])
        / (delt["interval_months"] / 12)
    )

    for c in ["st29sv_relchange_emp", "st37sv_relchange_emp", "st149sv_relchange_emp"]:
        delt[c + "_isna"] = delt[c].isna().astype(int)

    keep = [
        "subjectid",
        "interval_months",

        "st29sv_base", "st29sv_follow",
        "st29sv_relchange_emp", "st29sv_relchange_emp_isna",

        "st149sv_base", "st149sv_follow",
        "st149sv_relchange_emp", "st149sv_relchange_emp_isna",

        "st37sv_base", "st37sv_follow",
        "st37sv_relchange_emp", "st37sv_relchange_emp_isna",
    ]

    for extra in ["converter", "group_base", "group_follow"]:
        if extra in delt.columns:
            keep.append(extra)

    ds = snap.merge(delt[keep], on="subjectid", how="inner").rename(columns={
        "st29sv_base": "hippo_base",
        "st149sv_base": "entorh_base",
        "st37sv_base": "vent_base",

        "st29sv_relchange_emp": "hippo_relchange_emp",
        "st149sv_relchange_emp": "entorh_relchange_emp",
        "st37sv_relchange_emp": "vent_relchange_emp",

        "st29sv_relchange_emp_isna": "hippo_relchange_emp_isna",
        "st149sv_relchange_emp_isna": "entorh_relchange_emp_isna",
        "st37sv_relchange_emp_isna": "vent_relchange_emp_isna",
    })

    if "converter" in ds.columns:
        y = ds["converter"].astype(int).values
    elif "group_follow" in ds.columns:
        y = ds["group_follow"].astype(str).str.lower().str.contains("ad").astype(int).values
    elif "group_base" in ds.columns:
        y = ds["group_base"].astype(str).str.lower().str.contains("progress").astype(int).values
    else:
        raise ValueError(f"No converter/group labels found in {snapshot_path.name} / {delta_path.name}")

    feats_base = ["hippo_base", "entorh_base", "vent_base"]
    feats_delta = [
        "hippo_relchange_emp",
        "entorh_relchange_emp",
        "vent_relchange_emp",
        "hippo_relchange_emp_isna",
        "entorh_relchange_emp_isna",
        "vent_relchange_emp_isna",
    ]
    feats_all = feats_base + feats_delta

    ds["_has_delta_any"] = ds[
        ["hippo_relchange_emp", "entorh_relchange_emp", "vent_relchange_emp"]
    ].notna().any(axis=1)

    ds["SubjectID"] = ds["subjectid"].astype(str)

    return ds, y, feats_base, feats_all


# =========================================================
# pair 자동 찾기
# =========================================================
def find_pairs():
    pattern = re.compile(r"^A_s(\d)\(([^)]+)\)\.csv$", re.IGNORECASE)
    found = {}

    for p in INPUT_DIR.glob("A_s*.csv"):
        m = pattern.match(p.name)
        if not m:
            continue
        seed = m.group(1)
        window = m.group(2)
        delta_name = f"A_m{seed}({window}).csv"
        delta_path = INPUT_DIR / delta_name
        if delta_path.exists():
            found[(seed, window)] = (p, delta_path)

    return sorted(found.items(), key=lambda x: (x[0][1], x[0][0]))


# =========================================================
# 한 쌍 실행
# =========================================================
def run_one_pair(seed, window, snap_path, delt_path):
    ds, y, feats_base, feats_all = build_table_A(snap_path, delt_path)

    groups = ds["SubjectID"].astype(str).values
    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=1 - TEST_SIZE,
        random_state=RANDOM_STATE
    )
    tr, te = next(gss.split(ds, y, groups))

    res = []
    pred_rows = []

    sets = [("Snapshot only", feats_base), ("Snapshot+Change", feats_all)]

    for tag, feats in sets:
        Xtr_df = ds.iloc[tr].copy()
        Xte_df = ds.iloc[te].copy()

        if tag == "Snapshot+Change":
            Xtr_df = Xtr_df[Xtr_df["_has_delta_any"]]
            Xte_df = Xte_df[Xte_df["_has_delta_any"]]

        Xtr = ensure_X(Xtr_df, feats)
        Xte = ensure_X(Xte_df, feats)

        ytr = y[Xtr_df.index]
        yte = y[Xte_df.index]
        g_te = groups[Xte_df.index]

        pipe = build_pipeline()
        cal, _ = safe_calibrated(pipe, Xtr, ytr, method=CALIB_METHOD, inner_cv=3)
        p = safe_predict_proba(cal, Xte)

        metr = metrics_all(yte, p)
        boot = group_bootstrap_metrics(Xte_df, yte, p, g_te, B=BOOTSTRAP_B, seed=RANDOM_STATE)

        res.append({
            "seed": seed,
            "window": window,
            "Set": tag,
            "Model": "Logistic",
            "ROC_AUC": metr["roc_auc"],
            "ROC_AUC_CI_low": boot["roc_auc"][1][0],
            "ROC_AUC_CI_high": boot["roc_auc"][1][1],
            "PR_AP": metr["pr_auc"],
            "PR_AP_CI_low": boot["pr_auc"][1][0],
            "PR_AP_CI_high": boot["pr_auc"][1][1],
            "Brier": metr["brier"],
            "Brier_CI_low": boot["brier"][1][0],
            "Brier_CI_high": boot["brier"][1][1],
            "Cal_slope": metr["cal_slope"],
            "Cal_intercept": metr["cal_intercept"],
            "n_test": len(yte),
        })

        pred_rows.append(pd.DataFrame({
            "SubjectID": Xte_df["SubjectID"].values,
            "y_true": yte,
            "p_pred": p,
            "set": tag,
            "seed": seed,
            "window": window
        }))

    res_df = pd.DataFrame(res)

    pred_df = pd.concat(pred_rows, ignore_index=True)
    base = pred_df[pred_df["set"] == "Snapshot only"].copy()
    chg = pred_df[pred_df["set"] == "Snapshot+Change"].copy()

    inter = set(base["SubjectID"]) & set(chg["SubjectID"])
    if len(inter) >= 10:
        b = base[base["SubjectID"].isin(inter)].sort_values("SubjectID")
        c = chg[chg["SubjectID"].isin(inter)].sort_values("SubjectID")

        d, z, pv = delong_2sample_test(
            b["y_true"].values,
            b["p_pred"].values,
            c["p_pred"].values
        )

        delong_df = pd.DataFrame([{
            "seed": seed,
            "window": window,
            "Model": "Logistic",
            "DeltaAUC": d,
            "z": z,
            "p_value": pv,
            "n": len(inter)
        }])
    else:
        delong_df = pd.DataFrame([{
            "seed": seed,
            "window": window,
            "Model": "Logistic",
            "DeltaAUC": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "n": len(inter)
        }])

    return res_df, delong_df


# =========================================================
# 메인
# =========================================================
if __name__ == "__main__":
    pairs = find_pairs()

    if not pairs:
        print(f"No A_s*/A_m* pairs found in: {INPUT_DIR}")
        raise SystemExit(1)

    print(f"Found {len(pairs)} pairs in {INPUT_DIR}\n")

    all_summary = []
    all_delong = []

    for (seed, window), (snap_path, delt_path) in pairs:
        print(f"[RUN] seed={seed}, window={window}")
        print(f"      snapshot={snap_path.name}")
        print(f"      delta   ={delt_path.name}")

        res_df, delong_df = run_one_pair(seed, window, snap_path, delt_path)

        summary_path = OUT_DIR / f"summary_seed{seed}_{window}.csv"
        delong_path = OUT_DIR / f"delong_seed{seed}_{window}.csv"

        res_df.to_csv(summary_path, index=False)
        delong_df.to_csv(delong_path, index=False)

        all_summary.append(res_df)
        all_delong.append(delong_df)

        print(f"      -> saved {summary_path.name}")
        print(f"      -> saved {delong_path.name}\n")

    master_summary = pd.concat(all_summary, ignore_index=True)
    master_delong = pd.concat(all_delong, ignore_index=True)

    master_summary.to_csv(OUT_DIR / "master_summary.csv", index=False)
    master_delong.to_csv(OUT_DIR / "master_delong.csv", index=False)

    print("All done.")
    print(f"Saved to: {OUT_DIR}")