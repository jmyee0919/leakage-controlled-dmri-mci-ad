# export_adni_filtered_primary_predictions.py
# ---------------------------------------------------------
# Standalone exporter for the PRIMARY ADNI analysis setting
# used in the paper:
#   cohort       = filtered
#   calibration  = sigmoid
#   split        = 70/30 hold-out
#   random_state = 42
#   models       = Logistic, Snapshot only vs Snapshot+Change
# ---------------------------------------------------------
# Put this script in the SAME folder as:
#   filtered_snapshot.csv
#   filtered_mridelta.csv
# Then run:
#   python export_adni_filtered_primary_predictions.py
# It will save:
#   adni_filtered_primary_predictions.csv
# ---------------------------------------------------------

import os
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42
BASE_DIR = os.getcwd()

SNAP_PATH = os.path.join(BASE_DIR, 'filtered_snapshot.csv')
DELT_PATH = os.path.join(BASE_DIR, 'filtered_mridelta.csv')
OUT_PATH  = os.path.join(BASE_DIR, 'adni_filtered_primary_predictions.csv')


def ensure_X(df, feats):
    return df.reindex(columns=feats).apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)


def build_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value=0.0)),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=5000, solver='lbfgs', random_state=RANDOM_STATE))
    ])


def safe_calibrated(pipe, X, y, method='sigmoid', inner_cv=3):
    try:
        cal = CalibratedClassifierCV(pipe, method=method, cv=inner_cv)
        cal.fit(X, y)
        return cal
    except Exception:
        pipe.fit(X, y)
        return pipe


def safe_predict_proba(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        s = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-s))
    pred = model.predict(X)
    return np.asarray(pred, dtype=float)


def build_single_table_real(snap_file, delt_file):
    snap = pd.read_csv(snap_file)
    delt = pd.read_csv(delt_file)

    snap.columns = [c.lower() for c in snap.columns]
    delt.columns = [c.lower() for c in delt.columns]

    for df in [snap, delt]:
        df['subjectid'] = df['subjectid'].astype(str)

    if 'interval_months' not in delt.columns:
        raise ValueError('filtered_mridelta.csv must contain interval_months')

    delt['interval_months'] = (
        pd.to_numeric(delt['interval_months'], errors='coerce')
        .replace(0, np.nan)
        .fillna(0.1)
    )

    # empirical annualized change
    delt['st29sv_relchange_emp'] = (((delt['st29sv_follow'] - delt['st29sv_base']) / delt['st29sv_base']) / (delt['interval_months'] / 12))
    delt['st37sv_relchange_emp'] = (((delt['st37sv_follow'] - delt['st37sv_base']) / delt['st37sv_base']) / (delt['interval_months'] / 12))

    if {'st149sv_base', 'st149sv_follow'}.issubset(delt.columns):
        delt['st149sv_relchange_emp'] = (((delt['st149sv_follow'] - delt['st149sv_base']) / delt['st149sv_base']) / (delt['interval_months'] / 12))
    else:
        delt['st149sv_relchange_emp'] = np.nan
        delt['st149sv_base'] = np.nan
        delt['st149sv_follow'] = np.nan

    for c in ['st29sv_relchange_emp', 'st149sv_relchange_emp', 'st37sv_relchange_emp']:
        delt[c + '_isna'] = delt[c].isna().astype(int)

    keep = [
        'subjectid', 'interval_months',
        'st29sv_base', 'st29sv_follow', 'st29sv_relchange_emp', 'st29sv_relchange_emp_isna',
        'st149sv_base', 'st149sv_follow', 'st149sv_relchange_emp', 'st149sv_relchange_emp_isna',
        'st37sv_base', 'st37sv_follow', 'st37sv_relchange_emp', 'st37sv_relchange_emp_isna'
    ]
    for extra in ['converter', 'group_base', 'group_follow']:
        if extra in delt.columns:
            keep.append(extra)

    ds = snap.merge(delt[keep], on='subjectid', how='inner').rename(columns={
        'st29sv_base': 'hippo_base',
        'st149sv_base': 'entorh_base',
        'st37sv_base': 'vent_base',
        'st29sv_relchange_emp': 'hippo_relchange_emp',
        'st149sv_relchange_emp': 'entorh_relchange_emp',
        'st37sv_relchange_emp': 'vent_relchange_emp',
        'st29sv_relchange_emp_isna': 'hippo_relchange_emp_isna',
        'st149sv_relchange_emp_isna': 'entorh_relchange_emp_isna',
        'st37sv_relchange_emp_isna': 'vent_relchange_emp_isna'
    })

    if 'converter' in ds.columns:
        y = ds['converter'].astype(int).values
    elif 'group_follow' in ds.columns:
        y = ds['group_follow'].astype(str).str.lower().str.contains('ad').astype(int).values
    elif 'group_base' in ds.columns:
        y = ds['group_base'].astype(str).str.lower().str.contains('progress').astype(int).values
    else:
        raise ValueError('No converter or group labels found.')

    feats_base = ['hippo_base', 'entorh_base', 'vent_base']
    feats_delta = [
        'hippo_relchange_emp', 'entorh_relchange_emp', 'vent_relchange_emp',
        'hippo_relchange_emp_isna', 'entorh_relchange_emp_isna', 'vent_relchange_emp_isna'
    ]
    feats_all = feats_base + feats_delta

    ds['_has_delta_any'] = ds[['hippo_relchange_emp', 'entorh_relchange_emp', 'vent_relchange_emp']].notna().any(axis=1)
    ds['SubjectID'] = ds['subjectid'].astype(str)

    return ds, y, feats_base, feats_all


def main():
    if not os.path.exists(SNAP_PATH):
        raise FileNotFoundError(f'Could not find: {SNAP_PATH}')
    if not os.path.exists(DELT_PATH):
        raise FileNotFoundError(f'Could not find: {DELT_PATH}')

    ds, y, feats_base, feats_all = build_single_table_real(SNAP_PATH, DELT_PATH)
    groups = ds['SubjectID'].astype(str).values

    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=RANDOM_STATE)
    tr, te = next(gss.split(ds, y, groups))

    # Snapshot-only
    Xtr_b = ensure_X(ds.iloc[tr], feats_base)
    Xte_b = ensure_X(ds.iloc[te], feats_base)
    ytr_b = y[tr]
    yte_b = y[te]
    id_b = ds.iloc[te]['SubjectID'].values

    model_b = safe_calibrated(build_pipeline(), Xtr_b, ytr_b, method='sigmoid', inner_cv=3)
    p_b = safe_predict_proba(model_b, Xte_b)

    # Snapshot + DeltaMRI (restricted to rows with any delta)
    Xtr_df_a = ds.iloc[tr].copy()
    Xte_df_a = ds.iloc[te].copy()
    Xtr_df_a = Xtr_df_a[Xtr_df_a['_has_delta_any']]
    Xte_df_a = Xte_df_a[Xte_df_a['_has_delta_any']]

    Xtr_a = ensure_X(Xtr_df_a, feats_all)
    Xte_a = ensure_X(Xte_df_a, feats_all)
    ytr_a = y[Xtr_df_a.index]
    yte_a = y[Xte_df_a.index]
    id_a = Xte_df_a['SubjectID'].values

    model_a = safe_calibrated(build_pipeline(), Xtr_a, ytr_a, method='sigmoid', inner_cv=3)
    p_a = safe_predict_proba(model_a, Xte_a)

    # paired intersection for DeLong-compatible plotting
    inter = sorted(set(id_b) & set(id_a))
    if len(inter) < 10:
        raise RuntimeError('Intersection between snapshot-only and snapshot+delta test subjects is too small.')

    out_b = pd.DataFrame({'SubjectID': id_b, 'y_true': yte_b, 'p_snapshot': p_b})
    out_a = pd.DataFrame({'SubjectID': id_a, 'y_true': yte_a, 'p_snapshot_change': p_a})

    # Collapse to one subject-level row before merging.
    # Without this, repeated rows per SubjectID create a many-to-many merge
    # and inflate the output far beyond the true test-subject count.
    out_b = (
        out_b.groupby('SubjectID', as_index=False)
        .agg(y_true=('y_true', 'max'),
             p_snapshot=('p_snapshot', 'mean'))
    )

    out_a = (
        out_a.groupby('SubjectID', as_index=False)
        .agg(y_true=('y_true', 'max'),
             p_snapshot_change=('p_snapshot_change', 'mean'))
    )

    out = out_b.merge(out_a[['SubjectID', 'p_snapshot_change']], on='SubjectID', how='inner')
    out = out[out['SubjectID'].isin(inter)].sort_values('SubjectID').reset_index(drop=True)

    out.to_csv(OUT_PATH, index=False)
    print('Saved:', OUT_PATH)
    print('Rows:', len(out))
    print('Unique subjects:', out['SubjectID'].nunique())


if __name__ == '__main__':
    main()
