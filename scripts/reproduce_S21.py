"""Reproduce Table S21 (sensitivity analyses added during revision) from the restricted ADNI cohort files.

Usage (run from any directory; no other files from earlier packages are needed):
    python reproduce_S21.py \
        --cohort  secondary_cohort_PRIVATE.pkl      # 121-participant clinical cohort (restricted)
        --icv     ICV_PRIMARY_121.csv               # per-participant ICV audit with follow-up ICV and field strength (restricted)
        --spec    ANALYSIS_SPECIFICATION.json       # locked v11 specification (included in this package)
        --out     S21_output/

The script stops with an error unless it first reproduces:
  * the post hoc QC cohort (106 participants / 35 events; training 76/26; holdout 30/9),
  * the published primary and five-covariate holdout estimates and intervals, and the repeated
    cross-validation summary (Table 2, Table 3, Table S5),
  * the same holdout predictions with a second, independently written implementation.
Only then are the Table S21 analyses run and written.
"""
import argparse, json, pickle, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV, _SigmoidCalibration
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold


warnings.filterwarnings('ignore', category=FutureWarning, message="'penalty' was deprecated", module=r'sklearn\.linear_model\._logistic')

class SafeUnpickler(pickle.Unpickler):
    # Only pandas/numpy objects are allowed when reading the restricted cohort pickle.
    def find_class(self, module, name):
        if module.split('.')[0] not in ('pandas', 'numpy') and not (module == 'builtins' and name == 'slice'):
            raise RuntimeError(f'Blocked unpickling of {module}.{name}')
        return super().find_class(module, name)

def load_cohort(path):
    with open(path, 'rb') as f:
        return SafeUnpickler(f).load()

# ---------------------------------------------------------------- implementation 1 (as locked in v11)
def build_predict(train, test, columns, clinical, clf):
    if clinical:
        mri = [c for c in columns if c not in clinical]
        imp = ColumnTransformer([('clinical', SimpleImputer(strategy='median'), clinical),
                                 ('MRI', SimpleImputer(strategy='constant', fill_value=0.), mri)], remainder='drop')
    else:
        imp = SimpleImputer(strategy='constant', fill_value=0.)
    pipe = Pipeline([('imputer', imp), ('scaler', StandardScaler()), ('clf', LogisticRegression(**clf))])
    m = CalibratedClassifierCV(estimator=pipe, method='sigmoid', cv=3, ensemble=True)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter('always')
        m.fit(train[columns], train.y.to_numpy(int))
    if any('converg' in str(w.message).lower() for w in recorded):
        raise RuntimeError('Convergence warning during reproduction')
    return m.predict_proba(test[columns])[:, list(m.classes_).index(1)]

# ---------------------------------------------------------------- implementation 2 (written separately)
def build_predict_manual(train, test, columns, clinical, clf):
    X = train[columns].to_numpy(float); y = train.y.to_numpy(int); Xt = test[columns].to_numpy(float)
    clin_idx = [columns.index(c) for c in clinical]
    probs = []
    for fit_idx, cal_idx in StratifiedKFold(3).split(X, y):
        fill = np.zeros(X.shape[1])
        for j in clin_idx:
            fill[j] = np.nanmedian(X[fit_idx, j])
        def impute(M):
            M = M.copy(); r, c = np.where(np.isnan(M)); M[r, c] = fill[c]; return M
        A = impute(X[fit_idx]); mu = A.mean(0); sd = A.std(0); sd[sd == 0] = 1
        z = lambda M: (impute(M) - mu) / sd
        lr = LogisticRegression(**clf).fit(z(X[fit_idx]), y[fit_idx])
        cal = _SigmoidCalibration().fit(lr.decision_function(z(X[cal_idx])), y[cal_idx])
        probs.append(cal.predict(lr.decision_function(z(Xt))))
    return np.mean(probs, axis=0)

def metrics(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.array([np.nan, np.nan, np.mean((p - y) ** 2)])
    auc = ((pos[:, None] > neg).sum() + .5 * (pos[:, None] == neg).sum()) / (len(pos) * len(neg))
    ap = 0.
    for t in np.unique(p)[::-1]:
        ap += (np.sum((p == t) & (y == 1)) / len(pos)) * (np.sum((p >= t) & (y == 1)) / np.sum(p >= t))
    return np.array([auc, ap, np.mean((p - y) ** 2)])

def analyse(c, arms, clinical, clf, repeated=True):
    c = c.reset_index(drop=True); tr = c[c.split.eq('train')]; te = c[c.split.eq('test')]; y = te.y.to_numpy(int)
    P = [build_predict(tr, te, cols, clinical, clf) for cols in arms.values()]
    P2 = [build_predict_manual(tr, te, cols, clinical, clf) for cols in arms.values()]
    second_impl_max_diff = float(max(np.max(np.abs(a - b)) for a, b in zip(P, P2)))
    idx = np.random.default_rng(42).integers(0, len(te), (500, len(te)))
    B = [np.array([metrics(y[i], p[i]) for i in idx]) for p in P]
    pt = [metrics(y, p) for p in P]
    out = dict(N=len(c), events=int(c.y.sum()), holdout=f'{len(te)}/{int(te.y.sum())}', second_implementation_max_abs_diff=second_impl_max_diff)
    for j, n in enumerate(['AUC', 'AP', 'Brier']):
        d = B[1][:, j] - B[0][:, j]; d = d[np.isfinite(d)]
        out.update({f'{n}_snapshot': pt[0][j], f'{n}_augmented': pt[1][j], f'd{n}': pt[1][j] - pt[0][j],
                    f'd{n}_lower95': np.quantile(d, .025), f'd{n}_upper95': np.quantile(d, .975)})
        if n == 'AUC':
            out['share_of_resamples_dAUC_exactly_zero'] = float(np.mean(d == 0))
    if repeated:
        R = {a: np.full((10, len(c)), np.nan) for a in arms}
        for k, (a, b) in enumerate(RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42).split(c, c.y)):
            for arm, cols in arms.items():
                R[arm][k // 3, b] = build_predict(c.iloc[a], c.iloc[b], cols, clinical, clf)
        yy = c.y.to_numpy(int); names = list(arms)
        d = np.array([metrics(yy, R[names[1]][r])[0] - metrics(yy, R[names[0]][r])[0] for r in range(10)])
        out.update(repeated_dAUC_mean=d.mean(), repeated_dAUC_SD=d.std(ddof=1), repeated_dAUC_min=d.min(),
                   repeated_dAUC_max=d.max(), repeated_negative=int((d < 0).sum()))
    return out

def check(label, got, expected, tol=5e-5):
    ok = abs(got - expected) <= tol
    print(f'  {"OK  " if ok else "FAIL"} {label}: {got:.6f} (published {expected:.4f})')
    if not ok:
        sys.exit(f'Reproduction check failed: {label}')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cohort', required=True); ap.add_argument('--icv', required=True)
    ap.add_argument('--spec', required=True); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out).resolve()
    inputs = [Path(a.cohort).resolve(), Path(a.icv).resolve(), Path(a.spec).resolve()]
    if any(out == p.parent or out in p.parents for p in inputs):
        raise ValueError('Choose an output directory separate from input directories')
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads(Path(a.spec).read_text()); clf = spec['classifier']; ref = {r['region']: r for r in spec['reference_distributions'] if r['reference'] == 'all_1474'}
    d = load_cohort(a.cohort)
    icv = pd.read_csv(a.icv)[['participant_key', 'followup_ICV_percent_difference', 'source_FIELD_STRENGTH']]
    d = d.merge(icv, on='participant_key', validate='one_to_one')
    # Post hoc QC rules exactly as specified (Methods 2.3)
    rule_a = d.followup_ICV_percent_difference.abs() > 5
    z = []
    for reg, (b, f) in {'hippo': ('hippo_base', 'hippo_follow'), 'entorh': ('entorh_base', 'entorh_follow'), 'vent': ('vent_base', 'vent_follow')}.items():
        x = np.log(d[f] / d[b]); r = ref[reg]; z.append((r['MAD_multiplier'] * (x - r['median']) / r['unscaled_MAD']).abs() > 4)
    rule_b = pd.concat(z, axis=1).any(axis=1)
    c = d[~(rule_a | rule_b)].reset_index(drop=True)
    tr, te = c[c.split.eq('train')], c[c.split.eq('test')]
    print('Step 1. QC cohort')
    print(f'  Rule A {int(rule_a.sum())}, Rule B {int(rule_b.sum())}, retained {len(c)}/{int(c.y.sum())}, training {len(tr)}/{int(tr.y.sum())}, holdout {len(te)}/{int(te.y.sum())}')
    if (int(rule_a.sum()), int(rule_b.sum()), len(c), int(c.y.sum()), len(tr), int(tr.y.sum()), len(te), int(te.y.sum())) != (11, 4, 106, 35, 76, 26, 30, 9):
        sys.exit('QC cohort does not match the published cohort')
    c['field_3T'] = (c.source_FIELD_STRENGTH == '3T').astype(float)
    c['lag_days'] = (c.event_date - c.information_date).dt.days
    prim = spec['analyses']['primary']; five = spec['analyses']['five_covariate']
    print('Step 2. Published results (Table 2, Table 3, Table S5)')
    r = analyse(c, prim['arms'], [], clf)
    for k, v in [('AUC_snapshot', .7725), ('AUC_augmented', .8519), ('dAUC', .0794), ('dAUC_lower95', -.0132), ('dAUC_upper95', .2162),
                 ('dAP', .1034), ('dBrier', -.0196), ('repeated_dAUC_mean', .0219), ('repeated_dAUC_SD', .0150)]:
        check('primary ' + k, r[k], v)
    check('primary second implementation max difference', r['second_implementation_max_abs_diff'], 0, tol=1e-8)
    r5 = analyse(c, five['arms'], five['clinical'], clf, repeated=False)
    for k, v in [('dAUC', .0847), ('dAUC_lower95', .0024), ('dAUC_upper95', .2100)]:
        check('five-covariate ' + k, r5[k], v)
    check('five-covariate second implementation max difference', r5['second_implementation_max_abs_diff'], 0, tol=1e-8)
    print('Step 3. Table S21 analyses')
    rows = []
    early5 = c[(c.y == 1) & (c.lag_days <= 30)].participant_key; early6 = c[(c.y == 1) & (c.lag_days <= 45)].participant_key
    print(f'  AD recorded <=30 days after index MRI: {len(early5)}; <=45 days: {len(early6)}')
    fs_arms = lambda arms: {k: ['field_3T'] + v for k, v in arms.items()}
    for label, cohort, arms, clin, rep in [
        ('Primary (reference)', c, prim['arms'], [], True),
        ('Primary (reference), five covariates', c, five['arms'], five['clinical'], False),
        ('Excluding AD <=30 days', c[~c.participant_key.isin(early5)], prim['arms'], [], True),
        ('Excluding AD <=30 days, five covariates', c[~c.participant_key.isin(early5)], five['arms'], five['clinical'], False),
        ('Excluding AD <=45 days', c[~c.participant_key.isin(early6)], prim['arms'], [], True),
        ('Excluding AD <=45 days, five covariates', c[~c.participant_key.isin(early6)], five['arms'], five['clinical'], False),
        ('Field strength added', c, fs_arms(prim['arms']), [], True),
        ('Field strength added, five covariates', c, fs_arms(five['arms']), five['clinical'], False)]:
        res = analyse(cohort, arms, clin, clf, repeated=rep); res['analysis'] = label; rows.append(res)
        print(f'  {label}: N {res["N"]}/{res["events"]}, holdout {res["holdout"]}, dAUC {res["dAUC"]:+.4f} ({res["dAUC_lower95"]:+.4f} to {res["dAUC_upper95"]:+.4f}); second implementation max diff {res["second_implementation_max_abs_diff"]:.1e}')
        if res['second_implementation_max_abs_diff'] > 1e-8:
            sys.exit('Second implementation disagrees for ' + label)
    pd.DataFrame(rows).to_csv(out / 'Table_S21_reproduced.csv', index=False, float_format='%.10g')
    print('Done. All checks passed; results written to', out / 'Table_S21_reproduced.csv')

if __name__ == '__main__':
    main()
