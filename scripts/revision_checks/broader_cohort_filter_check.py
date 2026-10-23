"""Reproduce the broader sensitivity cohort after post hoc QC (106/28) and count historical-filter status.
Inputs (restricted ADNI-derived, internal only): archived broader classified cohort (126), UCSFFSX7 source, QC reference (Table S3).
Outputs: aggregate counts only (no identifiers)."""
import sys, pandas as pd, numpy as np
coh, fs7, ref_csv, out = sys.argv[1:5]
b = pd.read_csv(coh); F = pd.read_csv(fs7, low_memory=False); F['d'] = pd.to_datetime(F.EXAMDATE).dt.date
def icv(rid, date, h, v, e):
    m = F[(F.RID == rid) & (F.d == pd.to_datetime(date).date())]
    m = m[np.isclose(m.ST29SV, h, atol=0.06) & np.isclose(m.ST37SV, v, atol=0.06) & np.isclose(m.ST24CV, e, atol=0.6)]
    u = m.ST10CV.dropna().unique(); assert len(u) == 1; return u[0]
i1 = np.array([icv(x.RID, x.baseline_date, x.hippo_base, x.vent_base, x.entorh_base) for x in b.itertuples()])
i2 = np.array([icv(x.RID, x.information_date, x.hippo_follow, x.vent_follow, x.entorh_follow) for x in b.itertuples()])
A = np.abs(i2 / i1 - 1) > 0.05
ref = pd.read_csv(ref_csv); ref = ref[ref.reference == 'all_1474'].set_index('region')
B = np.zeros(len(b), bool)
for r in ['hippo', 'entorh', 'vent']:
    z = 0.67448975 * (np.log(b[f'{r}_follow'] / b[f'{r}_base']) - ref.loc[r, 'median']) / ref.loc[r, 'unscaled_MAD']; B |= (np.abs(z) > 4).to_numpy()
keep = ~(A | B); y = b.y.to_numpy().astype(int)
sf = ~b.passes_selected_pair_sign_filter.astype(bool).to_numpy(); rf = ~b.passes_historical_snapshot_subject_filter.astype(bool).to_numpy()
hi = (b.hippo_follow > b.hippo_base).to_numpy(); vd = (b.vent_follow < b.vent_base).to_numpy()
test = (b.sensitivity_split == 'test').to_numpy()
rows = [('Broader classified cohort before QC', 126, len(b), y.sum()),
        ('Excluded by Rule A (ICV)', 0, A.sum(), y[A].sum()), ('Excluded by Rule B (robust z)', 0, B.sum(), y[B].sum()),
        ('Broader cohort after QC', 1, keep.sum(), y[keep].sum()), ('  holdout partition', 1, (keep & test).sum(), y[keep & test].sum()),
        ('  pair would fail the direction filter', 1, (keep & sf).sum(), y[keep & sf].sum()),
        ('    hippocampal volume increase', 1, (keep & hi).sum(), y[keep & hi].sum()),
        ('    ventricular volume decrease', 1, (keep & vd).sum(), y[keep & vd].sum()),
        ('  participant would fail the volume-range filter', 1, (keep & rf).sum(), y[keep & rf].sum()),
        ('  fails either historical filter', 1, (keep & (sf | rf)).sum(), y[keep & (sf | rf)].sum())]
df = pd.DataFrame([(a, int(n), int(e)) for a, _, n, e in rows], columns=['stage', 'N', 'AD_events'])
assert (int(keep.sum()), int(y[keep].sum()), int((keep & test).sum()), int(y[keep & test].sum())) == (106, 28, 33, 9)
df.to_csv(out, index=False); print(df.to_string(index=False))
