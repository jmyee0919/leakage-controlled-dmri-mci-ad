"""Regenerate the visit-level check of AD diagnoses recorded <=45 days after the index (second) MRI.

python make_early_ad_check.py --cohort secondary_cohort_PRIVATE.pkl --icv ICV_PRIMARY_121.csv \
    --spec ANALYSIS_SPECIFICATION.json --dxsum DXSUM.csv --fs7 UCSFFSX7_24Sep2026.csv --out EARLY_AD_VISIT_LEVEL_CHECK.csv

For each early event the index MRI is located in the UCSF FreeSurfer 7 table by RID and scan date (giving the
MRI's own PHASE/VISCODE/VISCODE2/IMAGEUID), and the AD diagnosis and the preceding MCI diagnosis are located in
ADNI DXSUM. 'same_visit_code' is True when the MRI row and the dementia DXSUM row carry the same VISCODE2.
"""
import argparse, json, pickle
import numpy as np, pandas as pd

class Safe(pickle.Unpickler):
    def find_class(self, m, n):
        if m.split('.')[0] not in ('pandas', 'numpy') and not (m == 'builtins' and n == 'slice'): raise RuntimeError(m)
        return super().find_class(m, n)

a = argparse.ArgumentParser()
for k in ['cohort', 'icv', 'spec', 'dxsum', 'fs7', 'out']: a.add_argument('--' + k, required=True)
a = a.parse_args()
d = Safe(open(a.cohort, 'rb')).load().merge(pd.read_csv(a.icv)[['participant_key', 'followup_ICV_percent_difference']], on='participant_key')
spec = json.load(open(a.spec)); ref = {r['region']: r for r in spec['reference_distributions'] if r['reference'] == 'all_1474'}
flag = d.followup_ICV_percent_difference.abs() > 5
for reg, (b, f) in {'hippo': ('hippo_base', 'hippo_follow'), 'entorh': ('entorh_base', 'entorh_follow'), 'vent': ('vent_base', 'vent_follow')}.items():
    r = ref[reg]; flag |= (r['MAD_multiplier'] * (np.log(d[f] / d[b]) - r['median']) / r['unscaled_MAD']).abs() > 4
c = d[~flag].copy(); assert (len(c), int(c.y.sum())) == (106, 35)
c['lag'] = (c.event_date - c.information_date).dt.days
e = c[(c.y == 1) & (c.lag <= 45)].sort_values('lag')
dx = pd.read_csv(a.dxsum, low_memory=False); dx['EXAMDATE'] = pd.to_datetime(dx.EXAMDATE)
fs = pd.read_csv(a.fs7, low_memory=False); fs['EXAMDATE'] = pd.to_datetime(fs.EXAMDATE)
rows = []
for _, r in e.iterrows():
    mri = fs[(fs.RID == r.RID) & (fs.EXAMDATE == r.information_date)].iloc[0]
    ad = dx[(dx.RID == r.RID) & (dx.EXAMDATE == r.event_date)].iloc[0]
    prev = dx[(dx.RID == r.RID) & (dx.EXAMDATE < r.information_date) & (dx.DIAGNOSIS == 'MCI')].sort_values('EXAMDATE').iloc[-1]
    rows.append(dict(participant_key=r.participant_key, RID=int(r.RID), PTID=ad.PTID, split=r.split,
        index_MRI_date=r.information_date.date(), index_MRI_PHASE=mri.PHASE, index_MRI_VISCODE=mri.VISCODE, index_MRI_VISCODE2=mri.VISCODE2, index_MRI_IMAGEUID=int(mri.IMAGEUID),
        AD_DXSUM_EXAMDATE=ad.EXAMDATE.date(), AD_DXSUM_VISCODE=ad.VISCODE, AD_DXSUM_VISCODE2=ad.VISCODE2, AD_DXSUM_DIAGNOSIS=ad.DIAGNOSIS,
        AD_DXSUM_DXAD=ad.DXAD, AD_DXSUM_DXDDUE=ad.DXDDUE, AD_DXSUM_DXAPP=ad.DXAPP, AD_DXSUM_row=int(ad.source_RDA_data_row_1based),
        days_MRI_to_AD_diagnosis=int(r.lag), same_visit_code=str(mri.VISCODE2) == str(ad.VISCODE2),
        prior_MCI_DXSUM_EXAMDATE=prev.EXAMDATE.date(), prior_MCI_VISCODE2=prev.VISCODE2, days_prior_MCI_to_MRI=int((r.information_date - prev.EXAMDATE).days)))
out = pd.DataFrame(rows); out.to_csv(a.out, index=False); print(out.to_string())
