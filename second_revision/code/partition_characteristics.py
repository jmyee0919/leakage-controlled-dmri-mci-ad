"""Describe primary training and holdout distributions without changing any fit."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

def summarize(d):
    tr=d[d.split.eq('train')];te=d[d.split.eq('test')]
    if (len(tr),len(te),int(tr.y.sum()),int(te.y.sum()))!=(76,30,26,9):raise ValueError('Frozen counts differ')
    rows=[]
    def continuous(key,label,factor=1):
        x=tr[key].dropna().astype(float)*factor;y=te[key].dropna().astype(float)*factor
        def fmt(z):return f'{z.mean():.2f} ({z.std(ddof=1):.2f}); {z.median():.2f} [{z.quantile(.25):.2f}, {z.quantile(.75):.2f}]; n={len(z)}'
        sd=np.sqrt((x.var(ddof=1)+y.var(ddof=1))/2)
        rows.append([label,fmt(x),fmt(y),f'{abs(x.mean()-y.mean())/sd:.3f}' if sd else '0.000'])
    def binary(key,label):
        x=tr[key].astype(float);y=te[key].astype(float);p=x.mean();q=y.mean();sd=np.sqrt((p*(1-p)+q*(1-q))/2)
        rows.append([label,f'{int(x.sum())}/{len(x)} ({p*100:.1f}%)',f'{int(y.sum())}/{len(y)} ({q*100:.1f}%)',f'{abs(p-q)/sd:.3f}' if sd else '0.000'])
    continuous('age','Age, years');binary('female','Female sex')
    continuous('education','Education, years');continuous('MMSE','MMSE score')
    continuous('ICV','Estimated intracranial volume, mL',.001)
    continuous('interval_months','Inter-scan interval, months')
    for prefix,label in [('hippo','Left hippocampus'),('entorh','Left entorhinal cortex'),('vent','Left lateral ventricle')]:
        continuous(prefix+'_base',label+' at baseline, mm³')
        continuous(prefix+'_follow',label+' at second MRI, mm³')
        continuous(prefix+'_relchange_emp',label+' annualized change, %/year',100)
    d=d.copy();d['MMSE_missing']=d.MMSE.isna().astype(int)
    tr=d[d.split.eq('train')];te=d[d.split.eq('test')]
    binary('MMSE_missing','MMSE missing');binary('y','Recorded AD within 24 months')
    return pd.DataFrame(rows,columns=['Characteristic','Training (N=76)','Holdout (N=30)','Absolute SMD'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--primary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    a.out.parent.mkdir(parents=True,exist_ok=True)
    summarize(pd.read_csv(a.primary,float_precision='round_trip')).to_csv(a.out,index=False)
if __name__=='__main__':main()
