"""Reproduce the second revision from authorized frozen cohort CSVs.

Run from the repository root:
python second_revision/code/reproduce_revision2.py --primary PRIMARY.csv \
  --broader BROADER.csv --out /private/reproduction

Input files are restricted and must never be committed. They contain one row
per person, the frozen row order and split, binary y, the baseline/follow-up
regional measurements, original annualized changes and flags, and (primary
only) age, female, education, MMSE and ICV. This script neither constructs nor
changes the cohort and does not infer missing source provenance.
"""
import argparse, hashlib, importlib.util, json, platform, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parents[2]
METRICS = ['roc_auc','average_precision','brier','calibration_intercept','calibration_slope']
BASE = ['hippo_base','entorh_base','vent_base']
FOLLOW = ['hippo_follow','entorh_follow','vent_follow']
DELTA = ['hippo_relchange_emp','entorh_relchange_emp','vent_relchange_emp',
         'hippo_relchange_emp_isna','entorh_relchange_emp_isna','vent_relchange_emp_isna']
CLINICAL = ['age','female','education','MMSE','ICV']
COMPARISONS = [('M2','M1'),('M3','M1'),('M4','M3'),('M4','M2')]

def calibration(y,p):
    if np.unique(y).size < 2: return np.nan,np.nan
    v=np.clip(p,1e-8,1-1e-8); x=np.log(v/(1-v))
    if np.unique(x).size<2: return np.nan,np.nan
    if x[y==1].min()>=x[y==0].max() or x[y==0].min()>=x[y==1].max(): return np.nan,np.nan
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        f=sm.GLM(y,sm.add_constant(x,has_constant='add'),family=sm.families.Binomial()).fit(maxiter=200,tol=1e-10)
    if not f.converged or not np.isfinite(f.params).all(): return np.nan,np.nan
    if any('separation' in str(w.message).lower() for w in ws): return np.nan,np.nan
    if np.max(np.abs(f.model.score(f.params)))>1e-5: return np.nan,np.nan
    return tuple(f.params)

def metrics(y,p):
    both=np.unique(y).size==2
    a,b=calibration(y,p)
    return np.array([roc_auc_score(y,p) if both else np.nan,
                     average_precision_score(y,p) if both else np.nan,
                     brier_score_loss(y,p),a,b])

def fit(train,test,columns,clinical,clf):
    if clinical:
        mri=[c for c in columns if c not in clinical]
        imp=ColumnTransformer([('clinical',SimpleImputer(strategy='median'),clinical),
            ('MRI',SimpleImputer(strategy='constant',fill_value=0.),mri)],remainder='drop')
    else: imp=SimpleImputer(strategy='constant',fill_value=0.)
    pipe=Pipeline([('imputer',imp),('scaler',StandardScaler()),('clf',LogisticRegression(**clf))])
    model=CalibratedClassifierCV(estimator=pipe,method='sigmoid',cv=3,ensemble=True)
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always'); model.fit(train[columns],train.y.to_numpy(int))
    if any('converg' in str(w.message).lower() for w in ws): raise RuntimeError('Convergence failure')
    return model.predict_proba(test[columns])[:,1],model

def check_input(d,n,events,test_n,test_events,clinical):
    required=['participant_key','split','y']+BASE+FOLLOW+DELTA+clinical
    if any(c not in d for c in required): raise ValueError('Missing required input columns')
    if not d.participant_key.is_unique or not d.split.isin(['train','test']).all(): raise ValueError('Invalid keys/splits')
    if not d.y.isin([0,1]).all(): raise ValueError('Invalid outcome')
    test=d[d.split.eq('test')]
    if (len(d),int(d.y.sum()),len(test),int(test.y.sum()))!=(n,events,test_n,test_events): raise ValueError('Frozen cohort counts differ')
    if not np.isfinite(d[BASE+FOLLOW+DELTA].to_numpy(float)).all(): raise ValueError('Invalid MRI inputs')
    if any(d[c].nunique()!=1 or d[c].iloc[0]!=0 for c in DELTA[3:]): raise ValueError('Unexpected availability flags')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--primary',type=Path,required=True);ap.add_argument('--broader',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out.resolve()
    if ROOT==out or ROOT in out.parents: raise ValueError('Outputs must be outside the public repository')
    out.mkdir(parents=True,exist_ok=True)
    spec=json.loads((ROOT/'configs/analysis_specification.json').read_text())
    helper_spec=importlib.util.spec_from_file_location('prior_reproduction',ROOT/'scripts/reproduce_S21.py')
    helper=importlib.util.module_from_spec(helper_spec);helper_spec.loader.exec_module(helper)
    primary=pd.read_csv(a.primary,float_precision='round_trip');broader=pd.read_csv(a.broader,float_precision='round_trip')
    check_input(primary,106,35,30,9,CLINICAL);check_input(broader,106,28,33,9,[])
    aggregate=ROOT/'second_revision/Data_file_S1'
    all_hold=[];all_repeats=[];params=[];preprocess=[];checks=[];maxdiff=0.
    for analysis,d,clinical in [('imaging',primary,[]),('five_covariate',primary,CLINICAL),('broader',broader,[])]:
        arms={k:clinical+v for k,v in {'M1':BASE,'M2':BASE+DELTA,'M3':FOLLOW,'M4':FOLLOW+DELTA}.items()}
        train=d[d.split.eq('train')];test=d[d.split.eq('test')];y=test.y.to_numpy(int)
        P={};point={};boot={}
        ix=np.random.default_rng(42).integers(0,len(test),(500,len(test)))
        private=pd.DataFrame({'participant_key':test.participant_key,'y':y})
        for arm,cols in arms.items():
            p,model=fit(train,test,cols,clinical,spec['classifier']);P[arm]=p;private[arm]=p
            manual=helper.build_predict_manual(train,test,cols,clinical,spec['classifier'])
            err=float(np.max(np.abs(p-manual)));maxdiff=max(maxdiff,err)
            if err>1e-10: raise AssertionError('Independent prediction implementation differs')
            for fold,cc in enumerate(model.calibrated_classifiers_,1):
                est=cc.estimator;lr=est['clf'];sc=est['scaler'];imp=est['imputer']
                names=clinical+[c for c in cols if c not in clinical] if clinical else cols
                fills=np.concatenate([imp.named_transformers_['clinical'].statistics_,imp.named_transformers_['MRI'].statistics_]) if clinical else imp.statistics_
                for name,beta,mu,sd,fill in zip(names,lr.coef_[0],sc.mean_,sc.scale_,fills):
                    params.append(dict(analysis=analysis,arm=arm,fold=fold,parameter=name,value=float(beta)))
                    preprocess.append(dict(analysis=analysis,arm=arm,fold=fold,feature=name,imputation=float(fill),mean=float(mu),scale=float(sd)))
                for name,value in [('intercept',lr.intercept_[0]),('sigmoid_a',cc.calibrators[0].a_),('sigmoid_b',cc.calibrators[0].b_)]:
                    params.append(dict(analysis=analysis,arm=arm,fold=fold,parameter=name,value=float(value)))
            point[arm]=metrics(y,p);boot[arm]=np.array([metrics(y[i],p[i]) for i in ix])
        private.to_csv(out/f'{analysis}_holdout_PRIVATE.csv',index=False,float_format='%.17g')
        for comp,pv,bv in [(arm,point[arm],boot[arm]) for arm in arms]+[(f'{x}-{z}',point[x]-point[z],boot[x]-boot[z]) for x,z in COMPARISONS]:
            for j,metric in enumerate(METRICS):
                v=bv[:,j];v=v[np.isfinite(v)]
                all_hold.append(dict(analysis=analysis,comparison=comp,metric=('delta_' if '-' in comp else '')+metric,
                    estimate=pv[j],lower95=np.quantile(v,.025) if len(v) else np.nan,upper95=np.quantile(v,.975) if len(v) else np.nan,
                    test_N=len(test),test_events=int(y.sum()),valid_draws=len(v)))
        R={arm:np.full((10,len(d)),np.nan) for arm in arms}
        for k,(tr,te) in enumerate(RepeatedStratifiedKFold(n_splits=3,n_repeats=10,random_state=42).split(d,d.y)):
            for arm,cols in arms.items():
                p=fit(d.iloc[tr],d.iloc[te],cols,clinical,spec['classifier'])[0]
                manual=helper.build_predict_manual(d.iloc[tr],d.iloc[te],cols,clinical,spec['classifier'])
                err=float(np.max(np.abs(p-manual)));maxdiff=max(maxdiff,err)
                if err>1e-10:raise AssertionError('Independent cross-validation implementation differs')
                R[arm][k//3,te]=p
        for r in range(10):
            auc={arm:roc_auc_score(d.y,R[arm][r]) for arm in arms}
            row=dict(analysis=analysis,repeat=r+1,**{'AUC_'+arm:v for arm,v in auc.items()})
            row.update({f'dAUC_{x}-{z}':auc[x]-auc[z] for x,z in COMPARISONS});all_repeats.append(row)
        print(analysis,'complete',flush=True)
    hold=pd.DataFrame(all_hold);repeats=pd.DataFrame(all_repeats)
    for analysis,file in [('imaging','Table_2_3_four_model_holdout.csv'),('five_covariate','Table_2_3_four_model_holdout.csv'),('broader','Table_S13_broader_four_model_holdout.csv')]:
        expected=pd.read_csv(aggregate/file);actual=hold[hold.analysis.eq(analysis)]
        if 'analysis' in expected and analysis!='broader':expected=expected[expected.analysis.eq(analysis)]
        for r in expected.itertuples():
            comp=r.comparison.replace(' minus ','-')
            q=actual[actual.comparison.eq(comp)&actual.metric.eq(r.metric)].iloc[0]
            for col in ['estimate','lower95','upper95','valid_draws']:
                err=abs(float(q[col])-float(getattr(r,col)))
                if not (err<=5.1e-7 or (pd.isna(q[col]) and pd.isna(getattr(r,col)))):raise AssertionError((analysis,comp,r.metric,col,err))
                checks.append(f'{analysis}/{comp}/{r.metric}/{col}')
    for analysis,file in [('imaging','Table_S11_four_model_each_repeat.csv'),('five_covariate','Table_S11_four_model_each_repeat.csv'),('broader','Table_S13_broader_each_repeat.csv')]:
        expected=pd.read_csv(aggregate/file)
        if analysis!='broader':expected=expected[expected.analysis.eq(analysis)]
        for r in expected.to_dict('records'):
            q=repeats[repeats.analysis.eq(analysis)&repeats.repeat.eq(r['repeat'])].iloc[0]
            for key in r:
                if key.startswith(('AUC_','dAUC_')):
                    if abs(float(q[key])-float(r[key]))>5.1e-7:raise AssertionError((analysis,r['repeat'],key,q[key],r[key]))
                    checks.append(f'{analysis}/repeat{r["repeat"]}/{key}')
    hold.to_csv(out/'reproduced_holdout.csv',index=False,float_format='%.17g')
    repeats.to_csv(out/'reproduced_each_repeat.csv',index=False,float_format='%.17g')
    pd.DataFrame(params).to_csv(out/'four_model_parameters.csv',index=False,float_format='%.17g')
    pd.DataFrame(preprocess).to_csv(out/'four_model_preprocessing.csv',index=False,float_format='%.17g')
    report=dict(status='PASS',checks=len(checks),maximum_manual_prediction_difference=maxdiff,
      scope='Frozen primary and broader cohorts; all 12 holdout arms and 360 outer cross-validation fits. Not raw-to-cohort reconstruction, image validation or external validation.',
      input_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.primary,a.broader]},
      environment=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,statsmodels=statsmodels.__version__))
    (out/'REPRODUCTION_REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
