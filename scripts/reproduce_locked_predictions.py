"""Isolated reproduction from archived frozen cohorts, never overwriting source.

This rebuilds the original, fully specified estimators for verification only.
It performs no model/feature/threshold/split selection and supplies no new
validation cohort. The source archive is restricted and must be supplied by
an authorized holder. Pickles must only be read from that trusted archive.
"""
from pathlib import Path
import argparse,json,hashlib,platform,warnings
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RepeatedStratifiedKFold

def build_predict(train,test,columns,clinical,classifier):
    if clinical:
        mri=[c for c in columns if c not in clinical]
        imputer=ColumnTransformer([('clinical',SimpleImputer(strategy='median'),clinical),('MRI',SimpleImputer(strategy='constant',fill_value=0.),mri)],remainder='drop')
    else:imputer=SimpleImputer(strategy='constant',fill_value=0.)
    pipe=Pipeline([('imputer',imputer),('scaler',StandardScaler()),('clf',LogisticRegression(**classifier))])
    model=CalibratedClassifierCV(estimator=pipe,method='sigmoid',cv=3,ensemble=True)
    with warnings.catch_warnings(record=True) as messages:
        warnings.simplefilter('always');model.fit(train[columns],train.y.to_numpy(int))
    if any('converg' in str(w.message).lower() for w in messages):raise RuntimeError('Convergence warning during exact reproduction')
    return model.predict_proba(test[columns])[:,list(model.classes_).index(1)],model

def main():
    pa=argparse.ArgumentParser();pa.add_argument('--source',type=Path,required=True);pa.add_argument('--output',type=Path,required=True);pa.add_argument('--tolerance',type=float,default=1e-6);a=pa.parse_args()
    S=a.source.resolve();a.output=a.output.resolve()
    if a.output==S or S in a.output.parents:raise ValueError('Output must be outside frozen source')
    P=S/'01_private';a.output.mkdir(parents=True,exist_ok=True)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()}
    spec=json.loads((S/'02_locked_analysis/ANALYSIS_SPECIFICATION.json').read_text());rows=[];base_count=0;prob_count=0
    for name,cfg in spec['analyses'].items():
        if not cfg['fit_allowed']:continue
        assert sha(P/f'{name}_cohort.pkl')==cfg['cohort_sha256']
        d=pd.read_pickle(P/f'{name}_cohort.pkl');tr=d[d.split.eq('train')];te=d[d.split.eq('test')];saved=pd.read_csv(P/f'{name}_predictions.csv',float_precision='round_trip')
        assert te.participant_key.tolist()==saved.participant_key.tolist()
        parameters=pd.read_csv(P/f'{name}_parameters.csv',float_precision='round_trip')
        for arm,cols in cfg['arms'].items():
            pp,model=build_predict(tr,te,cols,cfg['clinical'],spec['classifier']);err=float(np.max(abs(pp-saved[arm].to_numpy())));ce=[];se=[]
            for i,cc in enumerate(model.calibrated_classifiers_,1):
                z=parameters[parameters.arm.eq(arm)&parameters.fold.eq(i)]
                ce.extend(abs(cc.estimator['clf'].coef_[0]-z.coefficient.to_numpy()));ce.append(abs(cc.estimator['clf'].intercept_[0]-z.logistic_intercept.iloc[0]));se.extend(abs(np.array([cc.calibrators[0].a_,cc.calibrators[0].b_])-np.array([z.sigmoid_a.iloc[0],z.sigmoid_b.iloc[0]])))
            rows.append(dict(analysis=name,arm=arm,stage='inherited_holdout',probabilities=len(te),maximum_probability_error=err,maximum_logistic_parameter_error=max(ce),maximum_sigmoid_parameter_error=max(se)))
            base_count+=3;prob_count+=len(te)
        print(name+' reproduced',flush=True)
    cfg=spec['analyses']['primary'];d=pd.read_pickle(P/'primary_cohort.pkl').reset_index(drop=True);original=np.load(P/'REPEATED_PREDICTIONS_PRIVATE.npz');new={arm:np.full((10,len(d)),np.nan) for arm in cfg['arms']};members=pd.read_csv(P/'REPEATED_MEMBERSHIP_PRIVATE.csv')
    for i,(tridx,teidx) in enumerate(RepeatedStratifiedKFold(n_splits=3,n_repeats=10,random_state=42).split(d,d.y)):
        repeat=i//3;fold=i%3;tr=d.iloc[tridx];te=d.iloc[teidx]
        mem=members[members.repeat.eq(repeat+1)&members.fold.eq(fold+1)]
        assert set(te.participant_key)==set(mem.loc[mem.role.eq('outer_test'),'participant_key'])
        assert set(tr.participant_key)==set(mem.loc[mem.role.eq('outer_train'),'participant_key'])
        for arm,cols in cfg['arms'].items():
            new[arm][repeat,teidx]=build_predict(tr,te,cols,[],spec['classifier'])[0];base_count+=3;prob_count+=len(te)
    for arm,pp in new.items():
        assert np.isfinite(pp).all();err=float(np.max(abs(pp-original[arm.replace('+','_')])));rows.append(dict(analysis='primary',arm=arm,stage='ten_repeated_outer_partitions',probabilities=pp.size,maximum_probability_error=err,maximum_logistic_parameter_error=np.nan,maximum_sigmoid_parameter_error=np.nan))
    assert before=={str(p.relative_to(S)):sha(p) for p in S.rglob('*') if p.is_file()}
    frame=pd.DataFrame(rows);frame.to_csv(a.output/'LOCKED_PREDICTION_REPRODUCTION.csv',index=False);ok=bool(frame.maximum_probability_error.le(a.tolerance).all());r={'status':'PASS' if ok else 'MISMATCH','scope':'Isolated deterministic reproduction of the locked fits from frozen cohort artifacts, not raw source ingestion or independent validation. Original source/predictions unchanged.','base_logistic_models_rebuilt':base_count,'sigmoid_calibrators_rebuilt':base_count,'individual_probabilities_compared':prob_count,'maximum_probability_error':float(frame.maximum_probability_error.max()),'prediction_tolerance':a.tolerance,'protected_source_files':len(before),'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__}}
    (a.output/'LOCKED_REPRODUCTION_REPORT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if not ok:raise AssertionError('Predictions did not reproduce at declared tolerance')

if __name__=='__main__':main()
