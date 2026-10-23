"""Build/freeze the approved primary cohort. No model fitting or performance."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,warnings
sys.path.insert(0,str(Path('work/python_packages').resolve()))
import pandas as pd,numpy as np,pyreadr
from sklearn.model_selection import GroupShuffleSplit
ROOT=Path.cwd();OUT=ROOT/'outputs/step2A';SRC=ROOT/'work/sources';PRIVATE=ROOT/'work/step2A_private'
specfile=OUT/'02_corrections/APPROVED_TEMPORAL_SPECIFICATION.json'
spec=json.loads(specfile.read_text(encoding='utf8'))
assert spec['horizon_calendar_months']==24 and spec['diagnosis_hierarchy']['fallback_max_days']==180
assert not (OUT/'03_corrected_primary/corrected_primary_metrics.csv').exists()
source=SRC/'CSBJ_revision/adni synthetic paper/adnipocfinal/csv'
snap=pd.read_csv(source/'filtered_snapshot.csv');pairs=pd.read_csv(source/'filtered_mridelta.csv')
snap['subjectid']=snap.subjectid.astype(str);pairs['subjectid']=pairs.subjectid.astype(str)
pairs['source_row']=np.arange(len(pairs));pairs['dist_to_mid']=(pairs.interval_months-10.5).abs()
eligible=pairs[pairs.interval_months.between(3,18)].copy()
eligible=eligible[eligible.subjectid.isin(snap.subjectid)]
whole=eligible.sort_values(['subjectid','dist_to_mid','interval_months','examdate_base','examdate_follow','source_row'],kind='stable').drop_duplicates('subjectid')
# Retain original snapshot first-occurrence row order for training/calibration continuity.
order=snap.drop_duplicates('subjectid').subjectid
whole=whole.set_index('subjectid').reindex(order[order.isin(whole.subjectid)]).reset_index()
assert len(whole)==1474 and whole.subjectid.is_unique
fsfile=next(SRC.rglob('UCSFFSX7_24Sep2026.csv'))
fs=pd.read_csv(fsfile,low_memory=False)
fs['subjectid']=fs.PTID.apply(lambda x:str(int(''.join(c for c in str(x) if c.isdigit()))) if pd.notna(x) else '')
fs['date']=pd.to_datetime(fs.EXAMDATE,errors='coerce');fg=dict(tuple(fs.groupby(['subjectid','date'],sort=False)))
def load_rda(name):
 path=next(SRC.rglob(name+'.rda'))
 with warnings.catch_warnings():
  warnings.filterwarnings('ignore',message='invalid value encountered in cast')
  obj=pyreadr.read_r(str(path))
 return path,next(d for d in obj.values() if isinstance(d,pd.DataFrame)).copy()
dxfile,dx=load_rda('DXSUM');dx['date']=pd.to_datetime(dx.EXAMDATE,errors='coerce')
raw_dx_N=len(dx);qc_dx_N=int(dx.HAS_QC_ERROR.eq('Has QC error').sum())
dx=dx[dx.date.notna()&dx.DIAGNOSIS.notna()&dx.HAS_QC_ERROR.ne('Has QC error')].copy()
dx['ad_evidence']=dx.DIAGNOSIS.eq('Dementia')&(dx.DXAD.eq('Yes')|dx.DXDDUE.eq("Dementia due to Alzheimer's Disease"))
dx['other_evidence']=dx.DIAGNOSIS.eq('Dementia')&dx.DXDDUE.eq('Dementia due to other etiology')
def cat(g):
 cats=set(g.DIAGNOSIS.dropna())
 if len(cats)!=1:return 'CONFLICT'
 name=next(iter(cats))
 if name=='Dementia':
  ad=g.ad_evidence.any();other=g.other_evidence.any()
  if ad and other:return 'CONFLICT'
  return 'AD' if ad else 'OTHER_UNKNOWN_DEMENTIA'
 return name
canonical=[]
for (rid,date),g in dx.groupby(['RID','date'],observed=True,sort=False):
 canonical.append({'RID':rid,'date':date,'category':cat(g),'known_dementia':bool(g.DIAGNOSIS.eq('Dementia').any())})
timeline=pd.DataFrame(canonical).sort_values(['RID','date']);tg=dict(tuple(timeline.groupby('RID',sort=False)))
dg=dict(tuple(dx.groupby('RID',observed=True,sort=False)))
deaths={};deathflags=set();mortalityfiles=[];deathframes=[]
for name,flag in [('ADVERSE','SAEDEATH'),('RECADV','AEHDEATH')]:
 path,d=load_rda(name);mortalityfiles.append(path)
 d=d[d[flag].eq('Yes')|d.AEHDTHDT.notna()].copy();deathflags.update(d.RID)
 d['death']=pd.to_datetime(d.AEHDTHDT,errors='coerce');deathframes.append(d[['RID','death']].dropna())
path,d=load_rda('STUDYSUM');mortalityfiles.append(path);deathflags.update(d.loc[d.SDPRIMARY.eq('Death'),'RID'])
deaths=pd.concat(deathframes).groupby('RID').death.min().to_dict()
def norm(v):
 return str(v).strip().lower() if pd.notna(v) and str(v).strip().lower() not in ['','nan','-4'] else None
def scan_record(row,endpoint):
 date=pd.to_datetime(getattr(row,'examdate_'+endpoint));g=fg.get((row.subjectid,date))
 result={'date':date,'volume':np.nan,'RID':np.nan,'qc_fail':False,'visit':None,'source_rows':0,'visit_ambiguous':False}
 if g is None:return result
 m=np.ones(len(g),bool)
 for old,new in [('st29sv','ST29SV'),('st37sv','ST37SV'),('st149sv','ST149SV')]:
  m&=np.isclose(g[new],getattr(row,old+'_'+endpoint),rtol=0,atol=1e-6,equal_nan=True)
 g=g[m];result['source_rows']=len(g)
 vals=g.ST24CV.dropna().unique();rids=g.RID.dropna().unique()
 if len(vals)==1 and vals[0]>0:result['volume']=float(vals[0])
 if len(rids)==1:result['RID']=rids[0]
 result['qc_fail']=bool(g[['OVERALLQC','HIPPOQC','TEMPQC','VENTQC']].eq('Fail').any().any())
 visits={(norm(x.PHASE),norm(x.VISCODE2),norm(x.VISCODE)) for x in g.itertuples()}
 if len(visits)==1:result['visit']=next(iter(visits))
 else:result['visit_ambiguous']=bool(len(visits)>1)
 return result
def diagnosis_at(scan):
 result={'category':'UNAVAILABLE','method':'unavailable','gap_days':np.nan,'date':pd.NaT,'future_same_visit_rows_ignored':0}
 g=dg.get(scan['RID']);t=tg.get(scan['RID'])
 if g is None:return result
 same=None;visit=scan['visit'];date=scan['date']
 if visit is not None:
  phase,v2,v1=visit;phaseok=g.COLPROT.astype(str).str.strip().str.lower().eq(phase)
  if v2 is not None:
   same=g[phaseok&g.VISCODE2.map(norm).eq(v2)]
  elif v1 is not None:
   same=g[phaseok&g.VISCODE.map(norm).eq(v1)]
 if same is not None:
  result['future_same_visit_rows_ignored']=int((same.date>date).sum())
  same=same[same.date<=date]
  if len(same):
   cats=set(same.groupby('date',sort=False).apply(cat,include_groups=False))
   diagnostic_cat=next(iter(cats)) if len(cats)==1 else 'CONFLICT'
   latest=same.date.max()
   # A same-date cross-visit conflict invalidates rather than silently resolves status.
   if t[t.date.eq(latest)].category.eq('CONFLICT').any():diagnostic_cat='CONFLICT'
   return dict(result,category=diagnostic_cat,method='same_visit',date=latest,gap_days=int((date-latest).days))
 past=t[t.date<=date]
 if len(past):
  latest=past.iloc[-1];gap=int((date-latest.date).days)
  result.update(gap_days=gap,date=latest.date)
  if gap<=180:result.update(category=latest.category,method='fallback_le180')
  else:result.update(method='preceding_diagnosis_stale')
 return result
def outcome(t,info,death,undated):
 end=info+pd.DateOffset(months=24)
 if t is None:return 'insufficient_followup',pd.NaT,pd.NaT
 post=t[t.date>info].copy()
 if pd.notna(death):post=post[post.date<=death]
 dementias=post[post.category.isin(['AD','OTHER_UNKNOWN_DEMENTIA','CONFLICT'])]
 first=dementias.iloc[0] if len(dementias) else None
 if first is not None and first.date<=end:
  return ('AD_event' if first.category=='AD' else 'other_unknown_dementia_or_conflict'),first.date,pd.NaT
 if pd.notna(death) and death<=end:return 'death_before_horizon',pd.NaT,pd.NaT
 if undated:return 'undated_death',pd.NaT,pd.NaT
 confirm=post[(post.date>=end)&post.category.isin(['CN','MCI'])]
 if first is not None:confirm=confirm[confirm.date<first.date]
 if len(confirm):return 'non_event_confirmed',pd.NaT,confirm.date.min()
 return 'insufficient_followup',pd.NaT,pd.NaT
records=[]
for i,row in enumerate(whole.itertuples(index=False)):
 b=scan_record(row,'base');s=scan_record(row,'follow');bd=diagnosis_at(b);sd=diagnosis_at(s)
 rid=b['RID'];assert (pd.isna(rid) and pd.isna(s['RID'])) or rid==s['RID']
 t=tg.get(rid);death=deaths.get(rid,pd.NaT);undated=rid in deathflags and pd.isna(death)
 prior=bool(t is not None and t[(t.date<=s['date'])].known_dementia.any())
 conflict_prior=bool(t is not None and t[(t.date<=s['date'])].category.eq('CONFLICT').any())
 reverted=bool(t is not None and t[(t.date>b['date'])&(t.date<s['date'])].category.eq('CN').any())
 state,ev,conf=outcome(t,s['date'],death,undated)
 rec={'participant_key':f'P{i+1:06d}','SubjectID':row.subjectid,'RID':rid,'source_row':row.source_row,'baseline_date':b['date'],'information_date':s['date'],'horizon_date':s['date']+pd.DateOffset(months=24),'interval_months':row.interval_months,'hippo_base':row.st29sv_base,'hippo_follow':row.st29sv_follow,'vent_base':row.st37sv_base,'vent_follow':row.st37sv_follow,'entorh_base':b['volume'],'entorh_follow':s['volume'],'source_recoverable':bool(np.isfinite(b['volume'])&np.isfinite(s['volume'])&pd.notna(rid)),'documented_QC_failure':b['qc_fail'] or s['qc_fail'],'baseline_MCI':bd['category']=='MCI','information_MCI':sd['category']=='MCI','prior_dementia':prior,'prior_diagnosis_conflict':conflict_prior,'death_before_information':pd.notna(death) and death<=s['date'],'intervening_CN':reverted,'outcome_state':state,'event_date':ev,'confirmatory_non_event_date':conf,'death_date':death,'undated_death':undated}
 for prefix,diagnosis,scan in [('baseline',bd,b),('information',sd,s)]:
  for k,v in diagnosis.items():rec[prefix+'_diagnosis_'+k]=v
  rec[prefix+'_MRI_visit_ambiguous']=scan['visit_ambiguous']
  rec[prefix+'_MRI_source_rows']=scan['source_rows']
 records.append(rec)
c=pd.DataFrame(records);c['y']=c.outcome_state.map({'AD_event':1,'non_event_confirmed':0})
flow=[];active=pd.Series(True,index=c.index)
def step(name,criterion):
 global active
 before=int(active.sum());excluded=active&~criterion;active=active&criterion
 flow.append({'step':name,'entering_N':before,'excluded_N':int(excluded.sum()),'remaining_N':int(active.sum())})
step('Original filtered snapshot-joined whole-row selected pairs',pd.Series(True,index=c.index))
step('Correct left entorhinal recoverable at both selected scans',c.source_recoverable)
step('Exclude independently documented selected-scan QC Fail',~c.documented_QC_failure)
step('Exclude known dementia at/before information time',~c.prior_dementia)
step('Exclude known dated death at/before information time',~c.death_before_information)
step('Require approved current baseline-MRI MCI diagnosis',c.baseline_MCI)
step('Require approved current second-MRI MCI diagnosis',c.information_MCI)
c['riskset_eligible']=active
step('Require classified 24-calendar-month future binary outcome',c.y.notna())
c['analytic_eligible']=active
analytic=c[active].copy().reset_index(drop=True);analytic.y=analytic.y.astype(int)
for region in ['hippo','entorh','vent']:
 analytic[region+'_relchange_emp']=(analytic[region+'_follow']-analytic[region+'_base'])/analytic[region+'_base']/(analytic.interval_months/12)
 analytic[region+'_relchange_emp_isna']=analytic[region+'_relchange_emp'].isna().astype(int)
assert np.isfinite(analytic[['hippo_base','entorh_base','vent_base','hippo_relchange_emp','entorh_relchange_emp','vent_relchange_emp']].to_numpy()).all()
assert analytic.SubjectID.is_unique
train,test=next(GroupShuffleSplit(n_splits=1,train_size=.7,random_state=42).split(analytic,analytic.y,analytic.SubjectID))
analytic['split']='';analytic.loc[train,'split']='train';analytic.loc[test,'split']='test'
assert set(analytic.loc[train,'SubjectID']).isdisjoint(analytic.loc[test,'SubjectID'])
assert analytic.loc[train,'y'].nunique()==2 and analytic.loc[test,'y'].nunique()==2
diagcounts=[]
for pool,mask in [('selected_pairs',pd.Series(True,index=c.index)),('approved_MCI_riskset',c.riskset_eligible),('classified_primary',c.analytic_eligible)]:
 for time in ['baseline','information']:
  for (method,category),n in c[mask].groupby([time+'_diagnosis_method',time+'_diagnosis_category']).size().items():
   diagcounts.append({'pool':pool,'MRI_time':time,'method':method,'diagnosis':category,'N':int(n)})
pd.DataFrame(diagcounts).to_csv(OUT/'02_corrections/approved_diagnosis_timing_counts.csv',index=False)
pd.DataFrame(flow).to_csv(OUT/'02_corrections/corrected_cohort_flow.csv',index=False)
outcomes=c[c.riskset_eligible].outcome_state.value_counts().rename_axis('outcome_state').reset_index(name='N')
outcomes.to_csv(OUT/'02_corrections/approved_outcome_classification_counts.csv',index=False)
summary=[]
for label,g in [('total',analytic),('train',analytic[analytic.split.eq('train')]),('test',analytic[analytic.split.eq('test')])]:
 summary.append({'population':label,'N':len(g),'events':int(g.y.sum()),'non_events':int((g.y==0).sum()),'intervening_CN':int(g.intervening_CN.sum()),'evidentiary_status':'exploratory'})
pd.DataFrame(summary).to_csv(OUT/'03_corrected_primary/corrected_primary_cohort_summary.csv',index=False)
metadata={'freeze_utc':datetime.now(timezone.utc).isoformat(),'status':'COHORT FLOW AND SPLIT FROZEN BEFORE CORRECTED MODEL FITTING','specification_sha256':hashlib.sha256(specfile.read_bytes()).hexdigest(),'selected_pair_N':len(c),'riskset_N':int(c.riskset_eligible.sum()),'unclassified_riskset_N':int((c.riskset_eligible&c.y.isna()).sum()),'analytic_N':len(analytic),'summary':summary,'raw_DXSUM_N':raw_dx_N,'explicit_DXSUM_QC_error_rows_excluded':qc_dx_N,'conflicting_diagnosis_dates_N':int(timeline.category.eq('CONFLICT').sum()),'selected_pairs_prior_dementia_N':int(c.prior_dementia.sum()),'selected_pairs_baseline_no_current_MCI_N':int((~c.baseline_MCI).sum()),'selected_pairs_information_no_current_MCI_N':int((~c.information_MCI).sum()),'riskset_intervening_CN_N':int((c.riskset_eligible&c.intervening_CN).sum()),'analytic_intervening_CN_N':int(analytic.intervening_CN.sum()),'baseline_future_same_visit_rows_ignored':int(c.baseline_diagnosis_future_same_visit_rows_ignored.sum()),'information_future_same_visit_rows_ignored':int(c.information_diagnosis_future_same_visit_rows_ignored.sum()),'baseline_MRI_visit_ambiguous_N':int(c.baseline_MRI_visit_ambiguous.sum()),'information_MRI_visit_ambiguous_N':int(c.information_MRI_visit_ambiguous.sum()),'selected_pairs_QC_failure_N':int(c.documented_QC_failure.sum()),'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'filtered_snapshot.csv',source/'filtered_mridelta.csv',fsfile,dxfile,*mortalityfiles]},'corrected_models_fitted':0}
c.to_pickle(PRIVATE/'approved_cohort_audit_PRIVATE.pkl');analytic.to_pickle(PRIVATE/'approved_primary_cohort_PRIVATE.pkl')
metadata['private_cohort_sha256']=hashlib.sha256((PRIVATE/'approved_primary_cohort_PRIVATE.pkl').read_bytes()).hexdigest()
metadata['cohort_flow_sha256']=hashlib.sha256((OUT/'02_corrections/corrected_cohort_flow.csv').read_bytes()).hexdigest()
(OUT/'06_logs/approved_cohort_freeze.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print(pd.DataFrame(flow).to_string(index=False));print(pd.DataFrame(summary).to_string(index=False));print(outcomes.to_string(index=False))
print('Aggregate timing',pd.DataFrame(diagcounts).to_string(index=False))
