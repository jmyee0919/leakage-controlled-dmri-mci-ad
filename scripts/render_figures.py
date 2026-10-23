"""Render figures from public aggregate tables and curve coordinates. No model fitting.
python scripts/render_figures.py --output build/figures
"""
from pathlib import Path
import argparse,shutil,hashlib,json
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve,precision_recall_curve
from scipy.special import expit,logit

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
ROOT=Path(__file__).resolve().parents[1];O=args.output.resolve()
if O==ROOT or O in ROOT.parents:raise ValueError('Choose a build subfolder or an external output folder')
F=O/'figures';T=O/'tables';F.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
D=ROOT/'data/figure_inputs'
A=pd.read_csv(D/'ALL_PAIRED_RESULTS.csv',float_precision='round_trip')
rep=pd.read_csv(D/'REPEATED_VALIDATION.csv',float_precision='round_trip')
s21=pd.read_csv(ROOT/'data/aggregate/Table_S21_sensitivity_added_in_revision.csv',float_precision='round_trip')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,'legend.fontsize':8.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
blue='#17658b';orange='#bb522b';gray='#59636c';pale='#edf3f7'
def save(fig,name):
 for ext in ['pdf','svg','png']:
  path=F/f'{name}.{ext}';fig.savefig(path,dpi=600,facecolor='white')
  if ext=='svg':
   # Preserve intentional spaces in scientific labels across SVG renderers.
   s=path.read_text();path.write_text(s.replace('<svg ','<svg xml:space="preserve" ',1))
 plt.close(fig)
def box(ax,x,y,w,h,label,fc=pale,fontsize=9):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.006,rounding_size=0.008',facecolor=fc,edgecolor=gray,lw=.8));ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=fontsize)
def arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=gray,lw=1))
def flow_arrow(ax,x,y,xx,yy):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops=dict(arrowstyle='->',color=gray,lw=1,shrinkA=0,shrinkB=0,mutation_scale=8))
def result(analysis,arm,metric):return A[(A.analysis==analysis)&(A.comparison==arm)&(A.metric==metric)].iloc[0]
arms=['Snapshot','Snapshot+DeltaMRI'];labels=['Snapshot','Snapshot + ΔMRI']

fig,ax=plt.subplots(figsize=(180/25.4,98/25.4));fig.subplots_adjust(left=.02,right=.98,top=.97,bottom=.04);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.96,'Common prediction time at the second MRI',weight='bold',fontsize=11,va='top')
box(ax,.02,.52,.26,.25,'Baseline MRI\nMCI from dated assessment\nRegional anatomy')
box(ax,.37,.52,.27,.25,'Second MRI / index date\nMCI from dated assessment\nChange now available')
box(ax,.73,.52,.25,.25,'Next 24 calendar months\nRecorded AD or\nconfirmed non-event')
arrow(ax,.29,.64,.36,.64);arrow(ax,.65,.64,.72,.64)
ax.text(.325,.82,'3-18 months',ha='center',fontsize=8.5)
box(ax,.04,.11,.92,.27,'Both arms start the outcome horizon at the same index date\nSnapshot: baseline anatomy     Augmented: baseline anatomy + annualized change\nA preceding clinical assessment may be used; MRI date is not an onset date',fc='white',fontsize=8.5)
save(fig,'Figure1_landmark_timeline')

fig,ax=plt.subplots(figsize=(180/25.4,228/25.4));fig.subplots_adjust(left=.02,right=.98,top=.99,bottom=.02);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
ax.text(.02,.985,'Participant flow',fontsize=11,weight='bold',va='top')
box(ax,.025,.835,.45,.095,'Cross-sectional MRI records\n12,151 rows (3,231 participants)\nVolume ranges: 10,279 rows (2,732 participants)',fontsize=8.4)
box(ax,.525,.835,.45,.095,'Longitudinal MRI pairs\n6,213 rows (1,850 participants)\nChange direction: 3,281 rows (1,559 participants)',fontsize=8.4)
flow_arrow(ax,.25,.828,.28,.788);flow_arrow(ax,.75,.828,.46,.788)
box(ax,.03,.70,.68,.08,'Participant-ID join and 3-18-month interval\n3,141 pair rows from 1,474 participants')
flow_arrow(ax,.37,.692,.37,.669)
box(ax,.03,.585,.68,.075,'One complete pair nearest 10.5 months\n1,474 participants')
steps=[('1,469 after documented QC exclusions','5 excluded'),('1,045 without prior recorded dementia','424 excluded'),('1,039 without prior dated death','6 excluded'),('387 with last-known baseline MCI','652 excluded'),('205 with last-known index MCI','182 excluded'),('121 classified outcomes / 37 AD events','84 unclassified*')]
for i,(left,right) in enumerate(steps):
 y=.538-i*.061
 box(ax,.03,y-.017,.68,.034,left,fc='white',fontsize=8.6)
 ax.text(.76,y,right,va='center',fontsize=8.6)
 upper=.579 if i==0 else (.538-(i-1)*.061)-.023
 flow_arrow(ax,.37,upper-.0015,.37,y+.0245)
flow_arrow(ax,.37,.2085,.37,.1695)
box(ax,.03,.09,.68,.072,'106 after post hoc QC / 35 AD events\nTraining 76/26; internal holdout 30/9',fontsize=8.8)
ax.text(.76,.126,'15 excluded (2 events)\nRule A (ICV): 11\nRule B (regional): 4',va='center',fontsize=8.4)
ax.text(.03,.045,'*79 insufficient follow-up, 3 deaths, 2 other dementia or conflicting records.',fontsize=8.3)
save(fig,'Figure2_cohort_flow')


fig=plt.figure(figsize=(180/25.4,226/25.4));gs=fig.add_gridspec(2,2,height_ratios=[1,1.7],hspace=.55,wspace=.42,left=.105,right=.96,top=.95,bottom=.07)
ax=fig.add_subplot(gs[0,0]);ax.set_title('A  Primary internal holdout\n30 participants; 9 AD events',loc='left',weight='bold',fontsize=9)
for a,label,c in zip(arms,labels,[blue,orange]):
 curve=pd.read_csv(D/f'primary_{a.replace("+","_")}_ROC.csv');fpr=curve.false_positive_rate;tpr=curve.true_positive_rate;ax.plot(fpr,tpr,c=c,lw=1.6,label=f'{label}: {result("primary",a,"roc_auc").estimate:.4f}')
ax.plot([0,1],[0,1],':',c=gray,lw=.8);ax.set(xlim=(-.025,1.025),ylim=(-.025,1.04),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6),xlabel='False-positive rate',ylabel='True-positive rate');ax.legend(loc='lower right',frameon=False,fontsize=8)
ax=fig.add_subplot(gs[0,1]);ax.set_title('B  Repeated validation',loc='left',weight='bold');v=rep.delta_roc_auc.to_numpy();ax.axhline(0,c=gray,ls=':',lw=1);ax.scatter(np.arange(1,11),v,c=blue,s=21,zorder=3);ax.axhline(v.mean(),color=orange,ls='--',lw=1)
ax.set(xlabel='Repeat (same 106 participants)',ylabel='Paired AUC difference',xticks=[1,3,5,7,10],ylim=(-.012,.065));ax.text(.03,.97,f'Mean {v.mean():+.4f}; SD {v.std(ddof=1):.4f}\n1 of 10 repeats below zero',transform=ax.transAxes,va='top',fontsize=8)
ax=fig.add_subplot(gs[1,:]);pos=ax.get_position();ax.set_position([.42,pos.y0,.53,pos.height]);fig.text(.035,pos.y1+.045,'C  Paired AUC differences',weight='bold',fontsize=10);fig.text(.035,pos.y1+.018,'Comparison (evaluation N/events)',fontsize=8.5)
items=[('primary','Primary (30/9)'),('five_covariate','Five covariates (30/9)'),('complete_MMSE','Complete MMSE (25/8)'),('APOE','Five covariates + APOE (30/9)'),('interval_ge6','Interval ≥6 months (21/7)'),('interval_shared','Shared interval (30/9)'),('second_MRI_snapshot','Second-MRI Snapshot (30/9)'),('broader_QC','Broader sensitivity (33/9)'),('full121_clinical_ICV_QC_missing','All 121, ICV imputed (37/10)'),('excl5','Excluding AD ≤30 days (29/8)'),('field_strength','Field strength added (30/9)'),('original','Original 121, imaging (37/10)')]
historical=json.loads((D/'HISTORICAL_PRIMARY_SUMMARY.json').read_text())
for i,(name,label) in enumerate(items):
 y=len(items)-1-i
 if name=='original':e,lo,hi=[historical[k] for k in ['delta_AUC','delta_AUC_lower95','delta_AUC_upper95']]
 elif name=='excl5':
  r=s21[s21.analysis=='Excluding AD <=30 days'].iloc[0];e,lo,hi=r.dAUC,r.dAUC_lower95,r.dAUC_upper95
 elif name=='field_strength':
  r=s21[s21.analysis=='Field strength added'].iloc[0];e,lo,hi=r.dAUC,r.dAUC_lower95,r.dAUC_upper95
 else:r=A[(A.analysis==name)&(A.metric=='delta_roc_auc')].iloc[0];e,lo,hi=r.estimate,r.lower95,r.upper95
 ax.errorbar(e,y,xerr=[[e-lo],[hi-e]],fmt='s' if name=='original' else 'o',color=orange if name=='original' else blue if name=='primary' else gray,ms=4,capsize=2,lw=1.1)
 ax.text(-.735,y,label,va='center',fontsize=8.3,clip_on=False)
ax.axvline(0,ls=':',c=gray,lw=1);ax.set(yticks=[],xlim=(-.3,.3),ylim=(-.7,len(items)-.3),xticks=[-.3,-.15,0,.15,.3],xlabel='Augmented minus matched Snapshot AUC');ax.grid(axis='x',alpha=.15)
save(fig,'Figure3_primary_and_sensitivities')

fig,axs=plt.subplots(1,2,figsize=(180/25.4,100/25.4));fig.subplots_adjust(left=.09,right=.98,bottom=.14,top=.88,wspace=.34)
for a,label,c in zip(arms,labels,[blue,orange]):
 curve=pd.read_csv(D/f'five_covariate_{a.replace("+","_")}_ROC.csv');f=curve.false_positive_rate;t=curve.true_positive_rate;axs[0].plot(f,t,color=c,lw=1.6,label=f'{label}: {result("five_covariate",a,"roc_auc").estimate:.4f}')
 curve=pd.read_csv(D/f'five_covariate_{a.replace("+","_")}_PR.csv');prec=curve.precision;rec=curve.recall;axs[1].step(rec,prec,where='post',color=c,lw=1.6,label=f'{label}: {result("five_covariate",a,"average_precision").estimate:.4f}')
axs[0].plot([0,1],[0,1],':',c=gray,lw=.8);axs[1].axhline(9/30,c=gray,ls=':',lw=.8)
for ax,title,x,y in [(axs[0],'A  Adjusted ROC','False-positive rate','True-positive rate'),(axs[1],'B  Adjusted precision–recall','Recall','Precision')]:
 ax.set(title=title,xlabel=x,ylabel=y,xlim=(-.025,1.025),ylim=(-.025,1.04),xticks=np.linspace(0,1,6),yticks=np.linspace(0,1,6));ax.legend(frameon=False,loc='lower left',fontsize=8)
save(fig,'FigureS2_clinical_secondary')

# Re-express existing saved predictions as three descriptive calibration bins.
# No smoother or new clinical model; point intervals are Wilson binomial intervals.
fig,axs=plt.subplots(3,2,figsize=(180/25.4,237/25.4));fig.subplots_adjust(left=.10,right=.97,bottom=.09,top=.95,hspace=.80,wspace=.36);bins=[]
for row,(analysis,title) in enumerate([('primary','Primary'),('five_covariate','Five covariates'),('APOE','Five covariates + APOE')]):
 d=pd.read_csv(ROOT/'data/aggregate/Table_S17_calibration_bins.csv',float_precision='round_trip')
 for col,(arm,label,c) in enumerate(zip(arms,labels,[blue,orange])):
  ax=axs[row,col];subset=d[(d.analysis==analysis)&(d.arm==arm)]
  a=result(analysis,arm,'calibration_intercept');b=result(analysis,arm,'calibration_slope')
  for rr in subset.to_dict('records'):
   mean=rr['mean_prediction'];obs=rr['observed_fraction'];lo=rr['Wilson_lower95'];hi=rr['Wilson_upper95']
   ax.errorbar(mean,obs,yerr=[[obs-lo],[hi-obs]],xerr=[[mean-rr['minimum_prediction']],[rr['maximum_prediction']-mean]],fmt='o',color=c,capsize=2,ms=4,lw=1)
   bins.append(rr)
  grid=np.linspace(subset.minimum_prediction.min(),subset.maximum_prediction.max(),100);ax.plot(grid,expit(a.estimate+b.estimate*logit(grid)),color=c,lw=1.1)
  # Rendering padding exposes complete zero-event markers and Wilson caps.
  # Tick labels and data retain the probability domain [0, 1].
  ax.plot([0,1],[0,1],':',color=gray,lw=.9);ax.set(xlim=(-.025,1.025),ylim=(-.04,1.04),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='Predicted probability',ylabel='Observed AD fraction')
  ax.set_title(f'{"ABCDEF"[row*2+col]}  {title}\n{label}',loc='left',fontsize=9,weight='bold')
  ax.text(0,-.32,f'Intercept {a.estimate:.2f}; slope {b.estimate:.2f}\nSlope 95% CI {b.lower95:.2f} to {b.upper95:.2f}',transform=ax.transAxes,fontsize=8,va='top')
save(fig,'FigureS3_calibration_uncertainty');pd.DataFrame(bins).to_csv(T/'Table_S17_calibration_bins.csv',index=False,float_format='%.17g')
for ext in ['png','pdf']:shutil.copy2(ROOT/f'figures/supplementary/FigureS1_synthetic_summary.{ext}',F/f'FigureS1_synthetic_summary.{ext}')
manifest=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),origin='Unchanged synthetic control' if p.name.startswith('FigureS1') else 'Saved predictions and canonical results; no prognostic refit') for p in sorted(F.iterdir()) if p.suffix in ['.png','.pdf','.svg']]
(F/'FIGURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Five figure sets rendered from aggregate inputs; archived Figure S1 copied; no model refit.')
