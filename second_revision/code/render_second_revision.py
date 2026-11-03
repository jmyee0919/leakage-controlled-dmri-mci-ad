"""Recreate new second-revision analytical figures from verified inputs.

Figure 3 uses public curve coordinates and aggregates only. Figure S1 requires
the authorized frozen primary cohort; its scatter coordinates are not separately
redistributed. Archived publication PDFs preserve the exact submitted styling.
Older supplementary calibration/clinical/synthetic plots remain reproducible
with scripts/render_figures.py and public aggregate inputs (see README).
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'second_revision/Data_file_S1'
COLORS=['#2a78d6','#eb6834','#1b4f91','#a64414']

def save(fig,out,name):
    fig.savefig(out/(name+'.pdf'),facecolor='white')
    fig.savefig(out/(name+'.png'),dpi=180,facecolor='white')
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--primary',type=Path);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Liberation Serif','DejaVu Serif'],
      'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    h=pd.read_csv(DATA/'Table_2_3_four_model_holdout.csv')
    r=pd.read_csv(DATA/'Table_S11_four_model_each_repeat.csv');r=r[r.analysis.eq('imaging')]
    fig=plt.figure(figsize=(10,12),layout='constrained');gs=fig.add_gridspec(3,3,height_ratios=[1,1,1])
    axs=[fig.add_subplot(gs[0,i]) for i in range(3)]
    for ax,arms,label in [(axs[0],['M1','M2'],'A'),(axs[1],['M3','M4'],'B')]:
        for arm in arms:
            q=pd.read_csv(ROOT/f'second_revision/figure_source/{arm}_ROC.csv')
            auc=h.loc[h.analysis.eq('imaging')&h.comparison.eq(arm)&h.metric.eq('roc_auc'),'estimate'].iloc[0]
            ax.plot(q.false_positive_rate,q.true_positive_rate,color=COLORS[int(arm[1])-1],label=f'{arm} ({auc:.3f})')
        ax.plot([0,1],[0,1],':',color='.4');ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − specificity',ylabel='Sensitivity')
        ax.legend(loc='lower right',fontsize=8);ax.text(-.15,1.03,f'({label})',transform=ax.transAxes,fontweight='bold')
    ax=axs[2]
    for i,comp in enumerate(['M2-M1','M3-M1','M4-M3']):
        vals=r['dAUC_'+comp].to_numpy();ax.scatter(np.full(10,i)+np.linspace(-.1,.1,10),vals,s=17,color=COLORS[i+1])
        ax.plot([i-.2,i+.2],[vals.mean()]*2,color='black');ax.text(i,.059,f'{vals.mean():+.4f}',ha='center')
    ax.axhline(0,color='.4',ls=':');ax.set(xticks=[0,1,2],xticklabels=['M2 − M1','M3 − M1','M4 − M3'],ylabel='Paired ΔAUC, repeated CV',ylim=(-.02,.065))
    ax.text(-.15,1.03,'(C)',transform=ax.transAxes,fontweight='bold')
    ax=fig.add_subplot(gs[1:,:]);sec=pd.read_csv(DATA/'Table_4_secondary_sensitivity_paired.csv')
    # Use the public publication forest inputs, whose estimates are independently audited.
    f=pd.read_csv(ROOT/'data/figure_inputs/ALL_PAIRED_RESULTS.csv')
    rows=[]
    for analysis,label in [('primary','Primary M2 − M1'),('five_covariate','Five covariates'),('complete_MMSE','Complete MMSE'),('APOE','APOE added'),('interval_ge6','Interval ≥6 months'),('interval_shared','Shared interval'),('broader_QC','Broader cohort'),('full121_clinical_ICV_QC_missing','All 121, flagged ICV imputed')]:
        q=f[(f.analysis==analysis)&(f.metric=='delta_roc_auc')]
        if len(q):rows.append((label,q.iloc[0].estimate,q.iloc[0].lower95,q.iloc[0].upper95))
    q=h[h.analysis.eq('imaging')&h.comparison.eq('M4-M3')&h.metric.eq('delta_roc_auc')].iloc[0]
    rows.insert(1,('Updated anatomy M4 − M3',q.estimate,q.lower95,q.upper95))
    for y,(label,e,lo,hi) in enumerate(rows):
        ax.errorbar(e,y,xerr=[[e-lo],[hi-e]],fmt='o',color='black',capsize=2)
        ax.text(.32,y,f'{e:+.3f} ({lo:+.3f} to {hi:+.3f})',va='center',fontsize=8)
    ax.axvline(0,color='.4',ls=':');ax.set(yticks=np.arange(len(rows)),yticklabels=[t[0] for t in rows],xlim=(-.3,.58),xlabel='Paired holdout ΔAUC (95% conditional CI)')
    ax.invert_yaxis();ax.text(-.04,1.03,'(D)',transform=ax.transAxes,fontweight='bold');save(fig,a.out,'Figure_3_recreated')
    if a.primary:
        d=pd.read_csv(a.primary,float_precision='round_trip')
        if len(d)!=106 or int(d.y.sum())!=35:raise ValueError('Frozen primary counts differ')
        fig,axs=plt.subplots(2,2,figsize=(10,8),layout='constrained')
        x=d.interval_months
        axs[0,0].hist([x[d.y.eq(0)],x[d.y.eq(1)]],bins=np.arange(3,17,1),stacked=True,color=COLORS[:2],label=['Confirmed non-event (n=71)','Recorded AD (n=35)'],edgecolor='white')
        axs[0,0].legend(fontsize=8);axs[0,0].set_ylabel('Participants')
        for ax,(key,title) in zip(axs.flat[1:],[('hippo','Left hippocampus'),('entorh','Left entorhinal cortex'),('vent','Left lateral ventricle')]):
            for y,marker,color in [(0,'o',COLORS[0]),(1,'^',COLORS[1])]:
                t=d[d.y.eq(y)];ax.scatter(t.interval_months,t[key+'_relchange_emp']*100,s=16,marker=marker,color=color,edgecolor='white',linewidth=.3)
            ax.axhline(0,color='.4',ls=':');ax.set_title(title,loc='left',fontsize=9);ax.set_ylabel('Annualized change (%/year)')
        for i,ax in enumerate(axs.flat):
            ax.axvline(6,color='.5',ls=':');ax.set_xlabel('Inter-scan interval (months)');ax.text(-.12,1.03,'('+chr(65+i)+')',transform=ax.transAxes,fontweight='bold')
        save(fig,a.out,'Figure_S1_recreated')

if __name__=='__main__':main()
