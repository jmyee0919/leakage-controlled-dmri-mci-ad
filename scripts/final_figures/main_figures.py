"""Publication figures 1-3 (CSBJ style): serif type, no in-figure titles, (A)(B)(C) panels, no grid.
python fig_v3.py --source <figsrc> --out <dir>"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrowPatch
from sklearn.metrics import roc_curve

ap = argparse.ArgumentParser(); ap.add_argument('--source', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
INK, INK2, EDGE = '#111111', '#4a4a4a', '#2b3a4a'
BLUE_T, ORANGE_T = '#e3eefb', '#fde9df'

def save(fig, name):
    for ext in ('pdf', 'png', 'tiff'):
        kw = {'dpi': 600} if ext != 'pdf' else {}
        if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(a.out / f'{name}.{ext}', facecolor='white', **kw)
    plt.close(fig)

def box(ax, x, y, w, h, text, fc='white', ec=EDGE, lw=0.8, size=8.2, weight='normal', ha='center'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.012', fc=fc, ec=ec, lw=lw, transform=ax.transAxes))
    tx = x + w / 2 if ha == 'center' else x + 0.018
    ax.text(tx, y + h / 2, text, ha=ha, va='center', fontsize=size, color=INK, weight=weight, transform=ax.transAxes, linespacing=1.25)

def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle='-|>', mutation_scale=7, color=color, lw=lw, transform=ax.transAxes, shrinkA=0, shrinkB=0))

# ------------------------------------------------------------------ Figure 1: landmark design
fig = plt.figure(figsize=(180 / 25.4, 78 / 25.4)); ax = fig.add_axes([0, -0.03, 1, 1.2]); ax.axis('off')
x0, x1, x2, ya = 0.11, 0.40, 0.86, 0.58
ax.add_patch(Rectangle((x0, ya), x1 - x0, 0.085, fc=BLUE_T, ec='none', transform=ax.transAxes))
ax.add_patch(Rectangle((x1, ya), x2 - x1, 0.085, fc=ORANGE_T, ec='none', transform=ax.transAxes))
arrow(ax, 0.04, ya, 0.965, ya, color=INK, lw=1.0)
ax.text(0.965, ya - 0.065, 'Time', ha='right', va='top', fontsize=8, color=INK2, style='italic', transform=ax.transAxes)
ax.text((x0 + x1) / 2, ya + 0.0425, 'Change interval: 3–18 months', ha='center', va='center', fontsize=8, color='#1b4f91', transform=ax.transAxes)
ax.text((x1 + x2) / 2, ya + 0.0425, 'Outcome window: 24 calendar months', ha='center', va='center', fontsize=8, color='#9a3b12', transform=ax.transAxes)
for x, top, sub in [(x0, 'Baseline MRI', 'Recorded MCI'), (x1, 'Second MRI', 'Recorded MCI\nPrediction time (landmark)'), (x2, 'End of horizon', '')]:
    ax.plot([x, x], [ya - 0.04, ya + 0.085], color=INK, lw=1.2, transform=ax.transAxes, solid_capstyle='butt')
    ax.text(x, ya + 0.17, top, ha='center', va='bottom', fontsize=9, weight='bold', color=INK, transform=ax.transAxes)
    if sub: ax.text(x, ya + 0.165, sub, ha='center', va='top', fontsize=7.8, color=INK2, transform=ax.transAxes, linespacing=1.2)
ax.plot([x1, x1], [0.06, ya - 0.05], color=INK2, lw=0.7, ls=(0, (3, 2)), transform=ax.transAxes)
# model rows
yS, yD = 0.33, 0.16
ax.text(0.015, yS, 'Snapshot', ha='left', va='center', fontsize=8.5, weight='bold', color=BLUE, transform=ax.transAxes)
ax.text(0.015, yD, 'Snapshot\n+ ΔMRI', ha='left', va='center', fontsize=8.5, weight='bold', color=ORANGE, transform=ax.transAxes, linespacing=1.1)
ax.plot([x0], [yS], 'o', ms=5, color=BLUE, transform=ax.transAxes)
arrow(ax, x0 + 0.012, yS, x1 - 0.006, yS, color=BLUE, lw=0.9)
ax.text((x0 + x1) / 2, yS + 0.035, 'baseline regional volumes', ha='center', va='bottom', fontsize=7.8, color=INK2, transform=ax.transAxes)
ax.plot([x0, x1], [yD, yD], 'o', ms=5, color=ORANGE, transform=ax.transAxes)
ax.plot([x0, x1], [yD, yD], color=ORANGE, lw=2.2, transform=ax.transAxes, solid_capstyle='butt')
ax.text((x0 + x1) / 2, yD + 0.035, 'baseline volumes + annualized change', ha='center', va='bottom', fontsize=7.8, color=INK2, transform=ax.transAxes)
for y, c in [(yS, BLUE), (yD, ORANGE)]:
    arrow(ax, x1 + 0.006, y, x1 + 0.05, y, color=c, lw=0.9)
    ax.text(x1 + 0.057, y, 'predicted risk', ha='left', va='center', fontsize=7.8, color=INK2, transform=ax.transAxes)
box(ax, 0.60, 0.06, 0.385, 0.36,
    'Outcome classification\n'
    '• Event: first recorded AD diagnosis\n   within the window\n'
    '• Non-event: CN or MCI recorded at or\n   after the horizon, no dementia\n'
    '• Unclassified: excluded (not negative)', fc='white', size=7.6, ha='left')
save(fig, 'Fig1')

# ------------------------------------------------------------------ Figure 2: STROBE-style flow
fig = plt.figure(figsize=(170 / 25.4, 200 / 25.4)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
mx, mw, ex, ew, bh = 0.04, 0.44, 0.555, 0.43, 0.068
cx = mx + mw / 2
steps = [
    ('MRI pairs with a 3–18-month interval after volume-range\nand change-direction filters\n1,474 participants', None),
    ('One pair per participant (nearest 10.5 months)\nn = 1,474', 'Documented source QC failure (n = 5)'),
    ('n = 1,469', 'Dementia recorded at or before the second MRI (n = 424)\nDeath before the second MRI (n = 6)'),
    ('n = 1,039', 'No recorded MCI at baseline MRI (n = 652)\nNo recorded MCI at second MRI (n = 182)'),
    ('MCI risk set\nn = 205', 'Outcome not classifiable (n = 84)\n   insufficient follow-up 79; death 3;\n   other dementia or conflicting records 2'),
    ('Classified cohort\nn = 121 (37 AD events)', 'Post hoc quality control (n = 15; 2 events)\n   Rule A, ICV change > 5% (n = 11)\n   Rule B, regional robust z > 4 (n = 4)'),
    ('Primary cohort\nn = 106 (35 AD events)', None),
]
ys = [0.86, 0.735, 0.61, 0.485, 0.36, 0.235, 0.11]
heights = [0.085, bh, 0.05, 0.05, bh, bh, bh]
# source boxes
box(ax, 0.06, 0.945, 0.43, 0.045, 'Cross-sectional MRI records: 12,151 (3,231 participants)', size=7.6)
box(ax, 0.53, 0.945, 0.44, 0.045, 'Longitudinal MRI pairs: 6,213 (1,850 participants)', size=7.6)
arrow(ax, 0.275, 0.945, cx - 0.04, ys[0] + heights[0] / 2 + 0.002); arrow(ax, 0.75, 0.945, cx + 0.06, ys[0] + heights[0] / 2 + 0.002)
for k, ((txt, exc), y, h) in enumerate(zip(steps, ys, heights)):
    final = k == len(steps) - 1
    box(ax, mx, y - h / 2, mw, h, txt, fc=BLUE_T if final else 'white', ec=BLUE if final else EDGE, lw=1.1 if final else 0.8,
        weight='bold' if k in (4, 5, 6) else 'normal', size=8.2)
    if k < len(steps) - 1:
        yb, yt = y - h / 2, ys[k + 1] + heights[k + 1] / 2
        arrow(ax, cx, yb, cx, yt)
        if exc:  # exclusion leaves the flow after box k, before box k + 1
            ym = (yb + yt) / 2; eh = 0.075 if exc.count('\n') >= 2 else 0.055
            ax.plot([cx, ex], [ym, ym], color=EDGE, lw=0.9, transform=ax.transAxes)
            arrow(ax, ex - 0.03, ym, ex, ym)
            box(ax, ex, ym - eh / 2, ew, eh, exc, fc=ORANGE_T, ec='#c4581f', size=7.6, ha='left')
# split into partitions
yb = ys[-1] - heights[-1] / 2; ysplit = 0.035
ax.plot([cx, cx], [yb, ysplit + 0.03], color=EDGE, lw=0.9, transform=ax.transAxes)
ax.plot([0.2, 0.72], [ysplit + 0.03, ysplit + 0.03], color=EDGE, lw=0.9, transform=ax.transAxes)
for xc, t in [(0.2, 'Training: n = 76 (26 events)'), (0.72, 'Internal holdout: n = 30 (9 events)')]:
    arrow(ax, xc, ysplit + 0.03, xc, ysplit + 0.018)
    box(ax, xc - 0.17, ysplit - 0.022, 0.34, 0.04, t, size=8, weight='bold')
ax.set_ylim(0, 1); save(fig, 'Fig2')

# ------------------------------------------------------------------ Figure 3: performance
S = a.source; A = pd.read_csv(S / '03_results/ALL_PAIRED_RESULTS.csv'); pred = pd.read_csv(S / '01_private/primary_predictions.csv'); rep = pd.read_csv(S / '03_results/REPEATED_VALIDATION.csv')
fig = plt.figure(figsize=(180 / 25.4, 205 / 25.4))
axA = fig.add_axes([0.085, 0.695, 0.36, 0.265]); axB = fig.add_axes([0.6, 0.695, 0.37, 0.265])
for arm, lab, c, ls in [('Snapshot', 'Snapshot', BLUE, '-'), ('Snapshot+DeltaMRI', 'Snapshot + ΔMRI', ORANGE, '-')]:
    f, t, _ = roc_curve(pred.y, pred[arm]); auc = A[(A.analysis == 'primary') & (A.comparison == arm) & (A.metric == 'roc_auc')].estimate.iloc[0]
    axA.step(f, t, where='post', color=c, lw=1.6, ls=ls, label=f'{lab} (AUC {auc:.3f})')
axA.plot([0, 1], [0, 1], color=INK2, lw=0.6, ls=(0, (2, 2)))
axA.set(xlim=(0, 1), ylim=(0, 1), xlabel='1 − specificity', ylabel='Sensitivity', aspect='equal', xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
axA.legend(loc='lower right', frameon=False, handlelength=1.6)
axA.text(-0.2, 1.035, '(A)', transform=axA.transAxes, fontsize=10, weight='bold')
v = rep.delta_roc_auc.to_numpy(); r = np.arange(1, 11)
axB.axhline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
axB.axhline(v.mean(), color=ORANGE, lw=1.0)
axB.plot(r, v, 'o', ms=5.5, color=BLUE, mec='white', mew=0.6)
axB.text(3.3, v.mean() + 0.0012, f'Mean {v.mean():+.3f}', va='bottom', ha='left', fontsize=7.8, color=INK2)
axB.set(xlim=(0.5, 10.5), ylim=(-0.01, 0.055), xticks=r, xlabel='Cross-validation repeat', ylabel='Paired ΔAUC')
axB.text(-0.2, 1.035, '(B)', transform=axB.transAxes, fontsize=10, weight='bold')
# forest
groups = [('Primary analysis', [('primary', 'Imaging model (reference)')]),
          ('Clinical adjustment', [('five_covariate', 'Five covariates'), ('complete_MMSE', 'Five covariates, complete MMSE'), ('APOE', 'Five covariates + APOE ε4')]),
          ('Timing and acquisition', [('excl5', 'Excluding AD ≤ 30 days after MRI'), ('field_strength', 'Field strength added')]),
          ('Interval and anatomy', [('interval_ge6', 'Minimum interval 6 months'), ('interval_shared', 'Interval added to both arms'), ('second_MRI_snapshot', 'Second-MRI anatomy')]),
          ('Cohort definition', [('full121_clinical_ICV_QC_missing', 'All 121, flagged ICV imputed'), ('broader_QC', 'Broader, no range or direction filter'), ('original', 'Original 121, imaging')])]
rows = []
for g, items in groups:
    rows.append(('header', g, None))
    for k, lab in items: rows.append(('item', lab, k))
axC = fig.add_axes([0.42, 0.055, 0.33, 0.53]); n = len(rows)
axC.set_ylim(-0.7, n - 0.3); axC.set_xlim(-0.3, 0.3)
NE = {'primary': '30/9', 'five_covariate': '30/9', 'complete_MMSE': '25/8', 'APOE': '30/9', 'excl5': '29/8', 'field_strength': '30/9', 'interval_ge6': '21/7',
      'interval_shared': '30/9', 'second_MRI_snapshot': '30/9', 'full121_clinical_ICV_QC_missing': '37/10', 'broader_QC': '33/9', 'original': '37/10'}
for i, (kind, lab, key) in enumerate(rows):
    y = n - 1 - i
    if kind == 'header':
        fig.text(0.03, axC.transData.transform((0, y))[1] / fig.dpi / fig.get_size_inches()[1], lab, fontsize=8.3, weight='bold', va='center', transform=fig.transFigure)
        continue
    if key == 'original': e, lo, hi = .0667, -.0127, .1653
    else:
        rr = A[(A.analysis == key) & (A.metric == 'delta_roc_auc')].iloc[0]; e, lo, hi = rr.estimate, rr.lower95, rr.upper95
    c, mk = (BLUE, 's') if key == 'primary' else (ORANGE, 'D') if key == 'original' else (INK, 'o')
    axC.plot([lo, hi], [y, y], color=c, lw=1.1, solid_capstyle='butt'); axC.plot(e, y, mk, ms=5 if mk != 'D' else 4.5, color=c, mec='white', mew=0.5, zorder=3)
    ty = axC.transData.transform((0, y))[1] / fig.dpi / fig.get_size_inches()[1]
    fig.text(0.045, ty, f'{lab}', fontsize=7.9, va='center', color=INK)
    fig.text(0.37, ty, NE[key], fontsize=7.9, va='center', ha='center', color=INK2)
    fig.text(0.985, ty, f'{e:+.3f} ({lo:+.3f} to {hi:+.3f})'.replace('+0.000', '0.000').replace('-', '−'), fontsize=7.9, va='center', ha='right', color=INK)
axC.axvline(0, color=INK2, lw=0.6, ls=(0, (2, 2)))
axC.set(ylim=(-0.7, n - 0.3), yticks=[], xlim=(-0.3, 0.3), xticks=[-0.3, -0.15, 0, 0.15, 0.3], xlabel='Paired ΔAUC (Snapshot + ΔMRI minus Snapshot)')
axC.spines['left'].set_visible(False)
top = axC.transData.transform((0, n - 0.1))[1] / fig.dpi / fig.get_size_inches()[1] + 0.012
fig.text(0.37, top, 'Holdout\nN/events', fontsize=7.6, ha='center', va='bottom', color=INK2, linespacing=1.1)
fig.text(0.985, top, 'ΔAUC (95% CI)', fontsize=7.8, ha='right', va='bottom', color=INK2)
fig.text(0.03, top + 0.035, '(C)', fontsize=10, weight='bold')
save(fig, 'Fig3')
print('Figures 1-3 written to', a.out)
