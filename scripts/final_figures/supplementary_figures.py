"""Supplementary figures S1-S3 in the same style as fig_v3.py. Saved predictions only; no model fitting.
python figS_v3.py --source <figsrc> --synthetic <synthetic_plot_summary.csv> --out <dir>"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from scipy.special import expit, logit

ap = argparse.ArgumentParser(); ap.add_argument('--source', type=Path, required=True)
ap.add_argument('--synthetic', type=Path, required=True); ap.add_argument('--out', type=Path, required=True)
a = ap.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Liberation Serif', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'axes.linewidth': 0.7, 'xtick.major.width': 0.7, 'ytick.major.width': 0.7, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
BLUE, ORANGE, INK, INK2 = '#2a78d6', '#eb6834', '#111111', '#4a4a4a'
S = a.source; A = pd.read_csv(S / '03_results/ALL_PAIRED_RESULTS.csv', float_precision='round_trip'); P = S / '01_private'
arms = ['Snapshot', 'Snapshot+DeltaMRI']; labels = ['Snapshot', 'Snapshot + ΔMRI']; cols = [BLUE, ORANGE]

def res(analysis, arm, metric):
    return A[(A.analysis == analysis) & (A.comparison == arm) & (A.metric == metric)].iloc[0]

def save(fig, name):
    for ext in ('pdf', 'png', 'tiff'):
        kw = {'dpi': 600} if ext != 'pdf' else {}
        if ext == 'tiff': kw['pil_kwargs'] = {'compression': 'tiff_lzw'}
        fig.savefig(a.out / f'{name}.{ext}', facecolor='white', **kw)
    plt.close(fig)

def plabel(ax, s):
    ax.text(-0.16, 1.06, s, transform=ax.transAxes, fontsize=10, weight='bold', va='bottom', ha='left', color=INK)

# ---------------------------------------------------------------- Figure S1: calibration, three equal-size bins
fig, axs = plt.subplots(3, 2, figsize=(160 / 25.4, 200 / 25.4))
fig.subplots_adjust(left=.11, right=.97, bottom=.06, top=.95, hspace=.48, wspace=.38); bins = []
for row, (analysis, title) in enumerate([('primary', 'Primary model'), ('five_covariate', 'Five covariates'), ('APOE', 'Five covariates + APOE ε4')]):
    d = pd.read_csv(P / f'{analysis}_predictions.csv', float_precision='round_trip')
    for col, (arm, label, c) in enumerate(zip(arms, labels, cols)):
        ax = axs[row, col]; p = d[arm].to_numpy(); y = d.y.to_numpy(); order = np.argsort(p, kind='stable')
        ia = res(analysis, arm, 'calibration_intercept'); sb = res(analysis, arm, 'calibration_slope')
        ax.plot([0, 1], [0, 1], ls=(0, (1, 1.5)), color=INK2, lw=.7)
        grid = np.linspace(p.min(), p.max(), 100); ax.plot(grid, expit(ia.estimate + sb.estimate * logit(grid)), color=c, lw=1.2)
        for k, idx in enumerate(np.array_split(order, 3), 1):
            n = len(idx); ev = int(y[idx].sum()); mean = p[idx].mean(); obs = ev / n; z = 1.95996398454; den = 1 + z * z / n
            cen = (obs + z * z / (2 * n)) / den; half = z * np.sqrt(obs * (1 - obs) / n + z * z / (4 * n * n)) / den
            lo, hi = max(0., min(cen - half, obs)), min(1., max(cen + half, obs))
            ax.errorbar(mean, obs, yerr=[[obs - lo], [hi - obs]], xerr=[[mean - p[idx].min()], [p[idx].max() - mean]],
                        fmt='o', color=c, mfc='white', mew=1.1, capsize=2, ms=4.2, lw=.9, zorder=3)
            bins.append(dict(analysis=analysis, arm=arm, bin=k, N=n, events=ev, mean_prediction=mean, minimum_prediction=p[idx].min(),
                             maximum_prediction=p[idx].max(), observed_fraction=obs, Wilson_lower95=lo, Wilson_upper95=hi))
        ax.set(xlim=(-.025, 1.025), ylim=(-.04, 1.04), xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel='Predicted probability', ylabel='Observed AD fraction')
        plabel(ax, '(' + 'ABCDEF'[row * 2 + col] + ')')
        ax.text(-0.16 + 0.11, 1.06, f'{title}: {label}', transform=ax.transAxes, fontsize=8.5, va='bottom', color=INK)
        ax.text(0.03, 0.97, f'Intercept {ia.estimate:.2f}\nSlope {sb.estimate:.2f} (95% CI {sb.lower95:.2f} to {sb.upper95:.2f})',
                transform=ax.transAxes, fontsize=7.4, va='top', ha='left', color=INK2, linespacing=1.3)
save(fig, 'FigureS1_calibration'); pd.DataFrame(bins).to_csv(a.out / 'calibration_bins_check.csv', index=False, float_format='%.17g')

# ---------------------------------------------------------------- Figure S2: five-covariate ROC and PR
q = pd.read_csv(P / 'five_covariate_predictions.csv', float_precision='round_trip')
fig, axs = plt.subplots(1, 2, figsize=(170 / 25.4, 80 / 25.4)); fig.subplots_adjust(left=.09, right=.98, bottom=.16, top=.90, wspace=.32)
axs[0].plot([0, 1], [0, 1], ls=(0, (1, 1.5)), color=INK2, lw=.7); axs[1].axhline(q.y.mean(), ls=(0, (1, 1.5)), color=INK2, lw=.7)
axs[1].text(0.99, q.y.mean() + .02, f'Event rate {q.y.mean():.2f}', ha='right', va='bottom', fontsize=7.5, color=INK2)
for arm, label, c in zip(arms, labels, cols):
    f, t, _ = roc_curve(q.y, q[arm]); axs[0].plot(f, t, color=c, lw=1.5, label=f'{label} (AUC {res("five_covariate", arm, "roc_auc").estimate:.3f})')
    pr, rc, _ = precision_recall_curve(q.y, q[arm]); axs[1].step(rc, pr, where='post', color=c, lw=1.5, label=f'{label} (AP {res("five_covariate", arm, "average_precision").estimate:.3f})')
for ax, lab, xl, yl in [(axs[0], '(A)', 'False-positive rate', 'True-positive rate'), (axs[1], '(B)', 'Recall', 'Precision')]:
    ax.set(xlabel=xl, ylabel=yl, xlim=(0, 1), ylim=(0, 1.02), xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6)); ax.set_aspect('equal')
    ax.legend(frameon=False, loc='lower right' if lab == '(A)' else 'lower left', handlelength=1.6); plabel(ax, lab)
save(fig, 'FigureS2_clinical_secondary')

# ---------------------------------------------------------------- Figure S3: synthetic control
syn = pd.read_csv(a.synthetic); wins = ['3-18', '6-24', '12-30']; xs = np.arange(3); wl = ['3–18', '6–24', '12–30']
def g(w, m): r = syn[(syn.window == w) & (syn.metric == m)].iloc[0]; return r['mean'], r['sd']
fig, axs = plt.subplots(2, 2, figsize=(160 / 25.4, 130 / 25.4)); fig.subplots_adjust(left=.11, right=.98, bottom=.10, top=.94, hspace=.55, wspace=.35)
for ax, lab, m, yl in zip(axs.flat[:3], ['(A)', '(B)', '(C)'], ['AUC', 'AP', 'Brier'], ['AUC', 'Average precision', 'Brier score']):
    for k, (arm, c, mk, off) in enumerate([('Snapshot', BLUE, 'o', -.09), ('Snapshot_DeltaMRI', ORANGE, '^', .09)]):
        mv = [g(w, f'{arm}_{m}') for w in wins]
        ax.errorbar(xs + off, [v[0] for v in mv], yerr=[v[1] for v in mv], fmt=mk, color=c, ms=5, capsize=2, lw=.9, label=labels[k])
    ax.set(xticks=xs, xticklabels=wl, xlabel='Simulated window (months)', ylabel=yl, xlim=(-.45, 2.45)); plabel(ax, lab)
    if m == 'AUC': ax.legend(frameon=False, loc='center', bbox_to_anchor=(0.5, 0.56))
ax = axs.flat[3]; mv = [g(w, 'DeltaAUC') for w in wins]
ax.errorbar(xs, [v[0] for v in mv], yerr=[v[1] for v in mv], fmt='s', color=INK, ms=4.5, capsize=2, lw=.9)
ax.set(xticks=xs, xticklabels=wl, xlabel='Simulated window (months)', ylabel='ΔAUC', xlim=(-.45, 2.45), ylim=(0, None)); plabel(ax, '(D)')
save(fig, 'FigureS3_synthetic_summary')
print('ok')
