#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# KDE
try:
    from scipy.stats import gaussian_kde  # type: ignore
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


# ------------------------------------------------------------
# Density helper
# ------------------------------------------------------------
def density(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    x = x[np.isfinite(x)]
    x = np.clip(x, 0.0, 1.0)

    if x.size < 10:
        return np.zeros_like(grid)

    if HAS_SCIPY:
        return gaussian_kde(x, bw_method="scott")(grid)

    # fallback: smoothed histogram
    hist, edges = np.histogram(x, bins=80, range=(0, 1), density=True)
    centers = (edges[:-1] + edges[1:]) / 2
    k = np.exp(-0.5 * (np.arange(-15, 16) / 3.0) ** 2)
    k /= k.sum()
    sm = np.convolve(hist, k, mode="same")
    return np.interp(grid, centers, sm, left=0, right=0)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poc_root", default="data/processed")
    ap.add_argument("--out_root", default="figures/supplementary")
    ap.add_argument("--dpi", type=int, default=600)

    # Explicit columns (confirmed from your CSV)
    ap.add_argument("--group_col", default="Group")
    ap.add_argument("--pred_snapshot_col", default="snapshot_only_logistic_oof")
    ap.add_argument("--pred_change_col", default="snapshot+change_logistic_oof")
    args = ap.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    merged_dir = Path(args.poc_root) / "실험 결과값들" / "oof 관련" / "merged file"
    files = [
        merged_dir / "predictions_wide_oof_seed42.csv",
        merged_dir / "predictions_wide_seed43.csv",
        merged_dir / "predictions_oof_wide_seed44.csv",
    ]

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)

    # sanity
    for c in [args.group_col, args.pred_snapshot_col, args.pred_change_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}'. Available: {list(df.columns)}")

    group = df[args.group_col].astype(str)
    p_snap = pd.to_numeric(df[args.pred_snapshot_col], errors="coerce").to_numpy()
    p_chg  = pd.to_numeric(df[args.pred_change_col],  errors="coerce").to_numpy()

    m = np.isfinite(p_snap) & np.isfinite(p_chg)
    group = group[m]
    p_snap = p_snap[m]
    p_chg  = p_chg[m]

    # --------------------------------------------------------
    # Determine S2A / S2B automatically by probability level
    # --------------------------------------------------------
    means = (
        pd.DataFrame({"group": group, "p": p_chg})
        .groupby("group")["p"]
        .mean()
        .sort_values(ascending=False)
    )

    if len(means) < 2:
        raise RuntimeError(
            f"Need ≥2 groups in '{args.group_col}'. Found: {means.index.tolist()}"
        )

    # Highest-prob group = converters (S2A)
    conv_group = means.index[0]
    non_group  = means.index[1]

    mask_conv = (group == conv_group).to_numpy()
    mask_non  = (group == non_group).to_numpy()

    # --------------------------------------------------------
    # Density estimation
    # --------------------------------------------------------
    grid = np.linspace(0, 1, 500)

    # S2A
    y1a = density(p_snap[mask_conv], grid)
    y2a = density(p_chg [mask_conv], grid)
    # S2B
    y1b = density(p_snap[mask_non], grid)
    y2b = density(p_chg [mask_non], grid)

    # --------------------------------------------------------
    # Plot (stacked like your reference)
    # --------------------------------------------------------
    plt.figure(figsize=(9.5, 9.0))

    axA = plt.subplot(2, 1, 1)
    axB = plt.subplot(2, 1, 2)

    # ---- S2A ----
    axA.plot(grid, y1a, label="snapshot_only_logistic_oof")
    axA.plot(grid, y2a, label="snapshot+change_logistic_oof")
    axA.set_xlabel("Predicted probability")
    axA.set_ylabel("Density")
    axA.set_xlim(0, 1)
    axA.grid(True, alpha=0.3)
    axA.legend(loc="upper left", frameon=True)

    # ---- S2B ----
    axB.plot(grid, y1b, label="snapshot_only_logistic_oof")
    axB.plot(grid, y2b, label="snapshot+change_logistic_oof")
    axB.set_xlabel("Predicted probability")
    axB.set_ylabel("Density")
    axB.set_xlim(0, 1)
    axB.grid(True, alpha=0.3)
    axB.legend(loc="upper right", frameon=True)

    # --------------------------------------------------------
    # Panel labels: bottom-left, bold, readable
    # --------------------------------------------------------
    label_kw = dict(
        fontsize=22,
        fontweight="bold",
        color="black",
        ha="left",
        va="bottom",
        zorder=10,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2.5),
    )

    axA.text(0.02, 0.04, "S1A", transform=axA.transAxes, **label_kw)
    axB.text(0.02, 0.04, "S1B", transform=axB.transAxes, **label_kw)

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    out_pdf = out_root / "SuppFig_S2.pdf"
    out_jpg = out_root / "SuppFig_S2.jpg"

    plt.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.savefig(out_jpg, dpi=args.dpi, bbox_inches="tight")
    plt.close()

    print("[OK] Groups (mean prob, change model):")
    print(means)
    print("[OK] S2A =", conv_group)
    print("[OK] S2B =", non_group)
    print("[OK] Saved:" )
    print(" ", out_pdf)
    print(" ", out_jpg)


if __name__ == "__main__":
    main()
