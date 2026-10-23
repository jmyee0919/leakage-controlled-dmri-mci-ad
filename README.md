# Short interval MRI change and Alzheimer disease progression

Code and aggregate results for **Short-interval MRI volume change and 24-month Alzheimer's disease progression in a selected MCI cohort: an exploratory landmark analysis**.

The revised study compares baseline regional MRI with baseline MRI plus annualized change at a common second-MRI landmark. The primary analysis includes 106 participants and 35 recorded AD events; its internal holdout includes 30 participants and 9 events. The primary paired AUC difference is +0.0794 (95% bootstrap interval −0.0132 to +0.2162). This is an exploratory evaluation with substantial uncertainty, rather than a validated clinical tool.

## Revision snapshot

Version **2.0.0**, 23 October 2026. The repository URL retains its original name to preserve existing citations. That historical name is not a claim that the revised design eliminates all leakage or selection bias. The original submission is retained in Git history at [ce31eac](https://github.com/jmyee0919/leakage-controlled-dmri-mci-ad/tree/ce31eac014e862d2efb70be5dcb89df4d409fc36); its code and results must not be mixed with this revision.

## Contents

| Folder | Purpose |
| --- | --- |
| `scripts/` | Public figure rendering, locked-prediction reproduction, and S21 QC/sensitivity reproduction |
| `configs/` | Unchanged locked analysis specification |
| `data/aggregate/` | Supplementary Data 1 tables underlying the final manuscript |
| `data/figure_inputs/` | Aggregate curves, performance summaries and repeat-level values used for plotting |
| `data/synthetic/` | Revised synthetic specification, seeds, saved metrics and structural diagnostics |
| `models/parameters/` | Fitted coefficients, preprocessing values and sigmoid parameters for the ten locked comparisons |
| `figures/main/`, `figures/supplementary/` | Final PDF/PNG figures supplied with the revision |
| `verification/` | Aggregate numerical verification records; earlier runs are labelled `PRIOR_` |
| `docs/` | Data access, execution scope and availability statements |

No ADNI participant-level data, source diagnosis rows, MRI files, individual predictions, manuscript drafts or response letters are distributed here. Curve-coordinate CSVs contain plotted coordinates without thresholds, identifiers or individual prediction vectors.

## Install

Use Python 3.12.14 and the recorded verification dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate with `.venv\Scripts\activate`. These are the independently verified environment versions, which differ from the archived original fitting environment described in the manuscript.

## Run without restricted data

```bash
python scripts/verify_public_release.py
python scripts/render_figures.py --output build/public_figures
```

The first command checks the public file inventory, figure inputs and numerical anchors. The second redraws Figures 1–3, S2 and S3 from public aggregate inputs. Figure S1 is copied from its archived final PDF/PNG, as explicitly recorded in the figure manifest. It is not a fresh synthetic-generator run. Rendering can produce small font/legend differences across environments; final journal figure assets are preserved unchanged.

## Run with authorized restricted inputs

The following commands require the study's frozen derived cohort files. Downloading arbitrary current ADNI exports does not automatically recreate those archived inputs. The required schemas and limitations are described in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

```bash
python scripts/reproduce_locked_predictions.py \
  --source /path/to/Step3E1 \
  --output /path/to/new_locked_checks

python scripts/reproduce_S21.py \
  --cohort /path/to/secondary_cohort_PRIVATE.pkl \
  --icv /path/to/ICV_PRIMARY_121.csv \
  --spec configs/analysis_specification.json \
  --out /path/to/new_S21_checks
```

The second script reconstructs QC from the 121-participant cohort, first verifies the primary and five-covariate results, and then reproduces the five-/six-early-event exclusions and field-strength analyses. The original train/test assignment is retained. The frozen-prediction check rebuilds all ten inherited-holdout comparisons and the repeated primary cross-validation predictions. Neither command performs new MRI measurement or creates an external validation cohort.

## Reproducibility scope

The archived independent check reproduced 2,712 probabilities with maximum error 3.33 × 10⁻¹⁶. The S21 checks compare a separately written implementation. Numerical reproduction from frozen inputs does not independently establish clinical label validity or eliminate design bias.

The complete executable revised synthetic generator and the full historical raw-to-cohort ingestion workflow were not available in the supplied final materials. The synthetic specification and saved outputs are included; their provenance is explained in Supplementary Methods S6.6 and S8. No missing generator is replaced with an invented implementation.

## Data access

ADNI data are accessed through the [LONI Image and Data Archive](https://adni.loni.usc.edu/data-samples/adni-data/) following approval and acceptance of the ADNI Data Use Agreement. Raw and derived participant-level data remain governed by that agreement. Summary results are supplied here and in Supplementary Data 1. See [docs/DATA_ACCESS.md](docs/DATA_ACCESS.md).

## License and citation

The existing MIT software license is retained. It does not grant rights to ADNI participant-level data. Cite the associated manuscript and identify the exact repository commit used. `CITATION.cff` supplies software citation metadata; no publication DOI or archival DOI is claimed.
