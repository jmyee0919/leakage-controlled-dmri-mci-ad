# Execution scope and input requirements

The public release contains executable figure and verification scripts, a frozen analysis specification, model parameters and aggregate results. It separates reproduction from frozen derived inputs from reconstruction of an ADNI cohort from raw source exports.

## Public commands

`python scripts/verify_public_release.py` checks SHA-256 integrity, CSV structure, the required figure set and the primary/S21 numerical anchors. It is not a clinical validation test.

`python scripts/render_figures.py --output build/public_figures` uses only public aggregate data. The ROC and PR inputs are coordinates of already-published curves; no subject identifiers, thresholds or individual probability vectors are included. The calibration display uses the three-bin aggregates and stored joint intercept/slope estimates. Figure S1 is copied unchanged from the archived asset; five other figure sets are rendered afresh.

## Restricted frozen prediction check

`scripts/reproduce_locked_predictions.py` expects an authorized Step3E1 directory containing:

- `02_locked_analysis/ANALYSIS_SPECIFICATION.json` with the exact recorded analysis definitions;
- `01_private/<analysis>_cohort.pkl` for each of the ten permitted fits;
- `01_private/<analysis>_predictions.csv` and `<analysis>_parameters.csv`;
- `01_private/REPEATED_PREDICTIONS_PRIVATE.npz` and `REPEATED_MEMBERSHIP_PRIVATE.csv`.

The cohort tables require the ordered predictor columns in the specification, binary `y`, `participant_key`, and the inherited `split` (`train`/`test`). The broader sensitivity preserves its separate archived assignment. The script validates cohort hashes and reconstructs the calibrated ensemble, preserving predictor order. It does not ingest raw DXSUM or FreeSurfer exports.

## Restricted S21 check

`scripts/reproduce_S21.py` needs the trusted 121-participant `secondary_cohort_PRIVATE.pkl`, an ICV audit CSV containing `participant_key`, `followup_ICV_percent_difference` and `source_FIELD_STRENGTH`, and the public locked specification. Cohort columns include the ordered predictors, `participant_key`, `split`, binary `y`, `event_date`, `information_date`, and baseline/follow-up values for the three regional volumes. Only trusted archived pickle files should be loaded.

QC excludes absolute ICV differences over 5% and absolute robust z scores over 4 for regional log ratios, using the locked reference medians and MADs. The script checks 106/35 retained participants/events and 76/26 versus 30/9 train/holdout support before fitting. It checks published reference estimates before reporting the S21 settings. A second implementation independently computes the calibrated predictions.

Both matched arms use the same 500 bootstrap samples within each analysis. A reduced cohort regenerates arrays at its own size using seed 42; it does not reuse the original-size index array. Ten-repeat three-fold validation is performed for imaging comparisons, regenerating folds with seed 42 on the relevant cohort. Clinical-adjusted repeats are not reported in S21. Output folders are separate from input folders.

## Model exports and records

`models/parameters/` contains the ten original coefficient and preprocessing exports. Predictions average the three sigmoid-calibrated fold probabilities, as specified in Supplementary Methods S3. These parameter files are not a clinical deployment recommendation.

`verification/PRIOR_LOCKED_*` records the earlier independent check of the unchanged locked specification. `verification/S21_*` records a separate verified reproduction of the added sensitivities. Its `MRI_visit_DX_verified: false` field describes that calculation-only audit; it does not contradict the authors' later, separately retained visit-level source review. Restricted diagnosis evidence is intentionally absent from this public release.

The complete executable revised synthetic generator and raw-to-cohort ingestion chain are not supplied. No end-to-end raw-data or fresh synthetic-generation reproduction is claimed. Changes made for this repository concern paths, public plotting inputs, output guards and convergence reporting; the paper's estimators and results were not selected or changed.
