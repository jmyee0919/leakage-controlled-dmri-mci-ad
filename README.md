# Leakage-controlled short-interval DeltaMRI improves MCI-to-AD prediction

This repository contains code, aggregate results, and figure-generation assets for the manuscript:

**Leakage-controlled short-interval ΔMRI improves MCI-to-AD prediction: synthetic positive-control validation and ADNI cohort evaluation**

The analysis tests whether short-interval longitudinal MRI-derived annualized atrophy rates (DeltaMRI / ΔMRI) add prognostic information beyond baseline volumetric MRI for predicting progression from mild cognitive impairment (MCI) to Alzheimer's disease (AD).

## Repository status

This is a cleaned, submission-oriented code release. It intentionally excludes ADNI participant-level data and subject-level derived files.

## Main analysis components

1. **Synthetic positive-control experiments** across 3-18, 6-24, and 12-30 month windows.
2. **Primary ADNI clinical analysis** using the filtered ADNI cohort with sigmoid calibration.
3. **ADNI sensitivity analyses** across cohort definitions and calibration settings.
4. **Figure/table generation** for the manuscript-facing outputs.

## Directory structure

```text
configs/                 Analysis configuration files
src/dmri_mci_ad/          Reusable Python utilities
scripts/synthetic/        Synthetic-data analysis scripts
scripts/adni/             ADNI analysis scripts requiring authorized ADNI data
scripts/figures/          R/Python figure-generation scripts
apps/                     Streamlit exploratory app
results/frozen/           Aggregate manuscript-facing results only
figures/main/             Final main manuscript figures
figures/supplementary/    Supplementary figure outputs where available
data/                     Empty raw/processed placeholders and data instructions
manuscript/               Figure-caption mapping and manuscript-adjacent notes
```

## Data availability and ADNI restrictions

This repository does **not** redistribute ADNI participant-level data.

To reproduce the ADNI analyses, users must obtain access to ADNI data through the official ADNI/LONI data access process and place the required files in `data/raw/`. Participant-level ADNI files, derived subject-level CSVs, prediction files, and visit-level outputs are excluded from this repository.

Examples of excluded files include:

- `ADNIMERGE.csv`
- UCSF/FreeSurfer ADNI exports
- `adni_snapshot_all.csv`
- `adni_deltaMRI_clean.csv`
- `filtered_snapshot.csv`
- `filtered_mridelta.csv`
- subject-level prediction/OOF files

Only aggregate manuscript-facing result tables are included under `results/frozen/`.

## Installation

Using conda:

```bash
conda env create -f environment.yml
conda activate dmri-mci-ad
```

Or with pip:

```bash
pip install -r requirements.txt
```

## Reproducing analyses

### Synthetic analyses

```bash
python scripts/synthetic/add_entorhinal_to_synthetic.py
python scripts/synthetic/run_batch_synthetic.py
```

The scripts assume the expected synthetic input CSVs are available in the configured input folder. The synthetic generator/augmentation logic is ADNI-independent.

### ADNI primary analysis

1. Obtain authorized ADNI/LONI data.
2. Place required raw files in `data/raw/`.
3. Build the leakage-safe snapshot and DeltaMRI files following the manuscript methods.
4. Run:

```bash
python scripts/adni/export_adni_filtered_primary_predictions.py
```

The ADNI script expects authorized derived inputs such as `filtered_snapshot.csv` and `filtered_mridelta.csv` in the working directory. These files are not included.

### Figures

The manuscript-facing figures are provided in `figures/main/`. Figure scripts are in `scripts/figures/`. Some figure scripts require intermediate prediction files that are excluded when they are ADNI-derived.

## Frozen aggregate results

`results/frozen/` contains aggregate, manuscript-facing metrics:

- `paper_results_synthetic.csv`
- `paper_results_adni_aggregate.csv`
- `synthetic_master_summary.csv`
- `synthetic_master_delong.csv`

These are included for exact cross-referencing with the paper figures and tables.

## License

Code is released under the MIT License. ADNI data are not included and remain governed by the ADNI Data Use Agreement.

## Citation

Please cite the associated manuscript. A `CITATION.cff` file is provided and should be updated with DOI information after publication.
