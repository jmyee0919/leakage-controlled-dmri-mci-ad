# Leakage-controlled short-interval ΔMRI improves MCI-to-AD prediction

## Overview
TBD

## Repository contents
TBD

## Data availability
TBD

## Installation
conda env create -f environment.yml
conda activate dmri-mci-ad

or

pip install -r requirements.txt

## Reproducing synthetic analyses
python scripts/01_generate_synthetic.py --config configs/synthetic_3_18.yaml
python scripts/03_run_synthetic_windows.py

## Reproducing ADNI analyses
1. Obtain ADNI files from LONI.
2. Place them in data/raw/.
3. Run:
python scripts/02_prepare_adni_from_loni_exports.py
python scripts/04_run_adni_primary_and_sensitivity.py

## Reproducing figures and tables
python scripts/05_make_main_figures.py
python scripts/06_make_supplementary_figures.py
python scripts/07_export_tables.py

## Citation
Use CITATION.cff.

## License
Code license only. ADNI data are separately governed by the ADNI Data Use Agreement.

# Data access

This repository does not redistribute ADNI participant-level data.

To reproduce the ADNI analyses, users must obtain access to the Alzheimer’s Disease Neuroimaging Initiative (ADNI) data through the official LONI/ADNI data access process and place the required files in `data/raw/`.

Expected ADNI input files include:
- ADNIMERGE.csv
- UCSFFSX7.csv
- UCSFFSL51.csv

These files are not included in this repository because they are governed by the ADNI Data Use Agreement. The code is provided to reproduce the analysis after authorized users obtain the data directly from ADNI/LONI.
