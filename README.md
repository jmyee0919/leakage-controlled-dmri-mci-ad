# Short-Interval MRI Volume Change and Recorded AD Diagnosis

Version **3.0.0** is the second-revision code snapshot for *Short-Interval MRI Volume Change and 24-Month Recorded Alzheimer's Disease Diagnosis in Mild Cognitive Impairment: An Exploratory Landmark Analysis*.

Use **[second_revision/](second_revision/)** for the current submitted four-model comparisons, revised table numbering, figures and verification records. Version 3.0.0 is a source-code snapshot identified by its exact Git commit; it is not a claim that a GitHub Release or a new DOI has been created.

The earlier version 2.0.1 remains available at commit `92513375b16f47b5b0382894f763726500b8c885`. Root `data/aggregate/`, `final_submission/`, `figures/`, `models/parameters/` and the older scripts preserve that legacy material; their numbering does not reproduce all second-revision results. See [the archived version 2.0.1 README](docs/README_v2.0.1.md).

## Current reproduction

Install `requirements.txt` with Python 3.12.14. Run these public checks without participant data:

```bash
python scripts/verify_public_release.py
python second_revision/code/verify_revision2.py
python second_revision/code/render_second_revision.py --out /tmp/revision2_figures
```

Clinical reruns require authorized frozen inputs, as described in [second_revision/README.md](second_revision/README.md). No participant-level ADNI data, individual predictions, source diagnosis rows, MRI images, manuscript or reviewer-response files are redistributed here.

The reported association is exploratory. The increment from change beyond second-MRI anatomy is small and inconsistent across repeated partitions; acquisition imbalance, ascertainment selection, small samples and calibration instability limit inference. Internal reproduction is not external validation or evidence for clinical use.

## Availability and scope

The repository distributes code, aggregate tables, model parameter exports, curve coordinates without identifiers or probability thresholds, final figure PDFs, specifications and verification records. The second-revision four-model analysis has been rerun from frozen cohorts. Full historical raw-to-cohort reconstruction, image adjudication and the complete executable synthetic generator are unavailable. The archived synthetic specification and seed-level summaries are preserved under `data/synthetic/`; those simulations have not been regenerated in this final verification.

Cite the manuscript and the exact commit used; see `CITATION.cff`. Code is MIT licensed. ADNI data access remains governed by the ADNI Data Use Agreement and is not granted by this code license.
