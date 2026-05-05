# Datasets

This directory is intentionally empty except for this README. The three
benchmark datasets used in the paper are **not redistributed** in this
repository — download them directly from the public sources below, or run
the helper script at the repository root:

```bash
bash download_data.sh
```

## Sources

| Dataset | File(s) expected in `data/` | Source | Licence |
| --- | --- | --- | --- |
| **Adult** (Census Income) | `adult.data`, `adult.test` | <https://archive.ics.uci.edu/dataset/2/adult> | CC BY 4.0 |
| **COMPAS** (two-year recidivism) | `compas-scores-two-years.csv` | <https://github.com/propublica/compas-analysis> | Released alongside the 2016 ProPublica investigation |
| **Bank Marketing** | `bank-full.csv` (from `bank.zip`) | <https://archive.ics.uci.edu/dataset/222/bank+marketing> | CC BY 4.0 |

## Preprocessing

Once the raw files are in place, run:

```bash
python preprocess_data.py
```

This writes three parquet files to this directory:

- `adult_processed.parquet`
- `compas_processed.parquet`
- `bm_processed.parquet`

These are the files that `run_experiments.py` and `verify_uniqueness.py`
actually consume. Preprocessing steps (categorical binning, target renaming)
follow standard practice in the fairness literature and are summarised in
Section 4.1 of the paper.
