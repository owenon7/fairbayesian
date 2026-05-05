# Fair Bayesian Classifier

Reference implementation and experimental pipeline for the paper
*"A Bayesian Framework for Statistically Consistent Prediction"*
(Owen O'Neill, submitted to the *Journal of Artificial Intelligence Research*, 2026).

The framework enforces statistical consistency between a classifier's predictions
and the observed data at every subgroup granularity the data supports, and
abstains when no deterministic prediction is consistent with the evidence.

## Contents

| File | Purpose |
| --- | --- |
| `fb_functions.py` | Core framework: `DataPrep`, `GenVs`, `CalcMs`, `CalcBounds`, `AdjustEnfs` |
| `pipeline.py` | MIP solve, two-stage solution selection (d-objective + v-node log-likelihood), baseline training |
| `preprocess_data.py` | Regenerate the preprocessed parquet files from the raw CSVs |
| `run_experiments.py` | End-to-end per-dataset driver: produces every paper table, node-level dataframe, illustrative example and figure |
| `verify_uniqueness.py` | Pre-flight checks: MIP gap = 0, unique top solution by both selectors, pool-size stability |
| `download_data.sh` | Fetches the three raw benchmark datasets from their public sources |
| `data/` | Populated by `download_data.sh` and `preprocess_data.py`. See [data/README.md](data/README.md) for sources and licences. |
| `results/` | Per-dataset output directory populated by `run_experiments.py` (gitignored) |

## Requirements

- **Python** 3.9 (tested on 3.9.6)
- **Operating system** macOS 26.3 (code is pure Python; any OS with a working
  Gurobi install should work, but results were generated on macOS)
- **Hardware** Any modern machine with at least 16 GB RAM. Reference timings
  were produced on a 2020 MacBook Pro (Apple M1, 16 GB RAM).
- **Gurobi Optimizer 12.0** with a valid license. **Free academic licenses are
  available at <https://www.gurobi.com/academia/>.** The framework's MIP
  formulation is solver-independent in principle, but the code currently calls
  Gurobi directly through `gurobipy`.

Python package versions are pinned in `requirements.txt`.

## Setup

```bash
# Clone / unpack this directory, then:
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install and activate a Gurobi license (academic users only):
# https://www.gurobi.com/features/academic-named-user-license/
grbgetkey <your-license-key>
```

## Reproducing the Paper Results

The three benchmark datasets are **not redistributed** in this repository.
Download them first and preprocess into the parquet format the pipeline
consumes:

```bash
bash download_data.sh      # fetches Adult, COMPAS and Bank Marketing into data/
python preprocess_data.py  # writes data/*_processed.parquet
```

The paper's tables, figures, node-level dataframes and illustrative examples
are then produced by a single entry point:

```bash
# Pre-flight: verify MIP optimality and selector uniqueness on all datasets.
python verify_uniqueness.py

# Full pipeline: produces results/<dataset>/ and a cross-dataset rollup.
python run_experiments.py --dataset all
```

Per-dataset runs are available for iterative work:

```bash
python run_experiments.py --dataset adult
python run_experiments.py --dataset compas
python run_experiments.py --dataset bm
```

Useful flags:

- `--no-alpha-sweep` skip the alpha sensitivity sweep (reuses cached sweep CSV if present)
- `--skip-baselines` skip DT/NN/PMC baseline training
- `--skip-pmc` train DT and NN but skip the (slow) PMC baseline
- `--force` rebuild `CalcMs` and baseline caches
- `--skip-verify` bypass the uniqueness pre-flight

All random seeds are fixed (`BASELINE_SEED = 42`, `MIP_SEED = 42`). The
framework itself is deterministic given `alpha` and the prior; the only
stochastic components are baseline training (scikit-learn) and Gurobi's
solution pool sampling, both of which are seeded.

## Datasets

All three benchmarks are publicly available and are included here for
reproducibility:

- **Adult** — UCI Machine Learning Repository. <https://archive.ics.uci.edu/ml/datasets/Adult>
- **COMPAS** — ProPublica. <https://github.com/propublica/compas-analysis>
- **Bank Marketing** — UCI Machine Learning Repository. <https://archive.ics.uci.edu/ml/datasets/bank+marketing>

Preprocessing (categorical binning, one-hot encoding details) is described in
Section 4.1 of the paper and performed by `DataPrep` in `fb_functions.py`.

## Output

Each `python run_experiments.py --dataset <ds>` run writes the following under
`results/<ds>/`:

- `tables/` — every paper table as CSV (`tab_node_counts.csv`, `tab_d01_error.csv`,
  `tab_dnf_summary.csv`, `tab_vnode_error.csv`, `tab_accuracy.csv`,
  `tab_multical.csv`, `tab_alpha_sweep.csv`)
- `dnodes.parquet`, `vnodes.parquet` — full node-level dataframes with
  predictions from every model
- `examples/candidates_*.csv` — candidate subgroups for the illustrative examples
- `figures/` — d-node size distributions and the race/d-node breakdown plot
- `summary.json` — run metadata (alpha*, MIP gap, pool size, agreement between
  selectors, runtimes)
- `report.md` — human-readable summary

A cross-dataset rollup is written to `results/report.md` when `--dataset all`
is used.

## License

MIT — see [LICENSE](LICENSE).

## Citation

If you use this code, please cite the paper:

```bibtex
@article{oneill2026fairbayesian,
  author  = {O'Neill, Owen and Costello, Fintan},
  title   = {A Bayesian Framework for Statistically Consistent Prediction},
  journal = {Journal of Artificial Intelligence Research},
  year    = {2026},
  note    = {Under review.}
}
```
