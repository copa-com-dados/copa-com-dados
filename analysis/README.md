# Paper Analysis Notebooks

This directory is the only new analysis area in the repository. It does not
edit project source files, Stan programs, input data, or public website
outputs. Model fitting and tournament simulation remain in the original
`src/` implementation.

| Path | Purpose |
|---|---|
| `01_snapshots_and_evaluation.ipynb` | Cutoff-aware snapshots, native-export reference comparison, and evaluation from saved products. |
| `02_exploratory_data_analysis.ipynb` | Input-data EDA and, after FULL snapshots exist, forecast-dependent EDA. |
| `paper_pipeline.py` | Shared orchestration and table/figure helpers. |
| `figure_manifest.md` | Template that is replaced after an explicit EDA run. |

## Modes

- `PLAN` is the default. It does not read source data or create artifacts.
- `FAST` is an explicit no-MCMC preflight. It writes only an input/dependency
  audit under `artifacts/preflight/`; it creates no posterior, forecast, or
  evaluation result.
- `FULL` uses the project's `train_model_up_to()` and
  `BayesianWorldCup2026` implementation at every documented cutoff. It uses
  100,000 tournament simulations by default, saves Parquet checkpoints, and
  skips completed snapshots on rerun.

No notebook mode was executed while preparing this deliverable.

## Running FULL

From the repository root, install the project environment and the two extra
runtime requirements:

```bash
uv sync --extra jupyter
uv pip install pyarrow
uv run python -c "import cmdstanpy; cmdstanpy.install_cmdstan()"
```

Set `MODE = "FULL"` in `01_snapshots_and_evaluation.ipynb` and run that
notebook first. Then set `MODE = "FULL"` in
`02_exploratory_data_analysis.ipynb` to create the figure/table products.

`validate.py` remains disabled by default. It requires
`data/raw/results.csv`; if that input is absent, the workflow records the
blocked condition rather than synthesising a historical backtest.

## Source-data assumption

The FULL evaluation expects `data/results.csv` to contain all scores for the 2026 World Cup. 
The workflow reports missing values if present, but never imputes a score, tiebreak winner, or champion.
