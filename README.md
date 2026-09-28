# Time-series Forecasting of Energy Consumption

This project forecasts hourly electricity consumption for the PJM East (`PJME`) region. It
compares simple time-series baselines with decision tree and random forest regressors using one
shared, leakage-safe feature pipeline.

## Current capabilities

- Cleans the raw PJME CSV and records every preprocessing decision in a quality report.
- Validates types, sorts timestamps, resolves duplicates, and fills missing clock hours causally.
- Flags imputed rows and statistical outliers without deleting valid demand peaks.
- Builds calendar, U.S. federal holiday, lag, and trailing-average features.
- Preserves chronological order when creating train and test periods.
- Evaluates persistence, 24-hour moving average, decision tree, and random forest forecasts.
- Reports MAE, RMSE, and MAPE and exports predictions and feature importances.
- Includes automated tests and GitHub Actions continuous integration.

## Project layout

```text
data/raw/                  included PJME source dataset
data/processed/            cleaned dataset and data-quality report
models/                    trained models (not committed)
notebooks/                 exploratory analyses
reports/figures/           generated figures (not committed)
src/energy_forecasting/    reusable data and modeling pipeline
tests/                     automated tests
```

## Setup

Use Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The repository includes `data/raw/PJME_hourly.csv`, the PJM East series from version 3 of Kaggle's
[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data)
dataset. It contains 145,366 source observations from January 2002 through August 2018 and is
licensed CC0/Public Domain. See [`data/README.md`](data/README.md) for its provenance and checksum.

## Preprocess the data

The committed clean dataset can be reproduced from the raw source with:

```bash
energy-preprocess
```

This writes `data/processed/PJME_hourly_clean.csv` and
`data/processed/data_quality_report.json`. The process:

1. Parses timestamps and energy values into consistent types.
2. Sorts observations chronologically.
3. Averages duplicate timestamps.
4. Creates a continuous hourly index.
5. Fills missing targets from the same hour one week earlier, then one day earlier, then the most
   recent prior observation. It never uses a future value to fill an earlier row.
6. Flags imputed values and values outside three interquartile ranges. It retains unusual demand
   peaks because they may represent real weather or usage events.

## Run the experiment

```bash
energy-forecast --data data/processed/PJME_hourly_clean.csv
```

Results are written to `reports/metrics.csv`, `reports/predictions.csv`, and per-model feature
importance files. Fitted tree models are written to `models/`.

### Initial benchmark

The default chronological 80/20 run produced the following holdout results:

| Model | MAE (MW) | RMSE (MW) | MAPE |
| --- | ---: | ---: | ---: |
| Persistence | 1,070.5 | 1,374.6 | 3.48% |
| 24-hour moving average | 3,643.0 | 4,557.5 | 12.01% |
| Decision Tree | 383.2 | 542.7 | 1.22% |
| Random Forest | **304.2** | **422.4** | **0.97%** |

These are initial single-holdout results, not final model-selection estimates. Time-series
cross-validation and tuning remain roadmap items.

Run the quality checks with:

```bash
ruff check .
pytest
```

## Modeling rules

This is time-series data, so rows must never be randomly shuffled across train and test periods.
Lag and rolling features use prior observations only. Model comparisons use the same chronological
holdout and feature matrix to keep results directly comparable.

## Roadmap

- Add an exploratory notebook with hourly, weekly, seasonal, and holiday visualizations.
- Add time-series cross-validation and model hyperparameter tuning.
- Compare PCA/NMF loadings with tree-based feature importances.
- Evaluate weather-derived heating and cooling degree-day features.
- Produce final comparison figures, report, and presentation.
