# Time-series Forecasting of Energy Consumption

This project forecasts hourly electricity consumption for the PJM East (`PJME`) region. It
compares simple time-series baselines with decision tree and random forest regressors using one
shared, leakage-safe feature pipeline.

## Current capabilities

- Validates, sorts, and deduplicates the PJME hourly CSV.
- Builds calendar, U.S. federal holiday, lag, and trailing-average features.
- Preserves chronological order when creating train and test periods.
- Evaluates persistence, 24-hour moving average, decision tree, and random forest forecasts.
- Reports MAE, RMSE, and MAPE and exports predictions and feature importances.
- Includes automated tests and GitHub Actions continuous integration.

## Project layout

```text
data/raw/                  included PJME source dataset
data/processed/            generated datasets (not committed)
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

## Run the experiment

```bash
energy-forecast --data data/raw/PJME_hourly.csv
```

Results are written to `reports/metrics.csv`, `reports/predictions.csv`, and per-model feature
importance files. Fitted tree models are written to `models/`.

### Initial benchmark

The default chronological 80/20 run produced the following holdout results:

| Model | MAE (MW) | RMSE (MW) | MAPE |
| --- | ---: | ---: | ---: |
| Persistence | 1,079.0 | 1,381.8 | 3.50% |
| 24-hour moving average | 3,688.9 | 4,607.2 | 12.13% |
| Decision Tree | 377.6 | 533.4 | 1.19% |
| Random Forest | **303.8** | **421.9** | **0.96%** |

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
