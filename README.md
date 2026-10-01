# Time-series Forecasting of Energy Consumption

This project forecasts hourly electricity consumption for the PJM East (`PJME`) region. It
compares simple time-series baselines with decision tree and random forest regressors using one
shared, leakage-safe feature pipeline.

## Current capabilities

- Cleans the raw PJME CSV and records every preprocessing decision in a quality report.
- Validates types, sorts timestamps, resolves duplicates, and fills missing clock hours causally.
- Flags imputed rows and statistical outliers without deleting valid demand peaks.
- Builds calendar, U.S. federal holiday, lag, and trailing-average features.
- Builds population-weighted heating and cooling degree-hour features from NOAA weather for five
  PJME cities, with leak-free gap filling and 24-hour, 16-day, and 96-day window features.
- Preserves chronological order when creating train and test periods.
- Evaluates persistence, 24-hour moving average, decision tree, and random forest forecasts.
- Reports MAE, RMSE, and MAPE and exports predictions and feature importances.
- Includes automated tests and GitHub Actions continuous integration.

## Project layout

```text
data/raw/                  included PJME source dataset
data/raw/weather/          NOAA hourly weather observations (gzip)
data/raw/population/       Census county population files
data/processed/            cleaned dataset and data-quality report
docs/                      method documentation
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

## Build weather features

```bash
energy-weather
```

This reads the NOAA and Census files in `data/raw/` and writes
`data/processed/weather_features.csv` and `data/processed/weather_quality_report.json`. Hourly
temperatures for Philadelphia, Newark, Baltimore, Washington, and Harrisburg are aligned to PJME's
hours, gaps are filled from the nearest station using only earlier data, and heating and cooling
degree hours are weighted by county population. See
[`docs/weather_features.md`](docs/weather_features.md) for sources and methods.

## Run the experiment

```bash
energy-forecast --data data/processed/PJME_hourly_clean.csv
```

Results are written to `reports/metrics.csv`, `reports/predictions.csv`, and per-model feature
importance files. Fitted tree models are written to `models/`.

The 12 weather window features are included by default, lagged one hour so a row never sees the
weather of the hour it predicts. Run `energy-weather` first, or add `--no-weather` to use the
energy-only feature set.

### Initial benchmark

The default chronological 80/20 run produced the following holdout results:

| Model | MAE (MW) | RMSE (MW) | MAPE |
| --- | ---: | ---: | ---: |
| Persistence | 1,074.8 | 1,378.3 | 3.49% |
| 24-hour moving average | 3,660.8 | 4,577.2 | 12.05% |
| Decision Tree | 380.0 | 544.3 | 1.20% |
| Random Forest | **297.4** | **415.3** | **0.94%** |

These are the weather-enabled default run's initial single-holdout results, not final
model-selection estimates. Time-series cross-validation and tuning remain roadmap items.

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
- Choose the forecast horizon (next hour, next day) and how weather is provided for it.
- Produce final comparison figures, report, and presentation.
