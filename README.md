# Time-series Forecasting of Energy Consumption

This project forecasts hourly electricity consumption for the PJM East (`PJME`) region. It
compares simple time-series baselines with decision tree and random forest regressors using one
shared, leakage-safe feature pipeline.

## Current capabilities

Final deliverables: [written report](reports/final_report.md),
[editable report](reports/CSC422_Final_Report.docx), and
[final slides](https://docs.google.com/presentation/d/1G8gHuEEzzCfA9qqNX6xhGup07aBLsyYSOagQm47aPeo/edit).
The latest three-horizon results are in [reports/horizons](reports/horizons/summary.md).

- Cleans the raw PJME CSV and records every preprocessing decision in a quality report.
- Validates types, sorts timestamps, resolves duplicates, and fills missing clock hours causally.
- Flags imputed rows and statistical outliers without deleting valid demand peaks.
- Builds calendar, U.S. federal holiday, lag, and trailing-average features.
- Builds population-weighted heating and cooling degree-hour features from NOAA weather for five
  PJME cities, with leak-free gap filling and 24-hour, 16-day, and 96-day window features.
- Preserves chronological order when creating train and test periods.
- Evaluates persistence, 24-hour moving average, decision tree, and random forest forecasts.
- Reports MAE, RMSE, and MAPE and exports predictions and feature importances.
- Runs exploratory analysis, PCA, and NNMF on complete hourly, weekly, and calendar-year load
  profiles.
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
reports/pca/               generated EDA/PCA figures and tables (not committed)
reports/nnmf/                generated NNMF figures and tables (not committed)
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

These are the original weather-enabled benchmark results. Cross-validation and tuning are now
complete. The newer horizon experiment expands the tree search and reports 371.7 MW one-hour
Decision Tree MAE; the Random Forest result remains 297.4 MW.

## Run the final experiment

The final experiment implements the remaining modeling work identified in the midterm report:
expanding-window cross-validation, deterministic tree-model tuning, a controlled energy-only
versus weather-enabled comparison, final feature importances, prediction-versus-actual plots, and
residual analysis.

```bash
energy-final
```

Both feature sets use the same timestamps, chronological 80/20 split, and tuned model settings.
Hyperparameters and the final model family are selected only from the training period; the final
20% holdout remains untouched until evaluation. Results are written to `reports/final/`, including
the fold-level validation record, metrics, selected parameters, a readable summary, and four
figures. The fitted final model is written to `models/`.

The completed run selected the Random Forest by validation MAE. On the untouched holdout, weather
features reduced its MAE from 305.5 MW to 297.4 MW (a 2.67% improvement), with a final RMSE of
415.3 MW and MAPE of 0.94%. See [`reports/final/summary.md`](reports/final/summary.md) for the full
controlled comparison and interpretation.

Run the quality checks with:

```bash
ruff check .
pytest
```

## Run exploratory analysis and PCA

From the repository root, run:

```bash
python src/PCA/pca.py
```

The script groups the hourly series into complete 24-hour daily profiles, standardizes each hour,
and calculates principal components explaining at least 95% of the variance. It also produces
hourly, weekday/weekend, seasonal, loading, and score visualizations under `reports/pca/`.

## Run NNMF

From the repository root, run:

```bash
python src/NNMF/nnmf.py
```

The NNMF script fits three complete profile horizons from the hourly series:

- `hour`: one row per day with 24 hourly columns;
- `day`: one row per Monday-Sunday week with 7 daily columns; and
- `week`: one row per calendar year with 365 daily columns. February 29 is removed from leap
  years so all calendar years share the same daily positions and weekday structure.

Each horizon fits nonnegative factors after min-max normalization to [0, 1]. Reconstruction
quality is evaluated after converting the fitted output back to MW, and component profiles are
reported as MW contributions above the dataset minimum. The script selects the smallest number
of factors reaching 95% reconstruction quality, defined as

`1 - ||X - WH||² / ||X - mean(X)||²`.

If no rank through the profile width reaches the target, it reports a warning and uses the rank with the best
reconstruction quality.

Figures and tables are written under `reports/nnmf/hour/`, `reports/nnmf/day/`, and
`reports/nnmf/week/`, including:

- a scree-style rank plot showing reconstruction quality and reconstruction error;
- factor activation plots and horizon-specific factor loadings (line and heatmap views);
- a top-factor summary ranked by mean activation, with each factor profile and its three
  highest-activation representative profiles; and
- CSV tables containing factor activations, component profiles, rank-selection metrics, and
  metadata.

These visualizations are descriptive analysis only and are not connected to the forecasting
models.

## Modeling rules

Run `energy-horizons` after building weather features to reproduce the 1-hour, 1-day, and 1-week
point forecasts, purged validation, permutation importance, and feature reconstruction rankings.
This is a long training job; committed tables and figures support review without retraining.
Weather reduces Random Forest MAE by 2.67% at one hour and 1.71% at one day, with no useful
weekly improvement. See the final report for evaluation limitations and corrected NNMF results.

This is time-series data, so rows must never be randomly shuffled across train and test periods.
Lag and rolling features use prior observations only. Model comparisons use the same chronological
holdout and feature matrix to keep results directly comparable.

## Submission status

The implemented analyses, final report, and five-slide presentation are prepared for team review.
Confirm the instructor's final submission format and deadline, rehearse the speaker notes, and
submit through the course system. No course submission has been made by this repository workflow.

Future research extensions include target-time calendar features, forecast weather inputs, and a
new independent evaluation period. These are documented limitations rather than completed work.
