# Time Series Forecasting of Energy Consumption

CSC 422 Final Project Report · Fall 2026 · Team 8

David Valenta, Cole Dail, Peter Holt, Thomas Mitchell

## 1 Introduction and objective

We built a reproducible forecasting pipeline for electricity demand in the PJM East region. Our main question was whether tree models could improve on simple forecasts based on recent demand, and whether regional weather would add useful information. We also used PCA and nonnegative matrix factorization (NNMF) to understand patterns in the data and compare reconstruction with predictive importance.

Random Forest achieved the lowest measured error at all three forecast horizons. With weather, its one-hour MAE was 297.4 MW and its MAPE was 0.94%. Error increased substantially at one day and one week. Weather helped at the shorter horizons, but its weekly result was effectively unchanged. This suggests that recent demand is especially useful for the next hour, while historical weather alone gives us limited information about demand a week away.

## 2 Dataset and preprocessing

We used the PJME series from Kaggle's Hourly Energy Consumption dataset. It contains 145,366 source observations from January 2002 through August 2018. The target is electricity load in megawatts, rather than energy accumulated over an interval.

We parsed and sorted timestamps, averaged four duplicate rows, and created a continuous hourly index. Thirty missing hours were filled from the same hour one week earlier, then one day earlier, and finally the preceding observation when necessary. This produced 145,392 hourly rows. We retained 27 possible outliers with flags because unusually high demand may be a real event rather than a recording error.

Peter's weather pipeline processes NOAA observations for Philadelphia, Newark, Baltimore, Washington, and Harrisburg. It checks observation quality, aligns UTC weather records to local PJME hour-ending timestamps, and fills station gaps from nearby donors using offsets estimated from preceding observations. Census county population weights combine the regional heating and cooling degree hours relative to 65 degrees Fahrenheit.

The forecasting models use eight calendar variables, three demand lags, two trailing averages, and optionally twelve weather summaries. Weather features describe weighted means, variability, and slopes over 24 hours, 16 days, and 96 days. All demand and weather predictors use observations available before the predicted hour. Quality flags are retained for auditing rather than used as model predictors.

## 3 Methodology and evaluation

We compared persistence, a trailing 24-hour average, Decision Tree regression, and Random Forest regression. Persistence uses the latest available load. The trees can capture interactions among recent load, calendar timing, and weather without requiring a linear relationship.

The horizon experiment predicts a single hourly load one hour, one day, or one week after the latest available observation. These are point forecasts, not forecasts of a complete future day or week. A feature row at time t uses observations through t minus one hour and predicts the load at t, t plus 23 hours, or t plus 167 hours.

Both feature sets share the same forecast origins and split boundary, April 28, 2015 at 15:00. Later forecast origins form the evaluation period, ending in August 2018; its last valid origin is earlier for longer horizons because a future target must exist. Training labels that cross the split are removed. Three expanding-window validation folds select hyperparameters by mean MAE. Gaps of 23 or 167 hours prevent training labels from extending into the validation block at longer horizons. The same chosen settings are used in each energy-only and weather comparison, isolating the feature change within that setting.

The Decision Tree search covers depths 3 through 12, 14, and 16 with leaf sizes 2, 8, and 32. Random Forest evaluates four combinations of tree count, depth, leaf size, and feature sampling. Selected tree depth becomes shallower as the horizon increases: 16, 9, and 7. Random Forest uses 200 trees at each horizon, depth 20 for one hour, and unlimited depth with 80% feature sampling for one day and one week.

MAE measures the mean absolute forecast miss in MW. RMSE also uses MW but penalizes large misses more strongly. MAPE averages each hour's absolute percentage error; it is not MAE divided by average load. We report these metrics alongside permutation importance, which measures the increase in evaluation MAE after shuffling a feature.

## 4 Forecasting results

The following table reports weather-enabled models from Peter's horizon experiment. The earlier final experiment used a narrower Decision Tree grid; its older 380.0 MW tree MAE is superseded here by 371.7 MW.

| Horizon | Model | MAE MW | RMSE MW | MAPE |
| --- | --- | ---: | ---: | ---: |
| 1 hour | Persistence | 1,074.8 | 1,378.3 | 3.49% |
| 1 hour | 24-hour average | 3,660.8 | 4,577.2 | 12.05% |
| 1 hour | Decision Tree | 371.7 | 530.9 | 1.17% |
| 1 hour | Random Forest | 297.4 | 415.3 | 0.94% |
| 1 day | Persistence | 2,192.7 | 3,018.3 | 6.94% |
| 1 day | 24-hour average | 4,271.3 | 5,438.4 | 14.01% |
| 1 day | Decision Tree | 1,869.3 | 2,578.8 | 5.89% |
| 1 day | Random Forest | 1,755.6 | 2,394.4 | 5.54% |
| 1 week | Persistence | 3,434.0 | 4,711.5 | 10.66% |
| 1 week | 24-hour average | 4,772.2 | 6,208.7 | 15.48% |
| 1 week | Decision Tree | 3,139.4 | 4,159.8 | 9.94% |
| 1 week | Random Forest | 2,916.0 | 3,864.4 | 9.24% |

Adding weather lowers Random Forest MAE from 305.5 to 297.4 MW at one hour, an improvement of 2.67%. At one day it lowers MAE from 1,786.2 to 1,755.6 MW, an improvement of 1.71%. At one week it changes MAE from 2,915.7 to 2,916.0 MW, a slight worsening of about 0.01%. We therefore do not claim a weekly benefit from weather. We did not perform significance tests, so these are measured differences on this evaluation period rather than established population effects.

The one-hour prediction plot closely follows the daily load cycle. Its residual plot is concentrated around zero but contains occasional large errors. The low average error should therefore be considered alongside the larger misses, particularly when interpreting reliability during unusual demand conditions.

## 5 PCA and NNMF findings

Thomas's profile PCA summarizes daily, weekly, and monthly load patterns. Daily PCA retains three components to exceed 95% variance: overall load level, morning versus afternoon demand, and the morning ramp. These descriptions help us interpret the daily cycle seen in the forecasts.

Cole corrected NNMF by applying min-max normalization before fitting and evaluating reconstruction after returning to MW. The current daily-profile analysis uses two factors and reaches 96.96% reconstruction quality. This replaces the midterm result of eight factors and 93.84%. The weekly and calendar-year profile analyses are available separately under reports/nnmf; their profile widths and meanings differ from the daily analysis.

Peter also fitted PCA and NNMF to the actual 25-column training feature table so their feature rankings could be compared with the models. This is a separate analysis: feature-table PCA keeps 13 components for about 95.5% variance, while feature-table NNMF uses 11 factors for about 95.5% reconstruction quality. Those counts must not be confused with profile decomposition counts.

| Rank | PCA reconstruction | NNMF reconstruction | One-hour RF prediction |
| ---: | --- | --- | --- |
| 1 | Year | Year | Prior-hour load |
| 2 | Federal holiday | Weekend | Hour of day |
| 3 | Weather variability over 24 hours | Federal holiday | Heating degree hours over 24 hours |
| 4 | Weather trend over 24 hours | Day of week | Day of week |
| 5 | Weather trend over 16 days | Month | Cooling degree hours over 24 hours |

The PCA column ranks how well retained components reconstruct each standardized feature, rather than its raw variance. NNMF ranks each feature's reconstruction quality. Random Forest ranks the increase in MAE after shuffling each feature. The rankings answer different questions: a calendar variable can be reconstructed well without adding much forecast accuracy. At one hour, PCA and RF permutation ranks have Spearman correlation around minus 0.52. Recent load and hour of day dominate forecasting even though they do not lead the reconstruction rankings.

At one day, day of week becomes more important. At one week, day of year and the 168-hour lag appear among Random Forest's leading predictors. This is consistent with longer-horizon models relying more on seasonal and weekly patterns as the latest observation becomes less informative.

## 6 Limitations and conclusion

Our strongest result is the one-hour Random Forest forecast. Longer horizons remain harder, and historical weather has little weekly benefit in the current setup. We used observed past weather rather than forecast weather. Calendar variables describe the feature row rather than the future target hour; adding target-time calendar variables would be a useful extension because those values are known in advance.

The later evaluation period was already examined in earlier experiments and also supplies depth-diagnostic curves and permutation importance. Hyperparameters are chosen through training-only cross-validation, but the reported evaluation is not a newly reserved independent test. The depth curves should be treated as descriptive checks rather than used to retune models on the test period. A future confirmation should use a separate untouched period or nested rolling evaluation. Random permutation can also overstate reliance when predictors are correlated or temporally dependent. NNMF iteration-limit warnings in some tests indicate a further convergence check is appropriate before interpreting every factor as stable.

We completed cleaning, regional weather integration, forecasting at three horizons, time-aware tuning, decomposition analyses, and readable comparison artifacts. Together, the results show that the latest demand and its position in the daily cycle explain most short-term predictive performance. Weather contributes a smaller benefit that depends on how far ahead we forecast.

## 7 Team contributions and reproducibility

David built and integrated the energy cleaning, feature, baseline, and tree-model pipelines; added final tuning and the weather comparison; and assembled the final results. Peter built the population-weighted weather pipeline and extended the experiments to three horizons, with purged validation, feature rankings, and charts. Thomas produced the exploratory analysis and PCA profiles. Cole implemented and corrected NNMF and extended its profile analyses. All four team members are listed as report authors; final review of the submission remains a shared responsibility.

From the repository root, install with `python -m pip install -e ".[dev]"`. Run `energy-preprocess`, `energy-weather`, `energy-final`, and `energy-horizons` in that order. Profile analyses run with `python src/PCA/pca.py` and `python src/NNMF/nnmf.py`. The horizon run is computationally expensive. Existing committed CSVs and figures allow the reported results to be inspected without retraining.

The current verification run passed 40 tests and Ruff. It produced NNMF convergence and dependency deprecation warnings, which are retained rather than hidden. Numerical results in this report are drawn from committed artifacts, not a newly rerun full horizon experiment.

## References and result sources

1. PJM Hourly Energy Consumption, Kaggle: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
2. NOAA Global Hourly data and Census population sources: see data/README.md and docs/weather_features.md.
3. Repository: https://github.com/aretegratatimferociter/CSC422-Energy-Consumption
4. Forecast metrics and validation: reports/horizons/metrics.csv, cv_results.csv, and final_model_errors.csv.
5. Reconstruction and predictive rankings: reports/horizons/feature_rank_side_by_side.csv and rank_agreement.csv.
6. Corrected daily-profile NNMF: reports/nnmf/hour/nnmf_summary.csv.

Final deck: https://docs.google.com/presentation/d/1NYYuDm3RbkNt6tLISEEJe-H_dGflHHpI/edit
