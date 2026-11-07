# Final Forecasting Experiment

This experiment completes the model-selection and weather-ablation work identified in the midterm
report. It uses expanding-window time-series cross-validation on the training period, followed by
one untouched chronological 20% holdout from 2015-04-28 through
2018-08-03. Energy-only and weather-enabled runs use identical timestamps, splits,
and tuned model settings.

## Selected model

The validation procedure selected **Random Forest**. Its tuned parameters
were `{"max_depth": 20, "max_features": 1.0, "min_samples_leaf": 2, "n_estimators": 200}`.

On the final holdout, adding weather improved MAE by **2.67%**, from
**305.5 MW** to **297.4 MW**. The weather-enabled result had an RMSE
of **415.3 MW** and a MAPE of **0.94%**.

## Holdout results

| Feature set | Model | MAE (MW) | RMSE (MW) | MAPE |
| --- | --- | ---: | ---: | ---: |
| Energy Only | Persistence | 1,074.8 | 1,378.3 | 3.49% |
| Energy Only | Moving Average 24H | 3,660.8 | 4,577.2 | 12.05% |
| Energy Only | Decision Tree | 382.5 | 542.5 | 1.21% |
| Energy Only | Random Forest | 305.5 | 425.4 | 0.97% |
| Weather Enabled | Persistence | 1,074.8 | 1,378.3 | 3.49% |
| Weather Enabled | Moving Average 24H | 3,660.8 | 4,577.2 | 12.05% |
| Weather Enabled | Decision Tree | 380.0 | 544.3 | 1.20% |
| Weather Enabled | Random Forest | 297.4 | 415.3 | 0.94% |

## Interpretation

The most important predictors in the final weather-enabled model were: lag_1, hour, dh_slope_24h, hdh_wma_24h, rolling_mean_24, day_of_week, cdh_wma_24h, is_weekend. This gives
the team a direct feature-importance comparison to discuss alongside PCA's overall-level,
morning-versus-afternoon, and morning-ramp components and NNMF's seasonal daily-load profiles.
PCA and NNMF remain descriptive analyses; they were not used to select hyperparameters or inspect
the final holdout.

## Generated figures

- `mae_comparison.png`: controlled energy-only versus weather-enabled error comparison.
- `actual_vs_predicted.png`: actual load and both final-model forecasts for the last 14 days.
- `residual_analysis.png`: residual scatter plot and residual distribution for the final model.
- `feature_importance.png`: ranked feature importances for the final weather-enabled model.

The full fold-level validation record is stored in `time_series_cv_results.csv`, and the exact
model choice is stored in `selected_model.json`.
