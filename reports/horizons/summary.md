# Feature importance across forecast horizons

Row t keeps its features (data through t - 1) and predicts load at t + horizon - 1:
1 hour ahead = shift 0 h, 1 day ahead = shift 23 h, 1 week ahead = shift 167 h.
All horizons share one split time; training rows whose target lands in the test period are
dropped, and tuning folds use a gap equal to the shift.

## Variance side (PCA and NNMF on the model feature table)

- PCA: 13 components keep
  95.5% of variance (PC1 alone
  27.9%).
- NNMF: rank 11 reaches
  95.5% reconstruction quality.
- Top features by PCA variance captured: year, is_holiday, dh_std_24h, dh_slope_24h, dh_slope_16d.
- Top features by NNMF variance captured: year, is_weekend, is_holiday, day_of_week, month.

## Holdout error

| horizon | feature_set | model | mae | mape_percent |
| --- | --- | --- | --- | --- |
| 1 hour | energy_only | persistence | 1,074.8 | 3.49% |
| 1 hour | energy_only | moving_average_24h | 3,660.8 | 12.05% |
| 1 hour | energy_only | decision_tree | 373.8 | 1.19% |
| 1 hour | energy_only | random_forest | 305.5 | 0.97% |
| 1 hour | weather_enabled | persistence | 1,074.8 | 3.49% |
| 1 hour | weather_enabled | moving_average_24h | 3,660.8 | 12.05% |
| 1 hour | weather_enabled | decision_tree | 371.7 | 1.17% |
| 1 hour | weather_enabled | random_forest | 297.4 | 0.94% |
| 1 day | energy_only | persistence | 2,192.7 | 6.94% |
| 1 day | energy_only | moving_average_24h | 4,271.3 | 14.01% |
| 1 day | energy_only | decision_tree | 1,852.0 | 5.83% |
| 1 day | energy_only | random_forest | 1,786.2 | 5.62% |
| 1 day | weather_enabled | persistence | 2,192.7 | 6.94% |
| 1 day | weather_enabled | moving_average_24h | 4,271.3 | 14.01% |
| 1 day | weather_enabled | decision_tree | 1,869.3 | 5.89% |
| 1 day | weather_enabled | random_forest | 1,755.6 | 5.54% |
| 1 week | energy_only | persistence | 3,434.0 | 10.66% |
| 1 week | energy_only | moving_average_24h | 4,772.2 | 15.48% |
| 1 week | energy_only | decision_tree | 3,112.9 | 9.83% |
| 1 week | energy_only | random_forest | 2,915.7 | 9.24% |
| 1 week | weather_enabled | persistence | 3,434.0 | 10.66% |
| 1 week | weather_enabled | moving_average_24h | 4,772.2 | 15.48% |
| 1 week | weather_enabled | decision_tree | 3,139.4 | 9.94% |
| 1 week | weather_enabled | random_forest | 2,916.0 | 9.24% |

## Most important features (permutation importance, energy + weather)

| horizon | model | top 5 (permutation) |
| --- | --- | --- |
| 1 hour | decision_tree | lag_1, hour, hdh_wma_24h, day_of_week, cdh_wma_24h |
| 1 hour | random_forest | lag_1, hour, hdh_wma_24h, day_of_week, cdh_wma_24h |
| 1 day | decision_tree | lag_1, day_of_week, hour, rolling_mean_24, is_weekend |
| 1 day | random_forest | lag_1, day_of_week, hour, rolling_mean_24, dh_slope_96d |
| 1 week | decision_tree | lag_1, hour, day_of_year, dh_slope_96d, hdh_wma_16d |
| 1 week | random_forest | lag_1, day_of_year, hour, lag_168, hdh_wma_96d |

## Rank agreement with the variance rankings (Spearman, permutation importance)

| model | horizon | pca | nnmf | target_pca |
| --- | --- | --- | --- | --- |
| decision_tree | 1 hour | -0.54 | -0.07 | 0.4 |
| decision_tree | 1 day | -0.37 | 0.33 | 0.21 |
| decision_tree | 1 week | -0.31 | -0.02 | 0.3 |
| random_forest | 1 hour | -0.52 | -0.05 | 0.39 |
| random_forest | 1 day | -0.52 | -0.06 | 0.54 |
| random_forest | 1 week | -0.49 | -0.08 | 0.43 |

## Notes

- `is_weekend` and `is_holiday` describe the forecast hour, not the target hour.
- `hour` and `day_of_week` are numeric codes, so PCA/NNMF treat 23:00 and 00:00 as far apart.
- Weather keeps its one-hour lag; at every horizon it only uses weather observed before the
  forecast time.
