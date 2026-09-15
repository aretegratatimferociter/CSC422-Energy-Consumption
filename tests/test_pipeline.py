import numpy as np
import pandas as pd

from energy_forecasting.data import TARGET_COLUMN, load_pjme_csv
from energy_forecasting.evaluation import chronological_split, regression_metrics
from energy_forecasting.features import build_features
from energy_forecasting.modeling import evaluate_models


def hourly_frame(periods: int = 500) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="h")
    values = 30_000 + 1_500 * np.sin(np.arange(periods) * 2 * np.pi / 24)
    return pd.DataFrame({TARGET_COLUMN: values}, index=index)


def test_loader_sorts_and_averages_duplicate_timestamps(tmp_path):
    source = tmp_path / "pjme.csv"
    source.write_text(
        "Datetime,PJME_MW\n2024-01-01 01:00:00,12\n2024-01-01 00:00:00,10\n"
        "2024-01-01 01:00:00,14\n"
    )
    loaded = load_pjme_csv(source)
    assert loaded.index.is_monotonic_increasing
    assert loaded.iloc[-1][TARGET_COLUMN] == 13


def test_loader_materializes_missing_clock_hours(tmp_path):
    source = tmp_path / "pjme.csv"
    source.write_text(
        "Datetime,PJME_MW\n2024-01-01 00:00:00,10\n2024-01-01 02:00:00,14\n"
    )
    loaded = load_pjme_csv(source)
    assert len(loaded) == 3
    assert pd.isna(loaded.loc["2024-01-01 01:00:00", TARGET_COLUMN])


def test_features_use_only_prior_consumption():
    frame = hourly_frame(200)
    featured = build_features(frame, lags=(1, 24), rolling_windows=(24,))
    timestamp = featured.index[0]
    source_position = frame.index.get_loc(timestamp)
    assert featured.loc[timestamp, "lag_1"] == frame.iloc[source_position - 1][TARGET_COLUMN]
    assert featured.loc[timestamp, "rolling_mean_24"] == frame.iloc[
        source_position - 24 : source_position
    ][TARGET_COLUMN].mean()


def test_chronological_split_preserves_order():
    train, test = chronological_split(hourly_frame(100), test_fraction=0.2)
    assert len(train) == 80
    assert len(test) == 20
    assert train.index.max() < test.index.min()


def test_metrics_are_zero_for_exact_prediction():
    actual = np.array([1.0, 2.0, 3.0])
    assert regression_metrics(actual, actual) == {"mae": 0.0, "rmse": 0.0, "mape_percent": 0.0}


def test_all_models_run_on_shared_features():
    frame = build_features(hourly_frame())
    train, test = chronological_split(frame)
    results = evaluate_models(train, test)
    assert {result.name for result in results} == {
        "persistence",
        "moving_average_24h",
        "decision_tree",
        "random_forest",
    }
    assert all(len(result.predictions) == len(test) for result in results)
