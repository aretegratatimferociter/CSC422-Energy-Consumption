"""Leakage-safe feature engineering for hourly forecasts."""

from collections.abc import Iterable

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from energy_forecasting.data import TARGET_COLUMN

DEFAULT_LAGS = (1, 24, 168)
DEFAULT_ROLLING_WINDOWS = (24, 168)


def build_features(
    frame: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
) -> pd.DataFrame:
    """Add calendar, holiday, lag, and trailing-average features.

    Every consumption-derived predictor is shifted by at least one hour, so the
    value being predicted can never enter its own feature row.
    """
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Expected target column {TARGET_COLUMN!r}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("Feature input must use a DatetimeIndex")

    result = frame.copy().sort_index()
    index = result.index
    result["hour"] = index.hour
    result["day_of_week"] = index.dayofweek
    result["day_of_year"] = index.dayofyear
    result["month"] = index.month
    result["quarter"] = index.quarter
    result["year"] = index.year
    result["is_weekend"] = (index.dayofweek >= 5).astype(int)

    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=index.min().normalize(), end=index.max().normalize())
    result["is_holiday"] = index.normalize().isin(holidays).astype(int)

    history = result[TARGET_COLUMN].shift(1)
    for lag in lags:
        if lag < 1:
            raise ValueError("Lag values must be positive")
        result[f"lag_{lag}"] = result[TARGET_COLUMN].shift(lag)
    for window in rolling_windows:
        if window < 1:
            raise ValueError("Rolling windows must be positive")
        result[f"rolling_mean_{window}"] = history.rolling(window=window).mean()

    return result.dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return predictor columns in stable input order."""
    return [column for column in frame.columns if column != TARGET_COLUMN]
