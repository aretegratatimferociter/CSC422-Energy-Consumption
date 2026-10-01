"""Leakage-safe feature engineering for hourly forecasts."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from energy_forecasting.data import TARGET_COLUMN, TIMESTAMP_COLUMN

DEFAULT_LAGS = (1, 24, 168)
DEFAULT_ROLLING_WINDOWS = (24, 168)

# Population-weighted weather window features produced by `energy-weather`.
# Author: plholt3
WEATHER_FEATURES = tuple(
    f"{measure}_{window}"
    for window in ("24h", "16d", "96d")
    for measure in ("hdh_wma", "cdh_wma", "dh_std", "dh_slope")
)
# Weather is shifted by this many hours so a row never sees the weather of the hour it predicts.
# Author: plholt3
DEFAULT_WEATHER_LAG = 1


def load_weather_features(path: str | Path) -> pd.DataFrame:
    """Read the weather window features from the CSV written by ``energy-weather``.

    Author: plholt3
    """
    weather = pd.read_csv(path, index_col=TIMESTAMP_COLUMN, parse_dates=True)
    missing = set(WEATHER_FEATURES) - set(weather.columns)
    if missing:
        raise ValueError(f"Weather file is missing columns: {sorted(missing)}")
    return weather[list(WEATHER_FEATURES)]


def add_weather_features(
    frame: pd.DataFrame, weather: pd.DataFrame, lag: int = DEFAULT_WEATHER_LAG
) -> pd.DataFrame:
    """Join weather features shifted back ``lag`` hours onto an hourly frame.

    Each weather feature is computed as of its own hour, so shifting by at least one hour means
    the row for hour t only sees weather through hour t - lag.

    Author: plholt3
    """
    if lag < 1:
        raise ValueError("Weather lag must be at least one hour")
    uncovered = frame.index.difference(weather.index)
    if len(uncovered):
        raise ValueError(f"Weather features are missing {len(uncovered)} hours, e.g. {uncovered[0]}")
    lagged = weather.shift(lag, freq="h").reindex(frame.index)
    return frame.join(lagged)


def build_features(
    frame: pd.DataFrame,
    lags: Iterable[int] = DEFAULT_LAGS,
    rolling_windows: Iterable[int] = DEFAULT_ROLLING_WINDOWS,
    weather: pd.DataFrame | None = None,
    weather_lag: int = DEFAULT_WEATHER_LAG,
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

    # Optional weather window features, lagged to avoid leakage. Author: plholt3
    if weather is not None:
        result = add_weather_features(result, weather, weather_lag)

    return result.dropna()


def feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return predictor columns in stable input order."""
    return [column for column in frame.columns if column != TARGET_COLUMN]
