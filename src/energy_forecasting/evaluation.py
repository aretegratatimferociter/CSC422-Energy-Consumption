"""Chronological splitting and regression metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def chronological_split(
    frame: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split an ordered frame without shuffling future observations into training."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(frame) < 2:
        raise ValueError("At least two observations are required")
    split_index = min(max(int(len(frame) * (1 - test_fraction)), 1), len(frame) - 1)
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def regression_metrics(actual: pd.Series | np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE, RMSE, and zero-safe MAPE."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    nonzero = actual_array != 0
    mape = (
        float(np.mean(np.abs((actual_array[nonzero] - predicted_array[nonzero]) / actual_array[nonzero])) * 100)
        if nonzero.any()
        else float("nan")
    )
    return {
        "mae": float(mean_absolute_error(actual_array, predicted_array)),
        "rmse": float(mean_squared_error(actual_array, predicted_array) ** 0.5),
        "mape_percent": mape,
    }
