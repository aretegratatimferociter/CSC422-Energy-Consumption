"""Model definitions and a shared evaluation workflow."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from energy_forecasting.data import TARGET_COLUMN
from energy_forecasting.evaluation import regression_metrics
from energy_forecasting.features import feature_columns


@dataclass
class ModelResult:
    name: str
    predictions: np.ndarray
    metrics: dict[str, float]
    model: RegressorMixin | None = None


def candidate_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    """Return comparable tree models using deterministic defaults."""
    return {
        "decision_tree": DecisionTreeRegressor(
            max_depth=16, min_samples_leaf=4, random_state=random_state
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def evaluate_models(train: pd.DataFrame, test: pd.DataFrame) -> list[ModelResult]:
    """Evaluate persistence, moving average, decision tree, and random forest."""
    columns = feature_columns(train)
    results = []

    baselines = {
        "persistence": test["lag_1"].to_numpy(),
        "moving_average_24h": test["rolling_mean_24"].to_numpy(),
    }
    for name, predictions in baselines.items():
        results.append(
            ModelResult(name, predictions, regression_metrics(test[TARGET_COLUMN], predictions))
        )

    for name, model in candidate_models().items():
        model.fit(train[columns], train[TARGET_COLUMN])
        predictions = model.predict(test[columns])
        results.append(
            ModelResult(name, predictions, regression_metrics(test[TARGET_COLUMN], predictions), model)
        )
    return results
