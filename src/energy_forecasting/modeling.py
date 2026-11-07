"""Model definitions, time-series tuning, and a shared evaluation workflow."""

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
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


def candidate_models(
    random_state: int = 42,
    parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, RegressorMixin]:
    """Return comparable tree models with deterministic defaults or tuned parameters."""
    parameters = parameters or {}
    tree_parameters = {
        "max_depth": 16,
        "min_samples_leaf": 4,
        "random_state": random_state,
        **parameters.get("decision_tree", {}),
    }
    forest_parameters = {
        "n_estimators": 200,
        "max_depth": 20,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "random_state": random_state,
        **parameters.get("random_forest", {}),
    }
    return {
        "decision_tree": DecisionTreeRegressor(**tree_parameters),
        "random_forest": RandomForestRegressor(**forest_parameters),
    }


def tuning_candidates() -> dict[str, list[dict[str, Any]]]:
    """Return a small, reproducible search space suitable for expanding-window CV."""
    return {
        "decision_tree": [
            {"max_depth": 12, "min_samples_leaf": 2},
            {"max_depth": 16, "min_samples_leaf": 4},
            {"max_depth": 24, "min_samples_leaf": 2},
            {"max_depth": None, "min_samples_leaf": 4},
        ],
        "random_forest": [
            {
                "n_estimators": 150,
                "max_depth": 16,
                "min_samples_leaf": 2,
                "max_features": 1.0,
            },
            {
                "n_estimators": 200,
                "max_depth": 20,
                "min_samples_leaf": 2,
                "max_features": 1.0,
            },
            {
                "n_estimators": 200,
                "max_depth": None,
                "min_samples_leaf": 2,
                "max_features": 0.8,
            },
            {
                "n_estimators": 250,
                "max_depth": 24,
                "min_samples_leaf": 1,
                "max_features": 0.8,
            },
        ],
    }


def tune_models_time_series(
    train: pd.DataFrame,
    n_splits: int = 3,
    candidates: dict[str, list[dict[str, Any]]] | None = None,
    random_state: int = 42,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    """Tune tree models with expanding-window validation and return every fold score.

    Candidate selection minimizes mean validation MAE. The splitter never trains on rows that
    occur after a validation row, which keeps model selection consistent with the final forecast.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")
    if len(train) <= n_splits:
        raise ValueError("Training data must contain more rows than time-series splits")

    columns = feature_columns(train)
    splitter = TimeSeriesSplit(n_splits=n_splits)
    candidates = candidates or tuning_candidates()
    records: list[dict[str, Any]] = []

    for model_name, model_candidates in candidates.items():
        if model_name not in {"decision_tree", "random_forest"}:
            raise ValueError(f"Unsupported model in tuning candidates: {model_name}")
        for candidate_index, model_parameters in enumerate(model_candidates):
            for fold, (fit_index, validation_index) in enumerate(splitter.split(train), start=1):
                model = candidate_models(
                    random_state=random_state,
                    parameters={model_name: model_parameters},
                )[model_name]
                fit = train.iloc[fit_index]
                validation = train.iloc[validation_index]
                model.fit(fit[columns], fit[TARGET_COLUMN])
                predictions = model.predict(validation[columns])
                metrics = regression_metrics(validation[TARGET_COLUMN], predictions)
                records.append(
                    {
                        "model": model_name,
                        "candidate": candidate_index,
                        "parameters": json.dumps(model_parameters, sort_keys=True),
                        "fold": fold,
                        "train_rows": len(fit),
                        "validation_rows": len(validation),
                        "validation_start": validation.index.min(),
                        "validation_end": validation.index.max(),
                        **metrics,
                    }
                )

    scores = pd.DataFrame(records)
    means = (
        scores.groupby(["model", "candidate", "parameters"], as_index=False)
        .agg(mean_mae=("mae", "mean"), mean_rmse=("rmse", "mean"), mean_mape=("mape_percent", "mean"))
        .sort_values(["model", "mean_mae", "mean_rmse"])
    )
    winners = means.groupby("model", sort=False).first().reset_index()
    selected = {
        row["model"]: json.loads(row["parameters"])
        for _, row in winners.iterrows()
    }
    scores = scores.merge(
        means,
        on=["model", "candidate", "parameters"],
        how="left",
        validate="many_to_one",
    )
    scores["selected"] = scores.apply(
        lambda row: json.loads(row["parameters"]) == selected[row["model"]], axis=1
    )
    return selected, scores.sort_values(["model", "candidate", "fold"]).reset_index(drop=True)


def evaluate_models(
    train: pd.DataFrame,
    test: pd.DataFrame,
    model_parameters: dict[str, dict[str, Any]] | None = None,
) -> list[ModelResult]:
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

    for name, model in candidate_models(parameters=model_parameters).items():
        model.fit(train[columns], train[TARGET_COLUMN])
        predictions = model.predict(test[columns])
        results.append(
            ModelResult(name, predictions, regression_metrics(test[TARGET_COLUMN], predictions), model)
        )
    return results
