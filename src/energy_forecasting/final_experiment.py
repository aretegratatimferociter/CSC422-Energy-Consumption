"""Run the leakage-safe final model selection and weather ablation experiment."""

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from energy_forecasting.data import TARGET_COLUMN, load_pjme_csv
from energy_forecasting.evaluation import chronological_split
from energy_forecasting.features import build_features, feature_columns, load_weather_features
from energy_forecasting.modeling import (
    ModelResult,
    evaluate_models,
    tune_models_time_series,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/PJME_hourly_clean.csv"),
    )
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/weather_features.csv"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/final"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--cv-splits", type=int, default=3)
    return parser.parse_args()


def aligned_feature_sets(data_path: Path, weather_path: Path) -> dict[str, pd.DataFrame]:
    """Build energy-only and weather-enabled frames over identical timestamps."""
    energy = load_pjme_csv(data_path)
    weather = load_weather_features(weather_path)
    frames = {
        "energy_only": build_features(energy),
        "weather_enabled": build_features(energy, weather=weather),
    }
    common_index = frames["energy_only"].index.intersection(frames["weather_enabled"].index)
    if common_index.empty:
        raise ValueError("Energy and weather feature sets do not share any timestamps")
    return {name: frame.loc[common_index].copy() for name, frame in frames.items()}


def _metrics_table(
    result_sets: dict[str, list[ModelResult]],
) -> pd.DataFrame:
    rows = [
        {"feature_set": feature_set, "model": result.name, **result.metrics}
        for feature_set, results in result_sets.items()
        for result in results
    ]
    metrics = pd.DataFrame(rows)
    energy_mae = (
        metrics[metrics["feature_set"] == "energy_only"]
        .set_index("model")["mae"]
        .to_dict()
    )
    metrics["mae_improvement_vs_energy_only_percent"] = metrics.apply(
        lambda row: 100 * (energy_mae[row["model"]] - row["mae"]) / energy_mae[row["model"]]
        if row["feature_set"] == "weather_enabled"
        else 0.0,
        axis=1,
    )
    return metrics


def _plot_metric_comparison(metrics: pd.DataFrame, output: Path) -> None:
    models = ["persistence", "moving_average_24h", "decision_tree", "random_forest"]
    display = {
        "persistence": "Persistence",
        "moving_average_24h": "24-hour average",
        "decision_tree": "Decision Tree",
        "random_forest": "Random Forest",
    }
    pivot = metrics.pivot(index="model", columns="feature_set", values="mae").loc[models]
    x = np.arange(len(models))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.35, 1]})
    axes[0].bar(
        x - width / 2,
        pivot["energy_only"],
        width,
        label="Energy only",
        color="#6B7280",
    )
    axes[0].bar(
        x + width / 2,
        pivot["weather_enabled"],
        width,
        label="Energy + weather",
        color="#2563EB",
    )
    axes[0].set_ylabel("MAE (MW)")
    axes[0].set_title("All models")
    axes[0].set_xticks(x, [display[model] for model in models], rotation=12)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    tree_models = ["decision_tree", "random_forest"]
    tree_pivot = pivot.loc[tree_models]
    tree_x = np.arange(len(tree_models))
    energy_bars = axes[1].bar(
        tree_x - width / 2,
        tree_pivot["energy_only"],
        width,
        color="#6B7280",
    )
    weather_bars = axes[1].bar(
        tree_x + width / 2,
        tree_pivot["weather_enabled"],
        width,
        color="#2563EB",
    )
    axes[1].bar_label(energy_bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].bar_label(weather_bars, fmt="%.1f", padding=3, fontsize=9)
    axes[1].set_title("Tree models (zoomed)")
    axes[1].set_xticks(tree_x, [display[model] for model in tree_models])
    axes[1].set_ylim(0, tree_pivot.to_numpy().max() * 1.18)
    axes[1].grid(axis="y", alpha=0.25)
    figure.suptitle("Final holdout error by feature set")
    figure.tight_layout()
    figure.savefig(output / "mae_comparison.png", dpi=180)
    plt.close(figure)


def _plot_feature_importance(
    importance: pd.DataFrame,
    final_model: str,
    output: Path,
) -> None:
    """Show the dominant lag feature and the smaller predictors at readable scales."""
    top = importance.head(15)
    remainder = top.iloc[1:].sort_values("importance")
    figure, axes = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [0.85, 1.5]})
    axes[0].barh(top.iloc[:2]["feature"], top.iloc[:2]["importance"], color="#2563EB")
    axes[0].set_title("Dominant predictors")
    axes[0].set_xlabel("Importance")
    axes[0].grid(axis="x", alpha=0.2)
    axes[1].barh(remainder["feature"], remainder["importance"], color="#2563EB")
    axes[1].set_title("Remaining top predictors (zoomed)")
    axes[1].set_xlabel("Importance")
    axes[1].grid(axis="x", alpha=0.2)
    figure.suptitle(f"{final_model.replace('_', ' ').title()} feature importance")
    figure.tight_layout()
    figure.savefig(output / "feature_importance.png", dpi=180)
    plt.close(figure)


def _plot_predictions(
    test: pd.DataFrame,
    results: dict[str, list[ModelResult]],
    final_model: str,
    output: Path,
) -> None:
    hours = min(24 * 14, len(test))
    index = test.index[-hours:]
    actual = test[TARGET_COLUMN].iloc[-hours:]
    energy_result = next(item for item in results["energy_only"] if item.name == final_model)
    weather_result = next(item for item in results["weather_enabled"] if item.name == final_model)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(index, actual, label="Actual", color="#111827", linewidth=1.7)
    axis.plot(
        index,
        energy_result.predictions[-hours:],
        label="Energy only",
        color="#6B7280",
        alpha=0.85,
    )
    axis.plot(
        index,
        weather_result.predictions[-hours:],
        label="Energy + weather",
        color="#2563EB",
        alpha=0.85,
    )
    axis.set_title(f"Actual versus {final_model.replace('_', ' ').title()} forecast (last 14 days)")
    axis.set_ylabel("PJME load (MW)")
    axis.legend(ncol=3)
    axis.grid(alpha=0.2)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(output / "actual_vs_predicted.png", dpi=180)
    plt.close(figure)


def _plot_residuals(
    test: pd.DataFrame,
    result: ModelResult,
    output: Path,
) -> None:
    residuals = test[TARGET_COLUMN].to_numpy() - result.predictions
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(result.predictions, residuals, s=5, alpha=0.18, color="#2563EB")
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xlabel("Predicted load (MW)")
    axes[0].set_ylabel("Residual: actual - predicted (MW)")
    axes[0].set_title("Residuals versus predictions")
    axes[1].hist(residuals, bins=60, color="#2563EB", alpha=0.85)
    axes[1].axvline(0, color="#111827", linewidth=1)
    axes[1].set_xlabel("Residual (MW)")
    axes[1].set_ylabel("Hours")
    axes[1].set_title("Residual distribution")
    figure.suptitle(f"Weather-enabled {result.name.replace('_', ' ').title()}")
    figure.tight_layout()
    figure.savefig(output / "residual_analysis.png", dpi=180)
    plt.close(figure)


def _write_summary(
    output: Path,
    metrics: pd.DataFrame,
    selected_parameters: dict[str, dict[str, Any]],
    final_model: str,
    test: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    final_rows = metrics[metrics["model"] == final_model].set_index("feature_set")
    energy = final_rows.loc["energy_only"]
    weather = final_rows.loc["weather_enabled"]
    improvement = weather["mae_improvement_vs_energy_only_percent"]
    direction = "improved" if improvement >= 0 else "worsened"
    top_features = ", ".join(importance.head(8)["feature"].tolist())

    table = metrics.copy()
    table["mae"] = table["mae"].map(lambda value: f"{value:,.1f}")
    table["rmse"] = table["rmse"].map(lambda value: f"{value:,.1f}")
    table["mape_percent"] = table["mape_percent"].map(lambda value: f"{value:.2f}%")
    table["feature_set"] = table["feature_set"].str.replace("_", " ").str.title()
    table["model"] = table["model"].str.replace("_", " ").str.title()
    markdown_rows = ["| Feature set | Model | MAE (MW) | RMSE (MW) | MAPE |", "| --- | --- | ---: | ---: | ---: |"]
    markdown_rows.extend(
        f"| {row.feature_set} | {row.model} | {row.mae} | {row.rmse} | {row.mape_percent} |"
        for row in table.itertuples()
    )

    text = f"""# Final Forecasting Experiment

This experiment completes the model-selection and weather-ablation work identified in the midterm
report. It uses expanding-window time-series cross-validation on the training period, followed by
one untouched chronological 20% holdout from {test.index.min():%Y-%m-%d} through
{test.index.max():%Y-%m-%d}. Energy-only and weather-enabled runs use identical timestamps, splits,
and tuned model settings.

## Selected model

The validation procedure selected **{final_model.replace('_', ' ').title()}**. Its tuned parameters
were `{json.dumps(selected_parameters[final_model], sort_keys=True)}`.

On the final holdout, adding weather {direction} MAE by **{abs(improvement):.2f}%**, from
**{energy['mae']:,.1f} MW** to **{weather['mae']:,.1f} MW**. The weather-enabled result had an RMSE
of **{weather['rmse']:,.1f} MW** and a MAPE of **{weather['mape_percent']:.2f}%**.

## Holdout results

{chr(10).join(markdown_rows)}

## Interpretation

The most important predictors in the final weather-enabled model were: {top_features}. This gives
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
"""
    (output / "summary.md").write_text(text)


def run(
    data_path: Path,
    weather_path: Path,
    output: Path,
    test_fraction: float = 0.2,
    cv_splits: int = 3,
) -> pd.DataFrame:
    """Run validation, controlled ablation, final evaluation, and artifact generation."""
    frames = aligned_feature_sets(data_path, weather_path)
    splits = {
        name: chronological_split(frame, test_fraction)
        for name, frame in frames.items()
    }
    weather_train, weather_test = splits["weather_enabled"]
    selected_parameters, cv_results = tune_models_time_series(
        weather_train,
        n_splits=cv_splits,
    )

    result_sets = {
        name: evaluate_models(train, test, model_parameters=selected_parameters)
        for name, (train, test) in splits.items()
    }
    metrics = _metrics_table(result_sets)

    selected_cv = cv_results[cv_results["selected"]]
    model_cv = selected_cv.groupby("model", as_index=False)["mean_mae"].first()
    final_model = model_cv.sort_values("mean_mae").iloc[0]["model"]
    final_result = next(
        item for item in result_sets["weather_enabled"] if item.name == final_model
    )

    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics.csv", index=False)
    cv_results.to_csv(output / "time_series_cv_results.csv", index=False)
    selection = {
        "selection_rule": "lowest mean MAE in expanding-window validation",
        "cv_splits": cv_splits,
        "test_fraction": test_fraction,
        "final_model": final_model,
        "parameters": selected_parameters,
    }
    (output / "selected_model.json").write_text(json.dumps(selection, indent=2) + "\n")

    model_directory = Path("models")
    model_directory.mkdir(exist_ok=True)
    joblib.dump(final_result.model, model_directory / f"final_{final_model}_weather.joblib")

    importance = pd.DataFrame(
        {
            "feature": feature_columns(weather_train),
            "importance": final_result.model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)

    _plot_feature_importance(importance, final_model, output)
    _plot_metric_comparison(metrics, output)
    _plot_predictions(weather_test, result_sets, final_model, output)
    _plot_residuals(weather_test, final_result, output)
    _write_summary(
        output,
        metrics,
        selected_parameters,
        final_model,
        weather_test,
        importance,
    )
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run(
        args.data,
        args.weather,
        args.output,
        args.test_fraction,
        args.cv_splits,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
