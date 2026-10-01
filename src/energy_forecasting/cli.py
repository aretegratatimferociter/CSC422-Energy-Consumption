"""Command-line entry point for the end-to-end experiment."""

import argparse
from pathlib import Path

import joblib
import pandas as pd

from energy_forecasting.data import TARGET_COLUMN, load_pjme_csv
from energy_forecasting.evaluation import chronological_split
from energy_forecasting.features import build_features, feature_columns, load_weather_features
from energy_forecasting.modeling import evaluate_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to PJME_hourly.csv")
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    # Weather feature options. Author: plholt3
    parser.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/weather_features.csv"),
        help="Weather features written by energy-weather",
    )
    parser.add_argument(
        "--no-weather", action="store_true", help="Run without weather features for comparison"
    )
    return parser.parse_args()


def run(
    data_path: Path,
    output: Path,
    test_fraction: float = 0.2,
    weather_path: Path | None = None,
) -> pd.DataFrame:
    """Run the shared pipeline and write reproducible result artifacts."""
    # Weather features are optional; None runs the energy-only feature set. Author: plholt3
    weather = load_weather_features(weather_path) if weather_path is not None else None
    frame = build_features(load_pjme_csv(data_path), weather=weather)
    train, test = chronological_split(frame, test_fraction)
    results = evaluate_models(train, test)

    output.mkdir(parents=True, exist_ok=True)
    model_directory = output.parent / "models"
    model_directory.mkdir(parents=True, exist_ok=True)

    metrics = pd.DataFrame({"model": item.name, **item.metrics} for item in results)
    metrics.to_csv(output / "metrics.csv", index=False)

    predictions = pd.DataFrame(index=test.index)
    predictions["actual"] = test[TARGET_COLUMN]
    for item in results:
        predictions[item.name] = item.predictions
        if item.model is not None:
            joblib.dump(item.model, model_directory / f"{item.name}.joblib")
            importance = pd.DataFrame(
                {"feature": feature_columns(train), "importance": item.model.feature_importances_}
            ).sort_values("importance", ascending=False)
            importance.to_csv(output / f"{item.name}_feature_importance.csv", index=False)
    predictions.to_csv(output / "predictions.csv")
    return metrics


def main() -> None:
    args = parse_args()
    weather_path = None if args.no_weather else args.weather  # Author: plholt3
    metrics = run(args.data, args.output, args.test_fraction, weather_path)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
