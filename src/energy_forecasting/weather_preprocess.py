"""Command-line entry point for building population-weighted weather features."""

# Author: plholt3

import argparse
import json
from pathlib import Path

import pandas as pd

from energy_forecasting.data import TIMESTAMP_COLUMN
from energy_forecasting.weather import build_weather_features, weather_report


def parse_args() -> argparse.Namespace:
    """Read input and output paths from the command line, with project defaults."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pjme",
        type=Path,
        default=Path("data/processed/PJME_hourly_clean.csv"),
        help="Cleaned PJME CSV whose hourly index the features are aligned to",
    )
    parser.add_argument("--weather-dir", type=Path, default=Path("data/raw/weather"))
    parser.add_argument("--population-dir", type=Path, default=Path("data/raw/population"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/processed/weather_features.csv")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("data/processed/weather_quality_report.json")
    )
    return parser.parse_args()


def main() -> None:
    """Build weather features on the PJME hourly index and write the CSV and quality report."""
    args = parse_args()
    pjme = pd.read_csv(args.pjme, usecols=[TIMESTAMP_COLUMN], parse_dates=[TIMESTAMP_COLUMN])
    index = pd.DatetimeIndex(pjme[TIMESTAMP_COLUMN])
    features, reports, populations = build_weather_features(
        index, args.weather_dir, args.population_dir
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, date_format="%Y-%m-%d %H:%M:%S")
    report = weather_report(reports, populations)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    for city in report["cities"]:
        print(city)


if __name__ == "__main__":
    main()
