"""Command-line entry point for PJME data preprocessing."""

import argparse
from pathlib import Path

from energy_forecasting.data import preprocess_pjme_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/PJME_hourly.csv"),
        help="Raw PJME CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/PJME_hourly_clean.csv"),
        help="Cleaned hourly CSV",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("data/processed/data_quality_report.json"),
        help="Data-quality report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = preprocess_pjme_csv(args.input, args.output, args.report)
    for key, value in report.to_dict().items():
        if key != "cleaning_policy":
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
