"""Loading, cleaning, and validation for the PJME hourly energy dataset."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

TIMESTAMP_COLUMN = "Datetime"
TARGET_COLUMN = "PJME_MW"
IMPUTED_COLUMN = "is_imputed"
OUTLIER_COLUMN = "is_outlier"


@dataclass(frozen=True)
class DataQualityReport:
    """Counts and boundaries recorded during preprocessing."""

    source_rows: int
    output_rows: int
    invalid_timestamp_rows: int
    invalid_target_rows: int
    non_positive_target_rows: int
    duplicate_rows_resolved: int
    missing_hours_added: int
    imputed_rows: int
    statistical_outlier_rows: int
    start_timestamp: str
    end_timestamp: str

    def to_dict(self) -> dict[str, int | str | dict[str, str]]:
        report: dict[str, int | str | dict[str, str]] = asdict(self)
        report["cleaning_policy"] = {
            "duplicates": "average values sharing the same timestamp",
            "missing_or_invalid_targets": "fill from 168h lag, then 24h lag, then prior hour",
            "outliers": "flag values outside 3 IQR; retain for modeling",
        }
        return report


def _validate_columns(frame: pd.DataFrame) -> None:
    missing = {TIMESTAMP_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def preprocess_pjme_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, DataQualityReport]:
    """Clean raw PJME observations and return an auditable hourly series.

    The cleaning policy is deliberately conservative. Invalid and non-positive
    targets are treated as missing. Missing hours are imputed from historical
    observations only, preventing future information from leaking backward.
    Statistical outliers are flagged but retained because real demand peaks are
    important forecasting observations.
    """
    _validate_columns(frame)
    source_rows = len(frame)
    working = frame[[TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    working[TIMESTAMP_COLUMN] = pd.to_datetime(working[TIMESTAMP_COLUMN], errors="coerce")
    working[TARGET_COLUMN] = pd.to_numeric(working[TARGET_COLUMN], errors="coerce")

    invalid_timestamp_rows = int(working[TIMESTAMP_COLUMN].isna().sum())
    invalid_target_rows = int(working[TARGET_COLUMN].isna().sum())
    non_positive = working[TARGET_COLUMN].notna() & (working[TARGET_COLUMN] <= 0)
    non_positive_target_rows = int(non_positive.sum())
    working.loc[non_positive, TARGET_COLUMN] = pd.NA
    working = working.dropna(subset=[TIMESTAMP_COLUMN])

    duplicate_rows_resolved = int(
        len(working) - working[TIMESTAMP_COLUMN].nunique(dropna=True)
    )
    hourly = (
        working.groupby(TIMESTAMP_COLUMN, as_index=False, sort=True)[TARGET_COLUMN]
        .mean()
        .set_index(TIMESTAMP_COLUMN)
        .sort_index()
    )
    if hourly.empty:
        raise ValueError("No valid timestamped observations were found")

    complete_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    missing_hours_added = int(len(complete_index) - len(hourly))
    hourly = hourly.reindex(complete_index).rename_axis(TIMESTAMP_COLUMN)
    missing_mask = hourly[TARGET_COLUMN].isna()

    filled = hourly[TARGET_COLUMN].copy()
    for lag_hours in (168, 24):
        filled = filled.fillna(filled.shift(lag_hours))
    filled = filled.ffill()
    if filled.isna().any():
        raise ValueError("Unable to impute target values without future observations")
    hourly[TARGET_COLUMN] = filled.astype(float)
    hourly[IMPUTED_COLUMN] = missing_mask.astype(bool)

    first_quartile, third_quartile = hourly[TARGET_COLUMN].quantile([0.25, 0.75])
    iqr = third_quartile - first_quartile
    lower_bound = first_quartile - 3 * iqr
    upper_bound = third_quartile + 3 * iqr
    hourly[OUTLIER_COLUMN] = (
        (hourly[TARGET_COLUMN] < lower_bound) | (hourly[TARGET_COLUMN] > upper_bound)
    )

    report = DataQualityReport(
        source_rows=source_rows,
        output_rows=len(hourly),
        invalid_timestamp_rows=invalid_timestamp_rows,
        invalid_target_rows=invalid_target_rows,
        non_positive_target_rows=non_positive_target_rows,
        duplicate_rows_resolved=duplicate_rows_resolved,
        missing_hours_added=missing_hours_added,
        imputed_rows=int(hourly[IMPUTED_COLUMN].sum()),
        statistical_outlier_rows=int(hourly[OUTLIER_COLUMN].sum()),
        start_timestamp=hourly.index.min().isoformat(),
        end_timestamp=hourly.index.max().isoformat(),
    )
    return hourly, report


def preprocess_pjme_csv(
    input_path: str | Path,
    output_path: str | Path,
    report_path: str | Path,
) -> DataQualityReport:
    """Clean a raw CSV and write the processed data and quality report."""
    cleaned, report = preprocess_pjme_frame(pd.read_csv(input_path))
    output_path = Path(output_path)
    report_path = Path(report_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, date_format="%Y-%m-%d %H:%M:%S")
    report_path.write_text(json.dumps(report.to_dict(), indent=2) + "\n")
    return report


def load_pjme_csv(path: str | Path) -> pd.DataFrame:
    """Load and preprocess PJME data for feature construction."""
    cleaned, _ = preprocess_pjme_frame(pd.read_csv(path))
    return cleaned[[TARGET_COLUMN]]
