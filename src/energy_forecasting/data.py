"""Loading and validation for the PJME hourly energy dataset."""

from pathlib import Path

import pandas as pd

TIMESTAMP_COLUMN = "Datetime"
TARGET_COLUMN = "PJME_MW"


def load_pjme_csv(path: str | Path) -> pd.DataFrame:
    """Load PJME data, normalize timestamps, and resolve duplicate hours.

    Duplicate wall-clock timestamps can occur around daylight-saving transitions.
    They are combined by averaging rather than silently keeping an arbitrary row.
    Missing hours remain missing; later feature construction drops rows without a
    target instead of filling the value from the future.
    """
    frame = pd.read_csv(path)
    missing = {TIMESTAMP_COLUMN, TARGET_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame[[TIMESTAMP_COLUMN, TARGET_COLUMN]].copy()
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[TARGET_COLUMN] = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN])
    frame = (
        frame.groupby(TIMESTAMP_COLUMN, as_index=False, sort=True)[TARGET_COLUMN]
        .mean()
        .set_index(TIMESTAMP_COLUMN)
        .sort_index()
    )

    if frame.empty:
        raise ValueError("No valid timestamped observations were found")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("Timestamps must be in chronological order")
    return frame
