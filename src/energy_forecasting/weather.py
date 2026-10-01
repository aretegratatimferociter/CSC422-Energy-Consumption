"""Population-weighted heating and cooling degree hours from NOAA hourly observations."""

# Author: plholt3

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOCAL_TIMEZONE = "America/New_York"
BASE_TEMPERATURE_F = 65.0
DONOR_OFFSET_HOURS = 48
MIN_DONOR_PAIRS = 6

# ISD report types, in order of preference: routine hourly METAR, then special METAR.
REPORT_TYPE_PRIORITY = {"FM-15": 0, "FM-16": 1}
# ISD temperature quality codes 2, 3, 6, and 7 mark suspect or erroneous values.
REJECTED_QUALITY_CODES = {"2", "3", "6", "7"}
MISSING_TEMPERATURE = "+9999"

# Each snapshot applies from its publication date until the next snapshot is published,
# so a timestamp is only weighted with population figures that existed at that time.
# Census counts use the April 1 statutory deadline by which every state had received them;
# the vintage 2015 county estimates were publicly released on March 24, 2016.
POPULATION_SNAPSHOTS = (
    ("2001-04-01", "co-est00int-tot.csv", "ESTIMATESBASE2000"),
    ("2011-04-01", "co-est00int-tot.csv", "CENSUS2010POP"),
    ("2016-03-24", "co-est2015-alldata.csv", "POPESTIMATE2015"),
)

# Window features: (suffix, window length in hours, hours per data point for std and slope).
# The 24-hour window uses hourly values; longer windows use rolling 24-hour means so their
# spread and trend describe day-to-day change rather than the daily temperature cycle.
WINDOWS = (("24h", 24, 1), ("16d", 16 * 24, 24), ("96d", 96 * 24, 24))
# Quarter weights for the weighted moving average, most recent quarter first.
QUARTER_WEIGHTS = (0.4, 0.3, 0.2, 0.1)


@dataclass(frozen=True)
class City:
    """A weather station, the counties (state FIPS, county FIPS) it represents, and the
    nearby stations used to fill its gaps, nearest first."""

    name: str
    station_prefix: str
    counties: tuple[tuple[int, int], ...]
    donors: tuple[str, ...]


CITIES = (
    City(
        name="philadelphia",
        station_prefix="PHL",
        counties=(
            (42, 17),  # Bucks, PA
            (42, 29),  # Chester, PA
            (42, 45),  # Delaware, PA
            (42, 91),  # Montgomery, PA
            (42, 101),  # Philadelphia, PA
            (34, 5),  # Burlington, NJ
            (34, 7),  # Camden, NJ
            (34, 15),  # Gloucester, NJ
            (34, 33),  # Salem, NJ
            (10, 3),  # New Castle, DE
            (24, 15),  # Cecil, MD
        ),
        donors=("ABE", "EWR", "BWI", "MDT", "DCA"),
    ),
    City(
        name="newark",
        station_prefix="EWR",
        counties=(
            (34, 3),  # Bergen, NJ
            (34, 13),  # Essex, NJ
            (34, 17),  # Hudson, NJ
            (34, 19),  # Hunterdon, NJ
            (34, 23),  # Middlesex, NJ
            (34, 25),  # Monmouth, NJ
            (34, 27),  # Morris, NJ
            (34, 29),  # Ocean, NJ
            (34, 31),  # Passaic, NJ
            (34, 35),  # Somerset, NJ
            (34, 37),  # Sussex, NJ
            (34, 39),  # Union, NJ
        ),
        donors=("ABE", "PHL", "BWI", "MDT", "DCA"),
    ),
    City(
        name="baltimore",
        station_prefix="BWI",
        counties=(
            (24, 3),  # Anne Arundel, MD
            (24, 5),  # Baltimore County, MD
            (24, 13),  # Carroll, MD
            (24, 25),  # Harford, MD
            (24, 27),  # Howard, MD
            (24, 35),  # Queen Anne's, MD
            (24, 510),  # Baltimore city, MD
        ),
        donors=("DCA", "MDT", "PHL", "ABE", "EWR"),
    ),
    City(
        name="washington",
        station_prefix="DCA",
        counties=(
            (11, 1),  # District of Columbia
            (24, 31),  # Montgomery, MD
            (24, 33),  # Prince George's, MD
        ),
        donors=("BWI", "MDT", "PHL", "ABE", "EWR"),
    ),
    City(
        name="harrisburg",
        station_prefix="MDT",
        counties=(
            (42, 41),  # Cumberland, PA
            (42, 43),  # Dauphin, PA
            (42, 99),  # Perry, PA
        ),
        donors=("BWI", "ABE", "PHL", "DCA", "EWR"),
    ),
)


@dataclass(frozen=True)
class CityWeatherReport:
    """Counts recorded while aligning one station to the target hourly index."""

    city: str
    observations_rejected: int
    hours_observed: int
    hours_filled_from_donor: dict[str, int]
    hours_carried_forward: int
    longest_gap_hours: int


def load_isd_temperature(paths: Sequence[str | Path]) -> tuple[pd.Series, int]:
    """Read NOAA Global Hourly CSVs and return temperature (deg C) by hour-ending local time.

    PJME labels each hour by its end in local prevailing time, so an observation at
    01:54 local is assigned to the 02:00 row. Converting to local time before labeling
    reproduces PJME's daylight-saving pattern: no 03:00 row on spring-forward days and
    two 02:00 rows on fall-back days, which are averaged as PJME's duplicates are.
    """
    frames = [
        pd.read_csv(path, usecols=["DATE", "REPORT_TYPE", "TMP"], dtype=str) for path in paths
    ]
    if not frames:
        raise ValueError("No weather files were provided")
    raw = pd.concat(frames, ignore_index=True)
    raw["REPORT_TYPE"] = raw["REPORT_TYPE"].str.strip()
    raw = raw[raw["REPORT_TYPE"].isin(REPORT_TYPE_PRIORITY)]

    value, quality = raw["TMP"].str.split(",", n=1, expand=True).T.to_numpy()
    rejected = (value == MISSING_TEMPERATURE) | pd.Series(quality).isin(
        REJECTED_QUALITY_CODES
    ).to_numpy()
    kept = raw[~rejected].copy()
    kept["temp_c"] = pd.to_numeric(value[~rejected]) / 10

    local = (
        pd.to_datetime(kept["DATE"])
        .dt.tz_localize("UTC")
        .dt.tz_convert(LOCAL_TIMEZONE)
        .dt.tz_localize(None)
    )
    kept["hour"] = local.dt.ceil("h")
    kept["priority"] = kept["REPORT_TYPE"].map(REPORT_TYPE_PRIORITY)

    best = kept.groupby("hour")["priority"].transform("min")
    preferred = kept[kept["priority"] == best]
    temperature = preferred.groupby("hour")["temp_c"].mean().sort_index()
    temperature.index.name = None
    return temperature, int(rejected.sum())


def fill_from_donors(
    target: pd.Series,
    donors: Mapping[str, pd.Series],
    index: pd.DatetimeIndex,
    offset_hours: int = DONOR_OFFSET_HOURS,
) -> tuple[pd.Series, pd.Series, dict[str, int], int]:
    """Fill missing hours from nearby stations using only information available at the time.

    A missing hour takes the first donor's reading for that same hour, shifted by the mean
    target-minus-donor difference over the preceding ``offset_hours``. Only real paired
    observations enter the offset, and nothing after the missing hour is used. An hour that
    no station reports, such as the nonexistent spring-forward hour, carries the previous
    hour forward; a longer unreported run raises.

    Returns the filled series, a mask of filled hours, counts per donor, and the number of
    hours carried forward.
    """
    span = pd.date_range(
        min(index.min(), target.index.min()), max(index.max(), target.index.max()), freq="h"
    )
    filled = target.reindex(span)
    missing = filled.isna()
    donor_counts = {}
    for name, donor in donors.items():
        donor = donor.reindex(span)
        offset = (
            (target.reindex(span) - donor)
            .rolling(offset_hours, min_periods=MIN_DONOR_PAIRS)
            .mean()
            .shift(1)
        )
        candidate = donor + offset
        usable = filled.isna() & candidate.notna()
        filled[usable] = candidate[usable]
        donor_counts[name] = int(usable[index].sum())

    remaining = filled.isna()
    filled = filled.ffill(limit=1)
    carried = int((remaining & filled.notna())[index].sum())
    filled = filled.reindex(index)
    if filled.isna().any():
        first = filled[filled.isna()].index[0]
        raise ValueError(f"No station could fill weather near {first}")
    return filled, missing.reindex(index), donor_counts, carried


def longest_gap(missing: pd.Series) -> int:
    """Length of the longest run of consecutive missing hours."""
    if not missing.any():
        return 0
    return int(missing.groupby((~missing).cumsum()).sum().max())


def degree_hours(temp_c: pd.Series, base_f: float = BASE_TEMPERATURE_F) -> pd.DataFrame:
    """Return heating and cooling degree hours relative to ``base_f``."""
    temp_f = temp_c * 9 / 5 + 32
    return pd.DataFrame(
        {
            "hdh": np.maximum(0.0, base_f - temp_f),
            "cdh": np.maximum(0.0, temp_f - base_f),
        },
        index=temp_c.index,
    )


def load_city_populations(
    population_dir: str | Path, cities: Sequence[City] = CITIES
) -> pd.DataFrame:
    """Return total county population per city for each snapshot publication date."""
    population_dir = Path(population_dir)
    sources: dict[str, pd.DataFrame] = {}
    rows = {}
    for effective_date, filename, column in POPULATION_SNAPSHOTS:
        if filename not in sources:
            sources[filename] = pd.read_csv(population_dir / filename, encoding="latin-1")
        source = sources[filename]
        keys = list(zip(source["STATE"], source["COUNTY"]))
        totals = {}
        for city in cities:
            selected = [key in city.counties for key in keys]
            found = int(sum(selected))
            if found != len(city.counties):
                raise ValueError(
                    f"{filename} has {found} of {len(city.counties)} counties for {city.name}"
                )
            totals[city.name] = int(source.loc[selected, column].sum())
        rows[pd.Timestamp(effective_date)] = totals
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def population_weights(populations: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Weight each city by the latest population snapshot published at or before each hour."""
    if index.min() < populations.index.min():
        raise ValueError("Target index begins before the earliest population snapshot")
    shares = populations.div(populations.sum(axis=1), axis=0)
    return shares.reindex(shares.index.union(index)).ffill().reindex(index)


def weighted_moving_average(series: pd.Series, window: int) -> pd.Series:
    """Average each quarter of the trailing window and combine them with QUARTER_WEIGHTS."""
    quarter = window // len(QUARTER_WEIGHTS)
    means = series.rolling(quarter).mean()
    return sum(weight * means.shift(i * quarter) for i, weight in enumerate(QUARTER_WEIGHTS))


def _window_points(series: pd.Series, window: int, step: int) -> np.ndarray:
    """Matrix of the trailing window's data points per hour, oldest first.

    With ``step`` > 1 each point is the mean of a rolling ``step``-hour block.
    """
    values = series.rolling(step).mean() if step > 1 else series
    count = window // step
    return np.column_stack(
        [values.shift(lag * step).to_numpy() for lag in range(count - 1, -1, -1)]
    )


def window_std_and_slope(series: pd.Series, window: int, step: int) -> tuple[pd.Series, pd.Series]:
    """Sample standard deviation and least-squares slope (per data point) over the window."""
    points = _window_points(series, window, step)
    positions = np.arange(points.shape[1], dtype=float)
    centered = positions - positions.mean()
    std = points.std(axis=1, ddof=1)
    slope = (points - points.mean(axis=1, keepdims=True)) @ centered / (centered @ centered)
    return pd.Series(std, index=series.index), pd.Series(slope, index=series.index)


def window_features(hdh: pd.Series, cdh: pd.Series) -> pd.DataFrame:
    """Build the 12 trailing-window features from weighted heating and cooling degree hours.

    Every value is computed as of its own hour, using that hour and earlier hours only.
    Hours without a full window of history are left missing.
    """
    combined = hdh + cdh
    features = pd.DataFrame(index=hdh.index)
    for suffix, window, step in WINDOWS:
        features[f"hdh_wma_{suffix}"] = weighted_moving_average(hdh, window)
        features[f"cdh_wma_{suffix}"] = weighted_moving_average(cdh, window)
        std, slope = window_std_and_slope(combined, window, step)
        features[f"dh_std_{suffix}"] = std
        features[f"dh_slope_{suffix}"] = slope
    return features


def _station_paths(weather_dir: Path, prefix: str) -> list[Path]:
    """List a station's yearly weather files (plain or gzip-compressed) in name order."""
    return sorted(weather_dir.glob(f"{prefix}_*.csv*"))


def build_weather_features(
    index: pd.DatetimeIndex,
    weather_dir: str | Path,
    population_dir: str | Path,
    cities: Sequence[City] = CITIES,
) -> tuple[pd.DataFrame, list[CityWeatherReport], pd.DataFrame]:
    """Build per-city temperatures, population-weighted degree hours, and window features."""
    weather_dir = Path(weather_dir)
    populations = load_city_populations(population_dir, cities)
    weights = population_weights(populations, index)

    prefixes = {city.station_prefix for city in cities} | {
        donor for city in cities for donor in city.donors
    }
    stations = {}
    rejected = {}
    for prefix in sorted(prefixes):
        paths = _station_paths(weather_dir, prefix)
        if not paths:
            raise ValueError(f"No weather files found for station {prefix}")
        stations[prefix], rejected[prefix] = load_isd_temperature(paths)

    features = pd.DataFrame(index=index)
    reports = []
    weighted_hdh = pd.Series(0.0, index=index)
    weighted_cdh = pd.Series(0.0, index=index)
    any_filled = pd.Series(False, index=index)
    for city in cities:
        donors = {prefix: stations[prefix] for prefix in city.donors}
        temperature, filled, donor_counts, carried = fill_from_donors(
            stations[city.station_prefix], donors, index
        )
        hours = degree_hours(temperature)

        features[f"temp_f_{city.name}"] = temperature * 9 / 5 + 32
        features[f"hdh_{city.name}"] = hours["hdh"]
        features[f"cdh_{city.name}"] = hours["cdh"]
        weighted_hdh += weights[city.name] * hours["hdh"]
        weighted_cdh += weights[city.name] * hours["cdh"]
        any_filled |= filled
        reports.append(
            CityWeatherReport(
                city=city.name,
                observations_rejected=rejected[city.station_prefix],
                hours_observed=int((~filled).sum()),
                hours_filled_from_donor=donor_counts,
                hours_carried_forward=carried,
                longest_gap_hours=longest_gap(filled),
            )
        )

    features["weighted_hdh"] = weighted_hdh
    features["weighted_cdh"] = weighted_cdh
    features["weighted_dh"] = weighted_hdh + weighted_cdh
    features = features.join(window_features(weighted_hdh, weighted_cdh))
    features["weather_filled"] = any_filled
    features.index.name = "Datetime"
    return features, reports, populations


def weather_report(
    reports: Sequence[CityWeatherReport], populations: pd.DataFrame
) -> dict[str, object]:
    """Summarize weather alignment and population inputs for the quality report."""
    snapshots: Mapping[str, dict[str, int]] = {
        date.strftime("%Y-%m-%d"): {name: int(value) for name, value in row.items()}
        for date, row in populations.iterrows()
    }
    return {
        "cities": [asdict(report) for report in reports],
        "population_snapshots": snapshots,
        "policy": {
            "source": "NOAA NCEI Global Hourly (ISD), report types FM-15 then FM-16",
            "rejected": "missing values and TMP quality codes 2, 3, 6, 7",
            "time_alignment": "UTC converted to America/New_York, labeled by hour ending",
            "gaps": (
                "nearest donor station's same-hour reading plus the mean difference over "
                f"the preceding {DONOR_OFFSET_HOURS} hours; no later data is used"
            ),
            "unreported_hours": (
                "an hour no station reports (such as the nonexistent spring-forward hour) "
                "carries the previous hour forward, at most one hour"
            ),
            "degree_hours": f"base {BASE_TEMPERATURE_F:g} F, hourly HDH and CDH",
            "population": "latest county snapshot published at or before each hour",
            "window_features": (
                "weighted HDH/CDH quarter-weighted moving averages (0.4/0.3/0.2/0.1) and "
                "combined degree-hour std and slope over 24h, 16d, and 96d"
            ),
        },
    }
