"""Tests for weather loading, gap filling, weather features, and their integration."""

# Author: plholt3

import numpy as np
import pandas as pd
import pytest

from energy_forecasting.data import TARGET_COLUMN
from energy_forecasting.features import WEATHER_FEATURES, add_weather_features, build_features
from energy_forecasting.weather import (
    City,
    build_weather_features,
    degree_hours,
    fill_from_donors,
    load_isd_temperature,
    population_weights,
    weighted_moving_average,
    window_features,
    window_std_and_slope,
)

HEADER = '"STATION","DATE","REPORT_TYPE","TMP"\n'


def isd_file(path, rows):
    """Write a minimal NOAA ISD CSV with (DATE, REPORT_TYPE, TMP) rows."""
    path.write_text(HEADER + "".join(f'"X","{d}","{r}","{t}"\n' for d, r, t in rows))
    return path


def hourly(values, start="2024-01-01"):
    """Build an hourly float series starting at ``start``."""
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="h"), dtype=float)


def test_observations_are_labeled_by_local_hour_ending(tmp_path):
    """A UTC reading is converted to Eastern time and labeled by the end of its hour."""
    # 2024-01-15 05:54 UTC is 00:54 EST, so it belongs to the 01:00 row.
    path = isd_file(tmp_path / "a.csv", [("2024-01-15T05:54:00", "FM-15", "+0100,5")])
    temperature, _ = load_isd_temperature([path])
    assert temperature.index[0] == pd.Timestamp("2024-01-15 01:00")
    assert temperature.iloc[0] == 10.0  # ISD reports tenths of a degree C


def test_daylight_saving_matches_pjme_hour_ending_pattern(tmp_path):
    """Spring skips 03:00 and fall averages the repeated 02:00, as PJME does."""
    spring = [(f"2024-03-10T{h:02d}:54:00", "FM-15", "+0000,5") for h in range(5, 9)]
    fall = [("2024-11-03T05:54:00", "FM-15", "+0100,5"), ("2024-11-03T06:54:00", "FM-15", "+0300,5")]
    temperature, _ = load_isd_temperature([isd_file(tmp_path / "a.csv", spring + fall)])
    spring_day = temperature.loc["2024-03-10"].index.hour.tolist()
    assert 3 not in spring_day and 2 in spring_day and 4 in spring_day
    assert temperature[pd.Timestamp("2024-11-03 02:00")] == 20.0


def test_rejects_missing_and_bad_quality_and_prefers_routine_reports(tmp_path):
    """Missing and suspect readings are dropped; routine reports beat special reports."""
    rows = [
        ("2024-01-15T05:10:00", "FM-16", "+0500,5"),
        ("2024-01-15T05:54:00", "FM-15", "+0100,5"),
        ("2024-01-15T06:54:00", "FM-15", "+9999,9"),
        ("2024-01-15T07:54:00", "FM-15", "+0400,3"),
        ("2024-01-15T08:00:00", "SOD", "+0700,5"),
    ]
    temperature, rejected = load_isd_temperature([isd_file(tmp_path / "a.csv", rows)])
    assert temperature.to_dict() == {pd.Timestamp("2024-01-15 01:00"): 10.0}
    assert rejected == 2


def test_gaps_fill_from_donor_plus_prior_offset():
    """A gap takes the donor's reading plus the average difference before the gap."""
    donor = hourly(np.arange(60))
    target = (donor + 2).drop(donor.index[50:53])
    filled, mask, counts, carried = fill_from_donors(target, {"D": donor}, donor.index)
    assert filled.iloc[50:53].tolist() == (donor.iloc[50:53] + 2).tolist()
    assert mask.sum() == 3 and counts == {"D": 3} and carried == 0


def test_gap_fill_never_uses_later_observations():
    """Changing readings after a gap does not change how the gap is filled."""
    donor = hourly(np.arange(60))
    target = (donor + 2).drop(donor.index[50:53])
    changed = target.copy()
    changed[changed.index > donor.index[52]] += 100
    original, *_ = fill_from_donors(target, {"D": donor}, donor.index)
    altered, *_ = fill_from_donors(changed, {"D": donor}, donor.index)
    assert original.iloc[50:53].tolist() == altered.iloc[50:53].tolist()


def test_unreported_hour_carries_forward_and_longer_runs_raise():
    """One hour no station reports repeats the prior hour; two in a row is an error."""
    donor = hourly(np.arange(60))
    target = donor + 2
    one_hour = target.drop(target.index[30])
    filled, _, counts, carried = fill_from_donors(
        one_hour, {"D": donor.drop(donor.index[30])}, donor.index
    )
    assert filled.iloc[30] == target.iloc[29] and carried == 1 and counts == {"D": 0}
    with pytest.raises(ValueError):
        fill_from_donors(
            target.drop(target.index[30:32]), {"D": donor.drop(donor.index[30:32])}, donor.index
        )


def test_degree_hours_use_65f_base():
    """Heating and cooling degree hours measure distance below and above 65 F."""
    hours = degree_hours(pd.Series([0.0, 18.0, 35.0]))  # 32F, 64.4F, 95F
    assert hours["hdh"].round(1).tolist() == [33.0, 0.6, 0.0]
    assert hours["cdh"].round(1).tolist() == [0.0, 0.0, 30.0]


def test_population_weights_never_use_future_snapshots():
    """A population snapshot only applies from its publication date onward."""
    populations = pd.DataFrame(
        {"a": [100, 100], "b": [100, 300]},
        index=pd.to_datetime(["2001-04-01", "2011-04-01"]),
    )
    index = pd.to_datetime(["2011-03-31 23:00", "2011-04-01 00:00"])
    weights = population_weights(populations, index)
    assert weights["b"].tolist() == [0.5, 0.75]


def test_weighted_moving_average_weights_recent_quarters_most():
    """Quarters are weighted 40/30/20/10 from newest to oldest."""
    # Oldest to newest quarters of 6 hours: 40, 30, 20, 10.
    series = hourly([40] * 6 + [30] * 6 + [20] * 6 + [10] * 6)
    wma = weighted_moving_average(series, 24)
    assert wma.iloc[-1] == pytest.approx(0.4 * 10 + 0.3 * 20 + 0.2 * 30 + 0.1 * 40)
    assert wma.iloc[:-1].isna().all()


def test_std_and_slope_on_hourly_and_daily_points():
    """Slope and standard deviation are correct on a straight-line series."""
    series = hourly(np.arange(24 * 20) * 2.0)
    std, slope = window_std_and_slope(series, 24, 1)
    assert slope.iloc[-1] == pytest.approx(2.0)
    assert std.iloc[-1] == pytest.approx(np.std(np.arange(24) * 2.0, ddof=1))
    _, daily_slope = window_std_and_slope(series, 16 * 24, 24)
    assert daily_slope.iloc[-1] == pytest.approx(48.0)  # rolling daily means rise 48 per day


def test_window_features_wait_for_full_history():
    """Window features stay blank until a full window of history exists."""
    hdh = hourly(np.ones(97 * 24))
    features = window_features(hdh, hdh * 0)
    assert len(features.columns) == 12
    assert features["hdh_wma_96d"].iloc[96 * 24 - 1 :].notna().all()
    assert features["hdh_wma_96d"].iloc[: 96 * 24 - 1].isna().all()
    assert features["hdh_wma_24h"].iloc[-1] == pytest.approx(1.0)
    assert features["dh_slope_16d"].iloc[-1] == pytest.approx(0.0)


def test_build_weather_features_end_to_end(tmp_path):
    """The full build produces weighted degree hours and fill counts from raw files."""
    weather_dir = tmp_path / "weather"
    population_dir = tmp_path / "population"
    weather_dir.mkdir()
    population_dir.mkdir()
    rows = [(f"2024-07-01T{h:02d}:54:00", "FM-15", "+0300,5") for h in range(4, 10)]
    isd_file(weather_dir / "TST_1_2024.csv", rows[:2] + rows[3:])
    isd_file(weather_dir / "DON_2_2024.csv", [(d, r, "+0200,5") for d, r, _ in rows])
    county = "STATE,COUNTY,ESTIMATESBASE2000,CENSUS2010POP,POPESTIMATE2015\n1,1,10,20,30\n"
    (population_dir / "co-est00int-tot.csv").write_text(county)
    (population_dir / "co-est2015-alldata.csv").write_text(county)

    city = City(name="test", station_prefix="TST", counties=((1, 1),), donors=("DON",))
    index = pd.date_range("2024-07-01 01:00", "2024-07-01 05:00", freq="h")
    features, reports, _ = build_weather_features(index, weather_dir, population_dir, [city])
    assert features["weighted_cdh"].round(1).tolist() == [21.0] * 5  # 30C = 86F
    assert features["cdh_test"].equals(features["weighted_cdh"])
    # The donor has too little prior overlap to set an offset, so the gap carries forward.
    assert features["weather_filled"].sum() == 1
    assert reports[0].hours_filled_from_donor == {"DON": 0}
    assert reports[0].hours_carried_forward == 1


def weather_frame(index):
    """Weather features whose values equal the hour number, to make shifts visible."""
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame({name: values for name in WEATHER_FEATURES}, index=index)


def test_weather_features_join_one_hour_behind():
    """The row for hour t carries the weather features computed as of hour t - 1."""
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    energy = pd.DataFrame({TARGET_COLUMN: np.linspace(30_000, 31_000, 200)}, index=index)
    featured = build_features(energy, weather=weather_frame(index))
    assert set(WEATHER_FEATURES) <= set(featured.columns)
    row = featured.index[0]
    assert featured.loc[row, "hdh_wma_24h"] == index.get_loc(row) - 1


def test_weather_must_cover_every_energy_hour():
    """Energy hours without a weather row raise instead of silently dropping."""
    index = pd.date_range("2024-01-01", periods=10, freq="h")
    energy = pd.DataFrame({TARGET_COLUMN: np.ones(10)}, index=index)
    with pytest.raises(ValueError):
        add_weather_features(energy, weather_frame(index[:-1]))


def test_build_features_without_weather_is_unchanged():
    """Leaving out weather gives the original energy-only feature set."""
    index = pd.date_range("2024-01-01", periods=200, freq="h")
    energy = pd.DataFrame({TARGET_COLUMN: np.linspace(30_000, 31_000, 200)}, index=index)
    featured = build_features(energy)
    assert not set(WEATHER_FEATURES) & set(featured.columns)
