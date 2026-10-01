# Weather Features: What We Built and Why

## Purpose

Electricity demand rises when people heat or cool their homes, so the proposal calls for a
weather feature: the difference between the temperature and 65°F, weighted by population.
This document explains where the weather and population data came from, how it was cleaned and
aligned with the PJME energy data, how gaps were filled, and which features we built.

Everything described here is produced by one command, `energy-weather`, and lives in
`src/energy_forecasting/weather.py`.

## Starting point: the PJME energy data

Before adding weather, we checked the cleaned PJME data in `data/processed/`:

- 145,392 hourly rows from January 2002 to August 2018, with no missing values.
- No rows were deleted. 4 duplicate hours were averaged, and 30 missing hours were filled.
- 29 of the 30 missing hours are daylight-saving clock changes, not bad data.
- The 27 flagged outliers are real heat-wave peaks (summer 2006 and 2011) and were kept.

## Data sources

| Data | Source | Link |
| --- | --- | --- |
| Hourly energy use | PJM Hourly Energy Consumption (Kaggle) | https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data |
| Hourly temperature | NOAA National Centers for Environmental Information (NCEI), Global Hourly / Integrated Surface Database | https://www.ncei.noaa.gov/data/global-hourly/access/ |
| Weather station list | NOAA ISD station history | https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv |
| County population, 2000 and 2010 | U.S. Census Bureau, Population Estimates Program (intercensal county totals) | https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv |
| County population, 2015 | U.S. Census Bureau, Population Estimates Program (vintage 2015 county totals) | https://www2.census.gov/programs-surveys/popest/datasets/2010-2015/counties/totals/co-est2015-alldata.csv |

The raw files are stored unchanged in `data/raw/weather/` (gzip-compressed) and
`data/raw/population/`. See `data/README.md` for file details.

## Choosing the cities

PJME ("PJM East") covers the original PJM area: eastern Pennsylvania, New Jersey, Maryland,
Delaware, and Washington DC. Pittsburgh and Richmond were considered but dropped because they are
served by Duquesne Light and Dominion, which PJM tracks as separate regions (DUQ and DOM).

We picked one large central city per part of the region, each with a major airport weather
station:

| City | Airport station | Area it represents (counties) |
| --- | --- | --- |
| Philadelphia | PHL | Philadelphia metro: 5 PA counties, 4 South Jersey counties, New Castle DE, Cecil MD |
| Newark | EWR | The 12 New Jersey counties of the New York metro area |
| Baltimore | BWI | Baltimore metro: 6 counties plus Baltimore City |
| Washington | DCA | District of Columbia, Montgomery MD, Prince George's MD |
| Harrisburg | MDT | Cumberland, Dauphin, and Perry counties, PA |

Northern Virginia is left out of Washington because it is Dominion territory. Harrisburg
replaced our first choice, Allentown, because Allentown's station had longer gaps (up to 42
hours). Allentown's data is still used as a backup for filling other stations' gaps.

## Population weights

Each city is weighted by the total population of its counties, so cities with more people count
more. Population changes over time, so we use three snapshots: the 2000 Census, the 2010
Census, and the 2015 Census estimate.

**Each snapshot is used only from the date it was published.** For example, the 2010 Census
was not released until early 2011, so hours in 2010 still use the 2000 numbers. This prevents
the model from seeing information that did not exist yet.

| Snapshot | Used from |
| --- | --- |
| 2000 Census | April 1, 2001 |
| 2010 Census | April 1, 2011 |
| 2015 estimate | March 24, 2016 |

In 2018 the weights are roughly Newark 36%, Philadelphia 33%, Baltimore 15%, Washington 14%,
and Harrisburg 3%.

## Cleaning and lining up the weather data

- **Readings used:** routine hourly airport reports, with special reports used when the routine
  one is missing. Readings marked missing or flagged as suspect by NOAA are removed.
- **Time zones:** NOAA records time in UTC; PJME uses local Eastern time. Each reading is
  converted to Eastern time and labeled by the end of its hour, the same way PJME labels its
  hours. This matches PJME exactly, including the daylight-saving changes.
- **Check:** summer demand lines up best with temperature at the same hour, which confirms the
  alignment is correct.

## How gaps were filled

Each station is missing a small number of hours (0.1 to 0.2% of the data). Every missing hour is
filled the same way:

1. Take the reading from the **nearest other station** for that same hour.
2. Adjust it by how much the two stations usually differ, using the average difference over
   the **48 hours before** the gap.
3. If the nearest station is also missing, try the next nearest, and so on.

**Only data from before the gap is used.** Drawing a straight line to the next reading would
use information from the future, so we avoided it. This matches how the PJME energy data was
filled.

In a handful of hours (8 per city), no station reported at all. These are mostly the
spring-forward daylight-saving hour, which does not exist on the clock. In those cases the
previous hour's value is repeated.

**Accuracy:** we hid 296 real stretches of 8 to 16 hours at Harrisburg and filled them with
this method. The filled values were off by **2.7°F on average** (median 2.0°F).

A single column, `weather_filled`, marks every hour that was filled, for reporting only.

## Degree hours

For each hour and city we compute:

- **Heating degree hours (HDH):** how far the temperature is **below** 65°F, or 0 if warmer.
- **Cooling degree hours (CDH):** how far the temperature is **above** 65°F, or 0 if colder.
- **Combined degree hours (DH):** HDH + CDH, the total heating or cooling need.

65°F is the standard point where people need neither heating nor cooling. These values are
"degree hours" rather than "degree days" because they are calculated hourly.

The five cities are then combined into **population-weighted** HDH, CDH, and DH.

## The 12 window features

These features summarize recent weather so the model can see trends, not just a single hour.
They are built only from the population-weighted series.

**Three time windows**, one for each kind of prediction:

| Window | Intended use |
| --- | --- |
| Last 24 hours | hourly predictions |
| Last 16 days | daily predictions |
| Last 96 days | weekly or monthly predictions |

**Four measures per window:**

| Measure | Built from | What it tells the model |
| --- | --- | --- |
| Weighted average of HDH | heating degree hours | how much heating has been needed recently |
| Weighted average of CDH | cooling degree hours | how much cooling has been needed recently |
| Standard deviation | combined degree hours | how much the weather has been swinging |
| Slope | combined degree hours | whether total heating/cooling need is rising or falling |

**Weighted average:** the window is split into four equal parts. The most recent part counts
40%, then 30%, 20%, and 10% for the oldest. Recent weather matters more because buildings hold
heat and cold for a while.

**Standard deviation and slope:** the 24-hour window uses each hour, so it captures the daily
swing. The 16- and 96-day windows use daily averages, so they capture day-to-day changes and
seasonal trends instead.

**Feature names:**

| Window | Features |
| --- | --- |
| 24 hours | `hdh_wma_24h`, `cdh_wma_24h`, `dh_std_24h`, `dh_slope_24h` |
| 16 days | `hdh_wma_16d`, `cdh_wma_16d`, `dh_std_16d`, `dh_slope_16d` |
| 96 days | `hdh_wma_96d`, `cdh_wma_96d`, `dh_std_96d`, `dh_slope_96d` |

Each feature uses only the current hour and earlier hours. The first 96 days of 2002 have no
96-day value because there is not enough history yet.

### Why only the weighted series

Temperatures in the five cities move almost together (correlation 0.97 to 0.99). Building the
same features for every city would add many nearly identical columns, which would also make the
feature-importance results harder to read. The population-weighted series also predicted demand
better than any single city in both summer and winter.

## Output files

`energy-weather` writes two files (generated, not committed):

- `data/processed/weather_features.csv`: one row per PJME hour, with each city's temperature
  and degree hours, the weighted degree hours, the 12 window features, and `weather_filled`.
- `data/processed/weather_quality_report.json`: counts of rejected readings and filled hours
  per city, the population snapshots, and the rules used.

## Modeling integration

The modeling pipeline includes the 12 weather window features by default. Each feature is shifted
one hour, so the prediction for hour `t` uses weather through hour `t - 1`. This matches the
current one-hour-ahead interpretation and avoids substituting observed current-hour weather for a
weather forecast. Use `energy-forecast --no-weather` for the energy-only comparison.

The team still needs to decide whether to add separate next-day forecasts. That horizon would
require weather forecasts or a longer weather lag rather than observed future conditions. A
similar trend and variability measure for energy use and a previous-year comparison also remain
possible extensions.
