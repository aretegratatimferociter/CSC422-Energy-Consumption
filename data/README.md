# Data

## Raw dataset

`raw/PJME_hourly.csv` is the PJM East regional series from version 3 of Kaggle's
[Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption/data)
dataset, published by Rob Mulla (`robikscube`). The Kaggle dataset is licensed CC0 1.0/Public
Domain.

- Downloaded: 2026-09-14
- Kaggle version: 3
- Original filename: `PJME_hourly.csv`
- Columns: `Datetime`, `PJME_MW`
- SHA-256: `4eb2b16d42bf07ec41ab55cb842191594cb69452725a6d3c0991658a628fde84`

The full Kaggle archive contains multiple PJM regions. This project intentionally uses only PJME
so that the geographic definition remains consistent across the modeled time range.
