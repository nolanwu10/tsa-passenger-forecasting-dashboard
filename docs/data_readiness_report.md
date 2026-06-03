# Data Readiness Report

## Source Status

- Raw TSA rows: 2,710
- Date range: `2019-01-01` to `2026-06-02`
- Expected daily rows in range: 2,710
- Missing dates: 0
- Duplicate dates: 0

The raw TSA dataset is continuous at daily grain and is ready to be the base table for feature work.

## YTD Signal

Through 2026-06-02, 2026 average daily throughput is 2,395,978. That is +0.53% versus 2025 and +7.20% versus 2019.

|    |   year |   days | total       | average   | avg_yoy_change   |
|---:|-------:|-------:|:------------|:----------|:-----------------|
|  0 |   2019 |    153 | 341,955,370 | 2,235,002 |                  |
|  1 |   2020 |    154 | 176,581,039 | 1,146,630 | -48.70%          |
|  2 |   2021 |    153 | 183,383,287 | 1,198,584 | +4.53%           |
|  3 |   2022 |    153 | 294,370,088 | 1,923,988 | +60.52%          |
|  4 |   2023 |    153 | 340,978,774 | 2,228,619 | +15.83%          |
|  5 |   2024 |    154 | 366,351,269 | 2,378,904 | +6.74%           |
|  6 |   2025 |    153 | 364,634,315 | 2,383,231 | +0.18%           |
|  7 |   2026 |    153 | 366,584,599 | 2,395,978 | +0.53%           |

## Annual Summary

|    |   year |   days | total       | average   | minimum   | maximum   |
|---:|-------:|-------:|:------------|:----------|:----------|:----------|
|  0 |   2019 |    365 | 848,102,043 | 2,323,567 | 1,591,158 | 2,882,915 |
|  1 |   2020 |    366 | 339,774,756 | 928,346   | 113,147   | 2,507,588 |
|  2 |   2021 |    365 | 585,250,987 | 1,603,427 | 508,467   | 2,458,325 |
|  3 |   2022 |    365 | 760,071,362 | 2,082,387 | 1,063,856 | 2,639,616 |
|  4 |   2023 |    365 | 858,548,196 | 2,352,187 | 1,534,786 | 2,908,785 |
|  5 |   2024 |    366 | 904,068,577 | 2,470,133 | 1,551,896 | 3,088,836 |
|  6 |   2025 |    365 | 906,735,976 | 2,484,208 | 1,559,165 | 3,134,613 |
|  7 |   2026 |    153 | 366,584,599 | 2,395,978 | 1,313,323 | 2,976,209 |

## Day-Of-Week Shape Since 2023

|    | day_of_week   |   days | average   |
|---:|:--------------|-------:|:----------|
|  0 | Monday        |    179 | 2,542,832 |
|  1 | Tuesday       |    179 | 2,138,850 |
|  2 | Wednesday     |    178 | 2,254,532 |
|  3 | Thursday      |    178 | 2,581,565 |
|  4 | Friday        |    178 | 2,630,957 |
|  5 | Saturday      |    178 | 2,231,910 |
|  6 | Sunday        |    179 | 2,634,082 |

## Busiest Observed Days

|      | Date       | Passengers   |
|-----:|:-----------|:-------------|
| 2525 | 2025-11-30 | 3,134,613    |
| 2364 | 2025-06-22 | 3,096,797    |
| 2161 | 2024-12-01 | 3,088,836    |
| 2392 | 2025-07-20 | 3,043,973    |
| 2378 | 2025-07-06 | 3,041,954    |
| 2399 | 2025-07-27 | 3,017,861    |
| 2474 | 2025-10-10 | 3,017,612    |
| 2014 | 2024-07-07 | 3,013,622    |
| 2334 | 2025-05-23 | 3,010,183    |
| 2385 | 2025-07-13 | 3,007,773    |

## Existing Derived Data

- `processed_features.csv` legacy table: {'exists': True, 'rows': 1157, 'min_date': '2023-01-01', 'max_date': '2026-03-02', 'columns': 30}
- `tsa_daily_calendar_features.csv`: {'exists': True, 'rows': 2710, 'min_date': '2019-01-01', 'max_date': '2026-06-02', 'columns': 50}
- `weather_history.csv`: {'weather_exists': True, 'weather_rows': 1, 'weather_min_date': '2019-01-01', 'weather_max_date': '2019-01-01', 'weather_columns': ['Date', 'Hub_Severe_Weather_Index']}

The legacy processed features are stale relative to raw TSA data and should not be used for new modeling until rebuilt.
The existing weather file is too sparse to be model-ready.

## Modeling Readiness Notes

- Use raw TSA as the canonical target table.
- Rebuild processed features from raw data in a deterministic pipeline.
- Treat 2020 and 2021 as structural-break years; include them carefully or downweight/exclude depending on the modeling objective.
- Preserve calendar and matched-history features, but define them in one feature-building module before training.
- Do not use legacy model artifacts as inputs to the reset modeling pipeline.

## External Data Candidates

- Calendar and holiday structure: federal holidays, observed holidays, Thanksgiving week, Christmas/New Year windows, spring break, school calendar proxies.
- Airline/network capacity: BTS or schedule-derived flight counts, available seats, cancellations, and delays.
- Weather: daily severe-weather indicators around major hubs, rebuilt with complete historical coverage.
- Macro/travel demand: gasoline prices, consumer sentiment, unemployment, airfare indexes, and travel search interest.
- Market context: if the goal includes Kalshi trading, keep market prices separate from target features to avoid leakage in passenger forecasts.
