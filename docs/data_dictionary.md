# Data Dictionary

## `raw_tsa`

Path: `data/raw/tsa_passenger_data.csv`

Expected grain: one row per calendar date.

Columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | date | Calendar date of TSA passenger throughput. |
| `Passengers` | integer | Total TSA traveler throughput for the date. |
| `source_url` | string | TSA source page the row came from. Optional but useful for provenance. |

Validation expectations:

- `Date` parses as a date.
- `Passengers` parses as a non-negative integer.
- Dates are unique after cleaning.
- Rows are sorted by date in downstream processed data.

## `external_weather`

Path: `data/external/weather_history.csv`

Expected grain: one row per date, after external weather aggregation.

Current columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | date | Calendar date. |
| `Hub_Severe_Weather_Index` | numeric | Aggregate severe-weather count or index across selected airport hubs. |

## `processed_features`

Path: `data/processed/processed_features.csv`

Expected grain: one row per calendar date.

This is a legacy processed dataset. It is preserved for reference, but the reset should define a new processed contract before additional model work.

Currently observed core columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | date | Calendar date. |
| `Passengers` | integer | Actual TSA passenger count. |
| `iso_year` | integer | ISO calendar year. |
| `iso_week` | integer | ISO calendar week. |
| `day_of_week` | integer | Monday = 0, Sunday = 6. |
| `Matched_1YA` ... `Matched_7YA` | numeric | Passenger counts matched by ISO week and weekday in prior years. |
| `Baseline_2019` | numeric | 2019 matched baseline for the same ISO week and weekday. |
| `is_holiday_window` | integer | Flag for dates near selected holidays. |
| `growth_trend` | numeric | Legacy rolling growth ratio. |

## `daily_calendar_features`

Path: `data/processed/tsa_daily_calendar_features.csv`

Expected grain: one row per calendar date.

This is the reset project feature table for calendar and holiday-only modeling preparation.

Core columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | date | Calendar date. |
| `Passengers` | integer | Actual TSA passenger count. |
| `year`, `quarter`, `month`, `day` | integer | Basic calendar parts. |
| `day_of_year`, `day_of_week` | integer | Seasonal and weekly position features. Monday is `0`, Sunday is `6`. |
| `iso_year`, `iso_week`, `iso_day` | integer | ISO calendar features for week-aligned history. |
| `is_weekend` | integer | Saturday/Sunday flag. |
| `is_federal_holiday` | integer | Observed U.S. federal holiday flag. |
| `is_high_impact_holiday` | integer | Major travel holiday flag. |
| `is_holiday_window` | integer | Within +/- 3 days of a major travel holiday. |
| `holiday_name` | string | Federal holiday name when applicable. |
| `is_thanksgiving_week` | integer | Within +/- 3 days of Thanksgiving. |
| `days_from_thanksgiving` | integer | Signed distance from Thanksgiving in the same year. |
| `days_until_christmas`, `days_until_new_year`, `days_until_july_4` | integer | Forward distance to key fixed-date holidays. |

## `bts_monthly_air_traffic`

Path: `data/external/bts_monthly_air_traffic.csv`

Source: BTS Monthly Transportation Statistics, Socrata dataset `crem-w557`.

Expected grain: one row per calendar month.

Key columns:

| Column | Type | Description |
| --- | --- | --- |
| `month_start` | date | First day of the source month. |
| `bts_mts_domestic_air_traffic_nsa` | numeric | Non-seasonally adjusted domestic U.S. airline traffic. |
| `bts_mts_total_air_traffic_nsa` | numeric | Non-seasonally adjusted total U.S. airline traffic. |
| `bts_mts_marketing_on_time_pct` | numeric | Monthly on-time percentage. |

## `bts_t100_annual_capacity`

Path: `data/external/bts_t100_annual_capacity.csv`

Source: BTS AFF T100 Segment Summary By Carrier, Socrata dataset `q4tb-tbff`.

Expected grain: one row per calendar year.

Key columns:

| Column | Type | Description |
| --- | --- | --- |
| `year` | integer | Calendar year. |
| `bts_t100_domestic_departures` | numeric | Annual domestic departures. |
| `bts_t100_domestic_passengers` | numeric | Annual domestic passengers. |
| `bts_t100_domestic_seats` | numeric | Annual domestic seats. |
| `bts_t100_domestic_load_factor` | numeric | Domestic passengers divided by domestic seats. |

## `bts_on_time_daily`

Path: `data/external/bts_on_time_daily.csv`

Source: BTS TranStats Reporting Carrier On-Time Performance.

Expected grain: one row per calendar date.

Key columns:

| Column | Type | Description |
| --- | --- | --- |
| `Date` | date | Flight date. |
| `bts_flights_scheduled` | integer | Count of scheduled reporting-carrier domestic flights. |
| `bts_cancel_rate` | numeric | Cancelled flights divided by scheduled flights. |
| `bts_dep_delay_15_rate` | numeric | Share of flights with departure delay of at least 15 minutes. |
| `bts_avg_dep_delay_minutes` | numeric | Average departure delay minutes across scheduled flights. |

## `daily_transport_features`

Path: `data/processed/tsa_daily_transport_features.csv`

Expected grain: one row per calendar date.

This table joins TSA calendar features to BTS external transport context. It includes raw same-day BTS columns for analysis plus safer lagged/rolling variants for modeling.

Modeling-safe examples:

| Column | Description |
| --- | --- |
| `bts_cancel_rate_lag1` | Previous available day cancellation rate. |
| `bts_cancel_rate_roll7_lag1` | Seven-day rolling cancellation rate, shifted by one day. |
| `bts_avg_dep_delay_minutes_roll7_lag1` | Seven-day rolling average departure delay, shifted by one day. |
| `bts_mts_domestic_air_traffic_nsa_lag2_months` | Monthly domestic air traffic shifted forward two months to reduce release-timing leakage. |
| `bts_t100_capacity_reference_year` | Published T-100 year used for annual capacity features. |
| `bts_t100_est_daily_domestic_seats` | Annual domestic seats spread across days in the year. |
