# Transport Enrichment Report

## Sources Selected

| Source | Use | Local output | Coverage fetched |
| --- | --- | --- | --- |
| BTS TranStats Reporting Carrier On-Time Performance | Daily delays, cancellations, scheduled flights | `data/external/bts_on_time_daily.csv` | 2026-01-01 to 2026-02-28 |
| BTS Monthly Transportation Statistics (`crem-w557`) | Monthly air traffic and monthly on-time context | `data/external/bts_monthly_air_traffic.csv` | 2019-01 to 2026-02 |
| BTS AFF T100 Segment Summary By Carrier (`q4tb-tbff`) | Annual domestic/total seats, departures, passengers, load factor | `data/external/bts_t100_annual_capacity.csv` | 2014 to 2025 |

References:

- BTS Monthly Transportation Statistics: https://dev.socrata.com/foundry/data.bts.gov/crem-w557
- BTS T-100 Segment Summary By Carrier: https://data.bts.gov/Aviation/AFF-T100-Segment-Summary-By-Carrier/q4tb-tbff
- BTS TranStats On-Time Performance fields: https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ

## Outputs Created

`data/processed/tsa_daily_transport_features.csv`

- Rows: 2,702
- Columns: 106
- Date range: 2019-01-01 to 2026-05-25
- Validation status: passed

The table keeps the existing TSA calendar/holiday fields and adds BTS transport columns. It includes direct observed values for analysis and lagged/rolling variants intended for modeling.

## Coverage Notes

- Monthly BTS air traffic covers 2,616 of 2,702 TSA dates after joining. Missing dates are March-May 2026 because BTS monthly air traffic currently has real values through February 2026.
- Annual T-100 capacity covers all 2,702 TSA rows after carrying forward the latest published capacity year. For 2026 dates, the feature table uses 2025 as `bts_t100_capacity_reference_year`.
- Daily on-time delay data currently covers 59 TSA dates, January-February 2026. The downloader supports wider pulls, but the full 2024-2026 pull exceeded the command timeout because monthly ZIPs are large.

## Initial Delay Signal

For January-February 2026:

- Mean scheduled reporting-carrier flights per day: about 17,950
- Mean cancellation rate: about 3.52%
- Mean departure delay: about 15.22 minutes
- Mean 15-minute departure-delay rate: about 19.45%
- Max observed cancellation rate in the fetched window: about 46.48%

On the overlapping 59 days, TSA passengers had positive correlation with scheduled flights and negative correlation with cancellation rate. This is directionally useful, but the overlap is too short to rely on yet.

## Modeling Guidance

Use these columns first:

- `bts_t100_est_daily_domestic_seats`
- `bts_t100_domestic_load_factor`
- `bts_mts_domestic_air_traffic_nsa_lag2_months`
- `bts_mts_marketing_on_time_pct_lag2_months`
- `bts_cancel_rate_lag1`
- `bts_cancel_rate_roll7_lag1`
- `bts_avg_dep_delay_minutes_roll7_lag1`
- `bts_dep_delay_15_rate_roll7_lag1`

Avoid same-day delay columns in forecasting models unless the prediction target is explicitly after the operating day and those values would already be known.

## Next Data Work

1. Pull a wider on-time window in smaller batches, for example one year at a time, and append daily aggregates.
2. Add airport-hub-level delay features for major hubs instead of only national aggregates.
3. Replace annual capacity with monthly capacity if a stable directly accessible DB20/T-100 monthly endpoint is added.
4. Add explicit feature availability dates for each external source before final model training.

