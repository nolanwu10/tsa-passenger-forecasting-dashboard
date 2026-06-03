# Live Weekly Predictor Feature Analysis

## Working Dataset

- Source table: `data\processed\tsa_daily_transport_features.csv`
- Rows: 2,702
- Columns before temporary history analysis: 106
- Date range: `2019-01-01` to `2026-05-25`

## Live-Safe Feature Groups

- Calendar and holiday features: fully known for future dates.
- Matched historical TSA features: known before forecast time, if built from prior years only.
- Recent TSA actuals: known only through the latest TSA-published date.
- Capacity context: use latest published annual T-100 values carried forward.
- Monthly traffic context: use lagged monthly values only, because current-month values are not live.
- BTS daily delay data: not currently live enough for same-week forecasting; use lagged values only where published.

## Baseline Forecast Strength

Baselines evaluated on 2024-01-01 through 2026-05-25 where each baseline had required inputs.

| baseline                    |   n | mae     | mape    | bias     | weekly_total_mape   |
|:----------------------------|----:|:--------|:--------|:---------|:--------------------|
| blend_growth_lag7           | 873 | 87,415  | +3.67%  | -20,595  | +1.80%              |
| matched_1ya_growth_adjusted | 876 | 87,792  | +3.71%  | 4,571    | +1.73%              |
| matched_1ya                 | 876 | 114,736 | +4.73%  | -56,979  | +3.18%              |
| blend_1ya_2ya               | 876 | 147,868 | +6.04%  | -105,460 | +4.71%              |
| lag_7                       | 876 | 151,480 | +6.46%  | -1,270   | +3.52%              |
| baseline_2019               | 873 | 169,423 | +6.75%  | -154,457 | +6.26%              |
| matched_2ya                 | 876 | 257,778 | +10.48% | -218,583 | +9.31%              |

## Strong Feature Signals

| feature                                      | n     |   corr_with_passengers |
|:---------------------------------------------|:------|-----------------------:|
| matched_1ya_growth_adjusted                  | 1,241 |              0.889153  |
| baseline_2019                                | 1,237 |              0.867317  |
| matched_1ya                                  | 1,241 |              0.853179  |
| lag_7                                        | 1,241 |              0.776364  |
| matched_2ya                                  | 1,241 |              0.660199  |
| roll_7_lag1                                  | 1,241 |              0.587032  |
| roll_28_lag1                                 | 1,241 |              0.495032  |
| days_until_new_year                          | 1,241 |             -0.239898  |
| iso_week                                     | 1,241 |              0.235812  |
| days_until_christmas                         | 1,241 |             -0.212932  |
| day_of_week                                  | 1,241 |              0.188487  |
| bts_mts_domestic_air_traffic_nsa_lag2_months | 1,216 |              0.163264  |
| bts_t100_est_daily_domestic_seats            | 1,241 |              0.143727  |
| is_holiday_window                            | 1,241 |              0.0760985 |
| bts_t100_domestic_load_factor                | 1,241 |             -0.0584662 |
| is_thanksgiving_week                         | 1,241 |              0.0330704 |

## Day-Of-Week Shape Since 2023

| day_name   |   count | mean      | std     |
|:-----------|--------:|:----------|:--------|
| Monday     |     178 | 2,542,094 | 229,967 |
| Tuesday    |     177 | 2,136,320 | 264,565 |
| Wednesday  |     177 | 2,253,852 | 247,517 |
| Thursday   |     177 | 2,581,116 | 264,169 |
| Friday     |     177 | 2,630,624 | 213,340 |
| Saturday   |     177 | 2,231,485 | 259,684 |
| Sunday     |     178 | 2,633,043 | 265,112 |

## Month Shape Since 2023

|   month |   count | mean      | std     |
|--------:|--------:|:----------|:--------|
|       1 |     124 | 2,074,131 | 288,841 |
|       2 |     113 | 2,209,867 | 290,275 |
|       3 |     124 | 2,460,115 | 240,372 |
|       4 |     120 | 2,439,294 | 243,280 |
|       5 |     118 | 2,524,596 | 253,606 |
|       6 |      90 | 2,678,198 | 205,767 |
|       7 |      93 | 2,678,067 | 213,653 |
|       8 |      93 | 2,529,092 | 265,086 |
|       9 |      90 | 2,376,714 | 300,054 |
|      10 |      93 | 2,485,594 | 308,848 |
|      11 |      90 | 2,393,004 | 325,262 |
|      12 |      93 | 2,444,821 | 300,405 |

## Holiday Window Shape Since 2023

|   is_holiday_window | count   | mean      | std     |
|--------------------:|:--------|:----------|:--------|
|                   0 | 1,106   | 2,421,589 | 317,212 |
|                   1 | 135     | 2,499,322 | 318,836 |

## Recommendation

Use a two-layer daily forecasting model for the live weekly predictor:

1. A deterministic baseline that creates a first-pass daily estimate from ISO-week/day matched history, recent TSA trend, and holiday/calendar rules.
2. A tabular residual model that predicts the correction to that baseline using only live-safe features.

The best first production model is gradient-boosted trees over daily rows, predicting daily passenger count or residual-to-baseline. For the current repo environment, `sklearn.ensemble.HistGradientBoostingRegressor` is a practical starting point. If we add a modeling dependency later, LightGBM or XGBoost are strong choices.

For weekly prediction, forecast each missing day independently, combine with known actual days already published by TSA, and sum/average the seven daily values.
