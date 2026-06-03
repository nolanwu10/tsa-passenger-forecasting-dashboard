# Live Weekly TSA Model Report

## Objective

Predict the Monday-Sunday average TSA passenger check-ins for a target week.

The model supports:

- Pre-week prediction with zero known days.
- Midweek prediction after known actual TSA days are already published.

## Architecture

The model forecasts each day independently, then sums or averages the seven daily predictions.

Layer 1: deterministic baseline

- ISO week/day matched prior-year TSA volume.
- Recent TSA growth trend versus prior-year matched dates.
- Seven-day lag and rolling recent TSA levels.
- Holiday-specific anchor logic for Thanksgiving and fixed-date holiday periods.

Layer 2: residual model

- `sklearn.ensemble.HistGradientBoostingRegressor`
- Target is `actual_passengers - deterministic_baseline`.
- Uses only live-safe features.

Layer 3: weekly confidence calibration

- Regime label for each target week.
- Historical walk-forward weekly average errors by known-day scenario and regime.
- Calibrated prediction intervals for the weekly average.
- Point correction is only enabled when walk-forward correction improves MAPE. In the latest test, global expanding point correction worsened MAPE, so it is disabled by default except where a regime-specific calibration entry proves better.

## Backtest Setup

Walk-forward weekly backtest:

- Test weeks: 72 Monday-Sunday weeks from 2025-01-06 through 2026-05-18.
- For each hidden week, the model is trained only on data before that week.
- Scenarios:
  - `known_days = 0`: before the week starts.
  - `known_days = 2`: Monday and Tuesday actuals are known.
  - `known_days = 3`: Monday through Wednesday actuals are known.

## Best Config

`hgb_abs_default`

## Backtest Results

| Scenario | Daily MAE | Daily MAPE | Weekly Total MAPE | Weekly Avg Abs Error | Within +/-50k | 80% Avg Error Width | 90% Avg Error Width |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before week starts | 76,479 | 3.31% | 1.74% | 41,262 | 73.6% | 63,157 | 81,124 |
| After Monday-Tuesday known | 51,968 | 2.23% | 1.56% | 36,791 | 80.6% | 48,008 | 81,510 |
| After Monday-Wednesday known | 43,205 | 1.83% | 1.34% | 31,678 | 84.7% | 42,930 | 53,821 |

The richer holiday/regime features improved the pre-week weekly MAPE from the prior `1.83%` to `1.74%`. Midweek accuracy remains strongest after Wednesday actuals are known.

## Regime Findings

Pre-week prediction is already usually inside +/-50k for summer, July 4, Memorial/Labor, Thanksgiving, and Christmas/New Year regimes in the current sample, but several of those groups have small sample sizes.

Hardest regimes:

- January/winter weeks.
- Normal weeks with no obvious travel-event anchor.
- Christmas/New Year and Thanksgiving when predicting midweek can still have large misses because the remaining travel days are highly asymmetric.

## Trading Confidence

For a strict +/-50k trading boundary:

- Pre-week forecasts are not reliable enough by default: 73.6% historical within +/-50k.
- After Monday-Tuesday, the model reaches about 80.6% within +/-50k.
- After Monday-Wednesday, the model reaches about 84.7% within +/-50k.

The prediction script now prints calibrated ranges and a `Trade-range confidence under 50k` flag. Treat that flag as a filter, not a guarantee.

## Artifacts

- Model: `artifacts/models/live_weekly_residual_model.joblib`
- Weekly backtest summary: `artifacts/reports/live_weekly_backtest_weekly.csv`
- Daily backtest rows: `artifacts/reports/live_weekly_backtest_results.csv`
- Direct weekly model: `artifacts/models/live_weekly_direct_model.joblib`
- Weekly ensemble model: `artifacts/models/live_weekly_ensemble.joblib`
- Weekly ensemble backtest: `artifacts/reports/live_weekly_ensemble_backtest_weekly.csv`

## Direct Weekly Model + Ensemble

A secondary model now predicts the Kalshi target directly: one Monday-Sunday row in, one weekly daily-average prediction out. It preserves day-specific feature blocks instead of averaging features together.

Latest walk-forward comparison:

| Scenario | Daily model avg abs error | Direct weekly avg abs error | Ensemble avg abs error | Daily within +/-50k | Direct within +/-50k | Ensemble within +/-50k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Before week starts | 41,262 | 47,705 | 41,881 | 73.61% | 62.50% | 69.44% |
| After Monday-Tuesday known | 36,791 | 42,471 | 36,264 | 80.56% | 66.67% | 79.17% |
| After Monday-Wednesday known | 31,678 | 33,455 | 30,287 | 84.72% | 77.78% | 86.11% |

Interpretation:

- The direct weekly model alone is not better than the daily model yet.
- The ensemble improves average error after Monday-Tuesday and Monday-Wednesday are known.
- Pre-week, the acceptance rule falls back toward the daily model because direct weekly features do not improve the trading boundary.
- Prediction output now shows all three values: daily model, direct weekly model, and accepted ensemble.

## Usage

Train and backtest:

```powershell
python scripts/backtest_live_weekly_model.py
```

Predict the next Monday-Sunday week after the latest TSA date:

```powershell
python scripts/predict_live_week.py
```

Predict a specific week before it starts:

```powershell
python scripts/predict_live_week.py --monday 2026-06-01 --known-days 0
```

Predict a week with Monday and Tuesday already known:

```powershell
python scripts/predict_live_week.py --monday 2026-06-01 --known-days 2
```

Backtest and retrain the direct weekly ensemble:

```powershell
python scripts/backtest_weekly_ensemble_model.py
```

## Notes

The model intentionally does not use same-day BTS delay values for future predictions. BTS delay data is not live enough for same-week forecasting in this pipeline.
