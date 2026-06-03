# TSA Passenger Forecasting Dashboard

Live-safe TSA passenger volume forecasting dashboard for weekly travel demand analysis.

The project forecasts the Monday-Sunday average TSA checkpoint passenger count using historical TSA data, calendar and holiday effects, transportation capacity/delay features, calibrated prediction intervals, and an optional Kalshi market comparison layer.

## Live App

Deploy with Streamlit Community Cloud using:

- Main file: `streamlit_app.py`
- Python version: 3.10 or newer
- Dependency file: `requirements.txt`

After deployment, add the generated `https://...streamlit.app` URL to your resume, GitHub profile, and LinkedIn Featured section.

## What The Dashboard Shows

- Current target week and latest TSA data date.
- Weekly average passenger forecast.
- Daily actuals and remaining-day predictions.
- Calibrated confidence ranges for the weekly average.
- Historical walk-forward model performance.
- Data and model artifact health checks.
- Optional live Kalshi contract comparison when credentials are configured.

## Model Summary

The forecasting stack uses:

- Calendar, ISO week, day-of-week, and travel-holiday features.
- Prior-year matched TSA passenger counts.
- Recent TSA trend and rolling passenger levels.
- BTS transportation features where available.
- A deterministic baseline plus a residual `HistGradientBoostingRegressor`.
- Weekly calibration by known-day scenario and travel regime.
- Optional direct-weekly model ensemble.

Latest documented walk-forward results:

| Scenario | Weekly total MAPE | Weekly avg abs error | Within +/-50k |
| --- | ---: | ---: | ---: |
| Before week starts | 1.74% | 41,262 | 73.6% |
| After Monday-Tuesday known | 1.56% | 36,791 | 80.6% |
| After Monday-Wednesday known | 1.34% | 31,678 | 84.7% |

See `docs/live_weekly_model_report.md` for the full model report and limitations.

## Local Setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will build missing processed feature files from committed raw/external data when needed.

## Data Pipeline

Useful commands:

```powershell
python scripts/fetch_tsa_data.py
python scripts/build_calendar_features.py
python scripts/build_transport_features.py
python scripts/backtest_live_weekly_model.py
python scripts/backtest_weekly_ensemble_model.py
python scripts/predict_live_week.py
```

Generated processed data, model artifacts, reports, and logs are intentionally ignored by Git. Rebuild them locally or on first app run.

## Optional Kalshi Setup

Kalshi credentials are optional. The public Streamlit app works without them.

For local use, copy `.env.example` to `.env` and set:

```powershell
KALSHI_KEY_ID=your-key-id
KALSHI_PRIVATE_KEY_PATH=C:\path\to\private_key.pem
KALSHI_HOST=https://external-api.kalshi.com
```

For Streamlit Cloud, use app secrets instead of committing credentials. Never commit `.env`, private keys, or certificates.

## Repository Hygiene

Before publishing:

- Confirm `.env` and `private_key.pem` are not tracked.
- Commit this folder as its own standalone GitHub repository.
- Keep `data/raw/tsa_passenger_data.csv` and small external source CSVs if you want the app to bootstrap on Streamlit Cloud.
- Do not commit generated logs, model artifacts, or private credentials.

## Project Layout

```text
TSAproject/
  streamlit_app.py           Streamlit Community Cloud entry point
  data/raw/                  TSA source data
  data/external/             Optional external enrichment data
  src/tsa_project/           Modeling, ingestion, dashboard, and market modules
  scripts/                   Command-line pipeline and analysis tasks
  docs/                      Model, data, and architecture reports
```
