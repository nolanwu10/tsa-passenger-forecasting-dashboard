from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tsa_project.dashboard import build_summary, kalshi_status, predict_current_week
from tsa_project.kalshi import build_market_dashboard


BACKTEST_SUMMARY = pd.DataFrame(
    [
        {
            "Scenario": "Before week starts",
            "Weekly total MAPE": "1.74%",
            "Weekly avg abs error": 41262,
            "Within +/-50k": "73.6%",
            "90% avg error width": 81124,
        },
        {
            "Scenario": "After Monday-Tuesday known",
            "Weekly total MAPE": "1.56%",
            "Weekly avg abs error": 36791,
            "Within +/-50k": "80.6%",
            "90% avg error width": 81510,
        },
        {
            "Scenario": "After Monday-Wednesday known",
            "Weekly total MAPE": "1.34%",
            "Weekly avg abs error": 31678,
            "Within +/-50k": "84.7%",
            "90% avg error width": 53821,
        },
    ]
)


def apply_streamlit_secrets() -> None:
    secrets_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        ROOT / ".streamlit" / "secrets.toml",
    ]
    if not any(path.exists() for path in secrets_paths):
        return
    secrets = st.secrets
    for key in ("KALSHI_KEY_ID", "KALSHI_PRIVATE_KEY_PATH", "KALSHI_HOST"):
        if key in secrets and key not in os.environ:
            os.environ[key] = str(secrets[key])


@st.cache_data(show_spinner=False, ttl=300)
def cached_summary() -> dict[str, object]:
    return build_summary()


@st.cache_data(show_spinner="Running TSA weekly forecast...", ttl=300)
def cached_prediction() -> dict[str, object]:
    return predict_current_week()


def format_int(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value):,.0f}"


def format_percent(value: object) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{float(value) * 100:.1f}%"


def render_metrics(summary: dict[str, object], prediction: dict[str, object]) -> None:
    current_week = summary["current_week"]
    last_week = summary["last_week"]
    forecast = prediction.get("dashboard_model_weekly_average") or prediction["predicted_weekly_average"]

    cols = st.columns(5)
    cols[0].metric(
        "Forecast weekly avg",
        format_int(forecast),
        prediction.get("dashboard_model_source", "daily model"),
    )
    cols[1].metric(
        "Current actual avg",
        format_int(current_week.get("average")),
        f"{current_week.get('known_days', 0)} known days",
    )
    cols[2].metric(
        "Last week avg",
        format_int(last_week.get("average")),
        f"{last_week['monday']} to {last_week['sunday']}",
    )
    cols[3].metric("Latest TSA date", str(summary["latest_tsa_date"]))
    cols[4].metric(
        "Trade-range confidence",
        format_percent(prediction.get("within_50k_rate")),
        prediction.get("calibration_source", "calibration"),
    )


def render_forecast(prediction: dict[str, object]) -> None:
    rows = pd.DataFrame(prediction["rows"]).copy()
    rows["Date"] = pd.to_datetime(rows["Date"])
    rows["Passengers"] = rows["predicted_passengers"].round(0).astype("Int64")

    st.subheader("Weekly Forecast Path")
    chart = rows.set_index("Date")[["predicted_passengers"]].rename(
        columns={"predicted_passengers": "Passengers"}
    )
    st.line_chart(chart)

    ranges = pd.DataFrame.from_dict(prediction.get("calibrated_ranges", {}), orient="index")
    if not ranges.empty:
        ranges.index.name = "confidence"
        ranges = ranges.reset_index()
        ranges["lower"] = ranges["lower"].round(0).astype("Int64")
        ranges["upper"] = ranges["upper"].round(0).astype("Int64")
        ranges["width"] = ranges["width"].round(0).astype("Int64")
        st.subheader("Calibrated Weekly Average Ranges")
        st.dataframe(ranges, use_container_width=True, hide_index=True)

    st.subheader("Daily Rows")
    st.dataframe(
        rows[["Date", "day_name", "type", "Passengers"]],
        use_container_width=True,
        hide_index=True,
    )


def render_data_health(summary: dict[str, object]) -> None:
    datasets = pd.DataFrame(summary["datasets"]).T.reset_index(names="dataset")
    st.subheader("Data Health")
    st.dataframe(datasets, use_container_width=True, hide_index=True)

    model = summary["model"]
    st.subheader("Model Artifact")
    st.json(
        {
            "exists": model["exists"],
            "path": model["path"],
            "updated_at": model["updated_at"],
        }
    )


def render_kalshi(prediction: dict[str, object]) -> None:
    status = kalshi_status()
    st.subheader("Kalshi Market Comparison")
    st.write(status["message"])
    if not status["enabled"]:
        st.info("The public dashboard runs without Kalshi credentials. Add Streamlit secrets to enable live market data.")
        return

    if st.button("Load live Kalshi markets"):
        with st.spinner("Fetching Kalshi markets..."):
            market_payload = build_market_dashboard(prediction)
        st.metric("Model forecast", format_int(market_payload["model_forecast"]))
        st.metric("Kalshi-implied forecast", format_int(market_payload.get("live_market_forecast")))
        st.metric("Blended forecast", format_int(market_payload["dashboard_blended_forecast"]))
        rows = pd.DataFrame(market_payload["markets"])
        if not rows.empty:
            st.dataframe(
                rows[
                    [
                        "ticker",
                        "threshold",
                        "yes_bid",
                        "yes_ask",
                        "market_probability",
                        "model_probability",
                        "adjusted_edge",
                        "adjusted_signal",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )


def main() -> None:
    st.set_page_config(
        page_title="TSA Passenger Forecasting",
        page_icon="TSA",
        layout="wide",
    )
    apply_streamlit_secrets()

    st.title("TSA Passenger Forecasting Dashboard")
    st.caption(
        "Live-safe weekly passenger volume forecasts using TSA history, calendar effects, "
        "transportation features, model calibration, and optional Kalshi market comparison."
    )

    refresh = st.button("Refresh forecast data")
    if refresh:
        cached_summary.clear()
        cached_prediction.clear()

    summary = cached_summary()
    prediction = cached_prediction()
    render_metrics(summary, prediction)

    forecast_tab, performance_tab, data_tab, kalshi_tab = st.tabs(
        ["Forecast", "Performance", "Data", "Kalshi"]
    )
    with forecast_tab:
        render_forecast(prediction)
    with performance_tab:
        st.subheader("Walk-forward Backtest Summary")
        st.dataframe(BACKTEST_SUMMARY, use_container_width=True, hide_index=True)
        st.caption("Backtest window: 72 Monday-Sunday weeks from 2025-01-06 through 2026-05-18.")
    with data_tab:
        render_data_health(summary)
    with kalshi_tab:
        render_kalshi(prediction)


if __name__ == "__main__":
    main()
