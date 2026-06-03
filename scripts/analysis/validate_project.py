from pathlib import Path
import contextlib
import io
import importlib.util
import logging
import sys
import warnings

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import RAW_TSA_PATH
from tsa_project.datasets import normalize_tsa_raw
from tsa_project.features import write_calendar_holiday_features
from tsa_project.live_weekly_model import (
    DEFAULT_CONFIGS,
    LiveFeatureFactory,
    MODEL_FEATURES,
    apply_expanding_weekly_correction,
    backtest_configs,
    load_modeling_data,
    make_model,
    make_supervised_dataset,
    predict_week,
)
from tsa_project.transport_features import write_transport_features


def log(message: str) -> None:
    print(f"[validate] {message}")


def next_completed_monday(data: pd.DataFrame) -> pd.Timestamp:
    latest = pd.Timestamp(data["Date"].max()).normalize()
    monday = latest - pd.Timedelta(days=latest.weekday())
    if monday + pd.Timedelta(days=6) > latest:
        monday = monday - pd.Timedelta(days=7)
    return monday


def check_raw_tsa() -> pd.DataFrame:
    if not RAW_TSA_PATH.exists():
        raise FileNotFoundError(f"Missing raw TSA data: {RAW_TSA_PATH}")

    raw = normalize_tsa_raw(pd.read_csv(RAW_TSA_PATH))
    expected_days = pd.date_range(raw["Date"].min(), raw["Date"].max(), freq="D")
    missing = expected_days.difference(raw["Date"])
    if not missing.empty:
        sample = ", ".join(date.date().isoformat() for date in missing[:5])
        raise ValueError(f"Raw TSA data has {len(missing)} missing date(s), starting with: {sample}")

    log(
        "raw TSA data ok: "
        f"{len(raw):,} rows from {raw['Date'].min().date()} to {raw['Date'].max().date()}"
    )
    return raw


def check_feature_build(raw: pd.DataFrame) -> pd.DataFrame:
    calendar = write_calendar_holiday_features(raw)
    transport = write_transport_features()
    latest = pd.Timestamp(transport["Date"].max()).normalize()
    feature_row = LiveFeatureFactory(transport).feature_row(latest, latest - pd.Timedelta(days=1))
    if list(feature_row) != MODEL_FEATURES:
        raise ValueError("Live feature factory output does not match MODEL_FEATURES order.")

    log(
        "feature build ok: "
        f"{len(calendar):,} calendar rows, {len(transport):,} transport rows"
    )
    return transport


def check_quick_backtest(data: pd.DataFrame) -> pd.DataFrame:
    summary, _ = backtest_configs(
        data,
        configs=[config for config in DEFAULT_CONFIGS if config.name == "hgb_abs_fast"],
        start_monday="2026-01-05",
        end_monday="2026-02-23",
        known_day_scenarios=(0, 2),
    )
    if summary.empty:
        raise ValueError("Quick backtest generated no rows.")

    summary = apply_expanding_weekly_correction(summary)
    preweek = summary[summary["known_days"] == 0]
    log(
        "quick backtest ok: "
        f"{len(summary):,} rows, pre-week MAPE {preweek['weekly_total_mape'].mean():.2%}"
    )
    return summary


def check_prediction_smoke(data: pd.DataFrame) -> None:
    monday = next_completed_monday(data)
    train = make_supervised_dataset(data[data["Date"] < monday]).tail(700)
    if len(train) < 120:
        raise ValueError(f"Not enough training rows for prediction smoke test before {monday.date()}")

    config = next(config for config in DEFAULT_CONFIGS if config.name == "hgb_abs_fast")
    model = make_model(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[MODEL_FEATURES], train["residual"])

    observed = data[data["Date"] < monday].copy()
    prediction = predict_week(model, observed, monday, known_days=0)
    if len(prediction) != 7 or prediction["predicted_passengers"].isna().any():
        raise ValueError("Prediction smoke test did not produce seven complete daily predictions.")

    log(
        "prediction smoke ok: "
        f"week of {monday.date()}, weekly average {prediction['predicted_passengers'].mean():,.0f}"
    )


def check_streamlit_import() -> None:
    logging.getLogger("streamlit").setLevel(logging.ERROR)
    logging.getLogger("streamlit.runtime.caching.cache_data_api").setLevel(logging.ERROR)
    app_path = ROOT / "streamlit_app.py"
    spec = importlib.util.spec_from_file_location("streamlit_app_validation", app_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load import spec for {app_path}")
    module = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stderr(io.StringIO()):
        spec.loader.exec_module(module)
    log("streamlit app import ok")


def main() -> None:
    raw = check_raw_tsa()
    check_feature_build(raw)
    data = load_modeling_data()
    check_quick_backtest(data)
    check_prediction_smoke(data)
    check_streamlit_import()
    log("project validation complete")


if __name__ == "__main__":
    main()
