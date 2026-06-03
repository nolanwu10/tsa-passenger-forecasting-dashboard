from pathlib import Path
import argparse
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")

from tsa_project.live_weekly_model import (
    MODEL_PATH,
    apply_weekly_calibration,
    load_modeling_data,
    lookup_calibration,
    predict_week,
    train_final_model,
    week_regime,
)
from tsa_project.weekly_ensemble_model import predict_weekly_ensemble


def next_monday_after(date_value: pd.Timestamp) -> pd.Timestamp:
    date_value = pd.Timestamp(date_value).normalize()
    days_until_monday = (7 - date_value.weekday()) % 7
    if days_until_monday == 0:
        days_until_monday = 7
    return date_value + pd.Timedelta(days=days_until_monday)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict TSA Monday-Sunday weekly average.")
    parser.add_argument("--monday", help="Target week Monday in YYYY-MM-DD format. Defaults to next Monday after latest TSA date.")
    parser.add_argument("--known-days", type=int, default=0, help="Known actual days in target week, e.g. 0 before week, 2 after Tuesday.")
    args = parser.parse_args()

    data = load_modeling_data()
    if args.monday:
        monday = pd.Timestamp(args.monday)
    else:
        monday = next_monday_after(data["Date"].max())

    if not MODEL_PATH.exists():
        train_final_model()

    payload = joblib.load(MODEL_PATH)
    model = payload["model"]
    calibration = payload.get("calibration")
    result = predict_week(model, data, monday, known_days=args.known_days)
    regime = week_regime(data, monday)
    calibration_entry = lookup_calibration(calibration, args.known_days, regime)
    result, weekly_correction = apply_weekly_calibration(result, calibration_entry)
    weekly_avg = result["predicted_passengers"].mean()
    weekly_total = result["predicted_passengers"].sum()

    print(result.to_string(index=False, formatters={
        "baseline": lambda v: "" if pd.isna(v) else f"{v:,.0f}",
        "residual_prediction": lambda v: "" if pd.isna(v) else f"{v:+,.0f}",
        "predicted_passengers": "{:,.0f}".format,
        "weekly_correction_applied_per_predicted_day": "{:+,.0f}".format,
    }))
    print(f"\nDaily-model calibrated weekly average: {weekly_avg:,.0f}")
    print(f"Daily-model calibrated weekly total: {weekly_total:,.0f}")
    print(f"Regime: {regime}")
    print(f"Calibration source: {calibration_entry.get('source', 'none')} (n={calibration_entry.get('n', 0)})")
    print(f"Weekly correction applied: {weekly_correction:+,.0f} passengers/day-average")
    intervals = calibration_entry.get("intervals", {}) or {}
    for level in ["0.68", "0.8", "0.9", "0.95"]:
        if level in intervals:
            width = intervals[level]
            print(f"{float(level):.0%} calibrated average range: {weekly_avg - width:,.0f} to {weekly_avg + width:,.0f} (+/-{width:,.0f})")
    within_50k = calibration_entry.get("within_50k_rate")
    if pd.notna(within_50k):
        print(f"Historical calibrated within +/-50k rate: {within_50k:.1%}")
        print(f"Trade-range confidence under 50k: {'YES' if within_50k >= 0.8 else 'NO'}")

    try:
        ensemble = predict_weekly_ensemble(data, monday, args.known_days)
    except FileNotFoundError as exc:
        print(f"\nWeekly ensemble unavailable: {exc}")
    else:
        print("\nWeekly ensemble comparison:")
        print(f"Daily model average: {ensemble['daily_model_avg']:,.0f}")
        print(f"Direct weekly model average: {ensemble['direct_weekly_avg']:,.0f}")
        print(f"Ensemble average: {ensemble['ensemble_avg']:,.0f}")
        print(f"Ensemble weight on daily model: {ensemble['ensemble_weight_daily']:.1f}")
        print(f"Ensemble weight source: {ensemble['ensemble_weight_source']}")
        ens_cal = ensemble.get("calibration", {})
        ens_intervals = ens_cal.get("intervals", {}) or {}
        for level in ["0.68", "0.8", "0.9", "0.95"]:
            if level in ens_intervals:
                width = ens_intervals[level]
                avg = ensemble["ensemble_avg"]
                print(
                    f"Ensemble {float(level):.0%} average range: "
                    f"{avg - width:,.0f} to {avg + width:,.0f} (+/-{width:,.0f})"
                )
        ens_within = ens_cal.get("within_50k_rate")
        if pd.notna(ens_within):
            print(f"Ensemble historical within +/-50k rate: {ens_within:.1%}")
            print(f"Ensemble trade-range confidence under 50k: {'YES' if ens_within >= 0.8 else 'NO'}")
