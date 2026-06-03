from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.live_weekly_model import (
    DEFAULT_CONFIGS,
    FEATURE_REPORT_PATH,
    WEEKLY_REPORT_PATH,
    apply_expanding_weekly_correction,
    backtest_configs,
    build_calibration,
    choose_best_config,
    load_modeling_data,
    train_final_model,
)


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


if __name__ == "__main__":
    data = load_modeling_data()
    summary, daily = backtest_configs(data, DEFAULT_CONFIGS)
    summary = apply_expanding_weekly_correction(summary)

    FEATURE_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(WEEKLY_REPORT_PATH, index=False)
    daily.to_csv(FEATURE_REPORT_PATH, index=False)

    leaderboard = (
        summary.groupby(["config", "known_days"])
        .agg(
            weeks=("week_monday", "count"),
            daily_mae=("daily_mae", "mean"),
            daily_mape=("daily_mape", "mean"),
            weekly_total_mape=("weekly_total_mape", "mean"),
            corrected_weekly_total_mape=("corrected_weekly_total_mape", "mean"),
            within_50k=("corrected_weekly_avg_error", lambda s: (s.abs() <= 50_000).mean()),
            weekly_avg_abs_error=("weekly_avg_error", lambda s: s.abs().mean()),
            corrected_weekly_avg_abs_error=("corrected_weekly_avg_error", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values(["known_days", "corrected_weekly_total_mape"])
    )

    print("Backtest leaderboard:")
    print(
        leaderboard.to_string(
            index=False,
            formatters={
                "daily_mae": "{:,.0f}".format,
                "daily_mape": fmt_pct,
                "weekly_total_mape": fmt_pct,
                "corrected_weekly_total_mape": fmt_pct,
                "within_50k": fmt_pct,
                "weekly_avg_abs_error": "{:,.0f}".format,
                "corrected_weekly_avg_abs_error": "{:,.0f}".format,
            },
        )
    )

    best = choose_best_config(summary)
    calibration = build_calibration(summary, best)
    trained = train_final_model(best, calibration=calibration)
    print(f"\nBest pre-week config: {best}")
    print(f"Saved final model: {trained['path']}")
    print(f"Saved weekly summary: {WEEKLY_REPORT_PATH}")
    print(f"Saved daily results: {FEATURE_REPORT_PATH}")
