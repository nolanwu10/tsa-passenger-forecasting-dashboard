from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.live_weekly_model import load_modeling_data
from tsa_project.weekly_ensemble_model import (
    ENSEMBLE_WEEKLY_REPORT_PATH,
    backtest_weekly_ensemble,
    train_final_weekly_ensemble,
)


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


if __name__ == "__main__":
    data = load_modeling_data()
    results = backtest_weekly_ensemble(data)
    ENSEMBLE_WEEKLY_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(ENSEMBLE_WEEKLY_REPORT_PATH, index=False)

    summary = (
        results.groupby("known_days")
        .agg(
            weeks=("week_monday", "count"),
            daily_avg_abs_error=("daily_abs_error", "mean"),
            direct_avg_abs_error=("direct_abs_error", "mean"),
            ensemble_avg_abs_error=("ensemble_abs_error", "mean"),
            daily_mape=("daily_mape", "mean"),
            direct_mape=("direct_mape", "mean"),
            ensemble_mape=("ensemble_mape", "mean"),
            daily_within_50k=("daily_abs_error", lambda s: (s <= 50_000).mean()),
            direct_within_50k=("direct_abs_error", lambda s: (s <= 50_000).mean()),
            ensemble_within_50k=("ensemble_abs_error", lambda s: (s <= 50_000).mean()),
            ensemble_p80=("ensemble_abs_error", lambda s: s.quantile(0.8)),
            ensemble_p90=("ensemble_abs_error", lambda s: s.quantile(0.9)),
        )
        .reset_index()
    )

    print("Weekly ensemble backtest:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "daily_avg_abs_error": "{:,.0f}".format,
                "direct_avg_abs_error": "{:,.0f}".format,
                "ensemble_avg_abs_error": "{:,.0f}".format,
                "daily_mape": fmt_pct,
                "direct_mape": fmt_pct,
                "ensemble_mape": fmt_pct,
                "daily_within_50k": fmt_pct,
                "direct_within_50k": fmt_pct,
                "ensemble_within_50k": fmt_pct,
                "ensemble_p80": "{:,.0f}".format,
                "ensemble_p90": "{:,.0f}".format,
            },
        )
    )

    print("\nWorst ensemble misses:")
    worst = results.sort_values("ensemble_abs_error", ascending=False).head(10)
    print(
        worst[
            [
                "week_monday",
                "known_days",
                "regime",
                "weekly_actual_avg",
                "daily_model_avg",
                "direct_weekly_avg",
                "ensemble_avg",
                "ensemble_weight_daily",
                "ensemble_abs_error",
            ]
        ].to_string(
            index=False,
            formatters={
                "weekly_actual_avg": "{:,.0f}".format,
                "daily_model_avg": "{:,.0f}".format,
                "direct_weekly_avg": "{:,.0f}".format,
                "ensemble_avg": "{:,.0f}".format,
                "ensemble_weight_daily": "{:.1f}".format,
                "ensemble_abs_error": "{:,.0f}".format,
            },
        )
    )

    artifacts = train_final_weekly_ensemble(data, results)
    print(f"\nSaved direct model: {artifacts['direct_path']}")
    print(f"Saved ensemble model: {artifacts['ensemble_path']}")
    print(f"Saved report: {ENSEMBLE_WEEKLY_REPORT_PATH}")

