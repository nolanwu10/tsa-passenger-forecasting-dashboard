from pathlib import Path
import argparse
import os
import sys
import warnings

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message="Could not find the number of physical cores*")

from tsa_project.kalshi_blend_backtest import (
    KALSHI_BLEND_BACKTEST_DAILY_PATH,
    KALSHI_BLEND_BACKTEST_SUMMARY_PATH,
    KALSHI_YTD_BACKTEST_PATH,
    run_kalshi_blend_backtest,
)


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backtest dashboard Kalshi sentiment blends for the daily model, direct weekly model, "
            "and weekly+daily ensemble."
        )
    )
    parser.add_argument(
        "--market-report",
        type=Path,
        default=KALSHI_YTD_BACKTEST_PATH,
        help="Historical market-day report with 50%% crossing market_predicted_avg values.",
    )
    parser.add_argument(
        "--min-train-weeks",
        type=int,
        default=80,
        help="Minimum prior weekly rows required before fitting a walk-forward direct weekly model.",
    )
    parser.add_argument(
        "--daily-output",
        type=Path,
        default=KALSHI_BLEND_BACKTEST_DAILY_PATH,
        help="Destination for per-market-day, per-variant backtest rows.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=KALSHI_BLEND_BACKTEST_SUMMARY_PATH,
        help="Destination for aggregate comparison metrics.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    base, results, summary = run_kalshi_blend_backtest(
        market_path=args.market_report,
        min_train_weeks=args.min_train_weeks,
    )

    args.daily_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.daily_output, index=False)
    summary.to_csv(args.summary_output, index=False)

    overall = summary[summary["grain"] == "overall"].copy()
    overall = overall.sort_values("final_mae")
    print("Kalshi dashboard ensemble backtest:")
    print(
        overall[
            [
                "variant",
                "rows",
                "model_mae",
                "market_mae",
                "final_mae",
                "model_win_rate_vs_market",
                "final_win_rate_vs_market",
                "final_win_rate_vs_model",
                "final_beats_both_rate",
                "final_within_50k",
                "p80_final_abs_error",
                "p90_final_abs_error",
                "worst_final_abs_error",
            ]
        ].to_string(
            index=False,
            formatters={
                "model_mae": fmt_count,
                "market_mae": fmt_count,
                "final_mae": fmt_count,
                "model_win_rate_vs_market": fmt_pct,
                "final_win_rate_vs_market": fmt_pct,
                "final_win_rate_vs_model": fmt_pct,
                "final_beats_both_rate": fmt_pct,
                "final_within_50k": fmt_pct,
                "p80_final_abs_error": fmt_count,
                "p90_final_abs_error": fmt_count,
                "worst_final_abs_error": fmt_count,
            },
        )
    )

    print("\nKnown-day breakdown:")
    by_known = summary[summary["grain"] == "known_days"].copy()
    by_known = by_known.sort_values(["known_days", "final_mae", "variant"])
    print(
        by_known[
            [
                "known_days",
                "variant",
                "rows",
                "model_mae",
                "final_mae",
                "final_within_50k",
                "final_win_rate_vs_market",
                "final_beats_both_rate",
            ]
        ].to_string(
            index=False,
            formatters={
                "model_mae": fmt_count,
                "final_mae": fmt_count,
                "final_within_50k": fmt_pct,
                "final_win_rate_vs_market": fmt_pct,
                "final_beats_both_rate": fmt_pct,
            },
        )
    )

    worst = results.sort_values("final_abs_error", ascending=False).head(10)
    print("\nWorst final misses:")
    print(
        worst[
            [
                "variant",
                "week_monday",
                "as_of_date",
                "known_days",
                "actual_weekly_avg",
                "model_forecast",
                "market_forecast",
                "final_forecast",
                "final_abs_error",
            ]
        ].to_string(
            index=False,
            formatters={
                "actual_weekly_avg": fmt_count,
                "model_forecast": fmt_count,
                "market_forecast": fmt_count,
                "final_forecast": fmt_count,
                "final_abs_error": fmt_count,
            },
        )
    )

    print(f"\nRows evaluated: {len(base)} market days x {results['variant'].nunique()} variants")
    print(f"Saved daily report: {args.daily_output}")
    print(f"Saved summary report: {args.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
