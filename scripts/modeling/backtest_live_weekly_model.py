from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
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


def parse_csv_ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return parsed


def parse_config_names(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one config name")
    available = {config.name for config in DEFAULT_CONFIGS}
    unknown = sorted(set(parsed) - available)
    if unknown:
        choices = ", ".join(sorted(available))
        raise argparse.ArgumentTypeError(
            f"unknown config(s): {', '.join(unknown)}. Expected one of: {choices}"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run walk-forward live weekly model backtests.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a reviewer-friendly smoke backtest with one fast config and a short 2026 window.",
    )
    parser.add_argument("--start", help="First Monday to backtest, in YYYY-MM-DD format.")
    parser.add_argument("--end", help="Last Monday to backtest, in YYYY-MM-DD format.")
    parser.add_argument(
        "--configs",
        type=parse_config_names,
        help="Comma-separated model config names. Defaults to all configs, or hgb_abs_fast in --quick mode.",
    )
    parser.add_argument(
        "--known-days",
        type=parse_csv_ints,
        help="Comma-separated known-day scenarios. Defaults to 0,1,2,3, or 0,2 in --quick mode.",
    )
    parser.add_argument(
        "--skip-train-final",
        action="store_true",
        help="Do not train and save the final model artifact after the backtest.",
    )
    parser.add_argument(
        "--train-final",
        action="store_true",
        help="Train and save the final model artifact even in --quick mode.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    configs_by_name = {config.name: config for config in DEFAULT_CONFIGS}
    if args.configs:
        configs = [configs_by_name[name] for name in args.configs]
    elif args.quick:
        configs = [configs_by_name["hgb_abs_fast"]]
    else:
        configs = DEFAULT_CONFIGS

    start_monday = args.start or ("2026-01-05" if args.quick else "2025-01-06")
    end_monday = args.end or ("2026-02-23" if args.quick else None)
    known_day_scenarios = args.known_days or ((0, 2) if args.quick else (0, 1, 2, 3))
    train_final = args.train_final or (not args.quick and not args.skip_train_final)

    data = load_modeling_data()
    summary, daily = backtest_configs(
        data,
        configs,
        start_monday=start_monday,
        end_monday=end_monday,
        known_day_scenarios=known_day_scenarios,
    )
    if summary.empty:
        raise SystemExit("No backtest rows were generated. Check --start, --end, and data coverage.")

    summary = apply_expanding_weekly_correction(summary)
    weekly_report_path = WEEKLY_REPORT_PATH
    feature_report_path = FEATURE_REPORT_PATH
    if args.quick:
        weekly_report_path = WEEKLY_REPORT_PATH.with_name("live_weekly_backtest_weekly_quick.csv")
        feature_report_path = FEATURE_REPORT_PATH.with_name("live_weekly_backtest_results_quick.csv")

    feature_report_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(weekly_report_path, index=False)
    daily.to_csv(feature_report_path, index=False)

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
    trained = train_final_model(best, calibration=calibration) if train_final else None
    print(f"\nBest pre-week config: {best}")
    if trained:
        print(f"Saved final model: {trained['path']}")
    else:
        print("Skipped final model training.")
    print(f"Saved weekly summary: {weekly_report_path}")
    print(f"Saved daily results: {feature_report_path}")
