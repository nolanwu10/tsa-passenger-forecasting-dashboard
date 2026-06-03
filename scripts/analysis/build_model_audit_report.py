from pathlib import Path
import argparse
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.live_weekly_model import (
    DEFAULT_CONFIGS,
    apply_expanding_weekly_correction,
    backtest_configs,
    load_modeling_data,
)


REPORT_PATH = ROOT / "docs" / "model_audit_summary.md"


def fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def markdown_table(df: pd.DataFrame, int_cols: set[str] | None = None, pct_cols: set[str] | None = None) -> str:
    out = df.copy()
    int_cols = int_cols or set()
    pct_cols = pct_cols or set()
    for col in out.columns:
        if col in int_cols:
            out[col] = out[col].map(fmt_int)
        if col in pct_cols:
            out[col] = out[col].map(fmt_pct)
    return out.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an auditable model metrics and baseline report.")
    parser.add_argument("--start", default="2026-01-05", help="First Monday in YYYY-MM-DD format.")
    parser.add_argument("--end", default="2026-02-23", help="Last Monday in YYYY-MM-DD format.")
    parser.add_argument("--config", default="hgb_abs_fast", help="Model config name to evaluate.")
    parser.add_argument(
        "--known-days",
        default="0,2",
        help="Comma-separated known-day scenarios to evaluate.",
    )
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Markdown output path.")
    return parser.parse_args()


def scenario_label(known_days: int) -> str:
    labels = {
        0: "Before week starts",
        1: "After Monday known",
        2: "After Monday-Tuesday known",
        3: "After Monday-Wednesday known",
    }
    return labels.get(int(known_days), f"After {known_days} known days")


def summarize_model(summary: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        summary.groupby("known_days")
        .agg(
            weeks=("week_monday", "count"),
            weekly_total_mape=("weekly_total_mape", "mean"),
            weekly_avg_abs_error=("weekly_avg_error", lambda s: s.abs().mean()),
            within_50k=("weekly_avg_error", lambda s: (s.abs() <= 50_000).mean()),
        )
        .reset_index()
    )
    grouped.insert(0, "scenario", grouped["known_days"].map(scenario_label))
    return grouped[["scenario", "weeks", "weekly_total_mape", "weekly_avg_abs_error", "within_50k"]]


def summarize_baseline(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (known_days, week_monday), group in daily.groupby(["known_days", "week_monday"]):
        actual_total = group["Passengers"].sum()
        if actual_total <= 0:
            continue
        model_total = group["predicted_passengers"].sum()
        baseline_prediction = group["baseline"].where(group["baseline"].notna(), group["Passengers"])
        baseline_total = baseline_prediction.sum()
        rows.append(
            {
                "known_days": int(known_days),
                "week_monday": week_monday,
                "model_weekly_total_mape": abs(model_total - actual_total) / actual_total,
                "baseline_weekly_total_mape": abs(baseline_total - actual_total) / actual_total,
                "model_weekly_avg_abs_error": abs(model_total - actual_total) / len(group),
                "baseline_weekly_avg_abs_error": abs(baseline_total - actual_total) / len(group),
            }
        )

    baseline = pd.DataFrame(rows)
    grouped = (
        baseline.groupby("known_days")
        .agg(
            weeks=("week_monday", "count"),
            deterministic_baseline_mape=("baseline_weekly_total_mape", "mean"),
            residual_model_mape=("model_weekly_total_mape", "mean"),
            mape_improvement=("baseline_weekly_total_mape", "mean"),
            deterministic_baseline_avg_abs_error=("baseline_weekly_avg_abs_error", "mean"),
            residual_model_avg_abs_error=("model_weekly_avg_abs_error", "mean"),
        )
        .reset_index()
    )
    grouped["mape_improvement"] = (
        grouped["deterministic_baseline_mape"] - grouped["residual_model_mape"]
    )
    grouped.insert(0, "scenario", grouped["known_days"].map(scenario_label))
    return grouped[
        [
            "scenario",
            "weeks",
            "deterministic_baseline_mape",
            "residual_model_mape",
            "mape_improvement",
            "deterministic_baseline_avg_abs_error",
            "residual_model_avg_abs_error",
        ]
    ]


def main() -> None:
    args = parse_args()
    known_days = tuple(int(item.strip()) for item in args.known_days.split(",") if item.strip())
    configs = {config.name: config for config in DEFAULT_CONFIGS}
    if args.config not in configs:
        choices = ", ".join(sorted(configs))
        raise SystemExit(f"Unknown --config {args.config!r}. Expected one of: {choices}")

    data = load_modeling_data()
    summary, daily = backtest_configs(
        data,
        configs=[configs[args.config]],
        start_monday=args.start,
        end_monday=args.end,
        known_day_scenarios=known_days,
    )
    if summary.empty:
        raise SystemExit("No audit rows were generated. Check date range and data coverage.")

    summary = apply_expanding_weekly_correction(summary)
    model_summary = summarize_model(summary)
    baseline_summary = summarize_baseline(daily)

    lines = [
        "# Model Audit Summary",
        "",
        "This report is generated by `python scripts/build_model_audit_report.py`.",
        "",
        "It is intentionally lightweight: the default window is a quick reviewer run, not the full historical backtest used for the headline README metrics.",
        "",
        "## Run Configuration",
        "",
        f"- Backtest Mondays: `{args.start}` through `{args.end}`",
        f"- Config: `{args.config}`",
        f"- Known-day scenarios: `{', '.join(str(day) for day in known_days)}`",
        f"- Source data max date: `{pd.Timestamp(data['Date'].max()).date()}`",
        "",
        "## Residual Model Metrics",
        "",
        markdown_table(
            model_summary,
            int_cols={"weeks", "weekly_avg_abs_error"},
            pct_cols={"weekly_total_mape", "within_50k"},
        ),
        "",
        "## Deterministic Baseline Comparison",
        "",
        "The deterministic baseline is the model's first layer before the residual gradient-boosted correction.",
        "",
        markdown_table(
            baseline_summary,
            int_cols={
                "weeks",
                "deterministic_baseline_avg_abs_error",
                "residual_model_avg_abs_error",
            },
            pct_cols={
                "deterministic_baseline_mape",
                "residual_model_mape",
                "mape_improvement",
            },
        ),
        "",
        "## Interpretation",
        "",
        "- Use this report to verify that the code path runs cleanly from a checkout.",
        "- Use the full backtest command for final model claims: `python scripts/backtest_live_weekly_model.py`.",
        "- The baseline comparison shows whether the ML residual layer improves the deterministic forecast on the selected window.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
