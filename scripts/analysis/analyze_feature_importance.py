from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.live_weekly_model import (
    CALENDAR_FEATURES,
    DEFAULT_CONFIGS,
    DELAY_FEATURES,
    EXTERNAL_FEATURES,
    HISTORY_FEATURES,
    MODEL_FEATURES,
    load_modeling_data,
    make_model,
    make_supervised_dataset,
)


REPORT_PATH = ROOT / "docs" / "feature_importance_report.md"


FEATURE_GROUPS = {
    "calendar_holiday": CALENDAR_FEATURES,
    "matched_history_recent_tsa": HISTORY_FEATURES,
    "external_transport_capacity": EXTERNAL_FEATURES,
    "bts_daily_delay": DELAY_FEATURES,
}


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
    parser = argparse.ArgumentParser(description="Analyze live weekly residual model feature importance.")
    parser.add_argument("--train-end", default="2026-01-01", help="Exclusive train cutoff date.")
    parser.add_argument("--test-end", default=None, help="Inclusive holdout end date. Defaults to latest data.")
    parser.add_argument("--config", default="hgb_abs_fast", help="Model config name.")
    parser.add_argument("--top-n", type=int, default=20, help="Number of individual features to report.")
    parser.add_argument("--output", type=Path, default=REPORT_PATH, help="Markdown output path.")
    return parser.parse_args()


def mae(values: pd.Series | np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def mape(actual: pd.Series, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - actual.to_numpy()) / actual.to_numpy()))


def permutation_delta(
    model,
    x_holdout: pd.DataFrame,
    y_holdout: pd.Series,
    columns: list[str],
    base_mae: float,
    rng: np.random.Generator,
) -> float:
    shuffled = x_holdout.copy()
    for column in columns:
        if column in shuffled:
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    pred = model.predict(shuffled)
    return mae(pred - y_holdout.to_numpy()) - base_mae


def null_rates(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_name, columns in FEATURE_GROUPS.items():
        present = [column for column in columns if column in data]
        if not present:
            continue
        values = data[present]
        rows.append(
            {
                "feature_group": group_name,
                "features": len(present),
                "avg_null_rate": float(values.isna().mean().mean()),
                "max_null_rate": float(values.isna().mean().max()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    configs = {config.name: config for config in DEFAULT_CONFIGS}
    if args.config not in configs:
        choices = ", ".join(sorted(configs))
        raise SystemExit(f"Unknown --config {args.config!r}. Expected one of: {choices}")

    data = load_modeling_data()
    supervised = make_supervised_dataset(data)
    train_cutoff = pd.Timestamp(args.train_end)
    test_end = pd.Timestamp(args.test_end) if args.test_end else pd.Timestamp(supervised["Date"].max())

    train = supervised[supervised["Date"] < train_cutoff].copy()
    holdout = supervised[
        (supervised["Date"] >= train_cutoff)
        & (supervised["Date"] <= test_end)
    ].copy()
    if len(train) < 120 or len(holdout) < 30:
        raise SystemExit(
            f"Not enough rows for feature analysis: train={len(train):,}, holdout={len(holdout):,}"
        )

    model = make_model(configs[args.config])
    model.fit(train[MODEL_FEATURES], train["residual"])

    x_holdout = holdout[MODEL_FEATURES]
    y_residual = holdout["residual"]
    residual_pred = model.predict(x_holdout)
    base_residual_mae = mae(residual_pred - y_residual.to_numpy())
    passenger_pred = holdout["baseline"].to_numpy() + residual_pred
    base_daily_mae = mae(passenger_pred - holdout["Passengers"].to_numpy())
    base_daily_mape = mape(holdout["Passengers"], passenger_pred)

    rng = np.random.default_rng(42)
    feature_rows = []
    for feature in MODEL_FEATURES:
        delta = permutation_delta(model, x_holdout, y_residual, [feature], base_residual_mae, rng)
        feature_rows.append(
            {
                "feature": feature,
                "group": next(
                    (group for group, columns in FEATURE_GROUPS.items() if feature in columns),
                    "other",
                ),
                "residual_mae_increase": delta,
            }
        )
    feature_importance = (
        pd.DataFrame(feature_rows)
        .sort_values("residual_mae_increase", ascending=False)
        .head(args.top_n)
    )

    group_rows = []
    rng = np.random.default_rng(84)
    for group_name, columns in FEATURE_GROUPS.items():
        present = [column for column in columns if column in MODEL_FEATURES]
        delta = permutation_delta(model, x_holdout, y_residual, present, base_residual_mae, rng)
        group_rows.append(
            {
                "feature_group": group_name,
                "features": len(present),
                "residual_mae_increase": delta,
            }
        )
    group_importance = pd.DataFrame(group_rows).sort_values("residual_mae_increase", ascending=False)

    lines = [
        "# Feature Importance Report",
        "",
        "This report is generated by `python scripts/analyze_feature_importance.py`.",
        "",
        "Importance is measured with holdout permutation: shuffle a feature or feature group, then measure the increase in residual-model MAE. Higher positive values indicate stronger model dependence.",
        "",
        "## Run Configuration",
        "",
        f"- Config: `{args.config}`",
        f"- Train rows: {fmt_int(len(train))} before `{train_cutoff.date()}`",
        f"- Holdout rows: {fmt_int(len(holdout))} from `{holdout['Date'].min().date()}` through `{holdout['Date'].max().date()}`",
        f"- Baseline plus residual daily MAE: {fmt_int(base_daily_mae)}",
        f"- Baseline plus residual daily MAPE: {fmt_pct(base_daily_mape)}",
        f"- Residual-model MAE on residual target: {fmt_int(base_residual_mae)}",
        "",
        "## Feature Group Importance",
        "",
        markdown_table(group_importance, int_cols={"features", "residual_mae_increase"}),
        "",
        "## Top Individual Features",
        "",
        markdown_table(feature_importance, int_cols={"residual_mae_increase"}),
        "",
        "## Feature Group Missingness",
        "",
        markdown_table(null_rates(supervised[MODEL_FEATURES]), int_cols={"features"}, pct_cols={"avg_null_rate", "max_null_rate"}),
        "",
        "## Interpretation",
        "",
        "- Calendar, matched-history, and recent TSA features should carry most of the useful signal.",
        "- Sparse BTS daily delay features should be treated as optional context unless their null rate and ablation impact improve.",
        "- Engineered ratio columns from the processed table are not part of `MODEL_FEATURES` and are therefore excluded from this live model.",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
