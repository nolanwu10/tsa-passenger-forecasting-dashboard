from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


DATA_PATH = ROOT / "data" / "processed" / "tsa_daily_transport_features.csv"
REPORT_PATH = ROOT / "docs" / "live_weekly_predictor_analysis.md"


def fmt_int(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return f"{value:+.2%}"


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy().sort_values("Date")
    lookup = work.set_index(["iso_year", "iso_week", "day_of_week"])["Passengers"]
    fallback = work["Passengers"].expanding().mean().shift(1)
    for lag_years in [1, 2, 3, 4, 5, 6, 7]:
        keys = pd.MultiIndex.from_arrays(
            [work["iso_year"] - lag_years, work["iso_week"], work["day_of_week"]]
        )
        work[f"matched_{lag_years}ya"] = lookup.reindex(keys).to_numpy()
    keys_2019 = pd.MultiIndex.from_arrays(
        [pd.Series(2019, index=work.index), work["iso_week"], work["day_of_week"]]
    )
    work["baseline_2019"] = lookup.reindex(keys_2019).to_numpy()
    work["lag_7"] = work["Passengers"].shift(7)
    work["roll_7_lag1"] = work["Passengers"].shift(1).rolling(7, min_periods=3).mean()
    work["roll_14_lag1"] = work["Passengers"].shift(1).rolling(14, min_periods=7).mean()
    work["roll_28_lag1"] = work["Passengers"].shift(1).rolling(28, min_periods=14).mean()
    work["growth_28_vs_1ya"] = (
        work["Passengers"].shift(1).rolling(28, min_periods=14).sum()
        / work["matched_1ya"].shift(1).rolling(28, min_periods=14).sum()
    )
    work["matched_1ya_growth_adjusted"] = work["matched_1ya"] * work["growth_28_vs_1ya"]
    work["fallback_expanding_mean"] = fallback
    return work


def evaluate_baselines(df: pd.DataFrame) -> pd.DataFrame:
    test = df[(df["Date"] >= "2024-01-01") & (df["Date"] <= "2026-05-25")].copy()
    baselines = {
        "matched_1ya": test["matched_1ya"],
        "matched_2ya": test["matched_2ya"],
        "baseline_2019": test["baseline_2019"],
        "lag_7": test["lag_7"],
        "matched_1ya_growth_adjusted": test["matched_1ya_growth_adjusted"],
        "blend_1ya_2ya": (0.7 * test["matched_1ya"]) + (0.3 * test["matched_2ya"]),
        "blend_growth_lag7": (
            0.6 * test["matched_1ya_growth_adjusted"]
            + 0.25 * test["lag_7"]
            + 0.15 * test["baseline_2019"]
        ),
    }
    rows = []
    for name, pred in baselines.items():
        mask = pred.notna() & test["Passengers"].notna()
        actual = test.loc[mask, "Passengers"]
        estimate = pred.loc[mask]
        err = estimate - actual
        rows.append(
            {
                "baseline": name,
                "n": int(mask.sum()),
                "mae": float(err.abs().mean()),
                "mape": float((err.abs() / actual).mean()),
                "bias": float(err.mean()),
                "weekly_total_mape": weekly_total_mape(test.loc[mask], estimate),
            }
        )
    return pd.DataFrame(rows).sort_values("mae")


def weekly_total_mape(test: pd.DataFrame, pred: pd.Series) -> float:
    weekly = test[["iso_year", "iso_week", "Passengers"]].copy()
    weekly["pred"] = pred.to_numpy()
    grouped = weekly.groupby(["iso_year", "iso_week"]).agg(
        actual=("Passengers", "sum"),
        pred=("pred", "sum"),
        days=("Passengers", "count"),
    )
    grouped = grouped[grouped["days"] >= 5]
    return float(((grouped["pred"] - grouped["actual"]).abs() / grouped["actual"]).mean())


def group_signal(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    recent = df[df["Date"] >= "2023-01-01"].copy()
    dow = recent.groupby("day_of_week")["Passengers"].agg(["count", "mean", "std"]).reset_index()
    dow["day_name"] = dow["day_of_week"].map(
        dict(enumerate(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))
    )
    month = recent.groupby("month")["Passengers"].agg(["count", "mean", "std"]).reset_index()
    holiday = recent.groupby("is_holiday_window")["Passengers"].agg(["count", "mean", "std"]).reset_index()
    thanksgiving = recent.groupby("is_thanksgiving_week")["Passengers"].agg(["count", "mean", "std"]).reset_index()
    return {
        "dow": dow[["day_name", "count", "mean", "std"]],
        "month": month,
        "holiday": holiday,
        "thanksgiving": thanksgiving,
    }


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    candidate_cols = [
        "matched_1ya",
        "matched_2ya",
        "baseline_2019",
        "lag_7",
        "roll_7_lag1",
        "roll_28_lag1",
        "matched_1ya_growth_adjusted",
        "day_of_week",
        "iso_week",
        "is_holiday_window",
        "is_thanksgiving_week",
        "days_until_christmas",
        "days_until_new_year",
        "bts_t100_est_daily_domestic_seats",
        "bts_t100_domestic_load_factor",
        "bts_mts_domestic_air_traffic_nsa_lag2_months",
    ]
    rows = []
    model_df = df[df["Date"] >= "2023-01-01"].copy()
    for col in candidate_cols:
        if col not in model_df:
            continue
        valid = model_df[["Passengers", col]].dropna()
        if len(valid) < 30:
            continue
        rows.append(
            {
                "feature": col,
                "n": len(valid),
                "corr_with_passengers": valid["Passengers"].corr(valid[col]),
            }
        )
    return pd.DataFrame(rows).sort_values("corr_with_passengers", key=lambda s: s.abs(), ascending=False)


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


def write_report() -> None:
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    enriched = add_history_features(df)
    baselines = evaluate_baselines(enriched)
    signals = group_signal(enriched)
    corr = correlation_table(enriched)

    lines = [
        "# Live Weekly Predictor Feature Analysis",
        "",
        "## Working Dataset",
        "",
        f"- Source table: `{DATA_PATH.relative_to(ROOT)}`",
        f"- Rows: {fmt_int(len(df))}",
        f"- Columns before temporary history analysis: {fmt_int(len(df.columns))}",
        f"- Date range: `{df['Date'].min().date()}` to `{df['Date'].max().date()}`",
        "",
        "## Live-Safe Feature Groups",
        "",
        "- Calendar and holiday features: fully known for future dates.",
        "- Matched historical TSA features: known before forecast time, if built from prior years only.",
        "- Recent TSA actuals: known only through the latest TSA-published date.",
        "- Capacity context: use latest published annual T-100 values carried forward.",
        "- Monthly traffic context: use lagged monthly values only, because current-month values are not live.",
        "- BTS daily delay data: not currently live enough for same-week forecasting; use lagged values only where published.",
        "",
        "## Baseline Forecast Strength",
        "",
        "Baselines evaluated on 2024-01-01 through 2026-05-25 where each baseline had required inputs.",
        "",
        markdown_table(
            baselines,
            int_cols={"n", "mae", "bias"},
            pct_cols={"mape", "weekly_total_mape"},
        ),
        "",
        "## Strong Feature Signals",
        "",
        markdown_table(
            corr.head(16),
            int_cols={"n"},
        ),
        "",
        "## Day-Of-Week Shape Since 2023",
        "",
        markdown_table(signals["dow"], int_cols={"count", "mean", "std"}),
        "",
        "## Month Shape Since 2023",
        "",
        markdown_table(signals["month"], int_cols={"count", "mean", "std"}),
        "",
        "## Holiday Window Shape Since 2023",
        "",
        markdown_table(signals["holiday"], int_cols={"count", "mean", "std"}),
        "",
        "## Recommendation",
        "",
        "Use a two-layer daily forecasting model for the live weekly predictor:",
        "",
        "1. A deterministic baseline that creates a first-pass daily estimate from ISO-week/day matched history, recent TSA trend, and holiday/calendar rules.",
        "2. A tabular residual model that predicts the correction to that baseline using only live-safe features.",
        "",
        "The best first production model is gradient-boosted trees over daily rows, predicting daily passenger count or residual-to-baseline. For the current repo environment, `sklearn.ensemble.HistGradientBoostingRegressor` is a practical starting point. If we add a modeling dependency later, LightGBM or XGBoost are strong choices.",
        "",
        "For weekly prediction, forecast each missing day independently, combine with known actual days already published by TSA, and sum/average the seven daily values.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    write_report()

