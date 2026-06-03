from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tsa_project.config import REPORT_ARTIFACTS_DIR
from tsa_project.kalshi import adaptive_blend
from tsa_project.live_weekly_model import load_modeling_data, week_regime
from tsa_project.weekly_ensemble_model import (
    choose_ensemble_weight,
    make_weekly_model,
    make_weekly_supervised_dataset,
    weekly_feature_columns,
)


KALSHI_YTD_BACKTEST_PATH = REPORT_ARTIFACTS_DIR / "kalshi_tsa_ytd_daily_model_market_backtest.csv"
KALSHI_BLEND_BACKTEST_DAILY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_dashboard_ensemble_blend_backtest_daily.csv"
)
KALSHI_BLEND_BACKTEST_SUMMARY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_dashboard_ensemble_blend_backtest_summary.csv"
)


@dataclass(frozen=True)
class ModelVariant:
    name: str
    forecast_column: str
    use_kalshi_blend: bool


MODEL_VARIANTS = (
    ModelVariant("weekly_daily_ensemble_plus_kalshi", "weekly_daily_ensemble_avg", True),
    ModelVariant("direct_weekly_plus_kalshi", "direct_weekly_avg", True),
    ModelVariant("daily_model_plus_kalshi", "daily_model_avg", True),
    ModelVariant("weekly_daily_ensemble_no_kalshi", "weekly_daily_ensemble_avg", False),
)


def load_market_backtest(path: Path = KALSHI_YTD_BACKTEST_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path, parse_dates=["week_monday", "week_sunday", "as_of_date"])
    rows = rows.dropna(
        subset=[
            "week_monday",
            "as_of_date",
            "market_predicted_avg",
            "actual_weekly_avg",
            "known_days_model_semantics",
        ]
    ).copy()
    rows["known_days_model_semantics"] = rows["known_days_model_semantics"].astype(int)
    return rows.sort_values(["week_monday", "known_days_model_semantics", "as_of_date"]).reset_index(
        drop=True
    )


def _fit_direct_weekly_forecast(
    matrix: pd.DataFrame,
    feature_cols: list[str],
    monday: pd.Timestamp,
    min_train_weeks: int,
) -> float:
    train = matrix[matrix["week_monday"] < monday].copy()
    test = matrix[matrix["week_monday"] == monday].copy()
    if len(train) < min_train_weeks or test.empty:
        return np.nan
    model = make_weekly_model()
    model.fit(train[feature_cols], train["weekly_actual_avg"])
    return float(model.predict(test[feature_cols])[0])


def build_base_forecasts(
    market_rows: pd.DataFrame,
    data: pd.DataFrame,
    min_train_weeks: int = 80,
) -> pd.DataFrame:
    known_day_values = sorted(int(value) for value in market_rows["known_days_model_semantics"].unique())
    weekly_matrices = {
        known_days: make_weekly_supervised_dataset(data, known_days)
        for known_days in known_day_values
    }
    feature_cols_by_known = {
        known_days: weekly_feature_columns(matrix)
        for known_days, matrix in weekly_matrices.items()
    }

    base_rows: list[dict[str, object]] = []
    for _, row in market_rows.iterrows():
        monday = pd.Timestamp(row["week_monday"]).normalize()
        known_days = int(row["known_days_model_semantics"])
        matrix = weekly_matrices[known_days]
        direct_avg = _fit_direct_weekly_forecast(
            matrix,
            feature_cols_by_known[known_days],
            monday,
            min_train_weeks=min_train_weeks,
        )
        daily_avg = float(row["model_predicted_avg_corrected"])
        actual = float(row["actual_weekly_avg"])
        market_avg = float(row["market_predicted_avg"])
        regime = week_regime(data, monday)

        prior = pd.DataFrame(base_rows)
        if math.isfinite(direct_avg):
            weight, weight_source = choose_ensemble_weight(prior, known_days, regime)
        else:
            weight, weight_source = 1.0, "fallback:daily"
            direct_avg = np.nan
        ensemble_avg = (
            daily_avg if pd.isna(direct_avg) else (weight * daily_avg) + ((1.0 - weight) * direct_avg)
        )

        base_rows.append(
            {
                "event_ticker": row.get("event_ticker"),
                "week_monday": monday.date().isoformat(),
                "week_sunday": pd.Timestamp(row["week_sunday"]).date().isoformat(),
                "as_of_date": pd.Timestamp(row["as_of_date"]).date().isoformat(),
                "market_candle_end_et": row.get("market_candle_end_et"),
                "known_days": known_days,
                "regime": regime,
                "actual_weekly_avg": actual,
                "weekly_actual_avg": actual,
                "market_forecast": market_avg,
                "market_abs_error": abs(market_avg - actual),
                "daily_model_avg": daily_avg,
                "direct_weekly_avg": direct_avg,
                "weekly_daily_ensemble_avg": float(ensemble_avg),
                "weekly_daily_ensemble_weight_daily": float(weight),
                "weekly_daily_ensemble_weight_source": weight_source,
                "market_contracts_with_candles": row.get("market_contracts_with_candles"),
                "market_volume_contracts": row.get("market_volume_contracts"),
            }
        )
    return pd.DataFrame(base_rows)


def _variant_result(base_row: pd.Series, variant: ModelVariant) -> dict[str, object] | None:
    model_forecast = base_row.get(variant.forecast_column)
    if pd.isna(model_forecast):
        return None

    model_forecast = float(model_forecast)
    market_forecast = float(base_row["market_forecast"])
    actual = float(base_row["actual_weekly_avg"])
    if variant.use_kalshi_blend:
        blend = adaptive_blend(model_forecast, market_forecast)
        final_forecast = float(blend["adjusted_weekly_average"])
        model_weight = float(blend["model_weight"])
        market_weight = float(blend["market_weight"])
        blend_direction = blend["direction"]
        model_market_gap = blend["gap"]
    else:
        final_forecast = model_forecast
        model_weight = 1.0
        market_weight = 0.0
        blend_direction = "model_only"
        model_market_gap = model_forecast - market_forecast

    model_abs_error = abs(model_forecast - actual)
    market_abs_error = abs(market_forecast - actual)
    final_abs_error = abs(final_forecast - actual)
    return {
        "variant": variant.name,
        "uses_kalshi_blend": variant.use_kalshi_blend,
        "model_source_column": variant.forecast_column,
        "model_forecast": model_forecast,
        "market_forecast": market_forecast,
        "final_forecast": final_forecast,
        "actual_weekly_avg": actual,
        "model_error": model_forecast - actual,
        "market_error": market_forecast - actual,
        "final_error": final_forecast - actual,
        "model_abs_error": model_abs_error,
        "market_abs_error": market_abs_error,
        "final_abs_error": final_abs_error,
        "model_mape": model_abs_error / actual,
        "market_mape": market_abs_error / actual,
        "final_mape": final_abs_error / actual,
        "model_weight": model_weight,
        "market_weight": market_weight,
        "blend_direction": blend_direction,
        "model_market_gap": model_market_gap,
        "model_beats_market": model_abs_error < market_abs_error,
        "final_beats_market": final_abs_error < market_abs_error,
        "final_beats_model": final_abs_error < model_abs_error,
        "final_beats_both": final_abs_error < min(model_abs_error, market_abs_error),
        "model_within_50k": model_abs_error <= 50_000,
        "market_within_50k": market_abs_error <= 50_000,
        "final_within_50k": final_abs_error <= 50_000,
    }


def build_variant_backtest(base_forecasts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    context_cols = [
        "event_ticker",
        "week_monday",
        "week_sunday",
        "as_of_date",
        "market_candle_end_et",
        "known_days",
        "regime",
        "weekly_daily_ensemble_weight_daily",
        "weekly_daily_ensemble_weight_source",
        "market_contracts_with_candles",
        "market_volume_contracts",
    ]
    for _, base_row in base_forecasts.iterrows():
        context = {column: base_row.get(column) for column in context_cols}
        for variant in MODEL_VARIANTS:
            result = _variant_result(base_row, variant)
            if result is not None:
                rows.append({**context, **result})
    return pd.DataFrame(rows)


def summarize_variant_backtest(results: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    groupings = [
        ("overall", ["variant"]),
        ("known_days", ["variant", "known_days"]),
    ]
    for grain, columns in groupings:
        grouped = results.groupby(columns, dropna=False)
        summary = grouped.agg(
            rows=("final_abs_error", "count"),
            first_as_of_date=("as_of_date", "min"),
            last_as_of_date=("as_of_date", "max"),
            model_mae=("model_abs_error", "mean"),
            market_mae=("market_abs_error", "mean"),
            final_mae=("final_abs_error", "mean"),
            model_mape=("model_mape", "mean"),
            market_mape=("market_mape", "mean"),
            final_mape=("final_mape", "mean"),
            model_within_50k=("model_within_50k", "mean"),
            market_within_50k=("market_within_50k", "mean"),
            final_within_50k=("final_within_50k", "mean"),
            model_win_rate_vs_market=("model_beats_market", "mean"),
            final_win_rate_vs_market=("final_beats_market", "mean"),
            final_win_rate_vs_model=("final_beats_model", "mean"),
            final_beats_both_rate=("final_beats_both", "mean"),
            median_final_abs_error=("final_abs_error", "median"),
            p80_final_abs_error=("final_abs_error", lambda values: values.quantile(0.8)),
            p90_final_abs_error=("final_abs_error", lambda values: values.quantile(0.9)),
            worst_final_abs_error=("final_abs_error", "max"),
        ).reset_index()
        summary.insert(0, "grain", grain)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True, sort=False)


def run_kalshi_blend_backtest(
    market_path: Path = KALSHI_YTD_BACKTEST_PATH,
    min_train_weeks: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_modeling_data()
    market_rows = load_market_backtest(market_path)
    base = build_base_forecasts(market_rows, data, min_train_weeks=min_train_weeks)
    results = build_variant_backtest(base)
    summary = summarize_variant_backtest(results)
    return base, results, summary
