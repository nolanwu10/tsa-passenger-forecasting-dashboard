from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tsa_project.config import MODEL_ARTIFACTS_DIR, REPORT_ARTIFACTS_DIR
from tsa_project.live_weekly_model import (
    DEFAULT_CONFIGS,
    WEEKLY_REPORT_PATH,
    INTERVAL_LEVELS,
    MODEL_FEATURES,
    MODEL_PATH as DAILY_MODEL_PATH,
    LiveFeatureFactory,
    apply_weekly_calibration,
    build_calibration,
    load_modeling_data,
    lookup_calibration,
    predict_week,
    fit_residual_model,
    train_final_model,
    week_regime,
)


DIRECT_WEEKLY_MODEL_PATH = MODEL_ARTIFACTS_DIR / "live_weekly_direct_model.joblib"
ENSEMBLE_MODEL_PATH = MODEL_ARTIFACTS_DIR / "live_weekly_ensemble.joblib"
ENSEMBLE_WEEKLY_REPORT_PATH = REPORT_ARTIFACTS_DIR / "live_weekly_ensemble_backtest_weekly.csv"

DAY_PREFIXES = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
KNOWN_DAY_SCENARIOS = tuple(range(8))
ENSEMBLE_WEIGHT_GRID = tuple(float(v) / 10.0 for v in range(11))


@dataclass(frozen=True)
class WeeklyModelConfig:
    name: str
    learning_rate: float = 0.05
    max_iter: int = 120
    max_leaf_nodes: int = 15
    l2_regularization: float = 0.03
    min_samples_leaf: int = 12


WEEKLY_CONFIG = WeeklyModelConfig("weekly_hgb_abs_default")


def make_weekly_model(config: WeeklyModelConfig = WEEKLY_CONFIG) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        min_samples_leaf=config.min_samples_leaf,
        random_state=42,
    )


def _issue_date_for_week(monday: pd.Timestamp, known_days: int) -> pd.Timestamp:
    return pd.Timestamp(monday).normalize() - pd.Timedelta(days=1) + pd.Timedelta(days=int(known_days))


def _actual_week_average(data: pd.DataFrame, monday: pd.Timestamp) -> float:
    week = data[(data["Date"] >= monday) & (data["Date"] <= monday + pd.Timedelta(days=6))]
    if len(week) != 7:
        return np.nan
    return float(week["Passengers"].mean())


def make_week_feature_row(
    data: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int,
) -> dict[str, float]:
    monday = pd.Timestamp(monday).normalize()
    known_days = int(known_days)
    issue_date = _issue_date_for_week(monday, known_days)
    observed = data[data["Date"] <= issue_date].copy()
    factory = LiveFeatureFactory(observed)
    by_date = data.set_index("Date")

    row: dict[str, float] = {
        "known_days": known_days,
        "week_iso_year": int(monday.isocalendar().year),
        "week_iso_week": int(monday.isocalendar().week),
        "week_month": int(monday.month),
    }
    for offset, prefix in enumerate(DAY_PREFIXES):
        target_date = monday + pd.Timedelta(days=offset)
        features = factory.feature_row(target_date, issue_date)
        for feature in MODEL_FEATURES:
            row[f"{prefix}_{feature}"] = features.get(feature, np.nan)
        is_known = int(offset < known_days and target_date in by_date.index)
        row[f"{prefix}_is_known_actual"] = is_known
        row[f"{prefix}_known_actual_passengers"] = (
            float(by_date.loc[target_date, "Passengers"]) if is_known else np.nan
        )
    return row


def make_weekly_supervised_dataset(
    data: pd.DataFrame,
    known_days: int,
    start_monday: str = "2023-01-02",
    end_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    max_monday = data["Date"].max() - pd.Timedelta(days=6)
    if end_before is not None:
        max_monday = min(max_monday, pd.Timestamp(end_before) - pd.Timedelta(days=7))
    mondays = pd.date_range(start_monday, max_monday, freq="W-MON")
    rows = []
    for monday in mondays:
        target = _actual_week_average(data, monday)
        if pd.isna(target):
            continue
        row = make_week_feature_row(data, monday, known_days)
        row["week_monday"] = monday
        row["weekly_actual_avg"] = target
        row["regime"] = week_regime(data, monday)
        rows.append(row)
    return pd.DataFrame(rows)


def weekly_feature_columns(train: pd.DataFrame) -> list[str]:
    excluded = {"week_monday", "weekly_actual_avg", "regime"}
    return [col for col in train.columns if col not in excluded]


def fit_direct_weekly_models(data: pd.DataFrame) -> dict[int, dict[str, object]]:
    models: dict[int, dict[str, object]] = {}
    for known_days in KNOWN_DAY_SCENARIOS:
        train = make_weekly_supervised_dataset(data, known_days)
        feature_cols = weekly_feature_columns(train)
        model = make_weekly_model()
        model.fit(train[feature_cols], train["weekly_actual_avg"])
        models[known_days] = {
            "model": model,
            "features": feature_cols,
            "config": WEEKLY_CONFIG,
        }
    return models


def predict_direct_weekly_average(
    model_payload: dict[str, object],
    data: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int,
) -> float:
    row = make_week_feature_row(data, monday, known_days)
    feature_cols = model_payload["features"]
    frame = pd.DataFrame([row]).reindex(columns=feature_cols)
    return float(model_payload["model"].predict(frame)[0])


def daily_model_weekly_average(model, data: pd.DataFrame, monday: pd.Timestamp, known_days: int) -> float:
    pred = predict_week(model, data, monday, known_days=known_days)
    return float(pred["predicted_passengers"].mean())


def best_weight(prior: pd.DataFrame) -> float:
    if prior.empty:
        return 1.0
    best = 1.0
    best_error = float("inf")
    actual = prior["weekly_actual_avg"]
    for weight in ENSEMBLE_WEIGHT_GRID:
        pred = (weight * prior["daily_model_avg"]) + ((1.0 - weight) * prior["direct_weekly_avg"])
        error = (pred - actual).abs().mean()
        if error < best_error:
            best_error = float(error)
            best = weight
    return best


def choose_ensemble_weight(
    prior_results: pd.DataFrame,
    known_days: int,
    regime: str,
    min_regime_n: int = 8,
) -> tuple[float, str]:
    if prior_results.empty or "known_days" not in prior_results.columns:
        return 1.0, "default:daily"
    prior = prior_results[prior_results["known_days"] == known_days].copy()
    regime_prior = prior[prior["regime"] == regime]
    if len(regime_prior) >= min_regime_n:
        return best_weight(regime_prior), f"regime:{regime}"
    return best_weight(prior), "global"


def apply_acceptance_rule(results: pd.DataFrame) -> dict[str, object]:
    accepted: dict[str, object] = {"known_days": {}}
    for known_days, group in results.groupby("known_days"):
        daily_mae = group["daily_error"].abs().mean()
        direct_mae = group["direct_error"].abs().mean()
        ensemble_mae = group["ensemble_error"].abs().mean()
        ensemble_within = (group["ensemble_error"].abs() <= 50_000).mean()
        daily_within = (group["daily_error"].abs() <= 50_000).mean()
        direct_within = (group["direct_error"].abs() <= 50_000).mean()
        accepted_weight = "ensemble"
        if ensemble_mae > min(daily_mae, direct_mae) and ensemble_within < max(daily_within, direct_within):
            accepted_weight = "daily" if daily_mae <= direct_mae else "direct"
        accepted["known_days"][str(int(known_days))] = {
            "mode": accepted_weight,
            "daily_mae": float(daily_mae),
            "direct_mae": float(direct_mae),
            "ensemble_mae": float(ensemble_mae),
            "daily_within_50k": float(daily_within),
            "direct_within_50k": float(direct_within),
            "ensemble_within_50k": float(ensemble_within),
        }
    return accepted


def build_ensemble_calibration(results: pd.DataFrame) -> dict[str, object]:
    calibration: dict[str, object] = {"global": {}, "by_regime": {}}
    for known_days, group in results.groupby("known_days"):
        key = str(int(known_days))
        errors = group["ensemble_error"]
        abs_errors = errors.abs()
        calibration["global"][key] = {
            "n": int(len(group)),
            "intervals": {str(level): float(abs_errors.quantile(level)) for level in INTERVAL_LEVELS},
            "within_50k_rate": float((abs_errors <= 50_000).mean()),
        }
        calibration["by_regime"][key] = {}
        for regime, regime_group in group.groupby("regime"):
            regime_abs = regime_group["ensemble_error"].abs()
            calibration["by_regime"][key][regime] = {
                "n": int(len(regime_group)),
                "intervals": {
                    str(level): float(regime_abs.quantile(level))
                    for level in INTERVAL_LEVELS
                },
                "within_50k_rate": float((regime_abs <= 50_000).mean()),
            }
    return calibration


def lookup_ensemble_calibration(
    calibration: dict[str, object],
    known_days: int,
    regime: str,
    min_regime_n: int = 8,
) -> dict[str, object]:
    key = str(int(known_days))
    regime_entry = calibration.get("by_regime", {}).get(key, {}).get(regime)
    if regime_entry and regime_entry.get("n", 0) >= min_regime_n:
        return {**regime_entry, "source": f"regime:{regime}"}
    return {**calibration.get("global", {}).get(key, {}), "source": "global"}


def backtest_weekly_ensemble(
    data: pd.DataFrame,
    start_monday: str = "2025-01-06",
    known_day_scenarios: tuple[int, ...] = KNOWN_DAY_SCENARIOS,
) -> pd.DataFrame:
    max_monday = data["Date"].max() - pd.Timedelta(days=6)
    mondays = pd.date_range(start_monday, max_monday, freq="W-MON")
    if WEEKLY_REPORT_PATH.exists():
        daily_backtest = pd.read_csv(WEEKLY_REPORT_PATH)
        if "config" in daily_backtest.columns:
            daily_backtest = daily_backtest[daily_backtest["config"] == "hgb_abs_default"].copy()
    else:
        daily_backtest = pd.DataFrame()
    weekly_matrices = {
        known_days: make_weekly_supervised_dataset(data, known_days)
        for known_days in known_day_scenarios
    }
    feature_cols_by_known = {
        known_days: weekly_feature_columns(matrix)
        for known_days, matrix in weekly_matrices.items()
    }
    rows = []
    for monday in mondays:
        if monday + pd.Timedelta(days=6) > data["Date"].max():
            continue
        for known_days in known_day_scenarios:
            matrix = weekly_matrices[known_days]
            train = matrix[matrix["week_monday"] < monday].copy()
            test = matrix[matrix["week_monday"] == monday].copy()
            if len(train) < 80 or test.empty:
                continue
            feature_cols = feature_cols_by_known[known_days]
            model = make_weekly_model()
            model.fit(train[feature_cols], train["weekly_actual_avg"])
            direct_avg = float(model.predict(test[feature_cols])[0])
            actual = _actual_week_average(data, monday)
            regime = week_regime(data, monday)
            daily_match = daily_backtest[
                (daily_backtest.get("week_monday", pd.Series(dtype=str)) == monday.date().isoformat())
                & (daily_backtest.get("known_days", pd.Series(dtype=int)).astype(int) == known_days)
            ]
            if daily_match.empty:
                daily_avg = np.nan
            else:
                daily_avg = float(daily_match.iloc[0]["weekly_pred_avg"])
            prior = pd.DataFrame(rows)
            weight, weight_source = choose_ensemble_weight(prior, known_days, regime)
            if pd.isna(daily_avg):
                weight = 0.0
                weight_source = "fallback:direct"
            ensemble_avg = (weight * daily_avg) + ((1.0 - weight) * direct_avg)
            rows.append(
                {
                    "week_monday": monday.date().isoformat(),
                    "known_days": known_days,
                    "regime": regime,
                    "daily_model_avg": daily_avg,
                    "direct_weekly_avg": direct_avg,
                    "ensemble_avg": ensemble_avg,
                    "ensemble_weight_daily": weight,
                    "ensemble_weight_source": weight_source,
                    "weekly_actual_avg": actual,
                    "daily_error": daily_avg - actual,
                    "direct_error": direct_avg - actual,
                    "ensemble_error": ensemble_avg - actual,
                    "daily_abs_error": abs(daily_avg - actual),
                    "direct_abs_error": abs(direct_avg - actual),
                    "ensemble_abs_error": abs(ensemble_avg - actual),
                    "daily_mape": abs(daily_avg - actual) / actual,
                    "direct_mape": abs(direct_avg - actual) / actual,
                    "ensemble_mape": abs(ensemble_avg - actual) / actual,
                }
            )
    return pd.DataFrame(rows)


def train_final_weekly_ensemble(data: pd.DataFrame, backtest_results: pd.DataFrame) -> dict[str, object]:
    direct_models = fit_direct_weekly_models(data)
    calibration = build_ensemble_calibration(backtest_results)
    acceptance = apply_acceptance_rule(backtest_results)
    DIRECT_WEEKLY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": direct_models, "known_days": KNOWN_DAY_SCENARIOS}, DIRECT_WEEKLY_MODEL_PATH)
    payload = {
        "direct_model_path": str(DIRECT_WEEKLY_MODEL_PATH),
        "daily_model_path": str(DAILY_MODEL_PATH),
        "calibration": calibration,
        "acceptance": acceptance,
        "weight_grid": ENSEMBLE_WEIGHT_GRID,
    }
    joblib.dump(payload, ENSEMBLE_MODEL_PATH)
    return {"direct_path": str(DIRECT_WEEKLY_MODEL_PATH), "ensemble_path": str(ENSEMBLE_MODEL_PATH)}


def predict_weekly_ensemble(
    data: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int,
) -> dict[str, object]:
    if not ENSEMBLE_MODEL_PATH.exists() or not DIRECT_WEEKLY_MODEL_PATH.exists():
        raise FileNotFoundError("Run scripts/backtest_weekly_ensemble_model.py first.")
    if not DAILY_MODEL_PATH.exists():
        raise FileNotFoundError("Daily model artifact is missing; run scripts/backtest_live_weekly_model.py first.")

    daily_payload = joblib.load(DAILY_MODEL_PATH)
    direct_payload = joblib.load(DIRECT_WEEKLY_MODEL_PATH)
    ensemble_payload = joblib.load(ENSEMBLE_MODEL_PATH)

    known_days = int(known_days)
    if known_days not in direct_payload["models"]:
        available = sorted(direct_payload["models"])
        known_days = max(day for day in available if day <= known_days)

    regime = week_regime(data, monday)
    daily_avg = daily_model_weekly_average(daily_payload["model"], data, monday, known_days)
    direct_avg = predict_direct_weekly_average(direct_payload["models"][known_days], data, monday, known_days)
    acceptance = ensemble_payload["acceptance"]["known_days"].get(str(known_days), {"mode": "ensemble"})
    mode = acceptance.get("mode", "ensemble")
    if mode == "daily":
        weight = 1.0
        weight_source = "acceptance:daily"
    elif mode == "direct":
        weight = 0.0
        weight_source = "acceptance:direct"
    else:
        historical = pd.read_csv(ENSEMBLE_WEEKLY_REPORT_PATH) if ENSEMBLE_WEEKLY_REPORT_PATH.exists() else pd.DataFrame()
        weight, weight_source = choose_ensemble_weight(historical, known_days, regime)

    ensemble_avg = (weight * daily_avg) + ((1.0 - weight) * direct_avg)
    calibration_entry = lookup_ensemble_calibration(
        ensemble_payload.get("calibration", {}),
        known_days,
        regime,
    )
    return {
        "known_days": known_days,
        "regime": regime,
        "daily_model_avg": daily_avg,
        "direct_weekly_avg": direct_avg,
        "ensemble_avg": ensemble_avg,
        "ensemble_weight_daily": weight,
        "ensemble_weight_source": weight_source,
        "calibration": calibration_entry,
        "acceptance": acceptance,
    }
