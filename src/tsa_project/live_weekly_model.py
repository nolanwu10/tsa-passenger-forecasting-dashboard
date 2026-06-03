from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tsa_project.config import DAILY_TRANSPORT_FEATURES_PATH, MODEL_ARTIFACTS_DIR, RAW_TSA_PATH, REPORT_ARTIFACTS_DIR
from tsa_project.features import build_calendar_holiday_features


MODEL_PATH = MODEL_ARTIFACTS_DIR / "live_weekly_residual_model.joblib"
FEATURE_REPORT_PATH = REPORT_ARTIFACTS_DIR / "live_weekly_backtest_results.csv"
WEEKLY_REPORT_PATH = REPORT_ARTIFACTS_DIR / "live_weekly_backtest_weekly.csv"


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

CALENDAR_FEATURES = [
    "quarter",
    "month",
    "day",
    "day_of_year",
    "day_of_week",
    "iso_week",
    "iso_day",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_federal_holiday",
    "is_high_impact_holiday",
    "is_holiday_window",
    "is_new_years_day",
    "is_july_4",
    "is_christmas_eve",
    "is_christmas_day",
    "is_nye",
    "days_from_thanksgiving",
    "is_thanksgiving",
    "is_thanksgiving_week",
    "is_thanksgiving_outbound",
    "is_thanksgiving_return",
    "days_from_easter",
    "is_easter_window",
    "days_from_mlk_day",
    "is_mlk_window",
    "days_from_presidents_day",
    "is_presidents_day_window",
    "days_from_memorial_day",
    "is_memorial_day_window",
    "days_from_labor_day",
    "is_labor_day_window",
    "days_from_nearest_july_4",
    "is_july_4_window",
    "days_from_nearest_christmas",
    "is_christmas_travel_window",
    "days_from_nearest_new_year",
    "is_new_year_travel_window",
    "is_spring_break_proxy",
    "days_until_christmas",
    "days_until_new_year",
    "days_until_july_4",
]

HISTORY_FEATURES = [
    "matched_1ya",
    "matched_2ya",
    "matched_3ya",
    "matched_4ya",
    "baseline_2019",
    "matched_mean_1_3ya",
    "matched_median_1_4ya",
    "holiday_anchor_1ya",
    "lag_7_target",
    "latest_known_passengers",
    "known_roll_7",
    "known_roll_14",
    "known_roll_28",
    "known_roll_56",
    "growth_7_vs_1ya",
    "growth_14_vs_1ya",
    "growth_28_vs_1ya",
    "target_days_ahead",
    "target_week_position",
    "known_days_in_target_week",
    "known_target_week_actual_avg",
    "known_target_week_expected_avg",
    "known_target_week_vs_expected",
    "known_target_week_error_avg",
    "known_target_week_last_actual",
    "known_target_week_trend",
    "deterministic_baseline",
]

EXTERNAL_FEATURES = [
    "bts_mts_domestic_air_traffic_nsa_lag2_months",
    "bts_mts_total_air_traffic_nsa_lag2_months",
    "bts_mts_marketing_on_time_pct_lag2_months",
    "bts_t100_est_daily_domestic_seats_safe",
    "bts_t100_domestic_load_factor_safe",
]

DELAY_FEATURES = [
    "bts_cancel_rate_lag1",
    "bts_cancel_rate_roll7_lag1",
    "bts_cancel_rate_roll28_lag1",
    "bts_avg_dep_delay_minutes_lag1",
    "bts_avg_dep_delay_minutes_roll7_lag1",
    "bts_dep_delay_15_rate_lag1",
    "bts_dep_delay_15_rate_roll7_lag1",
]

MODEL_FEATURES = CALENDAR_FEATURES + HISTORY_FEATURES + EXTERNAL_FEATURES + DELAY_FEATURES


@dataclass(frozen=True)
class ModelConfig:
    name: str
    learning_rate: float
    max_iter: int
    max_leaf_nodes: int
    l2_regularization: float
    min_samples_leaf: int = 20


DEFAULT_CONFIGS = [
    ModelConfig("hgb_abs_default", 0.05, 140, 15, 0.01, 20),
    ModelConfig("hgb_abs_smoother", 0.04, 180, 10, 0.05, 30),
    ModelConfig("hgb_abs_deeper", 0.05, 160, 31, 0.03, 20),
    ModelConfig("hgb_abs_fast", 0.08, 100, 15, 0.02, 25),
]

INTERVAL_LEVELS = (0.5, 0.68, 0.8, 0.9, 0.95)


def load_modeling_data(path: Path = DAILY_TRANSPORT_FEATURES_PATH) -> pd.DataFrame:
    if not path.exists() and path == DAILY_TRANSPORT_FEATURES_PATH:
        from tsa_project.features import write_calendar_holiday_features
        from tsa_project.transport_features import write_transport_features

        raw_tsa = pd.read_csv(RAW_TSA_PATH)
        write_calendar_holiday_features(raw_tsa)
        write_transport_features()
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


def make_model(config: ModelConfig) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="absolute_error",
        learning_rate=config.learning_rate,
        max_iter=config.max_iter,
        max_leaf_nodes=config.max_leaf_nodes,
        l2_regularization=config.l2_regularization,
        min_samples_leaf=config.min_samples_leaf,
        random_state=42,
    )


def _safe_mean(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return np.nan
    return float(series.mean())


def _safe_median(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    if series.empty:
        return np.nan
    return float(series.median())


class LiveFeatureFactory:
    def __init__(self, data: pd.DataFrame):
        self.data = data.sort_values("Date").copy()
        self.data["Date"] = pd.to_datetime(self.data["Date"])
        self.by_date = self.data.set_index("Date")
        self.by_iso = self.data.set_index(["iso_year", "iso_week", "day_of_week"])["Passengers"]
        self.t100_by_year = self._build_t100_lookup()

    def _build_t100_lookup(self) -> pd.DataFrame:
        cols = [
            "year",
            "bts_t100_est_daily_domestic_seats",
            "bts_t100_domestic_load_factor",
        ]
        available = [col for col in cols if col in self.data.columns]
        t100 = self.data[available].drop_duplicates(subset=["year"]).sort_values("year")
        return t100

    def feature_row(self, target_date: pd.Timestamp, issue_date: pd.Timestamp) -> dict[str, float]:
        target_date = pd.Timestamp(target_date).normalize()
        issue_date = pd.Timestamp(issue_date).normalize()
        target_calendar = self._calendar_for_date(target_date)
        history = self.data[self.data["Date"] <= issue_date].copy()
        latest_known = history["Passengers"].iloc[-1] if not history.empty else np.nan

        row: dict[str, float] = {
            feature: float(target_calendar.get(feature, np.nan))
            for feature in CALENDAR_FEATURES
            if feature != "holiday_name"
        }
        iso_year = int(target_calendar["iso_year"])
        iso_week = int(target_calendar["iso_week"])
        dow = int(target_calendar["day_of_week"])
        matched = {}
        for years_back in range(1, 5):
            matched[years_back] = self._iso_match(iso_year - years_back, iso_week, dow)
            row[f"matched_{years_back}ya"] = matched[years_back]
        row["baseline_2019"] = self._iso_match(2019, iso_week, dow)
        row["matched_mean_1_3ya"] = _safe_mean([matched[1], matched[2], matched[3]])
        row["matched_median_1_4ya"] = _safe_median([matched[1], matched[2], matched[3], matched[4]])
        row["holiday_anchor_1ya"] = self._holiday_anchor(target_date)
        row["lag_7_target"] = self._actual_on_or_before(target_date - pd.Timedelta(days=7), issue_date)
        row["latest_known_passengers"] = latest_known
        row["known_roll_7"] = self._known_roll(history, 7)
        row["known_roll_14"] = self._known_roll(history, 14)
        row["known_roll_28"] = self._known_roll(history, 28)
        row["known_roll_56"] = self._known_roll(history, 56)
        row["growth_7_vs_1ya"] = self._recent_growth(history, 7)
        row["growth_14_vs_1ya"] = self._recent_growth(history, 14)
        row["growth_28_vs_1ya"] = self._recent_growth(history, 28)
        row["target_days_ahead"] = int((target_date - issue_date).days)
        row["target_week_position"] = int(target_date.dayofweek)
        monday = target_date - pd.Timedelta(days=target_date.weekday())
        row["known_days_in_target_week"] = int(
            max(0, min(7, (issue_date - monday).days + 1))
        )
        row.update(self._target_week_known_context(monday, issue_date, row))

        row.update(self._external_features(target_date, issue_date))
        baseline = self._deterministic_baseline(row)
        row["deterministic_baseline"] = baseline
        return {feature: row.get(feature, np.nan) for feature in MODEL_FEATURES}

    def _calendar_for_date(self, target_date: pd.Timestamp) -> pd.Series:
        if target_date in self.by_date.index:
            return self.by_date.loc[target_date]

        empty = pd.DataFrame({"Date": [target_date], "Passengers": [0]})
        generated = build_calendar_holiday_features(empty).iloc[0]
        return generated

    def _iso_match(self, iso_year: int, iso_week: int, dow: int) -> float:
        return float(self.by_iso.get((iso_year, iso_week, dow), np.nan))

    def _holiday_anchor(self, target_date: pd.Timestamp) -> float:
        target = self._calendar_for_date(target_date)
        prior_year = target_date.year - 1
        if int(target.get("is_thanksgiving_week", 0)) == 1:
            offset = int(target["days_from_thanksgiving"])
            candidates = self.data[
                (self.data["year"] == prior_year)
                & (self.data["days_from_thanksgiving"] == offset)
            ]
            if not candidates.empty:
                return float(candidates["Passengers"].iloc[0])

        if int(target["month"]) == 12 or (int(target["month"]) == 1 and int(target["day"]) <= 3):
            candidate_date = target_date - pd.DateOffset(years=1)
            if candidate_date in self.by_date.index:
                return float(self.by_date.loc[candidate_date, "Passengers"])
        return np.nan

    def _actual_on_or_before(self, date_value: pd.Timestamp, issue_date: pd.Timestamp) -> float:
        if date_value <= issue_date and date_value in self.by_date.index:
            return float(self.by_date.loc[date_value, "Passengers"])
        return np.nan

    def _known_roll(self, history: pd.DataFrame, window: int) -> float:
        if history.empty:
            return np.nan
        return float(history["Passengers"].tail(window).mean())

    def _recent_growth(self, history: pd.DataFrame, window: int) -> float:
        if len(history) < max(7, window // 2):
            return np.nan
        recent = history.tail(window).copy()
        matched = []
        for _, record in recent.iterrows():
            matched.append(
                self._iso_match(
                    int(record["iso_year"]) - 1,
                    int(record["iso_week"]),
                    int(record["day_of_week"]),
                )
            )
        denominator = pd.Series(matched).dropna().sum()
        numerator = recent["Passengers"].sum()
        if denominator <= 0:
            return np.nan
        return float(numerator / denominator)

    def _target_week_known_context(
        self,
        monday: pd.Timestamp,
        issue_date: pd.Timestamp,
        target_row: dict[str, float],
    ) -> dict[str, float]:
        known = self.data[
            (self.data["Date"] >= monday)
            & (self.data["Date"] <= issue_date)
        ].copy()
        if known.empty:
            return {
                "known_target_week_actual_avg": np.nan,
                "known_target_week_expected_avg": np.nan,
                "known_target_week_vs_expected": np.nan,
                "known_target_week_error_avg": np.nan,
                "known_target_week_last_actual": np.nan,
                "known_target_week_trend": np.nan,
            }

        growth = target_row.get("growth_28_vs_1ya")
        if pd.isna(growth):
            growth = target_row.get("growth_14_vs_1ya")
        if pd.isna(growth):
            growth = 1.0

        expected = []
        for _, record in known.iterrows():
            matched = self._iso_match(
                int(record["iso_year"]) - 1,
                int(record["iso_week"]),
                int(record["day_of_week"]),
            )
            if pd.notna(matched):
                expected.append(matched * growth)
        expected_avg = _safe_mean(expected)
        actual_avg = float(known["Passengers"].mean())
        ratio = actual_avg / expected_avg if pd.notna(expected_avg) and expected_avg > 0 else np.nan
        trend = (
            float(known["Passengers"].iloc[-1] - known["Passengers"].iloc[0])
            if len(known) >= 2
            else 0.0
        )
        return {
            "known_target_week_actual_avg": actual_avg,
            "known_target_week_expected_avg": expected_avg,
            "known_target_week_vs_expected": ratio,
            "known_target_week_error_avg": actual_avg - expected_avg if pd.notna(expected_avg) else np.nan,
            "known_target_week_last_actual": float(known["Passengers"].iloc[-1]),
            "known_target_week_trend": trend,
        }

    def _external_features(self, target_date: pd.Timestamp, issue_date: pd.Timestamp) -> dict[str, float]:
        values = {}
        if target_date in self.by_date.index:
            target = self.by_date.loc[target_date]
            for col in [
                "bts_mts_domestic_air_traffic_nsa_lag2_months",
                "bts_mts_total_air_traffic_nsa_lag2_months",
                "bts_mts_marketing_on_time_pct_lag2_months",
            ]:
                values[col] = float(target[col]) if col in target and pd.notna(target[col]) else np.nan
        else:
            values.update(
                {
                    "bts_mts_domestic_air_traffic_nsa_lag2_months": np.nan,
                    "bts_mts_total_air_traffic_nsa_lag2_months": np.nan,
                    "bts_mts_marketing_on_time_pct_lag2_months": np.nan,
                }
            )

        safe_t100 = self.t100_by_year[self.t100_by_year["year"] <= target_date.year - 1]
        if safe_t100.empty:
            safe_t100 = self.t100_by_year[self.t100_by_year["year"] <= target_date.year]
        if safe_t100.empty:
            values["bts_t100_est_daily_domestic_seats_safe"] = np.nan
            values["bts_t100_domestic_load_factor_safe"] = np.nan
        else:
            latest = safe_t100.iloc[-1]
            values["bts_t100_est_daily_domestic_seats_safe"] = latest.get(
                "bts_t100_est_daily_domestic_seats",
                np.nan,
            )
            values["bts_t100_domestic_load_factor_safe"] = latest.get(
                "bts_t100_domestic_load_factor",
                np.nan,
            )

        if issue_date in self.by_date.index:
            issue = self.by_date.loc[issue_date]
            for col in DELAY_FEATURES:
                values[col] = float(issue[col]) if col in issue and pd.notna(issue[col]) else np.nan
        else:
            for col in DELAY_FEATURES:
                values[col] = np.nan
        return values

    def _deterministic_baseline(self, row: dict[str, float]) -> float:
        growth = row.get("growth_28_vs_1ya")
        if pd.isna(growth):
            growth = row.get("growth_14_vs_1ya")
        if pd.isna(growth):
            growth = 1.0

        components = []
        if pd.notna(row.get("matched_1ya")):
            components.append((0.65, row["matched_1ya"] * growth))
        if pd.notna(row.get("lag_7_target")):
            components.append((0.20, row["lag_7_target"]))
        if pd.notna(row.get("baseline_2019")):
            components.append((0.15, row["baseline_2019"] * growth))
        if pd.notna(row.get("holiday_anchor_1ya")):
            components.append((0.25, row["holiday_anchor_1ya"] * growth))

        valid = [(weight, value) for weight, value in components if pd.notna(value)]
        if not valid:
            return np.nan
        total_weight = sum(weight for weight, _ in valid)
        base = float(sum(weight * value for weight, value in valid) / total_weight)
        week_ratio = row.get("known_target_week_vs_expected")
        if pd.notna(week_ratio) and row.get("known_days_in_target_week", 0) > 0:
            clipped_ratio = float(np.clip(week_ratio, 0.88, 1.12))
            anchor_weight = min(0.55, 0.18 * row.get("known_days_in_target_week", 0))
            base = ((1.0 - anchor_weight) * base) + (anchor_weight * base * clipped_ratio)
        return base


def make_supervised_dataset(
    data: pd.DataFrame,
    start_date: str = "2023-01-01",
    end_before: pd.Timestamp | None = None,
) -> pd.DataFrame:
    factory = LiveFeatureFactory(data)
    rows = []
    work = data[data["Date"] >= pd.Timestamp(start_date)].copy()
    if end_before is not None:
        work = work[work["Date"] < pd.Timestamp(end_before)]
    for _, record in work.iterrows():
        target_date = record["Date"]
        issue_date = target_date - pd.Timedelta(days=1)
        features = factory.feature_row(target_date, issue_date)
        baseline = features["deterministic_baseline"]
        if pd.isna(baseline):
            continue
        actual = float(record["Passengers"])
        rows.append(
            {
                "Date": target_date,
                "Passengers": actual,
                "baseline": baseline,
                "residual": actual - baseline,
                **features,
            }
        )
    return pd.DataFrame(rows)


def fit_residual_model(data: pd.DataFrame, config: ModelConfig) -> HistGradientBoostingRegressor:
    train = make_supervised_dataset(data)
    model = make_model(config)
    model.fit(train[MODEL_FEATURES], train["residual"])
    return model


def predict_day(
    model: HistGradientBoostingRegressor,
    factory: LiveFeatureFactory,
    target_date: pd.Timestamp,
    issue_date: pd.Timestamp,
) -> dict[str, float]:
    features = factory.feature_row(target_date, issue_date)
    feature_df = pd.DataFrame([features], columns=MODEL_FEATURES)
    residual = float(model.predict(feature_df)[0])
    baseline = float(features["deterministic_baseline"])
    prediction = max(0.0, baseline + residual)
    return {
        "Date": pd.Timestamp(target_date).date().isoformat(),
        "day_name": DAY_NAMES[pd.Timestamp(target_date).weekday()],
        "baseline": baseline,
        "residual_prediction": residual,
        "predicted_passengers": prediction,
    }


def predict_week(
    model: HistGradientBoostingRegressor,
    data: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int = 0,
) -> pd.DataFrame:
    monday = pd.Timestamp(monday).normalize()
    known_days = int(max(0, min(7, known_days)))
    nominal_issue_date = monday - pd.Timedelta(days=1) + pd.Timedelta(days=known_days)
    latest_available = pd.Timestamp(data["Date"].max()).normalize()
    issue_date = min(nominal_issue_date, latest_available)
    factory = LiveFeatureFactory(data)
    rows = []
    by_date = data.set_index("Date")
    for offset in range(7):
        target_date = monday + pd.Timedelta(days=offset)
        if offset < known_days and target_date in by_date.index:
            actual = float(by_date.loc[target_date, "Passengers"])
            rows.append(
                {
                    "Date": target_date.date().isoformat(),
                    "day_name": DAY_NAMES[target_date.weekday()],
                    "type": "actual_known",
                    "baseline": np.nan,
                    "residual_prediction": np.nan,
                    "predicted_passengers": actual,
                }
            )
        else:
            prediction = predict_day(model, factory, target_date, issue_date)
            prediction["type"] = "predicted"
            rows.append(prediction)
    return pd.DataFrame(rows)


def week_regime(data: pd.DataFrame, monday: pd.Timestamp) -> str:
    monday = pd.Timestamp(monday).normalize()
    week = data[(data["Date"] >= monday) & (data["Date"] <= monday + pd.Timedelta(days=6))]
    if week.empty:
        calendar = build_calendar_holiday_features(
            pd.DataFrame(
                {
                    "Date": [monday + pd.Timedelta(days=i) for i in range(7)],
                    "Passengers": [0] * 7,
                }
            )
        )
        week = calendar

    checks = [
        ("thanksgiving", "is_thanksgiving_week"),
        ("christmas_new_year", "is_christmas_travel_window"),
        ("christmas_new_year", "is_new_year_travel_window"),
        ("july_4", "is_july_4_window"),
        ("memorial_labor", "is_memorial_day_window"),
        ("memorial_labor", "is_labor_day_window"),
        ("easter_spring_break", "is_easter_window"),
    ]
    for label, column in checks:
        if column in week and week[column].fillna(0).astype(int).sum() > 0:
            return label
    if "is_spring_break_proxy" in week and week["is_spring_break_proxy"].fillna(0).astype(int).sum() >= 4:
        return "easter_spring_break"
    if int(week["month"].mode().iloc[0]) == 1:
        return "january_winter"
    if int(week["month"].mode().iloc[0]) in (6, 7, 8):
        return "summer_peak"
    return "normal"


def backtest_week(
    data: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int,
    config: ModelConfig,
    start_date: str = "2023-01-01",
) -> tuple[dict[str, float], pd.DataFrame]:
    monday = pd.Timestamp(monday).normalize()
    train_data = data[data["Date"] < monday].copy()
    train = make_supervised_dataset(train_data, start_date=start_date)
    if len(train) < 120:
        raise ValueError(f"Not enough train rows before {monday.date()}")

    model = make_model(config)
    model.fit(train[MODEL_FEATURES], train["residual"])
    pred_df = predict_week(model, train_data, monday, known_days=known_days)
    actual = data[
        (data["Date"] >= monday)
        & (data["Date"] <= monday + pd.Timedelta(days=6))
    ][["Date", "Passengers"]].copy()
    actual["Date"] = actual["Date"].dt.date.astype(str)
    result = pred_df.merge(actual, on="Date", how="left")
    result["error"] = result["predicted_passengers"] - result["Passengers"]
    result["abs_error"] = result["error"].abs()
    result["ape"] = result["abs_error"] / result["Passengers"]
    weekly_actual = result["Passengers"].sum()
    weekly_pred = result["predicted_passengers"].sum()
    summary = {
        "week_monday": monday.date().isoformat(),
        "known_days": known_days,
        "config": config.name,
        "daily_mae": float(result["abs_error"].mean()),
        "daily_mape": float(result["ape"].mean()),
        "weekly_actual_avg": float(weekly_actual / len(result)),
        "weekly_pred_avg": float(weekly_pred / len(result)),
        "weekly_avg_error": float((weekly_pred - weekly_actual) / len(result)),
        "weekly_total_mape": float(abs(weekly_pred - weekly_actual) / weekly_actual),
    }
    return summary, result


def summarize_week_prediction(
    data: pd.DataFrame,
    pred_df: pd.DataFrame,
    monday: pd.Timestamp,
    known_days: int,
    config_name: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    monday = pd.Timestamp(monday).normalize()
    actual = data[
        (data["Date"] >= monday)
        & (data["Date"] <= monday + pd.Timedelta(days=6))
    ][["Date", "Passengers"]].copy()
    actual["Date"] = actual["Date"].dt.date.astype(str)
    result = pred_df.merge(actual, on="Date", how="left")
    result["error"] = result["predicted_passengers"] - result["Passengers"]
    result["abs_error"] = result["error"].abs()
    result["ape"] = result["abs_error"] / result["Passengers"]
    weekly_actual = result["Passengers"].sum()
    weekly_pred = result["predicted_passengers"].sum()
    summary = {
        "week_monday": monday.date().isoformat(),
        "known_days": known_days,
        "config": config_name,
        "regime": week_regime(data, monday),
        "daily_mae": float(result["abs_error"].mean()),
        "daily_mape": float(result["ape"].mean()),
        "weekly_actual_avg": float(weekly_actual / len(result)),
        "weekly_pred_avg": float(weekly_pred / len(result)),
        "weekly_avg_error": float((weekly_pred - weekly_actual) / len(result)),
        "weekly_total_mape": float(abs(weekly_pred - weekly_actual) / weekly_actual),
    }
    return summary, result


def backtest_configs(
    data: pd.DataFrame,
    configs: list[ModelConfig] = DEFAULT_CONFIGS,
    start_monday: str = "2025-01-06",
    end_monday: str | None = None,
    known_day_scenarios: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if end_monday is None:
        end_monday = str((data["Date"].max() - pd.Timedelta(days=6)).date())
    mondays = pd.date_range(start_monday, end_monday, freq="W-MON")
    supervised = make_supervised_dataset(data)
    summaries = []
    daily_results = []
    for config in configs:
        for monday in mondays:
            if monday + pd.Timedelta(days=6) > data["Date"].max():
                continue
            train = supervised[supervised["Date"] < monday]
            if len(train) < 120:
                continue
            train_data = data[data["Date"] < monday].copy()
            model = make_model(config)
            model.fit(train[MODEL_FEATURES], train["residual"])
            for known_days in known_day_scenarios:
                issue_date = monday - pd.Timedelta(days=1) + pd.Timedelta(days=known_days)
                observed_data = data[data["Date"] <= issue_date].copy()
                pred = predict_week(model, observed_data, monday, known_days=known_days)
                summary, daily = summarize_week_prediction(
                    data,
                    pred,
                    monday,
                    known_days,
                    config.name,
                )
                summaries.append(summary)
                daily["week_monday"] = summary["week_monday"]
                daily["known_days"] = known_days
                daily["config"] = config.name
                daily_results.append(daily)
    return pd.DataFrame(summaries), pd.concat(daily_results, ignore_index=True)


def choose_best_config(summary: pd.DataFrame) -> str:
    preweek = summary[summary["known_days"] == 0]
    scores = preweek.groupby("config")["weekly_total_mape"].mean().sort_values()
    return str(scores.index[0])


def apply_expanding_weekly_correction(summary: pd.DataFrame) -> pd.DataFrame:
    corrected_groups = []
    for _, group in summary.sort_values("week_monday").groupby(["config", "known_days"], dropna=False):
        group = group.copy().sort_values("week_monday")
        prior_bias = group["weekly_avg_error"].expanding().median().shift(1).fillna(0.0)
        group["weekly_correction"] = prior_bias
        group["corrected_weekly_avg_error"] = group["weekly_avg_error"] - group["weekly_correction"]
        group["corrected_weekly_total_mape"] = (
            group["corrected_weekly_avg_error"].abs() / group["weekly_actual_avg"]
        )
        corrected_groups.append(group)
    return pd.concat(corrected_groups, ignore_index=True)


def build_calibration(summary: pd.DataFrame, config_name: str) -> dict[str, object]:
    selected = summary[summary["config"] == config_name].copy()
    calibration: dict[str, object] = {"config": config_name, "global": {}, "by_regime": {}}
    for known_days, group in selected.groupby("known_days"):
        key = str(int(known_days))
        errors = group["weekly_avg_error"]
        raw_mape = float(group["weekly_total_mape"].mean())
        corrected_mape = float(group.get("corrected_weekly_total_mape", group["weekly_total_mape"]).mean())
        apply_point_correction = corrected_mape < raw_mape
        median_error = float(errors.median()) if apply_point_correction else 0.0
        adjusted_abs = (errors - median_error).abs()
        calibration["global"][key] = {
            "n": int(len(group)),
            "median_error": median_error,
            "apply_point_correction": apply_point_correction,
            "raw_weekly_total_mape": raw_mape,
            "corrected_weekly_total_mape": corrected_mape,
            "intervals": {str(level): float(adjusted_abs.quantile(level)) for level in INTERVAL_LEVELS},
            "within_50k_rate": float((adjusted_abs <= 50_000).mean()),
        }
        calibration["by_regime"][key] = {}
        for regime, regime_group in group.groupby("regime"):
            regime_errors = regime_group["weekly_avg_error"]
            regime_raw_mape = float(regime_group["weekly_total_mape"].mean())
            regime_corrected_mape = float(
                regime_group.get("corrected_weekly_total_mape", regime_group["weekly_total_mape"]).mean()
            )
            regime_apply_correction = regime_corrected_mape < regime_raw_mape
            regime_median = float(regime_errors.median()) if regime_apply_correction else 0.0
            regime_adjusted_abs = (regime_errors - regime_median).abs()
            calibration["by_regime"][key][regime] = {
                "n": int(len(regime_group)),
                "median_error": regime_median,
                "apply_point_correction": regime_apply_correction,
                "raw_weekly_total_mape": regime_raw_mape,
                "corrected_weekly_total_mape": regime_corrected_mape,
                "intervals": {
                    str(level): float(regime_adjusted_abs.quantile(level))
                    for level in INTERVAL_LEVELS
                },
                "within_50k_rate": float((regime_adjusted_abs <= 50_000).mean()),
            }
    return calibration


def lookup_calibration(
    calibration: dict[str, object] | None,
    known_days: int,
    regime: str,
    min_regime_n: int = 8,
) -> dict[str, object]:
    if not calibration:
        return {"median_error": 0.0, "intervals": {}, "within_50k_rate": np.nan, "source": "none", "n": 0}
    key = str(int(known_days))
    by_regime = calibration.get("by_regime", {}).get(key, {})
    regime_entry = by_regime.get(regime)
    if regime_entry and regime_entry.get("n", 0) >= min_regime_n:
        return {**regime_entry, "source": f"regime:{regime}"}
    global_entry = calibration.get("global", {}).get(key, {})
    return {**global_entry, "source": "global"}


def apply_weekly_calibration(
    pred_df: pd.DataFrame,
    calibration_entry: dict[str, object],
) -> tuple[pd.DataFrame, float]:
    result = pred_df.copy()
    correction = float(calibration_entry.get("median_error", 0.0) or 0.0)
    predicted_mask = result["type"] == "predicted"
    predicted_days = int(predicted_mask.sum())
    if predicted_days > 0:
        per_predicted_day = correction * 7.0 / predicted_days
        result.loc[predicted_mask, "predicted_passengers"] = (
            result.loc[predicted_mask, "predicted_passengers"] - per_predicted_day
        ).clip(lower=0)
        result["weekly_correction_applied_per_predicted_day"] = 0.0
        result.loc[predicted_mask, "weekly_correction_applied_per_predicted_day"] = per_predicted_day
    else:
        result["weekly_correction_applied_per_predicted_day"] = 0.0
    return result, correction


def train_final_model(
    config_name: str | None = None,
    calibration: dict[str, object] | None = None,
) -> dict[str, object]:
    data = load_modeling_data()
    configs = {config.name: config for config in DEFAULT_CONFIGS}
    config = configs[config_name] if config_name else DEFAULT_CONFIGS[0]
    model = fit_residual_model(data, config)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "config": config,
            "features": MODEL_FEATURES,
            "calibration": calibration,
        },
        MODEL_PATH,
    )
    return {"path": str(MODEL_PATH), "config": config.name, "features": MODEL_FEATURES}
