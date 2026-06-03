from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from tsa_project.config import DAILY_CALENDAR_FEATURES_PATH
from tsa_project.datasets import normalize_tsa_raw


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    days_until_weekday = (weekday - current.weekday()) % 7
    return current + timedelta(days=days_until_weekday + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    if month == 12:
        current = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        current = date(year, month + 1, 1) - timedelta(days=1)
    return current - timedelta(days=(current.weekday() - weekday) % 7)


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    actual = date(year, month, day)
    if actual.weekday() == 5:
        return actual - timedelta(days=1)
    if actual.weekday() == 6:
        return actual + timedelta(days=1)
    return actual


def federal_holidays_for_year(year: int) -> dict[date, str]:
    return {
        _observed_fixed_holiday(year, 1, 1): "New Year's Day",
        _nth_weekday(year, 1, 0, 3): "Martin Luther King Jr. Day",
        _nth_weekday(year, 2, 0, 3): "Presidents' Day",
        _last_weekday(year, 5, 0): "Memorial Day",
        _observed_fixed_holiday(year, 6, 19): "Juneteenth",
        _observed_fixed_holiday(year, 7, 4): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 10, 0, 2): "Columbus Day",
        _observed_fixed_holiday(year, 11, 11): "Veterans Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        _observed_fixed_holiday(year, 12, 25): "Christmas Day",
    }


def thanksgiving_date(year: int) -> date:
    return _nth_weekday(year, 11, 3, 4)


def easter_date(year: int) -> date:
    """Return Gregorian Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    correction = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * correction) // 451
    month = (h + correction - 7 * m + 114) // 31
    day = ((h + correction - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def days_until_next_fixed_holiday(value: pd.Timestamp, month: int, day: int) -> int:
    target = pd.Timestamp(year=value.year, month=month, day=day)
    if value.date() > target.date():
        target = pd.Timestamp(year=value.year + 1, month=month, day=day)
    return int((target - value).days)


def signed_days_from_holiday(value: pd.Timestamp, holiday: date) -> int:
    return int((value.date() - holiday).days)


def signed_days_from_nearest_fixed(value: pd.Timestamp, month: int, day: int) -> int:
    candidates = [
        date(value.year - 1, month, day),
        date(value.year, month, day),
        date(value.year + 1, month, day),
    ]
    distances = [signed_days_from_holiday(value, candidate) for candidate in candidates]
    return min(distances, key=abs)


def build_calendar_holiday_features(raw_tsa: pd.DataFrame) -> pd.DataFrame:
    df = normalize_tsa_raw(raw_tsa)
    dates = df["Date"].dt
    iso = dates.isocalendar()

    features = df[["Date", "Passengers"]].copy()
    features["year"] = dates.year.astype(int)
    features["quarter"] = dates.quarter.astype(int)
    features["month"] = dates.month.astype(int)
    features["day"] = dates.day.astype(int)
    features["day_of_year"] = dates.dayofyear.astype(int)
    features["day_of_week"] = dates.dayofweek.astype(int)
    features["iso_year"] = iso["year"].astype(int)
    features["iso_week"] = iso["week"].astype(int)
    features["iso_day"] = iso["day"].astype(int)
    features["is_weekend"] = features["day_of_week"].isin([5, 6]).astype(int)
    features["is_month_start"] = dates.is_month_start.astype(int)
    features["is_month_end"] = dates.is_month_end.astype(int)
    features["is_quarter_start"] = dates.is_quarter_start.astype(int)
    features["is_quarter_end"] = dates.is_quarter_end.astype(int)

    years = range(features["year"].min() - 1, features["year"].max() + 2)
    holiday_map: dict[date, str] = {}
    for year in years:
        holiday_map.update(federal_holidays_for_year(year))

    holiday_dates = set(holiday_map)
    high_impact_names = {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving",
        "Christmas Day",
    }
    high_impact_dates = {
        holiday_date
        for holiday_date, name in holiday_map.items()
        if name in high_impact_names
    }
    window_dates = {
        holiday_date + timedelta(days=offset)
        for holiday_date in high_impact_dates
        for offset in range(-3, 4)
    }

    features["holiday_name"] = features["Date"].dt.date.map(holiday_map).fillna("No Holiday")
    features["is_federal_holiday"] = features["Date"].dt.date.isin(holiday_dates).astype(int)
    features["is_high_impact_holiday"] = features["Date"].dt.date.isin(high_impact_dates).astype(int)
    features["is_holiday_window"] = features["Date"].dt.date.isin(window_dates).astype(int)

    features["is_new_years_day"] = ((features["month"] == 1) & (features["day"] == 1)).astype(int)
    features["is_july_4"] = ((features["month"] == 7) & (features["day"] == 4)).astype(int)
    features["is_christmas_eve"] = ((features["month"] == 12) & (features["day"] == 24)).astype(int)
    features["is_christmas_day"] = ((features["month"] == 12) & (features["day"] == 25)).astype(int)
    features["is_nye"] = ((features["month"] == 12) & (features["day"] == 31)).astype(int)

    thanksgiving_by_year = {year: thanksgiving_date(year) for year in years}
    easter_by_year = {year: easter_date(year) for year in years}
    mlk_by_year = {year: _nth_weekday(year, 1, 0, 3) for year in years}
    presidents_by_year = {year: _nth_weekday(year, 2, 0, 3) for year in years}
    memorial_by_year = {year: _last_weekday(year, 5, 0) for year in years}
    labor_by_year = {year: _nth_weekday(year, 9, 0, 1) for year in years}

    features["thanksgiving_date"] = features["year"].map(thanksgiving_by_year)
    features["days_from_thanksgiving"] = [
        signed_days_from_holiday(value, thanksgiving_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_thanksgiving"] = (features["days_from_thanksgiving"] == 0).astype(int)
    features["is_thanksgiving_week"] = features["days_from_thanksgiving"].between(-3, 3).astype(int)
    features["is_thanksgiving_outbound"] = features["days_from_thanksgiving"].between(-7, -1).astype(int)
    features["is_thanksgiving_return"] = features["days_from_thanksgiving"].between(1, 4).astype(int)
    features = features.drop(columns=["thanksgiving_date"])

    features["days_from_easter"] = [
        signed_days_from_holiday(value, easter_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_easter_window"] = features["days_from_easter"].between(-3, 3).astype(int)

    features["days_from_mlk_day"] = [
        signed_days_from_holiday(value, mlk_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_mlk_window"] = features["days_from_mlk_day"].between(-3, 3).astype(int)

    features["days_from_presidents_day"] = [
        signed_days_from_holiday(value, presidents_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_presidents_day_window"] = features["days_from_presidents_day"].between(-3, 3).astype(int)

    features["days_from_memorial_day"] = [
        signed_days_from_holiday(value, memorial_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_memorial_day_window"] = features["days_from_memorial_day"].between(-3, 3).astype(int)

    features["days_from_labor_day"] = [
        signed_days_from_holiday(value, labor_by_year[value.year])
        for value in features["Date"]
    ]
    features["is_labor_day_window"] = features["days_from_labor_day"].between(-3, 3).astype(int)

    features["days_from_nearest_july_4"] = features["Date"].map(
        lambda value: signed_days_from_nearest_fixed(value, 7, 4)
    )
    features["is_july_4_window"] = features["days_from_nearest_july_4"].between(-3, 3).astype(int)

    features["days_from_nearest_christmas"] = features["Date"].map(
        lambda value: signed_days_from_nearest_fixed(value, 12, 25)
    )
    features["is_christmas_travel_window"] = features["days_from_nearest_christmas"].between(-7, 4).astype(int)

    features["days_from_nearest_new_year"] = features["Date"].map(
        lambda value: signed_days_from_nearest_fixed(value, 1, 1)
    )
    features["is_new_year_travel_window"] = features["days_from_nearest_new_year"].between(-3, 3).astype(int)
    features["is_spring_break_proxy"] = (
        ((features["month"] == 3) & (features["day"] >= 8))
        | ((features["month"] == 4) & (features["day"] <= 20))
    ).astype(int)

    features["days_until_christmas"] = features["Date"].map(
        lambda value: days_until_next_fixed_holiday(value, 12, 25)
    )
    features["days_until_new_year"] = features["Date"].map(
        lambda value: days_until_next_fixed_holiday(value, 1, 1)
    )
    features["days_until_july_4"] = features["Date"].map(
        lambda value: days_until_next_fixed_holiday(value, 7, 4)
    )

    return features


def write_calendar_holiday_features(raw_tsa: pd.DataFrame, path: Path = DAILY_CALENDAR_FEATURES_PATH) -> pd.DataFrame:
    features = build_calendar_holiday_features(raw_tsa)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    return features
