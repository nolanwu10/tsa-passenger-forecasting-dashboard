from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from tsa_project.config import (
    DAILY_CALENDAR_FEATURES_PATH,
    DAILY_TRANSPORT_FEATURES_PATH,
    EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH,
    EXTERNAL_BTS_ON_TIME_DAILY_PATH,
    EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH,
    EXTERNAL_WEATHER_PATH,
    PROCESSED_FEATURES_PATH,
    RAW_TSA_PATH,
)


@dataclass(frozen=True)
class DatasetLocation:
    name: str
    path: Path


DATASETS = {
    "raw_tsa": DatasetLocation("raw_tsa", RAW_TSA_PATH),
    "external_weather": DatasetLocation("external_weather", EXTERNAL_WEATHER_PATH),
    "bts_on_time_daily": DatasetLocation("bts_on_time_daily", EXTERNAL_BTS_ON_TIME_DAILY_PATH),
    "bts_monthly_air_traffic": DatasetLocation(
        "bts_monthly_air_traffic",
        EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH,
    ),
    "bts_t100_annual_capacity": DatasetLocation(
        "bts_t100_annual_capacity",
        EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH,
    ),
    "processed_features": DatasetLocation("processed_features", PROCESSED_FEATURES_PATH),
    "daily_calendar_features": DatasetLocation(
        "daily_calendar_features",
        DAILY_CALENDAR_FEATURES_PATH,
    ),
    "daily_transport_features": DatasetLocation(
        "daily_transport_features",
        DAILY_TRANSPORT_FEATURES_PATH,
    ),
}


def read_csv_dataset(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{name}'. Expected one of: {choices}")

    location = DATASETS[name]
    if not location.path.exists():
        raise FileNotFoundError(f"{location.path} does not exist")

    return pd.read_csv(location.path)


def normalize_tsa_raw(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce")
    normalized["Passengers"] = pd.to_numeric(
        normalized["Passengers"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    normalized = normalized.dropna(subset=["Date", "Passengers"])
    normalized["Passengers"] = normalized["Passengers"].astype("int64")
    normalized = normalized.sort_values("Date").drop_duplicates(subset=["Date"])
    return normalized.reset_index(drop=True)


def summarize_dataframe(df: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
    }

    date_column = "Date" if "Date" in df.columns else "month_start" if "month_start" in df.columns else None
    if date_column is not None:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid_dates = dates.dropna()
        if not valid_dates.empty:
            summary["min_date"] = valid_dates.min().date().isoformat()
            summary["max_date"] = valid_dates.max().date().isoformat()
            summary["unique_dates"] = int(valid_dates.nunique())

    if "year" in df.columns and date_column is None:
        years = pd.to_numeric(df["year"], errors="coerce").dropna()
        if not years.empty:
            summary["min_year"] = int(years.min())
            summary["max_year"] = int(years.max())
            summary["unique_years"] = int(years.nunique())

    if "Passengers" in df.columns:
        passengers = pd.to_numeric(
            df["Passengers"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        valid_passengers = passengers.dropna()
        if not valid_passengers.empty:
            summary["min_passengers"] = int(valid_passengers.min())
            summary["max_passengers"] = int(valid_passengers.max())
            summary["mean_passengers"] = float(valid_passengers.mean())

    return summary


def inventory() -> list[dict[str, object]]:
    rows = []
    for name, location in DATASETS.items():
        item: dict[str, object] = {
            "dataset": name,
            "path": str(location.path),
            "exists": location.path.exists(),
        }
        if location.path.exists():
            item["size_bytes"] = location.path.stat().st_size
        rows.append(item)
    return rows
