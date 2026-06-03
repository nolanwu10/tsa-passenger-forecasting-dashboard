from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import requests

from tsa_project.config import (
    EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH,
    EXTERNAL_BTS_ON_TIME_DAILY_PATH,
    EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH,
)


ON_TIME_ZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
MTS_URL = "https://data.bts.gov/resource/crem-w557.json"
T100_CAPACITY_URL = "https://data.bts.gov/resource/q4tb-tbff.json"
REQUEST_TIMEOUT = 90


@dataclass(frozen=True)
class BtsSourceStatus:
    source: str
    rows: int
    min_date: str | None
    max_date: str | None
    path: str


def available_on_time_months(start_year: int, end_year: int) -> list[tuple[int, int]]:
    months: list[tuple[int, int]] = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = ON_TIME_ZIP_URL.format(year=year, month=month)
            response = requests.get(url, stream=True, timeout=30)
            prefix = response.raw.read(2) if response.status_code == 200 else b""
            response.close()
            if response.status_code == 200 and prefix == b"PK":
                months.append((year, month))
    return months


def fetch_on_time_month(year: int, month: int) -> pd.DataFrame:
    url = ON_TIME_ZIP_URL.format(year=year, month=month)
    response = requests.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    with ZipFile(BytesIO(response.content)) as archive:
        csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
        with archive.open(csv_name) as csv_file:
            return pd.read_csv(
                csv_file,
                usecols=[
                    "FlightDate",
                    "Cancelled",
                    "Diverted",
                    "DepDelayMinutes",
                    "ArrDelayMinutes",
                    "CarrierDelay",
                    "WeatherDelay",
                    "NASDelay",
                    "SecurityDelay",
                    "LateAircraftDelay",
                ],
                low_memory=False,
            )


def aggregate_on_time_daily(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["Date"] = pd.to_datetime(df["FlightDate"], errors="coerce")
    numeric_columns = [
        "Cancelled",
        "Diverted",
        "DepDelayMinutes",
        "ArrDelayMinutes",
        "CarrierDelay",
        "WeatherDelay",
        "NASDelay",
        "SecurityDelay",
        "LateAircraftDelay",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["delayed_departure_15"] = (df["DepDelayMinutes"] >= 15).astype(int)
    df["delayed_arrival_15"] = (df["ArrDelayMinutes"] >= 15).astype(int)
    df["flight_count"] = 1

    grouped = df.groupby("Date", as_index=False).agg(
        bts_flights_scheduled=("flight_count", "sum"),
        bts_flights_cancelled=("Cancelled", "sum"),
        bts_flights_diverted=("Diverted", "sum"),
        bts_dep_delay_total_minutes=("DepDelayMinutes", "sum"),
        bts_arr_delay_total_minutes=("ArrDelayMinutes", "sum"),
        bts_dep_delayed_15_count=("delayed_departure_15", "sum"),
        bts_arr_delayed_15_count=("delayed_arrival_15", "sum"),
        bts_carrier_delay_minutes=("CarrierDelay", "sum"),
        bts_weather_delay_minutes=("WeatherDelay", "sum"),
        bts_nas_delay_minutes=("NASDelay", "sum"),
        bts_security_delay_minutes=("SecurityDelay", "sum"),
        bts_late_aircraft_delay_minutes=("LateAircraftDelay", "sum"),
    )
    grouped["bts_cancel_rate"] = grouped["bts_flights_cancelled"] / grouped["bts_flights_scheduled"]
    grouped["bts_divert_rate"] = grouped["bts_flights_diverted"] / grouped["bts_flights_scheduled"]
    grouped["bts_dep_delay_15_rate"] = grouped["bts_dep_delayed_15_count"] / grouped["bts_flights_scheduled"]
    grouped["bts_arr_delay_15_rate"] = grouped["bts_arr_delayed_15_count"] / grouped["bts_flights_scheduled"]
    grouped["bts_avg_dep_delay_minutes"] = grouped["bts_dep_delay_total_minutes"] / grouped["bts_flights_scheduled"]
    grouped["bts_avg_arr_delay_minutes"] = grouped["bts_arr_delay_total_minutes"] / grouped["bts_flights_scheduled"]
    return grouped


def fetch_on_time_daily(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year, month in available_on_time_months(start_year, end_year):
        print(f"Fetching BTS on-time {year}-{month:02d}...")
        frames.append(aggregate_on_time_daily(fetch_on_time_month(year, month)))

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    return combined


def write_on_time_daily(start_year: int, end_year: int) -> BtsSourceStatus:
    data = fetch_on_time_daily(start_year, end_year)
    EXTERNAL_BTS_ON_TIME_DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(EXTERNAL_BTS_ON_TIME_DAILY_PATH, index=False)
    return _status("bts_on_time_daily", data, EXTERNAL_BTS_ON_TIME_DAILY_PATH)


def fetch_monthly_air_traffic(start_year: int = 2019) -> pd.DataFrame:
    params = {
        "$select": (
            "date,u_s_airline_traffic_total_non_seasonally_adjusted,"
            "u_s_airline_traffic_domestic_non_seasonally_adjusted,"
            "u_s_airline_traffic_international_non_seasonally_adjusted,"
            "system_use_u_s_airline_2,system_use_u_s_airline,"
            "system_use_u_s_airline_1,"
            "u_s_marketing_air_carriers_on_time_performance_percent"
        ),
        "$limit": 5000,
        "$order": "date",
    }
    response = requests.get(MTS_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = pd.DataFrame(response.json())
    data = data.rename(
        columns={
            "date": "month_start",
            "u_s_airline_traffic_total_non_seasonally_adjusted": "bts_mts_total_air_traffic_nsa",
            "u_s_airline_traffic_domestic_non_seasonally_adjusted": "bts_mts_domestic_air_traffic_nsa",
            "u_s_airline_traffic_international_non_seasonally_adjusted": "bts_mts_international_air_traffic_nsa",
            "system_use_u_s_airline_2": "bts_mts_total_air_traffic_sa",
            "system_use_u_s_airline": "bts_mts_domestic_air_traffic_sa",
            "system_use_u_s_airline_1": "bts_mts_international_air_traffic_sa",
            "u_s_marketing_air_carriers_on_time_performance_percent": "bts_mts_marketing_on_time_pct",
        }
    )
    data["month_start"] = pd.to_datetime(data["month_start"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    for column in data.columns:
        if column != "month_start":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    core_columns = [
        "bts_mts_total_air_traffic_nsa",
        "bts_mts_domestic_air_traffic_nsa",
        "bts_mts_international_air_traffic_nsa",
    ]
    data = data.dropna(subset=["month_start"])
    data = data[data["month_start"].dt.year >= start_year]
    data = data.dropna(subset=core_columns, how="all")
    data = data.sort_values("month_start")
    return data


def write_monthly_air_traffic() -> BtsSourceStatus:
    data = fetch_monthly_air_traffic()
    EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH, index=False)
    return _status("bts_monthly_air_traffic", data, EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH, "month_start")


def fetch_t100_annual_capacity() -> pd.DataFrame:
    params = {
        "$select": (
            "year,"
            "sum(domestic_departures) as bts_t100_domestic_departures,"
            "sum(domestic_passengers) as bts_t100_domestic_passengers,"
            "sum(domestic_seats) as bts_t100_domestic_seats,"
            "sum(total_departures) as bts_t100_total_departures,"
            "sum(total_passengers) as bts_t100_total_passengers,"
            "sum(total_seats) as bts_t100_total_seats"
        ),
        "$group": "year",
        "$order": "year",
        "$limit": 5000,
    }
    response = requests.get(T100_CAPACITY_URL, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = pd.DataFrame(response.json())
    data["year"] = pd.to_numeric(data["year"], errors="coerce").astype("Int64")
    for column in data.columns:
        if column != "year":
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data["bts_t100_domestic_load_factor"] = (
        data["bts_t100_domestic_passengers"] / data["bts_t100_domestic_seats"]
    )
    data["bts_t100_total_load_factor"] = (
        data["bts_t100_total_passengers"] / data["bts_t100_total_seats"]
    )
    return data.dropna(subset=["year"]).sort_values("year")


def write_t100_annual_capacity() -> BtsSourceStatus:
    data = fetch_t100_annual_capacity()
    EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH, index=False)
    return _status("bts_t100_annual_capacity", data, EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH, "year")


def _status(source: str, data: pd.DataFrame, path, date_column: str = "Date") -> BtsSourceStatus:
    min_date = None
    max_date = None
    if not data.empty and date_column in data.columns:
        values = data[date_column]
        if pd.api.types.is_datetime64_any_dtype(values):
            min_date = values.min().date().isoformat()
            max_date = values.max().date().isoformat()
        else:
            min_date = str(values.min())
            max_date = str(values.max())
    return BtsSourceStatus(source, len(data), min_date, max_date, str(path))
