import pandas as pd

from tsa_project.config import (
    DAILY_CALENDAR_FEATURES_PATH,
    DAILY_TRANSPORT_FEATURES_PATH,
    EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH,
    EXTERNAL_BTS_ON_TIME_DAILY_PATH,
    EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH,
)


def build_transport_features() -> pd.DataFrame:
    base = pd.read_csv(DAILY_CALENDAR_FEATURES_PATH)
    base["Date"] = pd.to_datetime(base["Date"], errors="coerce")
    base["month_start"] = base["Date"].dt.to_period("M").dt.to_timestamp()

    if EXTERNAL_BTS_ON_TIME_DAILY_PATH.exists():
        on_time = pd.read_csv(EXTERNAL_BTS_ON_TIME_DAILY_PATH)
        on_time["Date"] = pd.to_datetime(on_time["Date"], errors="coerce")
        base = base.merge(on_time, on="Date", how="left")

    if EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH.exists():
        monthly = pd.read_csv(EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH)
        monthly["month_start"] = pd.to_datetime(monthly["month_start"], errors="coerce")
        base = base.merge(monthly, on="month_start", how="left")
        lagged_monthly = monthly.copy()
        lagged_monthly["month_start"] = lagged_monthly["month_start"] + pd.DateOffset(months=2)
        lagged_monthly = lagged_monthly.rename(
            columns={
                column: f"{column}_lag2_months"
                for column in lagged_monthly.columns
                if column != "month_start"
            }
        )
        base = base.merge(lagged_monthly, on="month_start", how="left")

    if EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH.exists():
        annual = pd.read_csv(EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH)
        annual["year"] = pd.to_numeric(annual["year"], errors="coerce").astype("Int64")
        annual = annual.sort_values("year")
        all_years = pd.DataFrame({"year": sorted(base["year"].dropna().unique())})
        annual = all_years.merge(annual, on="year", how="left")
        annual["bts_t100_capacity_reference_year"] = annual["year"].where(
            annual["bts_t100_domestic_seats"].notna()
        )
        annual["bts_t100_capacity_reference_year"] = (
            annual["bts_t100_capacity_reference_year"].ffill()
        )
        annual = annual.ffill()
        base = base.merge(annual, on="year", how="left")

    engineered = base.copy()
    delay_signal_columns = [
        "bts_flights_scheduled",
        "bts_cancel_rate",
        "bts_divert_rate",
        "bts_dep_delay_15_rate",
        "bts_arr_delay_15_rate",
        "bts_avg_dep_delay_minutes",
        "bts_avg_arr_delay_minutes",
        "bts_weather_delay_minutes",
        "bts_nas_delay_minutes",
        "bts_late_aircraft_delay_minutes",
    ]
    for column in delay_signal_columns:
        if column not in engineered.columns:
            continue
        engineered[f"{column}_lag1"] = engineered[column].shift(1)
        engineered[f"{column}_roll7_lag1"] = engineered[column].shift(1).rolling(
            window=7,
            min_periods=3,
        ).mean()
        engineered[f"{column}_roll28_lag1"] = engineered[column].shift(1).rolling(
            window=28,
            min_periods=7,
        ).mean()

    if {"bts_t100_domestic_seats", "Passengers"}.issubset(engineered.columns):
        days_in_year = engineered["Date"].dt.is_leap_year.map({True: 366, False: 365})
        engineered["bts_t100_est_daily_domestic_seats"] = engineered["bts_t100_domestic_seats"] / days_in_year
        engineered["tsa_passengers_per_t100_daily_domestic_seat"] = (
            engineered["Passengers"] / engineered["bts_t100_est_daily_domestic_seats"]
        )

    if "bts_mts_domestic_air_traffic_nsa" in engineered.columns:
        days_in_month = engineered["Date"].dt.days_in_month
        engineered["bts_mts_est_daily_domestic_air_traffic_nsa"] = (
            engineered["bts_mts_domestic_air_traffic_nsa"] / days_in_month
        )
        engineered["tsa_passengers_per_mts_daily_domestic_traffic"] = (
            engineered["Passengers"] / engineered["bts_mts_est_daily_domestic_air_traffic_nsa"]
        )

    engineered = engineered.drop(columns=["month_start"])
    return engineered


def write_transport_features() -> pd.DataFrame:
    features = build_transport_features()
    DAILY_TRANSPORT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(DAILY_TRANSPORT_FEATURES_PATH, index=False)
    return features
