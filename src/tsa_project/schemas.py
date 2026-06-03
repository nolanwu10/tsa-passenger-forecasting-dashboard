from dataclasses import dataclass


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    logical_type: str
    required: bool = True
    description: str = ""


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    grain: str
    columns: tuple[ColumnSpec, ...]

    @property
    def required_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.required)


RAW_TSA_SPEC = DatasetSpec(
    name="raw_tsa",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date", description="Calendar date of TSA passenger throughput."),
        ColumnSpec("Passengers", "integer", description="Total TSA traveler throughput."),
        ColumnSpec("source_url", "string", required=False, description="TSA source page for the row."),
    ),
)

EXTERNAL_WEATHER_SPEC = DatasetSpec(
    name="external_weather",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date", description="Calendar date."),
        ColumnSpec(
            "Hub_Severe_Weather_Index",
            "numeric",
            description="Aggregate severe-weather index across selected airport hubs.",
        ),
    ),
)

PROCESSED_FEATURES_SPEC = DatasetSpec(
    name="processed_features",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date"),
        ColumnSpec("Passengers", "integer"),
        ColumnSpec("iso_year", "integer"),
        ColumnSpec("iso_week", "integer"),
        ColumnSpec("day_of_week", "integer"),
    ),
)

DAILY_CALENDAR_FEATURES_SPEC = DatasetSpec(
    name="daily_calendar_features",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date"),
        ColumnSpec("Passengers", "integer"),
        ColumnSpec("year", "integer"),
        ColumnSpec("month", "integer"),
        ColumnSpec("day", "integer"),
        ColumnSpec("day_of_week", "integer"),
        ColumnSpec("iso_year", "integer"),
        ColumnSpec("iso_week", "integer"),
        ColumnSpec("is_weekend", "integer"),
        ColumnSpec("is_federal_holiday", "integer"),
        ColumnSpec("is_holiday_window", "integer"),
    ),
)

DAILY_TRANSPORT_FEATURES_SPEC = DatasetSpec(
    name="daily_transport_features",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date"),
        ColumnSpec("Passengers", "integer"),
        ColumnSpec("bts_flights_scheduled", "numeric", required=False),
        ColumnSpec("bts_cancel_rate", "numeric", required=False),
        ColumnSpec("bts_avg_dep_delay_minutes", "numeric", required=False),
        ColumnSpec("bts_mts_domestic_air_traffic_nsa", "numeric", required=False),
        ColumnSpec("bts_t100_domestic_seats", "numeric", required=False),
    ),
)

BTS_ON_TIME_DAILY_SPEC = DatasetSpec(
    name="bts_on_time_daily",
    grain="one row per calendar date",
    columns=(
        ColumnSpec("Date", "date"),
        ColumnSpec("bts_flights_scheduled", "numeric"),
        ColumnSpec("bts_cancel_rate", "numeric"),
        ColumnSpec("bts_avg_dep_delay_minutes", "numeric"),
    ),
)

BTS_MONTHLY_AIR_TRAFFIC_SPEC = DatasetSpec(
    name="bts_monthly_air_traffic",
    grain="one row per calendar month",
    columns=(
        ColumnSpec("month_start", "date"),
        ColumnSpec("bts_mts_domestic_air_traffic_nsa", "numeric"),
    ),
)

BTS_T100_ANNUAL_CAPACITY_SPEC = DatasetSpec(
    name="bts_t100_annual_capacity",
    grain="one row per calendar year",
    columns=(
        ColumnSpec("year", "integer"),
        ColumnSpec("bts_t100_domestic_seats", "numeric"),
        ColumnSpec("bts_t100_domestic_departures", "numeric"),
    ),
)

DATASET_SPECS = {
    RAW_TSA_SPEC.name: RAW_TSA_SPEC,
    EXTERNAL_WEATHER_SPEC.name: EXTERNAL_WEATHER_SPEC,
    PROCESSED_FEATURES_SPEC.name: PROCESSED_FEATURES_SPEC,
    DAILY_CALENDAR_FEATURES_SPEC.name: DAILY_CALENDAR_FEATURES_SPEC,
    DAILY_TRANSPORT_FEATURES_SPEC.name: DAILY_TRANSPORT_FEATURES_SPEC,
    BTS_ON_TIME_DAILY_SPEC.name: BTS_ON_TIME_DAILY_SPEC,
    BTS_MONTHLY_AIR_TRAFFIC_SPEC.name: BTS_MONTHLY_AIR_TRAFFIC_SPEC,
    BTS_T100_ANNUAL_CAPACITY_SPEC.name: BTS_T100_ANNUAL_CAPACITY_SPEC,
}
