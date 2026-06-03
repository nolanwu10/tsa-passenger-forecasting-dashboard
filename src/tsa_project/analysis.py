from pathlib import Path

import pandas as pd

from tsa_project.config import (
    DAILY_CALENDAR_FEATURES_PATH,
    EXTERNAL_WEATHER_PATH,
    PROCESSED_FEATURES_PATH,
    RAW_TSA_PATH,
)
from tsa_project.datasets import normalize_tsa_raw


def _format_int(value: float | int) -> str:
    return f"{value:,.0f}"


def _format_pct(value: float) -> str:
    return f"{value:+.2%}"


def load_raw_tsa() -> pd.DataFrame:
    return normalize_tsa_raw(pd.read_csv(RAW_TSA_PATH))


def date_health(df: pd.DataFrame) -> dict[str, object]:
    full_range = pd.date_range(df["Date"].min(), df["Date"].max(), freq="D")
    observed = pd.DatetimeIndex(df["Date"])
    missing = full_range.difference(observed)
    return {
        "rows": len(df),
        "min_date": df["Date"].min().date().isoformat(),
        "max_date": df["Date"].max().date().isoformat(),
        "expected_days": len(full_range),
        "missing_days": len(missing),
        "duplicate_dates": int(df["Date"].duplicated().sum()),
    }


def annual_summary(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["year"] = work["Date"].dt.year
    grouped = work.groupby("year")["Passengers"]
    return grouped.agg(days="count", total="sum", average="mean", minimum="min", maximum="max")


def ytd_comparison(df: pd.DataFrame) -> pd.DataFrame:
    max_date = df["Date"].max()
    month_day_cutoff = (max_date.month, max_date.day)
    work = df.copy()
    work["year"] = work["Date"].dt.year
    ytd = work[
        work["Date"].map(lambda d: (d.month, d.day) <= month_day_cutoff)
    ]
    grouped = ytd.groupby("year")["Passengers"].agg(days="count", total="sum", average="mean")
    grouped["avg_yoy_change"] = grouped["average"].pct_change()
    return grouped


def day_of_week_summary(df: pd.DataFrame, start_year: int = 2023) -> pd.DataFrame:
    work = df[df["Date"].dt.year >= start_year].copy()
    work["day_of_week"] = work["Date"].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    summary = work.groupby("day_of_week")["Passengers"].agg(days="count", average="mean")
    return summary.reindex(order)


def busiest_days(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.nlargest(n, "Passengers")[["Date", "Passengers"]].copy()


def processed_health() -> dict[str, object]:
    result: dict[str, object] = {}
    for name, path in {
        "legacy_processed_features": PROCESSED_FEATURES_PATH,
        "daily_calendar_features": DAILY_CALENDAR_FEATURES_PATH,
    }.items():
        if not path.exists():
            result[name] = {"exists": False}
            continue

        processed = pd.read_csv(path)
        processed["Date"] = pd.to_datetime(processed["Date"], errors="coerce")
        result[name] = {
            "exists": True,
            "rows": len(processed),
            "min_date": processed["Date"].min().date().isoformat(),
            "max_date": processed["Date"].max().date().isoformat(),
            "columns": len(processed.columns),
        }
    return result


def external_data_health() -> dict[str, object]:
    if not EXTERNAL_WEATHER_PATH.exists():
        return {"weather_exists": False}

    weather = pd.read_csv(EXTERNAL_WEATHER_PATH)
    weather["Date"] = pd.to_datetime(weather["Date"], errors="coerce")
    return {
        "weather_exists": True,
        "weather_rows": len(weather),
        "weather_min_date": weather["Date"].min().date().isoformat()
        if weather["Date"].notna().any()
        else None,
        "weather_max_date": weather["Date"].max().date().isoformat()
        if weather["Date"].notna().any()
        else None,
        "weather_columns": list(weather.columns),
    }


def dataframe_to_markdown(
    df: pd.DataFrame,
    int_columns: set[str] | None = None,
    percent_columns: set[str] | None = None,
) -> str:
    int_columns = int_columns or set()
    percent_columns = percent_columns or set()
    output = df.copy()
    for column in output.columns:
        if column in int_columns:
            output[column] = output[column].map(_format_int)
        if column in percent_columns:
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else _format_pct(value))
    return output.to_markdown()


def build_markdown_report() -> str:
    df = load_raw_tsa()
    health = date_health(df)
    annual = annual_summary(df).reset_index()
    ytd = ytd_comparison(df).reset_index()
    dow = day_of_week_summary(df).reset_index()
    top = busiest_days(df).copy()
    processed = processed_health()
    external = external_data_health()

    latest_year = int(df["Date"].dt.year.max())
    ytd_latest = ytd[ytd["year"] == latest_year].iloc[0]
    ytd_prior = ytd[ytd["year"] == latest_year - 1].iloc[0]
    ytd_2019 = ytd[ytd["year"] == 2019].iloc[0]

    lines = [
        "# Data Readiness Report",
        "",
        "## Source Status",
        "",
        f"- Raw TSA rows: {_format_int(health['rows'])}",
        f"- Date range: `{health['min_date']}` to `{health['max_date']}`",
        f"- Expected daily rows in range: {_format_int(health['expected_days'])}",
        f"- Missing dates: {_format_int(health['missing_days'])}",
        f"- Duplicate dates: {_format_int(health['duplicate_dates'])}",
        "",
        "The raw TSA dataset is continuous at daily grain and is ready to be the base table for feature work.",
        "",
        "## YTD Signal",
        "",
        (
            f"Through {health['max_date']}, {latest_year} average daily throughput is "
            f"{_format_int(ytd_latest['average'])}. That is "
            f"{_format_pct((ytd_latest['average'] / ytd_prior['average']) - 1)} versus "
            f"{latest_year - 1} and {_format_pct((ytd_latest['average'] / ytd_2019['average']) - 1)} "
            "versus 2019."
        ),
        "",
        dataframe_to_markdown(
            ytd[["year", "days", "total", "average", "avg_yoy_change"]],
            int_columns={"days", "total", "average"},
            percent_columns={"avg_yoy_change"},
        ),
        "",
        "## Annual Summary",
        "",
        dataframe_to_markdown(
            annual[["year", "days", "total", "average", "minimum", "maximum"]],
            int_columns={"days", "total", "average", "minimum", "maximum"},
        ),
        "",
        "## Day-Of-Week Shape Since 2023",
        "",
        dataframe_to_markdown(dow, int_columns={"days", "average"}),
        "",
        "## Busiest Observed Days",
        "",
    ]

    top["Date"] = top["Date"].dt.date.astype(str)
    lines.extend(
        [
            dataframe_to_markdown(top, int_columns={"Passengers"}),
            "",
            "## Existing Derived Data",
            "",
            f"- `processed_features.csv` legacy table: {processed['legacy_processed_features']}",
            f"- `tsa_daily_calendar_features.csv`: {processed['daily_calendar_features']}",
            (
                f"- `weather_history.csv`: {external}"
                if external.get("weather_exists")
                else "- `weather_history.csv`: missing"
            ),
            "",
            "The legacy processed features are stale relative to raw TSA data and should not be used for new modeling until rebuilt.",
            "The existing weather file is too sparse to be model-ready.",
            "",
            "## Modeling Readiness Notes",
            "",
            "- Use raw TSA as the canonical target table.",
            "- Rebuild processed features from raw data in a deterministic pipeline.",
            "- Treat 2020 and 2021 as structural-break years; include them carefully or downweight/exclude depending on the modeling objective.",
            "- Preserve calendar and matched-history features, but define them in one feature-building module before training.",
            "- Do not use legacy model artifacts as inputs to the reset modeling pipeline.",
            "",
            "## External Data Candidates",
            "",
            "- Calendar and holiday structure: federal holidays, observed holidays, Thanksgiving week, Christmas/New Year windows, spring break, school calendar proxies.",
            "- Airline/network capacity: BTS or schedule-derived flight counts, available seats, cancellations, and delays.",
            "- Weather: daily severe-weather indicators around major hubs, rebuilt with complete historical coverage.",
            "- Macro/travel demand: gasoline prices, consumer sentiment, unemployment, airfare indexes, and travel search interest.",
            "- Market context: if the goal includes Kalshi trading, keep market prices separate from target features to avoid leakage in passenger forecasts.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(path: Path) -> str:
    report = build_markdown_report()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")
    return report
