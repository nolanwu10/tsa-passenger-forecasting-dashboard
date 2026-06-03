import pandas as pd

from tsa_project.datasets import normalize_tsa_raw
from tsa_project.schemas import DatasetSpec


def validate_required_columns(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    missing = sorted(set(spec.required_columns) - set(df.columns))
    if not missing:
        return []
    return [f"Missing required columns: {', '.join(missing)}"]


def validate_date_column(df: pd.DataFrame, column: str = "Date") -> list[str]:
    if column not in df.columns:
        return [f"Missing date column: {column}"]

    dates = pd.to_datetime(df[column], errors="coerce")
    errors = []
    if dates.isna().any():
        errors.append(f"{int(dates.isna().sum())} rows have invalid dates")
    if dates.dropna().duplicated().any():
        errors.append(f"{int(dates.dropna().duplicated().sum())} duplicate dates found")
    return errors


def validate_passengers_column(df: pd.DataFrame, column: str = "Passengers") -> list[str]:
    if column not in df.columns:
        return [f"Missing passenger column: {column}"]

    passengers = pd.to_numeric(
        df[column].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    errors = []
    if passengers.isna().any():
        errors.append(f"{int(passengers.isna().sum())} rows have invalid passenger counts")
    if (passengers.dropna() < 0).any():
        errors.append("Passenger counts include negative values")
    return errors


def validate_dataset(df: pd.DataFrame, spec: DatasetSpec) -> list[str]:
    errors = validate_required_columns(df, spec)

    if "Date" in spec.required_columns:
        errors.extend(validate_date_column(df))

    if "Passengers" in spec.required_columns:
        errors.extend(validate_passengers_column(df))

    if spec.name == "raw_tsa" and not errors:
        normalized = normalize_tsa_raw(df)
        if normalized.empty:
            errors.append("Raw TSA data is empty after normalization")

    return errors

