from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from tsa_project.config import RAW_TSA_PATH
from tsa_project.datasets import normalize_tsa_raw


BASE_URL = "https://www.tsa.gov/travel/passenger-volumes"
DEFAULT_YEARS = tuple(range(2019, 2026))
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


def source_urls(years: tuple[int, ...] = DEFAULT_YEARS) -> list[str]:
    return [BASE_URL, *(f"{BASE_URL}/{year}" for year in sorted(years, reverse=True))]


def fetch_tsa_table(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    tables = pd.read_html(StringIO(response.text))
    if not tables:
        raise ValueError(f"No HTML tables found at {url}")

    table = tables[0].iloc[:, :2].copy()
    table.columns = ["Date", "Passengers"]
    table["source_url"] = url
    return table


def fetch_tsa_passenger_data(years: tuple[int, ...] = DEFAULT_YEARS) -> pd.DataFrame:
    frames = [fetch_tsa_table(url) for url in source_urls(years)]
    combined = pd.concat(frames, ignore_index=True)
    return normalize_tsa_raw(combined)


def write_tsa_passenger_data(path: Path = RAW_TSA_PATH) -> pd.DataFrame:
    df = fetch_tsa_passenger_data()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

