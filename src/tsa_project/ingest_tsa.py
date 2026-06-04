from html.parser import HTMLParser
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


class TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self._table_depth = 0
        self._current_table: list[list[str]] | None = None
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            if self._table_depth == 0:
                self._current_table = []
            self._table_depth += 1
        elif tag == "tr" and self._table_depth > 0:
            self._current_row = []
        elif tag in {"td", "th"} and self._current_row is not None:
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"td", "th"} and self._current_cell is not None and self._current_row is not None:
            value = " ".join("".join(self._current_cell).split())
            self._current_row.append(value)
            self._current_cell = None
        elif tag == "tr" and self._current_row is not None and self._current_table is not None:
            if any(cell.strip() for cell in self._current_row):
                self._current_table.append(self._current_row)
            self._current_row = None
        elif tag == "table" and self._table_depth > 0:
            self._table_depth -= 1
            if self._table_depth == 0 and self._current_table is not None:
                self.tables.append(self._current_table)
                self._current_table = None


def source_urls(years: tuple[int, ...] = DEFAULT_YEARS) -> list[str]:
    return [BASE_URL, *(f"{BASE_URL}/{year}" for year in sorted(years, reverse=True))]


def extract_tsa_rows(html: str) -> list[list[str]]:
    parser = TableParser()
    parser.feed(html)
    parser.close()
    for table in parser.tables:
        rows = [
            row[:2]
            for row in table
            if len(row) >= 2 and row[0].strip().lower() != "date"
        ]
        if rows:
            return rows
    raise ValueError("No HTML tables found in TSA passenger volume page")


def fetch_tsa_table(url: str) -> pd.DataFrame:
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    table = pd.DataFrame(extract_tsa_rows(response.text), columns=["Date", "Passengers"])
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
