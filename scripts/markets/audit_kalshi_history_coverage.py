from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import REPORT_ARTIFACTS_DIR
from tsa_project.config import EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH
from tsa_project.kalshi_blend_backtest import KALSHI_YTD_BACKTEST_PATH


COVERAGE_OUTPUT = REPORT_ARTIFACTS_DIR / "kalshi_market_history_coverage.csv"


REQUIRED_CONTRACT_LEVEL_COLUMNS = {
    "ticker",
    "threshold",
    "yes_bid",
    "yes_ask",
    "market_probability",
}


def main() -> int:
    rows = pd.read_csv(KALSHI_YTD_BACKTEST_PATH, parse_dates=["week_monday", "as_of_date"])
    known_days = sorted(int(value) for value in rows["known_days_model_semantics"].dropna().unique())
    missing_known_days = [day for day in range(8) if day not in known_days]
    contract_columns_present = REQUIRED_CONTRACT_LEVEL_COLUMNS.issubset(rows.columns)
    candles = (
        pd.read_csv(EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH, parse_dates=["week_monday", "as_of_date"])
        if EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH.exists()
        else pd.DataFrame()
    )
    preweek_contract_rows = 0
    usable_contract_rows = 0
    usable_contract_events = 0
    if not candles.empty:
        preweek_contract_rows = int((candles["as_of_date"] < candles["week_monday"]).sum())
        usable_mask = candles[["yes_bid_close", "yes_ask_close"]].notna().any(axis=1)
        usable_contract_rows = int(usable_mask.sum())
        usable_contract_events = int(candles.loc[usable_mask, "event_ticker"].nunique())

    coverage = (
        rows.groupby("week_monday")
        .agg(
            event_ticker=("event_ticker", "first"),
            first_as_of_date=("as_of_date", "min"),
            last_as_of_date=("as_of_date", "max"),
            rows=("as_of_date", "count"),
            min_known_days=("known_days_model_semantics", "min"),
            max_known_days=("known_days_model_semantics", "max"),
            has_known_days_0=("known_days_model_semantics", lambda s: int((s == 0).any())),
            has_known_days_1=("known_days_model_semantics", lambda s: int((s == 1).any())),
            has_known_days_2=("known_days_model_semantics", lambda s: int((s == 2).any())),
            has_known_days_3=("known_days_model_semantics", lambda s: int((s == 3).any())),
            has_known_days_7=("known_days_model_semantics", lambda s: int((s == 7).any())),
        )
        .reset_index()
    )
    COVERAGE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(COVERAGE_OUTPUT, index=False)

    print("Kalshi historical coverage audit:")
    print(f"Rows: {len(rows):,}")
    print(f"Weeks: {rows['week_monday'].nunique():,}")
    print(f"Known-day states present: {known_days}")
    print(f"Known-day states missing: {missing_known_days}")
    print(f"Pre-week known_days=0 rows present: {0 in known_days}")
    print(f"Contract-level bid/ask columns present: {contract_columns_present}")
    if not contract_columns_present:
        missing = sorted(REQUIRED_CONTRACT_LEVEL_COLUMNS - set(rows.columns))
        print(f"Missing contract-level columns: {missing}")
    print(f"Contract candlestick file exists: {EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH.exists()}")
    if not candles.empty:
        print(f"Contract candlestick rows: {len(candles):,}")
        print(f"Contract events: {candles['event_ticker'].nunique():,}")
        print(f"Usable bid/ask candle rows: {usable_contract_rows:,}")
        print(f"Usable bid/ask events: {usable_contract_events:,}")
        print(f"Pre-week contract candle rows: {preweek_contract_rows:,}")
    print(f"Saved coverage report: {COVERAGE_OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
