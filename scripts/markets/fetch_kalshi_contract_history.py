from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import (
    EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH,
    EXTERNAL_KALSHI_TSA_MARKETS_PATH,
)
from tsa_project.kalshi_contract_history import (
    CONTRACT_HISTORY_COVERAGE_PATH,
    ContractHistoryConfig,
    fetch_contract_history,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch per-contract Kalshi TSA weekly market metadata and candlesticks."
    )
    parser.add_argument("--period-interval", type=int, default=1440, choices=[1, 60, 1440])
    parser.add_argument("--days-before-week", type=int, default=3)
    parser.add_argument("--days-after-week", type=int, default=1)
    parser.add_argument(
        "--max-events",
        type=int,
        help="Limit to the most recent N events for testing or incremental fetches.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    markets, candles = fetch_contract_history(
        ContractHistoryConfig(
            period_interval=args.period_interval,
            days_before_week=args.days_before_week,
            days_after_week=args.days_after_week,
            max_events=args.max_events,
        )
    )
    print("Kalshi contract history fetched:")
    print(f"Markets: {len(markets):,}")
    print(f"Candlestick rows: {len(candles):,}")
    print(f"Saved markets: {EXTERNAL_KALSHI_TSA_MARKETS_PATH}")
    print(f"Saved candlesticks: {EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH}")
    print(f"Saved coverage: {CONTRACT_HISTORY_COVERAGE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
