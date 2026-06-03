from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.kalshi_trade_backtest import (
    CONTRACT_TRADE_BACKTEST_GRID_PATH,
    CONTRACT_TRADE_BACKTEST_SUMMARY_PATH,
    CONTRACT_TRADE_BACKTEST_TRADES_PATH,
    DEFAULT_SIGNAL_VARIANT,
    KALSHI_BLEND_BACKTEST_DAILY_PATH,
    run_contract_trade_backtest,
)
from tsa_project.config import EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH


def fmt_money(value: float) -> str:
    return f"{value:+.2f}"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return "" if value != value else f"{value:.2%}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest TSA trades against actual historical Kalshi contract candlestick prices."
    )
    parser.add_argument("--source", type=Path, default=KALSHI_BLEND_BACKTEST_DAILY_PATH)
    parser.add_argument("--candles", type=Path, default=EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH)
    parser.add_argument("--signal-variant", default=DEFAULT_SIGNAL_VARIANT)
    parser.add_argument("--min-known-days", type=int, default=1)
    parser.add_argument("--max-known-days", type=int, default=6)
    parser.add_argument("--trades-output", type=Path, default=CONTRACT_TRADE_BACKTEST_TRADES_PATH)
    parser.add_argument("--summary-output", type=Path, default=CONTRACT_TRADE_BACKTEST_SUMMARY_PATH)
    parser.add_argument("--grid-output", type=Path, default=CONTRACT_TRADE_BACKTEST_GRID_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trades, summary, grid = run_contract_trade_backtest(
        source_path=args.source,
        candles_path=args.candles,
        signal_variant=args.signal_variant,
        min_known_days=args.min_known_days,
        max_known_days=args.max_known_days,
    )
    args.trades_output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.trades_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    grid.to_csv(args.grid_output, index=False)

    print("Selected actual-contract Kalshi TSA trade rule:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "min_gap": fmt_count,
                "min_estimated_edge": fmt_pct,
                "min_contract_volume": fmt_count,
                "min_open_interest": fmt_count,
                "min_entry_price": fmt_pct,
                "max_entry_price": fmt_pct,
                "trade_rate": fmt_pct,
                "win_rate": fmt_pct,
                "total_cost": "{:.2f}".format,
                "total_pnl": fmt_money,
                "roi": fmt_pct,
                "avg_pnl_per_trade": fmt_money,
                "median_estimated_edge": fmt_pct,
                "avg_abs_gap": fmt_count,
                "max_drawdown": fmt_money,
            },
        )
    )

    print("\nTop actual-contract grid candidates:")
    candidates = grid[grid["trades"] > 0].sort_values(["total_pnl", "roi"], ascending=False).head(15)
    print(
        candidates[
            [
                "min_gap",
                "min_estimated_edge",
                "min_contract_volume",
                "max_entry_price",
                "trades",
                "win_rate",
                "total_pnl",
                "roi",
                "max_drawdown",
            ]
        ].to_string(
            index=False,
            formatters={
                "min_gap": fmt_count,
                "min_estimated_edge": fmt_pct,
                "min_contract_volume": fmt_count,
                "max_entry_price": fmt_pct,
                "win_rate": fmt_pct,
                "total_pnl": fmt_money,
                "roi": fmt_pct,
                "max_drawdown": fmt_money,
            },
        )
    )

    traded = trades[trades["trade"]].copy()
    print("\nExecuted actual-contract trades:")
    if traded.empty:
        print("No trades passed the selected actual-contract rule.")
    else:
        print(
            traded[
                [
                    "as_of_date",
                    "week_monday",
                    "known_days",
                    "ticker",
                    "side",
                    "threshold",
                    "entry_price",
                    "estimated_edge",
                    "settlement",
                    "pnl",
                ]
            ].to_string(
                index=False,
                formatters={
                    "threshold": fmt_count,
                    "entry_price": fmt_pct,
                    "estimated_edge": fmt_pct,
                    "settlement": "{:.0f}".format,
                    "pnl": fmt_money,
                },
            )
        )

    print(f"\nSaved trades: {args.trades_output}")
    print(f"Saved summary: {args.summary_output}")
    print(f"Saved grid: {args.grid_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
