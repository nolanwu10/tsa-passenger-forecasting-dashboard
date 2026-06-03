from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.kalshi_trade_backtest import (
    DEFAULT_SIGNAL_VARIANT,
    KALSHI_BLEND_BACKTEST_DAILY_PATH,
    TRADE_BACKTEST_GRID_PATH,
    TRADE_BACKTEST_SUMMARY_PATH,
    TRADE_BACKTEST_TRADES_PATH,
    run_trade_backtest,
)


def fmt_money(value: float) -> str:
    return f"{value:+.2f}"


def fmt_count(value: float) -> str:
    return f"{value:,.0f}"


def fmt_pct(value: float) -> str:
    return "" if value != value else f"{value:.2%}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest simple Kalshi TSA trading rules against historical model-market disagreement."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=KALSHI_BLEND_BACKTEST_DAILY_PATH,
        help="Dashboard model/Kalshi blend backtest rows to use as signal history.",
    )
    parser.add_argument(
        "--signal-variant",
        default=DEFAULT_SIGNAL_VARIANT,
        help="Variant whose raw model forecast is compared against the Kalshi 50%% crossing.",
    )
    parser.add_argument(
        "--min-known-days",
        type=int,
        default=1,
        help="Earliest in-week known-day state to allow in the trading simulation.",
    )
    parser.add_argument(
        "--max-known-days",
        type=int,
        default=6,
        help="Latest known-day state to allow. Defaults to 6 to exclude fully known weeks.",
    )
    parser.add_argument("--trades-output", type=Path, default=TRADE_BACKTEST_TRADES_PATH)
    parser.add_argument("--summary-output", type=Path, default=TRADE_BACKTEST_SUMMARY_PATH)
    parser.add_argument("--grid-output", type=Path, default=TRADE_BACKTEST_GRID_PATH)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    trades, summary, grid = run_trade_backtest(
        source_path=args.source,
        signal_variant=args.signal_variant,
        min_known_days=args.min_known_days,
        max_known_days=args.max_known_days,
    )

    args.trades_output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.trades_output, index=False)
    summary.to_csv(args.summary_output, index=False)
    grid.to_csv(args.grid_output, index=False)

    print("Selected Kalshi TSA trade rule:")
    print(
        summary.to_string(
            index=False,
            formatters={
                "min_gap": fmt_count,
                "min_estimated_edge": fmt_pct,
                "min_market_volume": fmt_count,
                "half_spread": fmt_pct,
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

    print("\nTop grid candidates with at least 8 trades:")
    candidates = grid[grid["trades"] >= 8].copy()
    if candidates.empty:
        candidates = grid[grid["trades"] > 0].copy()
    candidates = candidates.sort_values(["total_pnl", "roi"], ascending=False).head(15)
    print(
        candidates[
            [
                "min_gap",
                "min_estimated_edge",
                "min_market_volume",
                "half_spread",
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
                "min_market_volume": fmt_count,
                "half_spread": fmt_pct,
                "win_rate": fmt_pct,
                "total_pnl": fmt_money,
                "roi": fmt_pct,
                "max_drawdown": fmt_money,
            },
        )
    )

    traded = trades[trades["trade"]].copy()
    print("\nExecuted trades:")
    if traded.empty:
        print("No trades passed the selected rule.")
    else:
        print(
            traded[
                [
                    "as_of_date",
                    "week_monday",
                    "known_days",
                    "side",
                    "model_market_gap",
                    "estimated_edge",
                    "market_volume_contracts",
                    "settlement",
                    "pnl",
                ]
            ].to_string(
                index=False,
                formatters={
                    "model_market_gap": fmt_count,
                    "estimated_edge": fmt_pct,
                    "market_volume_contracts": fmt_count,
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
