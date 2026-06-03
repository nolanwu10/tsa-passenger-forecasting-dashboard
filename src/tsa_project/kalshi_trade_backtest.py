from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from tsa_project.config import REPORT_ARTIFACTS_DIR
from tsa_project.config import EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH
from tsa_project.kalshi_blend_backtest import KALSHI_BLEND_BACKTEST_DAILY_PATH


TRADE_BACKTEST_TRADES_PATH = REPORT_ARTIFACTS_DIR / "kalshi_trade_strategy_backtest_trades.csv"
TRADE_BACKTEST_SUMMARY_PATH = REPORT_ARTIFACTS_DIR / "kalshi_trade_strategy_backtest_summary.csv"
TRADE_BACKTEST_GRID_PATH = REPORT_ARTIFACTS_DIR / "kalshi_trade_strategy_grid.csv"
CONTRACT_TRADE_BACKTEST_TRADES_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_contract_trade_strategy_backtest_trades.csv"
)
CONTRACT_TRADE_BACKTEST_SUMMARY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_contract_trade_strategy_backtest_summary.csv"
)
CONTRACT_TRADE_BACKTEST_GRID_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_contract_trade_strategy_grid.csv"
)

DEFAULT_SIGNAL_VARIANT = "weekly_daily_ensemble_no_kalshi"
DEFAULT_EVALUATION_VARIANTS = (
    "weekly_daily_ensemble_no_kalshi",
    "daily_model_plus_kalshi",
    "direct_weekly_plus_kalshi",
    "weekly_daily_ensemble_plus_kalshi",
)


@dataclass(frozen=True)
class TradeRule:
    signal_variant: str = DEFAULT_SIGNAL_VARIANT
    min_gap: float = 40_000.0
    min_estimated_edge: float = 0.05
    min_market_volume: float = 1_000.0
    half_spread: float = 0.02
    contract_count: int = 1
    min_known_days: int = 1
    max_known_days: int = 6


@dataclass(frozen=True)
class ContractTradeRule:
    signal_variant: str = DEFAULT_SIGNAL_VARIANT
    min_gap: float = 20_000.0
    min_estimated_edge: float = 0.05
    min_contract_volume: float = 0.0
    min_open_interest: float = 0.0
    min_entry_price: float = 0.02
    max_entry_price: float = 0.98
    contract_count: int = 1
    min_known_days: int = 1
    max_known_days: int = 6


def load_signal_rows(path: Path = KALSHI_BLEND_BACKTEST_DAILY_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path, parse_dates=["week_monday", "week_sunday", "as_of_date"])
    rows = rows.dropna(
        subset=[
            "variant",
            "model_forecast",
            "market_forecast",
            "actual_weekly_avg",
            "known_days",
        ]
    ).copy()
    rows["known_days"] = rows["known_days"].astype(int)
    rows["market_volume_contracts"] = pd.to_numeric(
        rows.get("market_volume_contracts", 0),
        errors="coerce",
    ).fillna(0.0)
    return rows.sort_values(["as_of_date", "variant"]).reset_index(drop=True)


def load_contract_candles(path: Path = EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH) -> pd.DataFrame:
    candles = pd.read_csv(path, parse_dates=["week_monday", "week_sunday", "as_of_date"])
    numeric_cols = [
        "threshold",
        "yes_bid_close",
        "yes_ask_close",
        "price_close",
        "volume_fp",
        "open_interest_fp",
    ]
    for col in numeric_cols:
        if col in candles.columns:
            candles[col] = pd.to_numeric(candles[col], errors="coerce")
    return candles.dropna(subset=["event_ticker", "ticker", "threshold", "as_of_date"]).copy()


def _prior_sigma(prior: pd.DataFrame, known_days: int, fallback_sigma: float = 50_000.0) -> tuple[float, str]:
    if prior.empty:
        return fallback_sigma, "fallback"
    same_known = prior[prior["known_days"] == known_days]
    source = same_known if len(same_known) >= 8 else prior
    source_name = "known_days" if len(same_known) >= 8 else "global"
    errors = pd.to_numeric(source["model_error"], errors="coerce").dropna()
    sigma = float(errors.std()) if len(errors) >= 8 else float("nan")
    if not math.isfinite(sigma) or sigma < 20_000:
        sigma = fallback_sigma
        source_name = "fallback"
    return sigma, source_name


def _probability_above(threshold: float, forecast: float, sigma: float) -> float:
    if sigma <= 0:
        return 1.0 if forecast > threshold else 0.0
    return 1.0 - NormalDist(mu=forecast, sigma=sigma).cdf(threshold)


def simulate_rule(signal_rows: pd.DataFrame, rule: TradeRule) -> pd.DataFrame:
    rows = signal_rows[signal_rows["variant"] == rule.signal_variant].copy()
    if rows.empty:
        raise ValueError(f"No rows found for signal variant: {rule.signal_variant}")

    simulated: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        known_days = int(row["known_days"])
        model_forecast = float(row["model_forecast"])
        market_forecast = float(row["market_forecast"])
        actual = float(row["actual_weekly_avg"])
        gap = model_forecast - market_forecast
        abs_gap = abs(gap)
        market_volume = float(row.get("market_volume_contracts") or 0.0)
        sigma, sigma_source = _prior_sigma(pd.DataFrame(prior_rows), known_days)
        prob_above = _probability_above(market_forecast, model_forecast, sigma)
        if gap >= 0:
            side = "YES"
            model_probability = prob_above
            settlement = 1.0 if actual > market_forecast else 0.0
        else:
            side = "NO"
            model_probability = 1.0 - prob_above
            settlement = 1.0 if actual <= market_forecast else 0.0

        entry_price = 0.5 + rule.half_spread
        estimated_edge = model_probability - entry_price
        passes_known_days = rule.min_known_days <= known_days <= rule.max_known_days
        passes_gap = abs_gap >= rule.min_gap
        passes_edge = estimated_edge >= rule.min_estimated_edge
        passes_volume = market_volume >= rule.min_market_volume
        should_trade = passes_known_days and passes_gap and passes_edge and passes_volume
        pnl_per_contract = settlement - entry_price if should_trade else 0.0
        cost = entry_price * rule.contract_count if should_trade else 0.0
        pnl = pnl_per_contract * rule.contract_count
        roi = pnl / cost if cost > 0 else np.nan

        simulated.append(
            {
                "event_ticker": row.get("event_ticker"),
                "week_monday": pd.Timestamp(row["week_monday"]).date().isoformat(),
                "week_sunday": pd.Timestamp(row["week_sunday"]).date().isoformat(),
                "as_of_date": pd.Timestamp(row["as_of_date"]).date().isoformat(),
                "known_days": known_days,
                "regime": row.get("regime"),
                "signal_variant": rule.signal_variant,
                "model_forecast": model_forecast,
                "market_forecast": market_forecast,
                "actual_weekly_avg": actual,
                "threshold": market_forecast,
                "model_market_gap": gap,
                "abs_gap": abs_gap,
                "side": side,
                "sigma": sigma,
                "sigma_source": sigma_source,
                "model_probability": model_probability,
                "entry_price": entry_price,
                "estimated_edge": estimated_edge,
                "market_volume_contracts": market_volume,
                "passes_known_days": passes_known_days,
                "passes_gap": passes_gap,
                "passes_edge": passes_edge,
                "passes_volume": passes_volume,
                "trade": should_trade,
                "settlement": settlement if should_trade else np.nan,
                "pnl_per_contract": pnl_per_contract,
                "contracts": rule.contract_count if should_trade else 0,
                "cost": cost,
                "pnl": pnl,
                "roi": roi,
                "winning_trade": bool(settlement == 1.0) if should_trade else np.nan,
            }
        )
        prior_rows.append(
            {
                "known_days": known_days,
                "model_error": model_forecast - actual,
            }
        )
    return pd.DataFrame(simulated)


def summarize_trades(trades: pd.DataFrame, rule: TradeRule) -> dict[str, object]:
    traded = trades[trades["trade"]].copy()
    total_cost = float(traded["cost"].sum()) if not traded.empty else 0.0
    total_pnl = float(traded["pnl"].sum()) if not traded.empty else 0.0
    roi = total_pnl / total_cost if total_cost else np.nan
    win_rate = float(traded["winning_trade"].mean()) if not traded.empty else np.nan
    return {
        "signal_variant": rule.signal_variant,
        "min_gap": rule.min_gap,
        "min_estimated_edge": rule.min_estimated_edge,
        "min_market_volume": rule.min_market_volume,
        "half_spread": rule.half_spread,
        "min_known_days": rule.min_known_days,
        "max_known_days": rule.max_known_days,
        "rows": int(len(trades)),
        "trades": int(len(traded)),
        "trade_rate": float(len(traded) / len(trades)) if len(trades) else np.nan,
        "win_rate": win_rate,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "roi": roi,
        "avg_pnl_per_trade": float(traded["pnl"].mean()) if not traded.empty else np.nan,
        "median_estimated_edge": float(traded["estimated_edge"].median()) if not traded.empty else np.nan,
        "avg_abs_gap": float(traded["abs_gap"].mean()) if not traded.empty else np.nan,
        "max_drawdown": max_drawdown(traded["pnl"].tolist()),
    }


def max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    drawdown = 0.0
    for pnl in pnls:
        equity += float(pnl)
        peak = max(peak, equity)
        drawdown = min(drawdown, equity - peak)
    return float(drawdown)


def grid_search(
    signal_rows: pd.DataFrame,
    signal_variant: str = DEFAULT_SIGNAL_VARIANT,
    min_gaps: tuple[float, ...] = (10_000, 20_000, 30_000, 40_000, 50_000, 75_000, 100_000),
    min_edges: tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.10, 0.15),
    min_volumes: tuple[float, ...] = (0, 1_000, 5_000, 10_000),
    half_spreads: tuple[float, ...] = (0.01, 0.02, 0.03, 0.05),
    min_known_days: int = 1,
    max_known_days: int = 6,
) -> pd.DataFrame:
    summaries = []
    for half_spread in half_spreads:
        for min_gap in min_gaps:
            for min_edge in min_edges:
                for min_volume in min_volumes:
                    rule = TradeRule(
                        signal_variant=signal_variant,
                        min_gap=min_gap,
                        min_estimated_edge=min_edge,
                        min_market_volume=min_volume,
                        half_spread=half_spread,
                        min_known_days=min_known_days,
                        max_known_days=max_known_days,
                    )
                    trades = simulate_rule(signal_rows, rule)
                    summaries.append(summarize_trades(trades, rule))
    return pd.DataFrame(summaries)


def pick_candidate_strategy(
    grid: pd.DataFrame,
    min_trades: int = 8,
    min_gap: float = 20_000.0,
    min_estimated_edge: float = 0.05,
    min_market_volume: float = 1_000.0,
    half_spread: float = 0.02,
) -> pd.Series:
    candidates = grid[
        (grid["trades"] >= min_trades)
        & (grid["min_gap"] >= min_gap)
        & (grid["min_estimated_edge"] >= min_estimated_edge)
        & (grid["min_market_volume"] >= min_market_volume)
        & (grid["half_spread"] == half_spread)
    ].copy()
    if candidates.empty:
        candidates = grid[
            (grid["trades"] > 0)
            & (grid["min_gap"] >= min_gap)
            & (grid["min_estimated_edge"] >= min_estimated_edge)
            & (grid["min_market_volume"] >= min_market_volume)
        ].copy()
    if candidates.empty:
        raise ValueError("No candidate strategy produced any trades.")
    candidates["score"] = (
        candidates["total_pnl"].fillna(-999)
        + candidates["roi"].fillna(-999)
        - candidates["max_drawdown"].abs().fillna(0) * 0.05
    )
    return candidates.sort_values(["score", "total_pnl", "roi"], ascending=False).iloc[0]


def run_trade_backtest(
    source_path: Path = KALSHI_BLEND_BACKTEST_DAILY_PATH,
    signal_variant: str = DEFAULT_SIGNAL_VARIANT,
    min_known_days: int = 1,
    max_known_days: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows = load_signal_rows(source_path)
    grid = grid_search(
        signal_rows,
        signal_variant=signal_variant,
        min_known_days=min_known_days,
        max_known_days=max_known_days,
    )
    candidate = pick_candidate_strategy(grid)
    rule = TradeRule(
        signal_variant=signal_variant,
        min_gap=float(candidate["min_gap"]),
        min_estimated_edge=float(candidate["min_estimated_edge"]),
        min_market_volume=float(candidate["min_market_volume"]),
        half_spread=float(candidate["half_spread"]),
        min_known_days=min_known_days,
        max_known_days=max_known_days,
    )
    trades = simulate_rule(signal_rows, rule)
    summary = pd.DataFrame([summarize_trades(trades, rule)])
    return trades, summary, grid


def _market_candle_date(row: pd.Series) -> pd.Timestamp:
    value = row.get("market_candle_end_et")
    if pd.isna(value):
        return pd.Timestamp(row["as_of_date"]).normalize()
    return pd.Timestamp(value).normalize().tz_localize(None)


def _contract_opportunities(
    row: pd.Series,
    candles_for_event_date: pd.DataFrame,
    sigma: float,
    rule: ContractTradeRule,
) -> list[dict[str, object]]:
    model_forecast = float(row["model_forecast"])
    actual = float(row["actual_weekly_avg"])
    opportunities: list[dict[str, object]] = []
    for _, candle in candles_for_event_date.iterrows():
        threshold = float(candle["threshold"])
        yes_ask = float(candle["yes_ask_close"]) if pd.notna(candle.get("yes_ask_close")) else np.nan
        yes_bid = float(candle["yes_bid_close"]) if pd.notna(candle.get("yes_bid_close")) else np.nan
        prob_yes = _probability_above(threshold, model_forecast, sigma)
        if math.isfinite(yes_ask):
            opportunities.append(
                {
                    "ticker": candle["ticker"],
                    "threshold": threshold,
                    "side": "YES",
                    "entry_price": yes_ask,
                    "model_probability": prob_yes,
                    "estimated_edge": prob_yes - yes_ask,
                    "settlement": 1.0 if actual > threshold else 0.0,
                    "contract_volume": float(candle.get("volume_fp") or 0.0),
                    "open_interest": float(candle.get("open_interest_fp") or 0.0),
                }
            )
        no_ask = 1.0 - yes_bid if math.isfinite(yes_bid) else np.nan
        if math.isfinite(no_ask):
            prob_no = 1.0 - prob_yes
            opportunities.append(
                {
                    "ticker": candle["ticker"],
                    "threshold": threshold,
                    "side": "NO",
                    "entry_price": no_ask,
                    "model_probability": prob_no,
                    "estimated_edge": prob_no - no_ask,
                    "settlement": 1.0 if actual <= threshold else 0.0,
                    "contract_volume": float(candle.get("volume_fp") or 0.0),
                    "open_interest": float(candle.get("open_interest_fp") or 0.0),
                }
            )
    return [
        item
        for item in opportunities
        if rule.min_entry_price <= float(item["entry_price"]) <= rule.max_entry_price
        and float(item["contract_volume"]) >= rule.min_contract_volume
        and float(item["open_interest"]) >= rule.min_open_interest
    ]


def simulate_contract_rule(
    signal_rows: pd.DataFrame,
    candles: pd.DataFrame,
    rule: ContractTradeRule,
) -> pd.DataFrame:
    rows = signal_rows[signal_rows["variant"] == rule.signal_variant].copy()
    if rows.empty:
        raise ValueError(f"No rows found for signal variant: {rule.signal_variant}")
    candles = candles.copy()
    candles["join_date"] = pd.to_datetime(candles["as_of_date"]).dt.date
    candle_groups = {
        key: group
        for key, group in candles.groupby(["event_ticker", "join_date"], dropna=False)
    }

    simulated: list[dict[str, object]] = []
    prior_rows: list[dict[str, object]] = []
    for _, row in rows.iterrows():
        known_days = int(row["known_days"])
        gap = float(row["model_forecast"]) - float(row["market_forecast"])
        market_candle_date = _market_candle_date(row).date()
        candles_for_date = candle_groups.get((row["event_ticker"], market_candle_date), pd.DataFrame())
        sigma, sigma_source = _prior_sigma(pd.DataFrame(prior_rows), known_days)
        opportunities = _contract_opportunities(row, candles_for_date, sigma, rule)
        best = max(opportunities, key=lambda item: item["estimated_edge"], default=None)
        passes_known_days = rule.min_known_days <= known_days <= rule.max_known_days
        passes_gap = abs(gap) >= rule.min_gap
        passes_edge = bool(best and float(best["estimated_edge"]) >= rule.min_estimated_edge)
        should_trade = bool(best and passes_known_days and passes_gap and passes_edge)
        if best:
            entry_price = float(best["entry_price"])
            settlement = float(best["settlement"])
            pnl_per_contract = settlement - entry_price if should_trade else 0.0
            cost = entry_price * rule.contract_count if should_trade else 0.0
            pnl = pnl_per_contract * rule.contract_count
            roi = pnl / cost if cost > 0 else np.nan
        else:
            entry_price = np.nan
            settlement = np.nan
            pnl_per_contract = 0.0
            cost = 0.0
            pnl = 0.0
            roi = np.nan
        simulated.append(
            {
                "event_ticker": row.get("event_ticker"),
                "week_monday": pd.Timestamp(row["week_monday"]).date().isoformat(),
                "week_sunday": pd.Timestamp(row["week_sunday"]).date().isoformat(),
                "as_of_date": pd.Timestamp(row["as_of_date"]).date().isoformat(),
                "market_candle_date": market_candle_date.isoformat(),
                "known_days": known_days,
                "regime": row.get("regime"),
                "signal_variant": rule.signal_variant,
                "model_forecast": float(row["model_forecast"]),
                "market_forecast": float(row["market_forecast"]),
                "actual_weekly_avg": float(row["actual_weekly_avg"]),
                "model_market_gap": gap,
                "abs_gap": abs(gap),
                "sigma": sigma,
                "sigma_source": sigma_source,
                "ticker": best.get("ticker") if best else None,
                "threshold": best.get("threshold") if best else np.nan,
                "side": best.get("side") if best else None,
                "entry_price": entry_price,
                "model_probability": best.get("model_probability") if best else np.nan,
                "estimated_edge": best.get("estimated_edge") if best else np.nan,
                "contract_volume": best.get("contract_volume") if best else np.nan,
                "open_interest": best.get("open_interest") if best else np.nan,
                "candidate_contracts": len(opportunities),
                "passes_known_days": passes_known_days,
                "passes_gap": passes_gap,
                "passes_edge": passes_edge,
                "trade": should_trade,
                "settlement": settlement if should_trade else np.nan,
                "pnl_per_contract": pnl_per_contract,
                "contracts": rule.contract_count if should_trade else 0,
                "cost": cost,
                "pnl": pnl,
                "roi": roi,
                "winning_trade": bool(settlement == 1.0) if should_trade else np.nan,
            }
        )
        prior_rows.append(
            {
                "known_days": known_days,
                "model_error": float(row["model_forecast"]) - float(row["actual_weekly_avg"]),
            }
        )
    return pd.DataFrame(simulated)


def summarize_contract_trades(trades: pd.DataFrame, rule: ContractTradeRule) -> dict[str, object]:
    traded = trades[trades["trade"]].copy()
    total_cost = float(traded["cost"].sum()) if not traded.empty else 0.0
    total_pnl = float(traded["pnl"].sum()) if not traded.empty else 0.0
    return {
        "mode": "contract_candlesticks",
        "signal_variant": rule.signal_variant,
        "min_gap": rule.min_gap,
        "min_estimated_edge": rule.min_estimated_edge,
        "min_contract_volume": rule.min_contract_volume,
        "min_open_interest": rule.min_open_interest,
        "min_entry_price": rule.min_entry_price,
        "max_entry_price": rule.max_entry_price,
        "min_known_days": rule.min_known_days,
        "max_known_days": rule.max_known_days,
        "rows": int(len(trades)),
        "rows_with_contract_candidates": int((trades["candidate_contracts"] > 0).sum()),
        "trades": int(len(traded)),
        "trade_rate": float(len(traded) / len(trades)) if len(trades) else np.nan,
        "win_rate": float(traded["winning_trade"].mean()) if not traded.empty else np.nan,
        "total_cost": total_cost,
        "total_pnl": total_pnl,
        "roi": total_pnl / total_cost if total_cost else np.nan,
        "avg_pnl_per_trade": float(traded["pnl"].mean()) if not traded.empty else np.nan,
        "median_estimated_edge": float(traded["estimated_edge"].median()) if not traded.empty else np.nan,
        "avg_abs_gap": float(traded["abs_gap"].mean()) if not traded.empty else np.nan,
        "max_drawdown": max_drawdown(traded["pnl"].tolist()),
    }


def contract_grid_search(
    signal_rows: pd.DataFrame,
    candles: pd.DataFrame,
    signal_variant: str = DEFAULT_SIGNAL_VARIANT,
    min_gaps: tuple[float, ...] = (10_000, 20_000, 30_000, 40_000, 50_000),
    min_edges: tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.10, 0.15),
    min_contract_volumes: tuple[float, ...] = (0, 1, 10, 100),
    max_entry_prices: tuple[float, ...] = (0.90, 0.95, 0.98),
    min_known_days: int = 1,
    max_known_days: int = 6,
) -> pd.DataFrame:
    summaries = []
    for min_gap in min_gaps:
        for min_edge in min_edges:
            for min_volume in min_contract_volumes:
                for max_entry_price in max_entry_prices:
                    rule = ContractTradeRule(
                        signal_variant=signal_variant,
                        min_gap=min_gap,
                        min_estimated_edge=min_edge,
                        min_contract_volume=min_volume,
                        max_entry_price=max_entry_price,
                        min_known_days=min_known_days,
                        max_known_days=max_known_days,
                    )
                    trades = simulate_contract_rule(signal_rows, candles, rule)
                    summaries.append(summarize_contract_trades(trades, rule))
    return pd.DataFrame(summaries)


def pick_contract_candidate_strategy(grid: pd.DataFrame, min_trades: int = 5) -> pd.Series:
    candidates = grid[
        (grid["trades"] >= min_trades)
        & (grid["min_gap"] >= 20_000)
        & (grid["min_estimated_edge"] >= 0.05)
        & (grid["min_contract_volume"] >= 1)
        & (grid["max_entry_price"] <= 0.95)
    ].copy()
    if candidates.empty:
        candidates = grid[grid["trades"] > 0].copy()
    if candidates.empty:
        raise ValueError("No contract-level strategy produced any trades.")
    candidates["score"] = (
        candidates["total_pnl"].fillna(-999)
        + candidates["roi"].fillna(-999)
        - candidates["max_drawdown"].abs().fillna(0) * 0.05
    )
    return candidates.sort_values(["score", "total_pnl", "roi"], ascending=False).iloc[0]


def run_contract_trade_backtest(
    source_path: Path = KALSHI_BLEND_BACKTEST_DAILY_PATH,
    candles_path: Path = EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH,
    signal_variant: str = DEFAULT_SIGNAL_VARIANT,
    min_known_days: int = 1,
    max_known_days: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_rows = load_signal_rows(source_path)
    candles = load_contract_candles(candles_path)
    grid = contract_grid_search(
        signal_rows,
        candles,
        signal_variant=signal_variant,
        min_known_days=min_known_days,
        max_known_days=max_known_days,
    )
    candidate = pick_contract_candidate_strategy(grid)
    rule = ContractTradeRule(
        signal_variant=signal_variant,
        min_gap=float(candidate["min_gap"]),
        min_estimated_edge=float(candidate["min_estimated_edge"]),
        min_contract_volume=float(candidate["min_contract_volume"]),
        min_open_interest=float(candidate["min_open_interest"]),
        min_entry_price=float(candidate["min_entry_price"]),
        max_entry_price=float(candidate["max_entry_price"]),
        min_known_days=min_known_days,
        max_known_days=max_known_days,
    )
    trades = simulate_contract_rule(signal_rows, candles, rule)
    summary = pd.DataFrame([summarize_contract_trades(trades, rule)])
    return trades, summary, grid
