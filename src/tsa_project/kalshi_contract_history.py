from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from tsa_project.config import (
    EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH,
    EXTERNAL_KALSHI_TSA_MARKETS_PATH,
    REPORT_ARTIFACTS_DIR,
)
from tsa_project.kalshi import TSA_WEEKLY_SERIES, kalshi_get
from tsa_project.kalshi_blend_backtest import KALSHI_YTD_BACKTEST_PATH


CONTRACT_HISTORY_COVERAGE_PATH = REPORT_ARTIFACTS_DIR / "kalshi_contract_history_coverage.csv"
PREWEEK_CONTRACT_SIGNAL_ROWS_PATH = REPORT_ARTIFACTS_DIR / "kalshi_preweek_contract_signal_rows.csv"


@dataclass(frozen=True)
class ContractHistoryConfig:
    period_interval: int = 1440
    days_before_week: int = 3
    days_after_week: int = 1
    max_events: int | None = None


def _to_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nested_value(row: dict[str, object], key: str, field: str) -> float | None:
    value = row.get(key)
    if isinstance(value, dict):
        return _to_float(value.get(field))
    return None


def _timestamp(date: pd.Timestamp) -> int:
    return int(pd.Timestamp(date).tz_localize("America/New_York").tz_convert("UTC").timestamp())


def event_tickers_from_backtest(path: Path = KALSHI_YTD_BACKTEST_PATH) -> pd.DataFrame:
    rows = pd.read_csv(path, parse_dates=["week_monday", "week_sunday", "as_of_date"])
    events = (
        rows.groupby(["event_ticker", "week_monday", "week_sunday"], as_index=False)
        .agg(first_as_of_date=("as_of_date", "min"), last_as_of_date=("as_of_date", "max"))
        .sort_values("week_monday")
    )
    return events


def fetch_event_markets(event_ticker: str, include_historical: bool = True) -> list[dict[str, object]]:
    markets: list[dict[str, object]] = []
    for endpoint in ("/trade-api/v2/markets", "/trade-api/v2/historical/markets"):
        if endpoint.endswith("historical/markets") and not include_historical:
            continue
        cursor: str | None = None
        for _ in range(10):
            params: dict[str, object] = {"event_ticker": event_ticker, "limit": 1000}
            if not endpoint.endswith("historical/markets"):
                params["series_ticker"] = TSA_WEEKLY_SERIES
            if cursor:
                params["cursor"] = cursor
            try:
                payload = kalshi_get(endpoint, params)
            except Exception:
                if endpoint.endswith("historical/markets"):
                    break
                raise
            page = payload.get("markets", [])
            if isinstance(page, list):
                for market in page:
                    if market.get("event_ticker") == event_ticker:
                        markets.append(market)
            cursor_value = payload.get("cursor")
            if not cursor_value:
                break
            cursor = str(cursor_value)
    deduped = {str(market.get("ticker")): market for market in markets if market.get("ticker")}
    return sorted(deduped.values(), key=lambda item: float(item.get("floor_strike") or 0))


def market_row(event: pd.Series, market: dict[str, object]) -> dict[str, object]:
    return {
        "event_ticker": event["event_ticker"],
        "week_monday": pd.Timestamp(event["week_monday"]).date().isoformat(),
        "week_sunday": pd.Timestamp(event["week_sunday"]).date().isoformat(),
        "ticker": market.get("ticker"),
        "market_type": market.get("market_type"),
        "title": market.get("title"),
        "subtitle": market.get("subtitle"),
        "status": market.get("status"),
        "floor_strike": _to_float(market.get("floor_strike")),
        "cap_strike": _to_float(market.get("cap_strike")),
        "yes_bid_dollars": _to_float(market.get("yes_bid_dollars")),
        "yes_ask_dollars": _to_float(market.get("yes_ask_dollars")),
        "last_price_dollars": _to_float(market.get("last_price_dollars")),
        "volume_fp": _to_float(market.get("volume_fp")),
        "open_interest_fp": _to_float(market.get("open_interest_fp")),
        "close_time": market.get("close_time"),
        "expiration_time": market.get("expiration_time"),
        "settlement_value": market.get("settlement_value"),
        "result": market.get("result"),
    }


def fetch_candlesticks(
    ticker: str,
    week_monday: pd.Timestamp,
    config: ContractHistoryConfig,
) -> list[dict[str, object]]:
    start = pd.Timestamp(week_monday).normalize() - pd.Timedelta(days=config.days_before_week)
    end = pd.Timestamp(week_monday).normalize() + pd.Timedelta(days=6 + config.days_after_week)
    params = {
        "start_ts": _timestamp(start),
        "end_ts": _timestamp(end + pd.Timedelta(days=1)),
        "period_interval": config.period_interval,
        "include_latest_before_start": "true",
    }
    endpoints = [
        f"/trade-api/v2/series/{TSA_WEEKLY_SERIES}/markets/{ticker}/candlesticks",
        f"/trade-api/v2/historical/markets/{ticker}/candlesticks",
    ]
    last_error: Exception | None = None
    for endpoint in endpoints:
        try:
            payload = kalshi_get(endpoint, params)
        except Exception as exc:
            last_error = exc
            continue
        candles = payload.get("candlesticks", [])
        if isinstance(candles, list):
            return candles
    if last_error is not None:
        raise last_error
    return []


def candle_rows(
    event: pd.Series,
    market: dict[str, object],
    candles: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    ticker = str(market.get("ticker"))
    threshold = _to_float(market.get("floor_strike"))
    for candle in candles:
        end_ts = candle.get("end_period_ts")
        end_period = (
            pd.Timestamp(int(end_ts), unit="s", tz="UTC").tz_convert("America/New_York")
            if end_ts is not None
            else pd.NaT
        )
        rows.append(
            {
                "event_ticker": event["event_ticker"],
                "week_monday": pd.Timestamp(event["week_monday"]).date().isoformat(),
                "week_sunday": pd.Timestamp(event["week_sunday"]).date().isoformat(),
                "ticker": ticker,
                "threshold": threshold,
                "end_period_ts": end_ts,
                "end_period_et": end_period.isoformat() if pd.notna(end_period) else None,
                "as_of_date": end_period.date().isoformat() if pd.notna(end_period) else None,
                "yes_bid_open": _nested_value(candle, "yes_bid", "open_dollars"),
                "yes_bid_low": _nested_value(candle, "yes_bid", "low_dollars"),
                "yes_bid_high": _nested_value(candle, "yes_bid", "high_dollars"),
                "yes_bid_close": _nested_value(candle, "yes_bid", "close_dollars"),
                "yes_ask_open": _nested_value(candle, "yes_ask", "open_dollars"),
                "yes_ask_low": _nested_value(candle, "yes_ask", "low_dollars"),
                "yes_ask_high": _nested_value(candle, "yes_ask", "high_dollars"),
                "yes_ask_close": _nested_value(candle, "yes_ask", "close_dollars"),
                "price_open": _nested_value(candle, "price", "open_dollars"),
                "price_low": _nested_value(candle, "price", "low_dollars"),
                "price_high": _nested_value(candle, "price", "high_dollars"),
                "price_close": _nested_value(candle, "price", "close_dollars"),
                "price_mean": _nested_value(candle, "price", "mean_dollars"),
                "price_previous": _nested_value(candle, "price", "previous_dollars"),
                "volume_fp": _to_float(candle.get("volume_fp")),
                "open_interest_fp": _to_float(candle.get("open_interest_fp")),
            }
        )
    return rows


def infer_crossing_from_candles(candles: pd.DataFrame) -> float | None:
    points = []
    for _, row in candles.iterrows():
        threshold = _to_float(row.get("threshold"))
        bid = _to_float(row.get("yes_bid_close"))
        ask = _to_float(row.get("yes_ask_close"))
        if threshold is None or bid is None or ask is None or ask < bid:
            continue
        points.append((threshold, max(0.0, min(1.0, (bid + ask) / 2.0))))
    if len(points) < 2:
        return None
    points = sorted(points)
    monotonic_points = []
    current_probability = 1.0
    for threshold, probability in points:
        current_probability = min(current_probability, probability)
        monotonic_points.append((threshold, current_probability))
    if monotonic_points[0][1] <= 0.5:
        return monotonic_points[0][0]
    for (low_threshold, low_probability), (high_threshold, high_probability) in zip(
        monotonic_points,
        monotonic_points[1:],
    ):
        if low_probability >= 0.5 and high_probability <= 0.5:
            if low_probability == high_probability:
                return (low_threshold + high_threshold) / 2.0
            return low_threshold + (
                (0.5 - low_probability)
                * (high_threshold - low_threshold)
                / (high_probability - low_probability)
            )
    return monotonic_points[-1][0]


def build_preweek_contract_signal_rows(
    candles_path: Path = EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH,
    weekly_ensemble_path: Path = REPORT_ARTIFACTS_DIR / "live_weekly_ensemble_backtest_weekly.csv",
) -> pd.DataFrame:
    output_columns = [
        "event_ticker",
        "week_monday",
        "week_sunday",
        "as_of_date",
        "market_candle_end_et",
        "known_days",
        "regime",
        "variant",
        "model_forecast",
        "market_forecast",
        "actual_weekly_avg",
        "model_error",
        "market_error",
        "model_market_gap",
    ]
    if not candles_path.exists() or not weekly_ensemble_path.exists():
        rows = pd.DataFrame(columns=output_columns)
        PREWEEK_CONTRACT_SIGNAL_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(PREWEEK_CONTRACT_SIGNAL_ROWS_PATH, index=False)
        return rows

    candles = pd.read_csv(candles_path, parse_dates=["week_monday", "week_sunday", "as_of_date"])
    weekly = pd.read_csv(weekly_ensemble_path, parse_dates=["week_monday"])
    weekly = weekly[weekly["known_days"] == 0].copy()
    if candles.empty or weekly.empty:
        rows = pd.DataFrame(columns=output_columns)
        PREWEEK_CONTRACT_SIGNAL_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(PREWEEK_CONTRACT_SIGNAL_ROWS_PATH, index=False)
        return rows

    preweek = candles[candles["as_of_date"] < candles["week_monday"]].copy()
    rows_out = []
    for (event_ticker, week_monday, week_sunday, as_of_date), group in preweek.groupby(
        ["event_ticker", "week_monday", "week_sunday", "as_of_date"],
        dropna=False,
    ):
        market_forecast = infer_crossing_from_candles(group)
        if market_forecast is None:
            continue
        model_match = weekly[weekly["week_monday"] == pd.Timestamp(week_monday)]
        if model_match.empty:
            continue
        model = model_match.iloc[0]
        model_forecast = float(model["ensemble_avg"])
        actual = float(model["weekly_actual_avg"])
        rows_out.append(
            {
                "event_ticker": event_ticker,
                "week_monday": pd.Timestamp(week_monday).date().isoformat(),
                "week_sunday": pd.Timestamp(week_sunday).date().isoformat(),
                "as_of_date": pd.Timestamp(as_of_date).date().isoformat(),
                "market_candle_end_et": pd.Timestamp(as_of_date).isoformat(),
                "known_days": 0,
                "regime": model.get("regime"),
                "variant": "weekly_daily_ensemble_no_kalshi",
                "model_forecast": model_forecast,
                "market_forecast": market_forecast,
                "actual_weekly_avg": actual,
                "model_error": model_forecast - actual,
                "market_error": market_forecast - actual,
                "model_market_gap": model_forecast - market_forecast,
            }
        )
    rows = pd.DataFrame(rows_out, columns=output_columns)
    PREWEEK_CONTRACT_SIGNAL_ROWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(PREWEEK_CONTRACT_SIGNAL_ROWS_PATH, index=False)
    return rows


def fetch_contract_history(
    config: ContractHistoryConfig = ContractHistoryConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    events = event_tickers_from_backtest()
    if config.max_events is not None:
        events = events.tail(config.max_events)

    market_rows_out: list[dict[str, object]] = []
    candle_rows_out: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for _, event in events.iterrows():
        markets = fetch_event_markets(str(event["event_ticker"]))
        for market in markets:
            market_rows_out.append(market_row(event, market))
            ticker = str(market.get("ticker"))
            try:
                candles = fetch_candlesticks(ticker, pd.Timestamp(event["week_monday"]), config)
            except Exception as exc:
                failures.append(
                    {
                        "event_ticker": event["event_ticker"],
                        "ticker": ticker,
                        "error": str(exc),
                    }
                )
                continue
            candle_rows_out.extend(candle_rows(event, market, candles))

    markets_df = pd.DataFrame(market_rows_out)
    candles_df = pd.DataFrame(candle_rows_out)
    EXTERNAL_KALSHI_TSA_MARKETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    markets_df.to_csv(EXTERNAL_KALSHI_TSA_MARKETS_PATH, index=False)
    candles_df.to_csv(EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH, index=False)

    coverage = pd.DataFrame(
        [
            {
                "events_requested": int(len(events)),
                "markets": int(len(markets_df)),
                "candlestick_rows": int(len(candles_df)),
                "failures": int(len(failures)),
                "period_interval": config.period_interval,
                "days_before_week": config.days_before_week,
                "days_after_week": config.days_after_week,
            }
        ]
    )
    coverage.to_csv(CONTRACT_HISTORY_COVERAGE_PATH, index=False)
    return markets_df, candles_df
