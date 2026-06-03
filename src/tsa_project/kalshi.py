from __future__ import annotations

import base64
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from urllib.parse import urlencode

import joblib
import pandas as pd
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from tsa_project.config import PROJECT_ROOT
from tsa_project.config import REPORT_ARTIFACTS_DIR
from tsa_project.live_weekly_model import MODEL_PATH, WEEKLY_REPORT_PATH


DEFAULT_BASE_URL = "https://external-api.kalshi.com"
TSA_WEEKLY_SERIES = "KXTSAW"
EDGE_THRESHOLD = 0.10
ORDERBOOK_DEPTH = 25
BLEND_NEUTRAL_GAP = 10_000.0
BLEND_BULLISH_MODEL_WEIGHT = 0.35
BLEND_BEARISH_MODEL_WEIGHT = 0.55
BLEND_NEUTRAL_MODEL_WEIGHT = 0.50
KALSHI_BLEND_BACKTEST_SUMMARY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_dashboard_ensemble_blend_backtest_summary.csv"
)
KALSHI_TRADE_STRATEGY_SUMMARY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_trade_strategy_backtest_summary.csv"
)
KALSHI_CONTRACT_TRADE_STRATEGY_SUMMARY_PATH = (
    REPORT_ARTIFACTS_DIR / "kalshi_contract_trade_strategy_backtest_summary.csv"
)


@dataclass(frozen=True)
class KalshiConfig:
    key_id: str | None
    private_key_path: Path | None
    base_url: str


def load_dotenv_file(path: Path = PROJECT_ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def get_config() -> KalshiConfig:
    load_dotenv_file()
    key_id = os.environ.get("KALSHI_KEY_ID")
    private_key = os.environ.get("KALSHI_PRIVATE_KEY_PATH")
    host = os.environ.get("KALSHI_HOST", DEFAULT_BASE_URL).rstrip("/")
    if host.endswith("/trade-api/v2"):
        host = host.removesuffix("/trade-api/v2")
    return KalshiConfig(
        key_id=key_id,
        private_key_path=Path(private_key) if private_key else None,
        base_url=host,
    )


def configured(config: KalshiConfig | None = None) -> bool:
    config = config or get_config()
    return bool(config.key_id and config.private_key_path and config.private_key_path.exists())


def _load_private_key(path: Path):
    return serialization.load_pem_private_key(path.read_bytes(), password=None)


def _signed_headers(config: KalshiConfig, method: str, path: str) -> dict[str, str]:
    if not config.key_id or not config.private_key_path:
        raise ValueError("Kalshi API key id or private key path is missing.")
    timestamp_ms = str(int(time.time() * 1000))
    signing_path = path.split("?", 1)[0]
    message = f"{timestamp_ms}{method.upper()}{signing_path}".encode("utf-8")
    private_key = _load_private_key(config.private_key_path)
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH,
        ),
        hashes.SHA256(),
    )
    return {
        "KALSHI-ACCESS-KEY": config.key_id,
        "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
        "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
    }


def kalshi_get(path: str, params: dict[str, object] | None = None) -> dict[str, object]:
    config = get_config()
    if not configured(config):
        raise RuntimeError("Kalshi is not configured. Set KALSHI_KEY_ID and KALSHI_PRIVATE_KEY_PATH.")

    query = f"?{urlencode(params)}" if params else ""
    headers = _signed_headers(config, "GET", path)
    response = requests.get(f"{config.base_url}{path}{query}", headers=headers, timeout=20)
    response.raise_for_status()
    return response.json()


def event_ticker_for_week(week_sunday: str | pd.Timestamp) -> str:
    date = pd.Timestamp(week_sunday)
    return f"{TSA_WEEKLY_SERIES}-{date.strftime('%y%b%d').upper()}"


def fetch_tsa_weekly_markets(week_sunday: str | pd.Timestamp) -> list[dict[str, object]]:
    event_ticker = event_ticker_for_week(week_sunday)
    cursor: str | None = None
    markets: list[dict[str, object]] = []
    for _ in range(5):
        params: dict[str, object] = {
            "series_ticker": TSA_WEEKLY_SERIES,
            "status": "open",
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        payload = kalshi_get("/trade-api/v2/markets", params)
        page = payload.get("markets", [])
        if isinstance(page, list):
            markets.extend(
                market for market in page if market.get("event_ticker") == event_ticker
            )
        cursor_value = payload.get("cursor")
        if not cursor_value:
            break
        cursor = str(cursor_value)
    return sorted(markets, key=lambda item: float(item.get("floor_strike") or 0), reverse=True)


def fetch_event_median_forecast(week_sunday: str | pd.Timestamp) -> dict[str, object] | None:
    event_ticker = event_ticker_for_week(week_sunday)
    sunday = pd.Timestamp(week_sunday).normalize()
    monday = sunday - pd.Timedelta(days=6)
    start = monday - pd.Timedelta(days=2)
    start_ts = int(start.tz_localize("America/New_York").tz_convert("UTC").timestamp())
    end_ts = int(pd.Timestamp.utcnow().ceil("D").timestamp())
    payload = kalshi_get(
        f"/trade-api/v2/series/{TSA_WEEKLY_SERIES}/events/{event_ticker}/forecast_percentile_history",
        {
            "percentiles": 5000,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": 1440,
        },
    )
    history = payload.get("forecast_history")
    if not isinstance(history, list) or not history:
        return None

    latest = max(history, key=lambda row: int(row.get("end_period_ts") or 0))
    points = latest.get("percentile_points")
    if not isinstance(points, list) or not points:
        return None
    point = points[0]
    formatted = point.get("formatted_forecast")
    try:
        forecast = float(formatted)
    except (TypeError, ValueError):
        raw = point.get("raw_numerical_forecast") or point.get("numerical_forecast")
        try:
            forecast = float(raw) / 1_000_000.0
        except (TypeError, ValueError):
            return None

    end_period_ts = latest.get("end_period_ts")
    end_period = (
        pd.Timestamp(int(end_period_ts), unit="s", tz="UTC").tz_convert("America/New_York").isoformat()
        if end_period_ts
        else None
    )
    return {
        "event_ticker": event_ticker,
        "market_median_forecast": forecast,
        "forecast_as_of": end_period,
        "source": "kalshi_forecast_percentile_history_p50",
    }


def adaptive_blend(
    model_average: float,
    market_average: float | None,
) -> dict[str, object]:
    if market_average is None or not math.isfinite(market_average):
        return {
            "available": False,
            "adjusted_weekly_average": model_average,
            "model_weight": 1.0,
            "market_weight": 0.0,
            "direction": "model_only",
            "gap": None,
            "rationale": "No live Kalshi-implied average was available, so the adjusted value uses the raw model.",
        }

    gap = model_average - market_average
    if abs(gap) <= BLEND_NEUTRAL_GAP:
        model_weight = BLEND_NEUTRAL_MODEL_WEIGHT
        direction = "neutral"
        rationale = "Model and market are close; the adjusted value uses a balanced blend."
    elif gap > 0:
        model_weight = BLEND_BULLISH_MODEL_WEIGHT
        direction = "model_bullish"
        rationale = (
            "Model is above Kalshi; YTD backtest showed bullish model disagreements were less reliable, "
            "so the blend leans toward the market."
        )
    else:
        model_weight = BLEND_BEARISH_MODEL_WEIGHT
        direction = "model_bearish"
        rationale = (
            "Model is below Kalshi; YTD backtest showed bearish model disagreements were more reliable, "
            "so the blend gives the model slightly more weight."
        )

    market_weight = 1.0 - model_weight
    adjusted = (model_weight * model_average) + (market_weight * market_average)
    return {
        "available": True,
        "adjusted_weekly_average": adjusted,
        "model_weight": model_weight,
        "market_weight": market_weight,
        "direction": direction,
        "gap": gap,
        "rationale": rationale,
    }


def infer_market_implied_average(markets: list[dict[str, object]]) -> dict[str, object] | None:
    points = []
    for market in markets:
        threshold = dollars_to_probability(market.get("floor_strike"))
        probability = midpoint_probability(market)
        if threshold is None or probability is None:
            continue
        if not math.isfinite(threshold) or not math.isfinite(probability):
            continue
        points.append((threshold, max(0.0, min(1.0, probability))))

    if len(points) < 2:
        return None

    points = sorted(points)
    monotonic_points = []
    current_probability = 1.0
    for threshold, probability in points:
        current_probability = min(current_probability, probability)
        monotonic_points.append((threshold, current_probability))

    forecast = None
    if monotonic_points[0][1] <= 0.5:
        forecast = monotonic_points[0][0]
    else:
        for (low_threshold, low_probability), (high_threshold, high_probability) in zip(
            monotonic_points,
            monotonic_points[1:],
        ):
            if low_probability >= 0.5 and high_probability <= 0.5:
                if low_probability == high_probability:
                    forecast = (low_threshold + high_threshold) / 2.0
                else:
                    forecast = low_threshold + (
                        (0.5 - low_probability)
                        * (high_threshold - low_threshold)
                        / (high_probability - low_probability)
                    )
                break
    if forecast is None:
        forecast = monotonic_points[-1][0]

    min_threshold, min_probability = monotonic_points[0]
    max_threshold, max_probability = monotonic_points[-1]
    return {
        "market_implied_average": forecast,
        "source": "live_contract_bid_ask_midpoint_50pct_crossing",
        "contract_count": len(monotonic_points),
        "min_threshold": min_threshold,
        "min_threshold_probability": min_probability,
        "max_threshold": max_threshold,
        "max_threshold_probability": max_probability,
        "method_note": (
            "Forecast is interpolated from the live threshold contract surface where "
            "P(weekly average > threshold) crosses 50%."
        ),
    }


def fetch_orderbook(ticker: str, depth: int = ORDERBOOK_DEPTH) -> dict[str, object]:
    payload = kalshi_get(
        f"/trade-api/v2/markets/{ticker}/orderbook",
        {"depth": depth},
    )
    orderbook = payload.get("orderbook_fp", {})
    return orderbook if isinstance(orderbook, dict) else {}


def dollars_to_probability(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def midpoint_probability(market: dict[str, object]) -> float | None:
    bid = dollars_to_probability(market.get("yes_bid_dollars"))
    ask = dollars_to_probability(market.get("yes_ask_dollars"))
    last = dollars_to_probability(market.get("last_price_dollars"))
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    return last


def parse_orderbook_levels(levels: object) -> list[dict[str, float]]:
    parsed = []
    if not isinstance(levels, list):
        return parsed
    for level in levels:
        if not isinstance(level, list) or len(level) < 2:
            continue
        try:
            parsed.append({"price": float(level[0]), "contracts": float(level[1])})
        except (TypeError, ValueError):
            continue
    return sorted(parsed, key=lambda row: row["price"])


def bids_to_opposite_asks(bids: list[dict[str, float]]) -> list[dict[str, float]]:
    asks = [
        {"price": 1.0 - bid["price"], "contracts": bid["contracts"]}
        for bid in bids
        if bid["price"] < 1.0
    ]
    return sorted(asks, key=lambda row: row["price"])


def levels_depth(levels: list[dict[str, float]]) -> float:
    return float(sum(level["contracts"] for level in levels))


def near_top_depth(levels: list[dict[str, float]], width: float = 0.05) -> float:
    if not levels:
        return 0.0
    top = levels[0]["price"]
    return float(sum(level["contracts"] for level in levels if level["price"] <= top + width))


def orderbook_metrics(orderbook: dict[str, object]) -> dict[str, object]:
    yes_bids = parse_orderbook_levels(orderbook.get("yes_dollars"))
    no_bids = parse_orderbook_levels(orderbook.get("no_dollars"))
    yes_asks = bids_to_opposite_asks(no_bids)
    no_asks = bids_to_opposite_asks(yes_bids)
    best_yes_bid = yes_bids[-1]["price"] if yes_bids else None
    best_no_bid = no_bids[-1]["price"] if no_bids else None
    best_yes_ask = yes_asks[0]["price"] if yes_asks else None
    best_no_ask = no_asks[0]["price"] if no_asks else None
    spread = (
        best_yes_ask - best_yes_bid
        if best_yes_bid is not None and best_yes_ask is not None
        else None
    )
    return {
        "best_yes_bid": best_yes_bid,
        "best_yes_ask": best_yes_ask,
        "best_no_bid": best_no_bid,
        "best_no_ask": best_no_ask,
        "spread": spread,
        "yes_bid_depth": levels_depth(yes_bids),
        "no_bid_depth": levels_depth(no_bids),
        "yes_ask_depth": levels_depth(yes_asks),
        "no_ask_depth": levels_depth(no_asks),
        "yes_ask_depth_near_top": near_top_depth(yes_asks),
        "no_ask_depth_near_top": near_top_depth(no_asks),
        "yes_bids": yes_bids,
        "no_bids": no_bids,
        "yes_asks": yes_asks,
        "no_asks": no_asks,
    }


def _load_model_calibration(known_days: int) -> dict[str, object] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        payload = joblib.load(MODEL_PATH)
    except Exception:
        return None
    calibration = payload.get("calibration")
    if not calibration:
        return None
    return calibration.get("global", {}).get(str(int(known_days)))


def estimate_weekly_sigma(known_days: int) -> tuple[float, str]:
    calibration_entry = _load_model_calibration(known_days)
    if calibration_entry:
        intervals = calibration_entry.get("intervals", {})
        interval_68 = intervals.get("0.68")
        if interval_68:
            return max(float(interval_68), 25_000.0), "model calibration"

    if WEEKLY_REPORT_PATH.exists():
        summary = pd.read_csv(WEEKLY_REPORT_PATH)
        subset = summary[summary["known_days"] == int(known_days)]
        source_days = int(known_days)
        if subset.empty and "known_days" in summary:
            available_days = sorted(int(day) for day in summary["known_days"].dropna().unique())
            if available_days:
                source_days = min(available_days, key=lambda day: abs(day - int(known_days)))
                subset = summary[summary["known_days"] == source_days]
        error_col = "corrected_weekly_avg_error" if "corrected_weekly_avg_error" in subset else "weekly_avg_error"
        if not subset.empty and error_col in subset:
            sigma = float(subset[error_col].dropna().std())
            if math.isfinite(sigma) and sigma > 0:
                return max(sigma, 25_000.0), f"backtest errors ({source_days} known days)"

    return 75_000.0, "fallback"


def probability_above_threshold(
    predicted_average: float,
    threshold: float,
    sigma: float,
) -> float:
    if sigma <= 0:
        return 1.0 if predicted_average > threshold else 0.0
    normal = NormalDist(mu=predicted_average, sigma=sigma)
    return 1.0 - normal.cdf(threshold)


def load_dashboard_blend_confidence(variant: str = "weekly_daily_ensemble_plus_kalshi") -> dict[str, object]:
    fallback = {
        "variant": variant,
        "source": "fallback",
        "p90_abs_error": 50_000.0,
        "within_50k_rate": None,
        "tradeable": False,
    }
    if not KALSHI_BLEND_BACKTEST_SUMMARY_PATH.exists():
        return fallback
    try:
        summary = pd.read_csv(KALSHI_BLEND_BACKTEST_SUMMARY_PATH)
    except Exception:
        return fallback
    match = summary[
        (summary.get("grain", pd.Series(dtype=str)) == "overall")
        & (summary.get("variant", pd.Series(dtype=str)) == variant)
    ]
    if match.empty:
        return fallback
    row = match.iloc[0]
    p90 = dollars_to_probability(row.get("p90_final_abs_error"))
    within_50k = dollars_to_probability(row.get("final_within_50k"))
    tradeable = bool(p90 is not None and p90 <= 50_000 and (within_50k is None or within_50k >= 0.8))
    return {
        "variant": variant,
        "source": str(KALSHI_BLEND_BACKTEST_SUMMARY_PATH.relative_to(PROJECT_ROOT)),
        "rows": int(row.get("rows", 0) or 0),
        "p90_abs_error": p90,
        "within_50k_rate": within_50k,
        "tradeable": tradeable,
    }


def load_trade_strategy_rule() -> dict[str, object] | None:
    strategy_path = (
        KALSHI_CONTRACT_TRADE_STRATEGY_SUMMARY_PATH
        if KALSHI_CONTRACT_TRADE_STRATEGY_SUMMARY_PATH.exists()
        else KALSHI_TRADE_STRATEGY_SUMMARY_PATH
    )
    if not strategy_path.exists():
        return None
    try:
        summary = pd.read_csv(strategy_path)
    except Exception:
        return None
    if summary.empty:
        return None
    row = summary.iloc[0]
    min_volume = row.get("min_market_volume", row.get("min_contract_volume", 0.0))
    half_spread = row.get("half_spread", None)
    if half_spread is None and "max_entry_price" in row:
        half_spread = max(float(row.get("max_entry_price") or 1.0) - 0.5, 0.0)
    return {
        "source": str(strategy_path.relative_to(PROJECT_ROOT)),
        "mode": row.get("mode", "crossing_approximation"),
        "signal_variant": row.get("signal_variant"),
        "min_gap": dollars_to_probability(row.get("min_gap")) or 0.0,
        "min_estimated_edge": dollars_to_probability(row.get("min_estimated_edge")) or 0.0,
        "min_market_volume": dollars_to_probability(min_volume) or 0.0,
        "half_spread": dollars_to_probability(half_spread) or 0.0,
        "min_entry_price": dollars_to_probability(row.get("min_entry_price")),
        "max_entry_price": dollars_to_probability(row.get("max_entry_price")),
        "min_known_days": int(row.get("min_known_days") or 0),
        "max_known_days": int(row.get("max_known_days") or 7),
        "historical_trades": int(row.get("trades") or 0),
        "historical_win_rate": dollars_to_probability(row.get("win_rate")),
        "historical_roi": dollars_to_probability(row.get("roi")),
        "historical_max_drawdown": dollars_to_probability(row.get("max_drawdown")),
    }


def evaluate_trade_strategy(
    model_average: float,
    market_average: float | None,
    known_days: int,
    sigma: float,
    market_volume: float,
) -> dict[str, object]:
    rule = load_trade_strategy_rule()
    if rule is None:
        return {
            "available": False,
            "recommendation": "WAIT",
            "reason": "No trade-rule backtest summary is available.",
        }
    if market_average is None or not math.isfinite(market_average):
        return {
            "available": True,
            "recommendation": "WAIT",
            "reason": "No Kalshi 50% crossing forecast is available.",
            "rule": rule,
        }
    gap = model_average - market_average
    if sigma <= 0:
        prob_above = 1.0 if model_average > market_average else 0.0
    else:
        prob_above = 1.0 - NormalDist(mu=model_average, sigma=sigma).cdf(market_average)
    if gap >= 0:
        side = "YES"
        model_probability = prob_above
    else:
        side = "NO"
        model_probability = 1.0 - prob_above
    entry_price = 0.5 + float(rule["half_spread"])
    estimated_edge = model_probability - entry_price
    checks = {
        "known_days": int(rule["min_known_days"]) <= known_days <= int(rule["max_known_days"]),
        "gap": abs(gap) >= float(rule["min_gap"]),
        "edge": estimated_edge >= float(rule["min_estimated_edge"]),
        "volume": market_volume >= float(rule["min_market_volume"]),
    }
    should_trade = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "available": True,
        "recommendation": "TRADE" if should_trade else "WAIT",
        "side": side,
        "model_probability": model_probability,
        "entry_price_assumption": entry_price,
        "estimated_edge": estimated_edge,
        "model_market_gap": gap,
        "market_volume_contracts": market_volume,
        "checks": checks,
        "failed_checks": failed,
        "reason": "Rule passed." if should_trade else f"Failed checks: {', '.join(failed)}.",
        "rule": rule,
    }


def evaluate_contract_trade_strategy(
    rows: list[dict[str, object]],
    model_average: float,
    market_average: float | None,
    known_days: int,
) -> dict[str, object]:
    rule = load_trade_strategy_rule()
    if rule is None:
        return {
            "available": False,
            "recommendation": "WAIT",
            "reason": "No trade-rule backtest summary is available.",
        }
    if rule.get("mode") != "contract_candlesticks":
        volume_values = []
        for row in rows:
            volume = row.get("volume")
            if volume is None:
                continue
            try:
                volume = float(volume)
            except (TypeError, ValueError):
                continue
            if math.isfinite(volume):
                volume_values.append(volume)
        return evaluate_trade_strategy(
            model_average,
            market_average,
            known_days,
            estimate_weekly_sigma(known_days)[0],
            float(sum(volume_values)),
        )
    if market_average is None or not math.isfinite(market_average):
        return {
            "available": True,
            "recommendation": "WAIT",
            "reason": "No Kalshi 50% crossing forecast is available.",
            "rule": rule,
        }
    gap = model_average - market_average
    opportunities = []
    for market in rows:
        model_probability = market.get("model_probability")
        if model_probability is None or not math.isfinite(float(model_probability)):
            continue
        yes_ask = market.get("yes_ask")
        yes_bid = market.get("yes_bid")
        volume = market.get("volume") or 0.0
        if yes_ask is not None and math.isfinite(float(yes_ask)):
            opportunities.append(
                {
                    "ticker": market.get("ticker"),
                    "threshold": market.get("threshold"),
                    "side": "YES",
                    "entry_price": float(yes_ask),
                    "model_probability": float(model_probability),
                    "estimated_edge": float(model_probability) - float(yes_ask),
                    "volume": float(volume),
                }
            )
        if yes_bid is not None and math.isfinite(float(yes_bid)):
            no_ask = 1.0 - float(yes_bid)
            opportunities.append(
                {
                    "ticker": market.get("ticker"),
                    "threshold": market.get("threshold"),
                    "side": "NO",
                    "entry_price": no_ask,
                    "model_probability": 1.0 - float(model_probability),
                    "estimated_edge": (1.0 - float(model_probability)) - no_ask,
                    "volume": float(volume),
                }
            )
    max_entry_price = dollars_to_probability(rule.get("max_entry_price"))
    if max_entry_price is None:
        max_entry_price = 1.0
    min_entry_price = dollars_to_probability(rule.get("min_entry_price"))
    if min_entry_price is None:
        min_entry_price = 0.0
    opportunities = [
        item
        for item in opportunities
        if min_entry_price <= item["entry_price"] <= max_entry_price
        and item["volume"] >= float(rule["min_market_volume"])
    ]
    best = max(opportunities, key=lambda item: item["estimated_edge"], default=None)
    checks = {
        "known_days": int(rule["min_known_days"]) <= known_days <= int(rule["max_known_days"]),
        "gap": abs(gap) >= float(rule["min_gap"]),
        "edge": bool(best and best["estimated_edge"] >= float(rule["min_estimated_edge"])),
        "volume": bool(best and best["volume"] >= float(rule["min_market_volume"])),
    }
    should_trade = all(checks.values())
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "available": True,
        "recommendation": "TRADE" if should_trade else "WAIT",
        "side": best.get("side") if best else None,
        "ticker": best.get("ticker") if best else None,
        "threshold": best.get("threshold") if best else None,
        "model_probability": best.get("model_probability") if best else None,
        "entry_price_assumption": best.get("entry_price") if best else None,
        "estimated_edge": best.get("estimated_edge") if best else None,
        "model_market_gap": gap,
        "market_volume_contracts": best.get("volume") if best else None,
        "checks": checks,
        "failed_checks": failed,
        "reason": "Rule passed." if should_trade else f"Failed checks: {', '.join(failed)}.",
        "rule": rule,
    }


def signal_for_edge(edge: float) -> str:
    if edge >= EDGE_THRESHOLD:
        return "Model favors YES"
    if edge <= -EDGE_THRESHOLD:
        return "Model favors NO"
    return "No clear edge"


def kelly_fraction(model_probability: float, ask_probability: float | None) -> float | None:
    if ask_probability is None or ask_probability <= 0 or ask_probability >= 1:
        return None
    fraction = (model_probability - ask_probability) / (1 - ask_probability)
    return max(0.0, fraction)


def market_url(market: dict[str, object]) -> str:
    event_ticker = market.get("event_ticker") or ""
    ticker = market.get("ticker") or ""
    return f"https://kalshi.com/markets/{event_ticker}?market={ticker}"


def build_market_dashboard(prediction: dict[str, object]) -> dict[str, object]:
    model_average_source = prediction.get("dashboard_model_source") or "daily_model"
    predicted_average = float(
        prediction.get("dashboard_model_weekly_average")
        or prediction.get("weekly_ensemble", {}).get("ensemble_avg")
        or prediction["predicted_weekly_average"]
    )
    known_days = int(prediction.get("known_days") or 0)
    sigma, sigma_source = estimate_weekly_sigma(known_days)
    markets = fetch_tsa_weekly_markets(str(prediction["week_sunday"]))
    live_market_average = infer_market_implied_average(markets)
    market_average = (
        float(live_market_average["market_implied_average"])
        if live_market_average and live_market_average.get("market_implied_average") is not None
        else None
    )
    blend = adaptive_blend(predicted_average, market_average)
    adjusted_average = float(blend["adjusted_weekly_average"])
    confidence = load_dashboard_blend_confidence()
    p90_width = confidence.get("p90_abs_error")
    dashboard_range = None
    if p90_width is not None and math.isfinite(float(p90_width)):
        p90_width = float(p90_width)
        dashboard_range = {
            "level": "0.90",
            "lower": adjusted_average - p90_width,
            "upper": adjusted_average + p90_width,
            "width": p90_width,
            "source": confidence.get("source"),
        }

    rows = []
    for market in markets:
        ticker = str(market.get("ticker") or "")
        orderbook = fetch_orderbook(ticker) if ticker else {}
        book = orderbook_metrics(orderbook)
        threshold = float(market.get("floor_strike") or 0)
        model_probability = probability_above_threshold(predicted_average, threshold, sigma)
        adjusted_probability = probability_above_threshold(adjusted_average, threshold, sigma)
        market_probability = midpoint_probability(market)
        ask_probability = dollars_to_probability(market.get("yes_ask_dollars"))
        bid_probability = dollars_to_probability(market.get("yes_bid_dollars"))
        edge = model_probability - market_probability if market_probability is not None else None
        adjusted_edge = (
            adjusted_probability - market_probability
            if market_probability is not None
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "event_ticker": market.get("event_ticker"),
                "title": market.get("title"),
                "subtitle": market.get("subtitle"),
                "threshold": threshold,
                "status": market.get("status"),
                "yes_bid": bid_probability,
                "yes_ask": ask_probability,
                "market_probability": market_probability,
                "model_probability": model_probability,
                "adjusted_probability": adjusted_probability,
                "edge": edge,
                "adjusted_edge": adjusted_edge,
                "signal": signal_for_edge(edge) if edge is not None else "No market price",
                "adjusted_signal": signal_for_edge(adjusted_edge)
                if adjusted_edge is not None
                else "No market price",
                "kelly_fraction": kelly_fraction(model_probability, ask_probability),
                "adjusted_kelly_fraction": kelly_fraction(adjusted_probability, ask_probability),
                "liquidity_dollars": dollars_to_probability(market.get("liquidity_dollars")),
                "volume_24h": dollars_to_probability(market.get("volume_24h_fp")),
                "volume": dollars_to_probability(market.get("volume_fp")),
                "open_interest": dollars_to_probability(market.get("open_interest_fp")),
                "updated_time": market.get("updated_time"),
                "market_url": market_url(market),
                "orderbook": book,
            }
        )

    priced = [row for row in rows if row["edge"] is not None]
    adjusted_priced = [row for row in rows if row["adjusted_edge"] is not None]
    trading_strategy = evaluate_contract_trade_strategy(
        rows,
        predicted_average,
        market_average,
        known_days,
    )
    best_yes = max(priced, key=lambda row: row["edge"], default=None)
    best_no = min(priced, key=lambda row: row["edge"], default=None)
    adjusted_best_yes = max(adjusted_priced, key=lambda row: row["adjusted_edge"], default=None)
    adjusted_best_no = min(adjusted_priced, key=lambda row: row["adjusted_edge"], default=None)
    return {
        "enabled": True,
        "status": "live",
        "series_ticker": TSA_WEEKLY_SERIES,
        "event_ticker": event_ticker_for_week(str(prediction["week_sunday"])),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "predicted_weekly_average": predicted_average,
        "model_forecast": predicted_average,
        "model_forecast_source": model_average_source,
        "daily_model_forecast": prediction.get("daily_model_weekly_average")
        or prediction.get("predicted_weekly_average"),
        "weekly_ensemble": prediction.get("weekly_ensemble"),
        "live_market_forecast": market_average,
        "live_market_forecast_as_of": pd.Timestamp.utcnow().isoformat(),
        "live_market_forecast_source": live_market_average.get("source") if live_market_average else None,
        "market_median_forecast": market_average,
        "market_median_forecast_as_of": pd.Timestamp.utcnow().isoformat(),
        "market_median_forecast_source": live_market_average.get("source") if live_market_average else None,
        "market_implied_average": market_average,
        "market_implied_average_source": live_market_average,
        "adjusted_weekly_average": adjusted_average,
        "dashboard_blended_forecast": adjusted_average,
        "model_market_gap": blend.get("gap"),
        "confidence_range": dashboard_range,
        "dashboard_backtest_confidence": confidence,
        "tradeable": bool(confidence.get("tradeable")),
        "trading_strategy": trading_strategy,
        "blend": blend,
        "known_days": known_days,
        "sigma": sigma,
        "sigma_source": sigma_source,
        "markets": rows,
        "summary": {
            "market_count": len(rows),
            "best_yes_ticker": best_yes["ticker"] if best_yes else None,
            "best_yes_edge": best_yes["edge"] if best_yes else None,
            "best_no_ticker": best_no["ticker"] if best_no else None,
            "best_no_edge": best_no["edge"] if best_no else None,
            "adjusted_best_yes_ticker": adjusted_best_yes["ticker"] if adjusted_best_yes else None,
            "adjusted_best_yes_edge": adjusted_best_yes["adjusted_edge"] if adjusted_best_yes else None,
            "adjusted_best_no_ticker": adjusted_best_no["ticker"] if adjusted_best_no else None,
            "adjusted_best_no_edge": adjusted_best_no["adjusted_edge"] if adjusted_best_no else None,
        },
    }
