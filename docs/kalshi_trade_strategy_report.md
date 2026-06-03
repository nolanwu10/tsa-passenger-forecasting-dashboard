# Kalshi TSA Trade Strategy Backtest

## Purpose

Forecast accuracy is not the same as tradable edge. This backtest adds a trading-rule layer on top of the TSA model/Kalshi comparison to answer a narrower question:

Can model-vs-market disagreement survive spread, liquidity, and settlement?

## Available Historical Market Data

The local Kalshi history currently contains:

- weekly event ticker
- as-of date
- Kalshi 50% crossing implied weekly average
- aggregate contract count / volume fields
- actual final TSA weekly average
- model and model/Kalshi blended forecasts

It does not currently contain full historical per-contract bid/ask/orderbook snapshots for every threshold. Because of that, the first strategy backtest uses a conservative 50% crossing approximation:

- threshold = Kalshi 50% crossing forecast
- buy YES if model is above Kalshi
- buy NO if model is below Kalshi
- entry price = `0.50 + assumed_half_spread`
- settlement = 1 if the final weekly average lands on the traded side of the crossing, else 0

This is not a replacement for a true contract-level PnL backtest. It is a disciplined first filter for whether the model disagreement is large enough to justify looking for trades.

## Selected Strategy Rule

The selected practical rule uses:

- signal model: raw weekly+daily ensemble, before Kalshi blend
- minimum model-market gap: 20,000 passengers
- minimum estimated probability edge: 5 percentage points
- minimum historical market volume: 1,000 contracts
- assumed half-spread: 2 percentage points
- known days allowed: 1 through 6
- fully known weeks are excluded

## Crossing Approximation Backtest Result

Rows evaluated: 147 historical market days.

Selected crossing-approximation rule:

- trades: 42
- trade rate: 28.57%
- win rate: 76.19%
- total PnL per 1-contract unit sizing: +10.16
- ROI on cost: 46.52%
- average PnL per trade: +0.24
- max drawdown: -1.60

## Actual Contract Candlestick Backtest Result

After adding the per-contract Kalshi candle fetcher, the real contract-price backtest uses actual listed thresholds and historical candle close prices:

- YES entry = historical `yes_ask_close`
- NO entry = `1 - yes_bid_close`
- selected contract = highest estimated model edge among live threshold contracts for that market day
- settlement = final weekly TSA average against that contract threshold

Fetched contract data currently covers:

- markets: 441
- candlestick rows: 2,974
- historical events: 21
- usable bid/ask events: 10

Older Jan/Feb historical contracts are discoverable through Kalshi's historical markets endpoint, but their daily candlesticks currently return null bid/ask/price fields. The actual-contract PnL result below therefore uses rows with usable contract prices, concentrated from mid-March through May.

Pre-week support has also been added through `scripts/build_kalshi_preweek_contract_signals.py`. The current fetched Kalshi contract history contains zero pre-week candle rows, so it writes an empty schema-valid report at `artifacts/reports/kalshi_preweek_contract_signal_rows.csv`. Once Kalshi exposes or we collect pre-week contract candles, this report can feed the same contract-level backtester with `known_days=0`.

Selected actual-contract rule:

- signal model: raw weekly+daily ensemble, before Kalshi blend
- minimum model-market gap: 20,000 passengers
- minimum estimated contract edge: 10 percentage points
- minimum contract candle volume: 1 contract
- max entry price: 90c
- known days allowed: 1 through 6

Backtest result:

- rows evaluated: 147
- rows with contract candidates: 62
- trades: 21
- trade rate: 14.29%
- win rate: 52.38%
- total PnL per 1-contract unit sizing: +6.15
- ROI on cost: 126.80%
- average PnL per trade: +0.29
- max drawdown: -0.87

Interpretation:

The model should not be treated as an always-on trader. The actual-contract backtest is more selective than the crossing approximation and finds fewer trades. The edge appears concentrated in lower-priced contracts where the model strongly disagrees with the market, but the win rate is only modest. This supports a selective alert tool before it supports any fully automated trading system.

## Files

- Backtest script: `scripts/backtest_kalshi_trade_strategy.py`
- Contract fetch script: `scripts/fetch_kalshi_contract_history.py`
- Actual-contract backtest script: `scripts/backtest_kalshi_contract_trade_strategy.py`
- Strategy module: `src/tsa_project/kalshi_trade_backtest.py`
- Trade rows: `artifacts/reports/kalshi_trade_strategy_backtest_trades.csv`
- Summary: `artifacts/reports/kalshi_trade_strategy_backtest_summary.csv`
- Parameter grid: `artifacts/reports/kalshi_trade_strategy_grid.csv`
- Contract markets: `data/external/kalshi_tsa_weekly_markets.csv`
- Contract candlesticks: `data/external/kalshi_tsa_weekly_market_candlesticks.csv`
- Actual-contract trade rows: `artifacts/reports/kalshi_contract_trade_strategy_backtest_trades.csv`
- Actual-contract summary: `artifacts/reports/kalshi_contract_trade_strategy_backtest_summary.csv`
- Actual-contract parameter grid: `artifacts/reports/kalshi_contract_trade_strategy_grid.csv`
- Pre-week contract signals: `artifacts/reports/kalshi_preweek_contract_signal_rows.csv`

## Next Data Upgrade

The next major improvement is to collect denser historical contract snapshots, ideally hourly or minute candles. The current actual-contract run uses daily candlesticks. Denser snapshots would let the simulator:

- choose real listed thresholds
- use intraday ask/bid prices
- apply liquidity/depth constraints per contract
- evaluate pre-week trading if the market has listed contracts before Monday
