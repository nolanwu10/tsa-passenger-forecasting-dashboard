# Script Layout

Root-level scripts are compatibility wrappers so existing commands keep working, for example:

```powershell
python scripts/validate_project.py
python scripts/backtest_live_weekly_model.py --quick
```

The actual implementations are grouped by purpose:

- `scripts/data/`: ingestion, raw-data inspection, calendar and transport feature builds.
- `scripts/modeling/`: live weekly training, prediction, and model backtests.
- `scripts/analysis/`: data readiness, model audit, validation, and feature-importance reports.
- `scripts/markets/`: Kalshi data collection and market-comparison backtests.

When adding a new script, put the implementation in the relevant subfolder. Add a root wrapper only if the command is part of the reviewer or user-facing workflow.
