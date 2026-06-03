from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.kalshi_contract_history import (
    PREWEEK_CONTRACT_SIGNAL_ROWS_PATH,
    build_preweek_contract_signal_rows,
)


def main() -> int:
    rows = build_preweek_contract_signal_rows()
    print("Pre-week contract signal rows built:")
    print(f"Rows: {len(rows):,}")
    print(f"Saved: {PREWEEK_CONTRACT_SIGNAL_ROWS_PATH}")
    if rows.empty:
        print("No pre-week contract candle rows are available in the current Kalshi history.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
