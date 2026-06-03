from pathlib import Path
import os
import sys
import traceback


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import RAW_TSA_PATH
from tsa_project.ingest_tsa import write_tsa_passenger_data


if __name__ == "__main__":
    try:
        data = write_tsa_passenger_data()
    except Exception as exc:
        allow_stale = os.environ.get("TSA_ALLOW_STALE_ON_FETCH_ERROR", "").lower() in {"1", "true", "yes"}
        if allow_stale and RAW_TSA_PATH.exists():
            print(f"TSA fetch failed; continuing with cached data at {RAW_TSA_PATH}: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            raise SystemExit(0)
        raise

    print(
        f"Wrote {len(data):,} rows to {RAW_TSA_PATH} "
        f"({data['Date'].min().date()} to {data['Date'].max().date()})"
    )
