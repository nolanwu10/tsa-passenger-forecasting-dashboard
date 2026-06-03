from pathlib import Path
import argparse
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.ingest_bts import (
    write_monthly_air_traffic,
    write_on_time_daily,
    write_t100_annual_capacity,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch BTS transport enrichment datasets.")
    parser.add_argument("--delay-start-year", type=int, default=2024)
    parser.add_argument("--delay-end-year", type=int, default=2026)
    args = parser.parse_args()

    for status in [
        write_monthly_air_traffic(),
        write_t100_annual_capacity(),
        write_on_time_daily(args.delay_start_year, args.delay_end_year),
    ]:
        print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

