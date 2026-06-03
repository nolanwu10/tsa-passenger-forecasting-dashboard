from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import DAILY_CALENDAR_FEATURES_PATH, RAW_TSA_PATH
from tsa_project.features import write_calendar_holiday_features


if __name__ == "__main__":
    raw = pd.read_csv(RAW_TSA_PATH)
    features = write_calendar_holiday_features(raw)
    print(
        f"Wrote {len(features):,} rows and {len(features.columns):,} columns to "
        f"{DAILY_CALENDAR_FEATURES_PATH}"
    )
    print(f"Date range: {features['Date'].min()} to {features['Date'].max()}")

