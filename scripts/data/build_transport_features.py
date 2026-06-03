from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from tsa_project.config import DAILY_TRANSPORT_FEATURES_PATH
from tsa_project.transport_features import write_transport_features


if __name__ == "__main__":
    features = write_transport_features()
    print(
        f"Wrote {len(features):,} rows and {len(features.columns):,} columns to "
        f"{DAILY_TRANSPORT_FEATURES_PATH}"
    )
    print(f"Date range: {features['Date'].min()} to {features['Date'].max()}")

