from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import pandas as pd

from tsa_project.live_weekly_model import (
    WEEKLY_REPORT_PATH,
    apply_expanding_weekly_correction,
    build_calibration,
    choose_best_config,
    train_final_model,
)


if __name__ == "__main__":
    calibration = None
    config_name = None
    if WEEKLY_REPORT_PATH.exists():
        summary = pd.read_csv(WEEKLY_REPORT_PATH)
        if "corrected_weekly_total_mape" not in summary.columns:
            summary = apply_expanding_weekly_correction(summary)
        config_name = choose_best_config(summary)
        calibration = build_calibration(summary, config_name)

    result = train_final_model(config_name, calibration=calibration)
    print(f"Wrote model to {result['path']}")
    print(f"Model config: {result['config']}")
    print(f"Feature count: {len(result['features'])}")
    print(f"Calibration: {'yes' if calibration else 'no'}")
