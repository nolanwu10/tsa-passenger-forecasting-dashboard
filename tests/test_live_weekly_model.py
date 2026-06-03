import numpy as np
import pandas as pd

from tsa_project.features import build_calendar_holiday_features
from tsa_project.live_weekly_model import LiveFeatureFactory, MODEL_FEATURES, predict_week


class ZeroResidualModel:
    def predict(self, frame):
        return np.zeros(len(frame))


def synthetic_modeling_data(start="2019-01-01", end="2026-06-30"):
    dates = pd.date_range(start, end, freq="D")
    raw = pd.DataFrame(
        {
            "Date": dates,
            "Passengers": [
                1_800_000
                + (date.year - 2019) * 45_000
                + date.dayofyear * 900
                + date.weekday() * 12_000
                for date in dates
            ],
        }
    )
    return build_calendar_holiday_features(raw)


def test_feature_row_schema_matches_declared_model_features():
    data = synthetic_modeling_data()
    factory = LiveFeatureFactory(data)

    row = factory.feature_row(pd.Timestamp("2026-06-15"), pd.Timestamp("2026-06-14"))

    assert list(row) == MODEL_FEATURES
    assert set(row) == set(MODEL_FEATURES)


def test_feature_row_does_not_use_future_passenger_values_after_issue_date():
    data = synthetic_modeling_data()
    target_date = pd.Timestamp("2026-06-18")
    issue_date = pd.Timestamp("2026-06-14")

    original_row = LiveFeatureFactory(data).feature_row(target_date, issue_date)

    changed = data.copy()
    future_mask = changed["Date"] > issue_date
    changed.loc[future_mask, "Passengers"] = changed.loc[future_mask, "Passengers"] + 9_000_000
    changed_row = LiveFeatureFactory(changed).feature_row(target_date, issue_date)

    assert original_row == changed_row


def test_predict_week_marks_known_days_as_actuals_and_future_days_as_predictions():
    data = synthetic_modeling_data()
    monday = pd.Timestamp("2026-06-15")

    result = predict_week(ZeroResidualModel(), data, monday, known_days=2)

    assert len(result) == 7
    assert result["type"].tolist()[:2] == ["actual_known", "actual_known"]
    assert result["type"].tolist()[2:] == ["predicted"] * 5

    by_date = data.set_index("Date")
    expected_monday = float(by_date.loc[monday, "Passengers"])
    expected_tuesday = float(by_date.loc[monday + pd.Timedelta(days=1), "Passengers"])
    assert result.loc[0, "predicted_passengers"] == expected_monday
    assert result.loc[1, "predicted_passengers"] == expected_tuesday
    assert result.loc[:1, "baseline"].isna().all()
    assert result.loc[2:, "predicted_passengers"].notna().all()
