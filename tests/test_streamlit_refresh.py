import pandas as pd

import streamlit_app


def test_refresh_forecast_data_fetches_and_rebuilds_features(monkeypatch):
    calls = []
    raw = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-06-01", "2026-06-02"]),
            "Passengers": [2_500_000, 2_600_000],
        }
    )
    calendar = raw.assign(year=[2026, 2026])
    transport = calendar.assign(extra_feature=[1.0, 2.0])

    def fake_write_tsa_passenger_data():
        calls.append("fetch")
        return raw

    def fake_write_calendar_holiday_features(frame):
        calls.append(("calendar", frame))
        return calendar

    def fake_write_transport_features():
        calls.append("transport")
        return transport

    monkeypatch.setattr(streamlit_app, "write_tsa_passenger_data", fake_write_tsa_passenger_data)
    monkeypatch.setattr(streamlit_app, "write_calendar_holiday_features", fake_write_calendar_holiday_features)
    monkeypatch.setattr(streamlit_app, "write_transport_features", fake_write_transport_features)

    result = streamlit_app.refresh_forecast_data()

    assert calls[0] == "fetch"
    assert calls[1][0] == "calendar"
    assert calls[1][1] is raw
    assert calls[2] == "transport"
    assert result == {
        "rows": 2,
        "latest_date": "2026-06-02",
        "raw_rows": 2,
        "calendar_rows": 2,
    }
