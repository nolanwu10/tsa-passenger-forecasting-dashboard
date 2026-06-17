import pandas as pd
import pytest
import requests

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
        "status": "refreshed",
        "message": "Fetched the latest TSA passenger data.",
        "rows": 2,
        "latest_date": "2026-06-02",
        "raw_rows": 2,
        "calendar_rows": 2,
    }


def test_refresh_forecast_data_uses_cached_raw_on_tsa_403(monkeypatch, tmp_path):
    calls = []
    raw_path = tmp_path / "tsa_passenger_data.csv"
    cached_raw = pd.DataFrame(
        {
            "Date": ["2026-06-01", "2026-06-02"],
            "Passengers": [2_500_000, 2_600_000],
        }
    )
    cached_raw.to_csv(raw_path, index=False)
    calendar = cached_raw.assign(Date=pd.to_datetime(cached_raw["Date"]), year=[2026, 2026])
    transport = calendar.assign(extra_feature=[1.0, 2.0])

    def fake_write_tsa_passenger_data():
        response = requests.Response()
        response.status_code = 403
        raise requests.HTTPError("403 Client Error: Forbidden", response=response)

    def fake_write_calendar_holiday_features(frame):
        calls.append(("calendar", frame))
        return calendar

    def fake_write_transport_features():
        calls.append("transport")
        return transport

    monkeypatch.setattr(streamlit_app, "RAW_TSA_PATH", raw_path)
    monkeypatch.setattr(streamlit_app, "write_tsa_passenger_data", fake_write_tsa_passenger_data)
    monkeypatch.setattr(streamlit_app, "write_calendar_holiday_features", fake_write_calendar_holiday_features)
    monkeypatch.setattr(streamlit_app, "write_transport_features", fake_write_transport_features)

    result = streamlit_app.refresh_forecast_data()

    assert calls[0][0] == "calendar"
    assert calls[0][1]["Passengers"].tolist() == [2_500_000, 2_600_000]
    assert calls[1] == "transport"
    assert result["status"] == "cached"
    assert "HTTP 403" in result["message"]
    assert result["latest_date"] == "2026-06-02"


def test_refresh_forecast_data_reraises_non_403_http_error(monkeypatch):
    def fake_write_tsa_passenger_data():
        response = requests.Response()
        response.status_code = 500
        raise requests.HTTPError("500 Server Error", response=response)

    monkeypatch.setattr(streamlit_app, "write_tsa_passenger_data", fake_write_tsa_passenger_data)

    with pytest.raises(requests.HTTPError):
        streamlit_app.refresh_forecast_data()
