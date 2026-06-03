from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from tsa_project.config import (
    DAILY_TRANSPORT_FEATURES_PATH,
    MODEL_ARTIFACTS_DIR,
    PROJECT_ROOT,
    RAW_TSA_PATH,
    REPORT_ARTIFACTS_DIR,
)
from tsa_project.kalshi import build_market_dashboard, configured as kalshi_configured
from tsa_project.live_weekly_model import (
    MODEL_FEATURES,
    MODEL_PATH,
    apply_weekly_calibration,
    load_modeling_data,
    lookup_calibration,
    predict_week,
    train_final_model,
    week_regime,
)
from tsa_project.weekly_ensemble_model import predict_weekly_ensemble


STATIC_DIR = PROJECT_ROOT / "dashboard_static"
LATEST_PREDICTION_PATH = REPORT_ARTIFACTS_DIR / "dashboard_latest_prediction.json"


@dataclass
class DashboardJob:
    id: str
    name: str
    status: str = "queued"
    logs: list[str] = field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None


class DashboardState:
    def __init__(self) -> None:
        self.jobs: dict[str, DashboardJob] = {}
        self.lock = threading.Lock()

    def create_job(self, name: str) -> DashboardJob:
        job = DashboardJob(id=uuid.uuid4().hex[:12], name=name)
        with self.lock:
            self.jobs[job.id] = job
        return job

    def get_job(self, job_id: str) -> DashboardJob | None:
        with self.lock:
            return self.jobs.get(job_id)

    def update_job(self, job_id: str, **changes: object) -> None:
        with self.lock:
            job = self.jobs[job_id]
            for key, value in changes.items():
                setattr(job, key, value)

    def append_log(self, job_id: str, message: str) -> None:
        with self.lock:
            self.jobs[job_id].logs.append(message)


STATE = DashboardState()


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def current_target_monday(today: date | None = None) -> pd.Timestamp:
    """Return the Monday for the calendar week the dashboard should target."""
    today = today or datetime.now().date()
    current_day = pd.Timestamp(today).normalize()
    return current_day - pd.Timedelta(days=current_day.weekday())


def to_json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [to_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def read_tsa_data() -> pd.DataFrame:
    path = DAILY_TRANSPORT_FEATURES_PATH if DAILY_TRANSPORT_FEATURES_PATH.exists() else RAW_TSA_PATH
    data = pd.read_csv(path, parse_dates=["Date"])
    data["Passengers"] = pd.to_numeric(data["Passengers"], errors="coerce")
    data = data.dropna(subset=["Date", "Passengers"])
    return data.sort_values("Date").reset_index(drop=True)


def summarize_dataset(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.exists(),
    }
    if not path.exists():
        return summary

    data = pd.read_csv(path)
    summary["rows"] = int(len(data))
    summary["columns"] = int(len(data.columns))
    summary["updated_at"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")
    if "Date" in data.columns:
        dates = pd.to_datetime(data["Date"], errors="coerce").dropna()
        if not dates.empty:
            summary["min_date"] = dates.min().date().isoformat()
            summary["max_date"] = dates.max().date().isoformat()
    return summary


def week_summary(data: pd.DataFrame, monday: pd.Timestamp) -> dict[str, object]:
    sunday = monday + pd.Timedelta(days=6)
    week = data[(data["Date"] >= monday) & (data["Date"] <= sunday)].copy()
    average = float(week["Passengers"].mean()) if not week.empty else None
    total = float(week["Passengers"].sum()) if not week.empty else None
    return {
        "monday": monday.date().isoformat(),
        "sunday": sunday.date().isoformat(),
        "days": int(len(week)),
        "average": average,
        "total": total,
    }


def build_summary() -> dict[str, object]:
    data = read_tsa_data()
    latest_date = pd.Timestamp(data["Date"].max()).normalize()
    target_monday = current_target_monday()
    last_week_monday = target_monday - pd.Timedelta(days=7)
    target_week = week_summary(data, target_monday)
    target_sunday = target_monday + pd.Timedelta(days=6)
    target_mask = (data["Date"] >= target_monday) & (data["Date"] <= target_sunday)
    target_week["known_days"] = int(len(data[target_mask]))

    return {
        "generated_at": utc_now(),
        "latest_tsa_date": latest_date.date().isoformat(),
        "target_week": target_week,
        "last_week": week_summary(data, last_week_monday),
        "current_week": target_week,
        "datasets": {
            "raw_tsa": summarize_dataset(RAW_TSA_PATH),
            "daily_transport_features": summarize_dataset(DAILY_TRANSPORT_FEATURES_PATH),
        },
        "model": {
            "path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "exists": MODEL_PATH.exists(),
            "updated_at": datetime.fromtimestamp(MODEL_PATH.stat().st_mtime).isoformat(timespec="seconds")
            if MODEL_PATH.exists()
            else None,
        },
        "kalshi": kalshi_status(),
    }


def kalshi_status() -> dict[str, object]:
    return {
        "enabled": kalshi_configured(),
        "status": "configured" if kalshi_configured() else "missing_credentials",
        "message": "Kalshi live market data is ready." if kalshi_configured() else "Set Kalshi credentials to enable live markets.",
        "planned_features": [
            "Live market lookup",
            "Contract probability display",
            "Model versus market spread",
            "Order ticket handoff",
        ],
    }


def predict_current_week() -> dict[str, object]:
    data = load_modeling_data()
    latest_date = pd.Timestamp(data["Date"].max()).normalize()
    monday = current_target_monday()
    known_days = int(len(data[(data["Date"] >= monday) & (data["Date"] <= latest_date)]))
    known_days = max(0, min(7, known_days))

    if not MODEL_PATH.exists():
        train_final_model()

    payload = joblib.load(MODEL_PATH)
    if payload.get("features") != MODEL_FEATURES:
        payload = train_final_model()
        payload = joblib.load(MODEL_PATH)
    calibration = payload.get("calibration")
    result = predict_week(payload["model"], data, monday, known_days=known_days)
    regime = week_regime(data, monday)
    calibration_entry = lookup_calibration(calibration, known_days, regime)
    result, weekly_correction = apply_weekly_calibration(result, calibration_entry)
    predicted_average = float(result["predicted_passengers"].mean())
    predicted_total = float(result["predicted_passengers"].sum())
    known = result[result["type"] == "actual_known"]
    predicted = result[result["type"] == "predicted"]
    intervals = calibration_entry.get("intervals", {}) or {}
    ranges = {
        level: {
            "lower": predicted_average - float(width),
            "upper": predicted_average + float(width),
            "width": float(width),
        }
        for level, width in intervals.items()
    }
    within_50k = calibration_entry.get("within_50k_rate")
    response = {
        "generated_at": utc_now(),
        "week_monday": monday.date().isoformat(),
        "week_sunday": (monday + pd.Timedelta(days=6)).date().isoformat(),
        "latest_tsa_date": latest_date.date().isoformat(),
        "known_days": known_days,
        "regime": regime,
        "calibration_source": calibration_entry.get("source", "none"),
        "calibration_n": calibration_entry.get("n", 0),
        "weekly_correction": weekly_correction,
        "calibrated_ranges": ranges,
        "within_50k_rate": float(within_50k) if pd.notna(within_50k) else None,
        "trade_range_confidence_under_50k": bool(pd.notna(within_50k) and within_50k >= 0.8),
        "known_running_average": float(known["predicted_passengers"].mean()) if not known.empty else None,
        "predicted_remaining_average": float(predicted["predicted_passengers"].mean())
        if not predicted.empty
        else None,
        "daily_model_weekly_average": predicted_average,
        "predicted_weekly_average": predicted_average,
        "predicted_weekly_total": predicted_total,
        "rows": result.to_dict(orient="records"),
    }
    try:
        ensemble = predict_weekly_ensemble(data, monday, known_days)
    except Exception as exc:
        response["weekly_ensemble"] = None
        response["dashboard_model_weekly_average"] = predicted_average
        response["dashboard_model_source"] = "daily_model_fallback"
        response["dashboard_model_error"] = str(exc)
    else:
        response["weekly_ensemble"] = ensemble
        response["dashboard_model_weekly_average"] = float(ensemble["ensemble_avg"])
        response["dashboard_model_source"] = "weekly_daily_ensemble"
    REPORT_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    LATEST_PREDICTION_PATH.write_text(json.dumps(to_json_safe(response), indent=2), encoding="utf-8")
    return response


def run_command(job_id: str, command: list[str]) -> None:
    STATE.append_log(job_id, f"$ {' '.join(command)}")
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if process.stdout.strip():
        STATE.append_log(job_id, process.stdout.strip())
    if process.stderr.strip():
        STATE.append_log(job_id, process.stderr.strip())
    if process.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {process.returncode}: {' '.join(command)}")


def train_model_for_job(job_id: str) -> None:
    run_command(job_id, [sys.executable, "scripts/backtest_live_weekly_model.py"])


def run_pipeline_job(job_id: str) -> None:
    STATE.update_job(job_id, status="running", started_at=utc_now())
    commands = [
        [sys.executable, "scripts/fetch_tsa_data.py"],
        [sys.executable, "scripts/build_calendar_features.py"],
        [sys.executable, "scripts/build_transport_features.py"],
    ]
    try:
        for command in commands:
            run_command(job_id, command)
        train_model_for_job(job_id)
        STATE.update_job(job_id, status="succeeded", finished_at=utc_now())
    except Exception as exc:
        STATE.append_log(job_id, traceback.format_exc())
        STATE.update_job(job_id, status="failed", finished_at=utc_now(), error=str(exc))


def job_to_dict(job: DashboardJob) -> dict[str, object]:
    return {
        "id": job.id,
        "name": job.name,
        "status": job.status,
        "logs": job.logs,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"[dashboard] {self.address_string()} - {format % args}\n")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/summary":
            self.write_json(HTTPStatus.OK, build_summary())
            return
        if path == "/api/kalshi":
            self.write_json(HTTPStatus.OK, kalshi_status())
            return
        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = STATE.get_job(job_id)
            if job is None:
                self.write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown job id"})
                return
            self.write_json(HTTPStatus.OK, job_to_dict(job))
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path == "/api/jobs/pipeline":
            job = STATE.create_job("Refresh TSA data and model")
            thread = threading.Thread(target=run_pipeline_job, args=(job.id,), daemon=True)
            thread.start()
            self.write_json(HTTPStatus.ACCEPTED, job_to_dict(job))
            return
        if path == "/api/predict":
            try:
                self.write_json(HTTPStatus.OK, predict_current_week())
            except Exception as exc:
                self.write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(exc), "traceback": traceback.format_exc()},
                )
            return
        if path == "/api/kalshi/markets":
            try:
                prediction = predict_current_week()
                self.write_json(HTTPStatus.OK, build_market_dashboard(prediction))
            except Exception as exc:
                self.write_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": str(exc), "traceback": traceback.format_exc()},
                )
            return
        self.write_json(HTTPStatus.NOT_FOUND, {"error": "Unknown endpoint"})

    def write_json(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(to_json_safe(payload), allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run_server(host: str, port: int) -> None:
    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"Dashboard static directory not found: {STATIC_DIR}")
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"TSA dashboard running at http://{host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TSA dashboard app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_server(args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
