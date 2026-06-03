from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
REPORT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reports"

RAW_TSA_PATH = RAW_DATA_DIR / "tsa_passenger_data.csv"
EXTERNAL_WEATHER_PATH = EXTERNAL_DATA_DIR / "weather_history.csv"
EXTERNAL_BTS_ON_TIME_DAILY_PATH = EXTERNAL_DATA_DIR / "bts_on_time_daily.csv"
EXTERNAL_BTS_MTS_AIR_TRAFFIC_PATH = EXTERNAL_DATA_DIR / "bts_monthly_air_traffic.csv"
EXTERNAL_BTS_T100_ANNUAL_CAPACITY_PATH = EXTERNAL_DATA_DIR / "bts_t100_annual_capacity.csv"
EXTERNAL_KALSHI_TSA_MARKETS_PATH = EXTERNAL_DATA_DIR / "kalshi_tsa_weekly_markets.csv"
EXTERNAL_KALSHI_TSA_CANDLESTICKS_PATH = EXTERNAL_DATA_DIR / "kalshi_tsa_weekly_market_candlesticks.csv"
PROCESSED_FEATURES_PATH = PROCESSED_DATA_DIR / "processed_features.csv"
DAILY_CALENDAR_FEATURES_PATH = PROCESSED_DATA_DIR / "tsa_daily_calendar_features.csv"
DAILY_TRANSPORT_FEATURES_PATH = PROCESSED_DATA_DIR / "tsa_daily_transport_features.csv"
