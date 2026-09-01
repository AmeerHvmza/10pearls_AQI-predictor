"""
Central configuration for the AQI Predictor project.
All secrets are read from environment variables — never hardcode API keys.
"""
import os


def _env_str(name: str, default: str) -> str:
    """GitHub Actions expands an undefined `vars.X` to an empty string rather
    than leaving it unset, so `os.getenv(name, default)` would return "" and
    the default would never apply. Treat blank as absent."""
    value = os.getenv(name, "")
    return value.strip() or default


def _env_float(name: str, default: float) -> float:
    raw = _env_str(name, str(default))
    try:
        return float(raw)
    except ValueError:
        print(f"[config] {name}={raw!r} is not a number; falling back to {default}")
        return default


# ---- Location -------------------------------------------------------
CITY_NAME = _env_str("AQI_CITY", "Karachi")
LATITUDE = _env_float("AQI_LAT", 24.8607)
LONGITUDE = _env_float("AQI_LON", 67.0011)

# Presets offered in the dashboard's location selector. The active city above
# is always available regardless of whether it appears here.
CITY_PRESETS = {
    "Karachi": (24.8607, 67.0011),
    "Lahore": (31.5204, 74.3587),
    "Islamabad": (33.6844, 73.0479),
    "Peshawar": (34.0151, 71.5249),
    "Quetta": (30.1798, 66.9750),
    "Delhi": (28.6139, 77.2090),
}

# ---- API keys (set these as environment variables / GitHub secrets) -
OPENWEATHER_API_KEY = _env_str("OPENWEATHER_API_KEY", "")

# OpenWeather is used for BOTH weather and air-pollution data because
# a single key covers both endpoints (simpler than juggling AQICN + a
# separate weather provider). Swap in AQICN by editing src/feature_pipeline.py
# if you prefer its ground-station data.
AIR_POLLUTION_URL = "https://api.openweathermap.org/data/2.5/air_pollution"
AIR_POLLUTION_HISTORY_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"
WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# ---- Network behaviour -------------------------------------------------
HTTP_TIMEOUT = _env_float("AQI_HTTP_TIMEOUT", 30)
HTTP_RETRIES = int(_env_float("AQI_HTTP_RETRIES", 3))
HTTP_BACKOFF_SECONDS = _env_float("AQI_HTTP_BACKOFF", 2)

# ---- Feature store ----------------------------------------------------
# Default: local parquet files (works with zero external accounts, and is
# what GitHub Actions will commit back to the repo or push to cheap storage
# e.g. an S3 bucket / Hopsworks). Swap FEATURE_STORE_BACKEND to "hopsworks"
# once you've created a free Hopsworks project — see README.
FEATURE_STORE_BACKEND = _env_str("FEATURE_STORE_BACKEND", "local")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
FEATURES_PATH = os.path.join(DATA_DIR, "features.parquet")

HOPSWORKS_API_KEY = _env_str("HOPSWORKS_API_KEY", "")
HOPSWORKS_PROJECT = _env_str("HOPSWORKS_PROJECT", "")
HOPSWORKS_FEATURE_GROUP = _env_str("HOPSWORKS_FEATURE_GROUP", "aqi_features")
HOPSWORKS_FG_VERSION = int(_env_float("HOPSWORKS_FG_VERSION", 1))
HOPSWORKS_MODEL_REGISTRY = _env_str("HOPSWORKS_MODEL_REGISTRY", "aqi_forecaster")

# ---- Model registry ----------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# ---- Forecast horizons (hours ahead) -----------------------------------
HORIZONS = [24, 48, 72]  # next 3 days

# A forecast anchored to a feature row older than this is reported as stale,
# because every lag feature it depends on describes the past, not the present.
STALE_AFTER_HOURS = _env_float("AQI_STALE_AFTER_HOURS", 6)

# ---- AQI hazard thresholds (US EPA scale, 0-500) -----------------------
# Ranges are half-open on the upper edge (lo <= aqi < hi) so that fractional
# values such as 50.4 cannot fall between two categories. The final category
# is closed at the top and absorbs anything above 500.
AQI_CATEGORIES = [
    (0, 51, "Good"),
    (51, 101, "Moderate"),
    (101, 151, "Unhealthy for Sensitive Groups"),
    (151, 201, "Unhealthy"),
    (201, 301, "Very Unhealthy"),
    (301, 501, "Hazardous"),
]
ALERT_THRESHOLD = 150  # trigger an alert at/above "Unhealthy"

# Colour per category, used by the dashboard. Hues follow the EPA's own
# published AQI colour scale.
AQI_CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffd700",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff2d2d",
    "Very Unhealthy": "#9d4bff",
    "Hazardous": "#9d174d",
}
