"""
Feature pipeline (Step 1+2+3 of the brief).

Fetches current weather + pollution data, computes model-ready features,
and writes them to the feature store. Designed to be run every hour
by GitHub Actions (see .github/workflows/feature_pipeline.yml).

Run manually:
    python src/feature_pipeline.py
"""
import os
import sys
import math
import time
import datetime as dt
import requests
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_feature_store, save_feature_store, to_hourly_grid


class UpstreamError(RuntimeError):
    """An API call failed in a way that retrying will not fix."""


def _get_json(url: str, params: dict) -> dict:
    """GET with bounded retries and clear messages for the failure modes the
    OpenWeather free tier actually produces (bad key, rate limit, blips)."""
    last_error = None
    for attempt in range(1, config.HTTP_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=config.HTTP_TIMEOUT)
        except requests.RequestException as exc:
            last_error = f"network error: {exc}"
        else:
            if r.status_code == 401:
                raise UpstreamError(
                    "OpenWeather rejected the API key (401). A newly created key "
                    "can take up to 2 hours to activate."
                )
            if r.status_code == 429:
                last_error = "rate limited (429)"
            elif r.status_code >= 500:
                last_error = f"upstream error ({r.status_code})"
            elif not r.ok:
                raise UpstreamError(f"{url} returned {r.status_code}: {r.text[:200]}")
            else:
                try:
                    return r.json()
                except ValueError:
                    last_error = "response was not valid JSON"

        if attempt < config.HTTP_RETRIES:
            delay = config.HTTP_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  {last_error}; retrying in {delay:.0f}s "
                  f"({attempt}/{config.HTTP_RETRIES - 1})")
            time.sleep(delay)

    raise UpstreamError(f"{url} failed after {config.HTTP_RETRIES} attempts: {last_error}")


def fetch_pollution(lat: float, lon: float) -> dict:
    data = _get_json(config.AIR_POLLUTION_URL, {
        "lat": lat, "lon": lon, "appid": config.OPENWEATHER_API_KEY
    })
    entries = data.get("list") or []
    if not entries:
        raise UpstreamError("Air-pollution response contained no readings.")
    entry = entries[0]
    if "components" not in entry:
        raise UpstreamError(f"Unexpected air-pollution payload: {str(entry)[:200]}")
    return {"aqi_raw": entry.get("main", {}).get("aqi"), **entry["components"]}


def fetch_weather(lat: float, lon: float) -> dict:
    data = _get_json(config.WEATHER_URL, {
        "lat": lat, "lon": lon, "appid": config.OPENWEATHER_API_KEY, "units": "metric"
    })
    if "main" not in data:
        raise UpstreamError(f"Unexpected weather payload: {str(data)[:200]}")
    return {
        "temp": data["main"].get("temp"),
        "humidity": data["main"].get("humidity"),
        "pressure": data["main"].get("pressure"),
        "wind_speed": data.get("wind", {}).get("speed", 0.0),
        "wind_deg": data.get("wind", {}).get("deg", 0),
        "clouds": data.get("clouds", {}).get("all", 0),
    }


# ---------------------------------------------------------------------------
# US EPA AQI
# ---------------------------------------------------------------------------
# Each entry: (concentration_low, concentration_high, aqi_low, aqi_high).
# The apparent gaps between buckets (12.0 -> 12.1, 35.4 -> 35.5, ...) are not
# gaps in the standard: the EPA requires the concentration to be TRUNCATED to
# the table's precision before lookup, which is what `_truncate` does below.
# Skipping that step lets values like 12.03 match no bucket at all.
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
    (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
]
PM10_BREAKPOINTS = [
    (0, 54, 0, 50), (55, 154, 51, 100), (155, 254, 101, 150),
    (255, 354, 151, 200), (355, 424, 201, 300),
    (425, 504, 301, 400), (505, 604, 401, 500),
]
O3_8H_BREAKPOINTS = [  # ppm
    (0.000, 0.054, 0, 50), (0.055, 0.070, 51, 100), (0.071, 0.085, 101, 150),
    (0.086, 0.105, 151, 200), (0.106, 0.200, 201, 300),
]
CO_8H_BREAKPOINTS = [  # ppm
    (0.0, 4.4, 0, 50), (4.5, 9.4, 51, 100), (9.5, 12.4, 101, 150),
    (12.5, 15.4, 151, 200), (15.5, 30.4, 201, 300),
    (30.5, 40.4, 301, 400), (40.5, 50.4, 401, 500),
]
SO2_1H_BREAKPOINTS = [  # ppb
    (0, 35, 0, 50), (36, 75, 51, 100), (76, 185, 101, 150),
    (186, 304, 151, 200), (305, 604, 201, 300),
    (605, 804, 301, 400), (805, 1004, 401, 500),
]
NO2_1H_BREAKPOINTS = [  # ppb
    (0, 53, 0, 50), (54, 100, 51, 100), (101, 360, 101, 150),
    (361, 649, 151, 200), (650, 1249, 201, 300),
    (1250, 1649, 301, 400), (1650, 2049, 401, 500),
]

# Ideal-gas conversion at 25 °C / 1 atm: ppb = (ug/m3) * 24.45 / molecular weight.
MOLAR_VOLUME = 24.45
MOLECULAR_WEIGHTS = {"o3": 48.00, "co": 28.01, "so2": 64.06, "no2": 46.0055}


def _ugm3_to_ppb(value: float, pollutant: str) -> float:
    return value * MOLAR_VOLUME / MOLECULAR_WEIGHTS[pollutant]


def _truncate(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def _sub_index(concentration, breakpoints, decimals: int):
    """Piecewise-linear AQI sub-index for one pollutant, or None if the
    reading is absent/invalid."""
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return None
    concentration = float(concentration)
    if concentration < 0:
        return None

    c = _truncate(concentration, decimals)
    for c_lo, c_hi, a_lo, a_hi in breakpoints:
        if c_lo <= c <= c_hi:
            return (a_hi - a_lo) / (c_hi - c_lo) * (c - c_lo) + a_lo

    # Above the top of the table the AQI is capped at the table's maximum
    # rather than defaulting to 500, which would misreport an out-of-range
    # ozone reading as the worst possible air quality.
    return float(breakpoints[-1][3]) if c > breakpoints[-1][1] else None


def aqi_sub_indices(components: dict) -> dict:
    """Per-pollutant AQI sub-indices from OpenWeather concentrations (ug/m3)."""
    subs = {}

    def add(name, value, breakpoints, decimals, converter=None):
        if value is None:
            return
        try:
            converted = converter(float(value)) if converter else float(value)
        except (TypeError, ValueError):
            return
        score = _sub_index(converted, breakpoints, decimals)
        if score is not None:
            subs[name] = score

    add("pm2_5", components.get("pm2_5"), PM25_BREAKPOINTS, 1)
    add("pm10", components.get("pm10"), PM10_BREAKPOINTS, 0)
    add("o3", components.get("o3"), O3_8H_BREAKPOINTS, 3,
        lambda v: _ugm3_to_ppb(v, "o3") / 1000)
    add("co", components.get("co"), CO_8H_BREAKPOINTS, 1,
        lambda v: _ugm3_to_ppb(v, "co") / 1000)
    add("so2", components.get("so2"), SO2_1H_BREAKPOINTS, 0,
        lambda v: _ugm3_to_ppb(v, "so2"))
    add("no2", components.get("no2"), NO2_1H_BREAKPOINTS, 0,
        lambda v: _ugm3_to_ppb(v, "no2"))
    return subs


def owm_aqi_to_us_aqi(components: dict) -> float:
    """Convert OpenWeather pollutant concentrations to the US EPA AQI (0-500).

    The overall AQI is the maximum of the per-pollutant sub-indices, which is
    the EPA's definition. OpenWeather's own 1-5 index is not that scale.

    Caveat worth stating in any write-up: the EPA defines these breakpoints
    against averaging windows we do not have here (24h for PM, 8h for O3/CO),
    so an instantaneous hourly reading is used as a proxy. That biases the
    result high during short spikes and is a limitation of the data source,
    not of this function.
    """
    subs = aqi_sub_indices(components)
    if not subs:
        return float("nan")
    return round(max(subs.values()), 1)


def dominant_pollutant(components: dict):
    """Which pollutant is setting the AQI right now."""
    subs = aqi_sub_indices(components)
    return max(subs, key=subs.get) if subs else None


def build_feature_row(timestamp: dt.datetime, pollution: dict, weather: dict) -> dict:
    row = {
        "timestamp": timestamp,
        "hour": timestamp.hour,
        "day": timestamp.day,
        "day_of_week": timestamp.weekday(),
        "month": timestamp.month,
        "is_weekend": int(timestamp.weekday() >= 5),
        **weather,
        **pollution,
    }
    row["aqi"] = owm_aqi_to_us_aqi(pollution)
    return row


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds lag / rate-of-change features once enough history exists.

    Expects a continuous hourly index (see `utils.to_hourly_grid`); the shifts
    below are positional, so a missing hour would silently misalign them.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_rolling_mean_6h"] = df["aqi"].rolling(6, min_periods=1).mean()
    return df


def run():
    if not config.OPENWEATHER_API_KEY:
        raise SystemExit(
            "OPENWEATHER_API_KEY is not set.\n"
            "  Local:  export OPENWEATHER_API_KEY=your_key   (PowerShell: $env:OPENWEATHER_API_KEY=\"your_key\")\n"
            "  CI:     add it under Settings > Secrets and variables > Actions > Secrets."
        )

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, minute=0, second=0, microsecond=0)

    try:
        pollution = fetch_pollution(config.LATITUDE, config.LONGITUDE)
        weather = fetch_weather(config.LATITUDE, config.LONGITUDE)
    except UpstreamError as exc:
        raise SystemExit(f"Feature pipeline aborted: {exc}")

    row = build_feature_row(now, pollution, weather)
    if pd.isna(row["aqi"]):
        raise SystemExit("Pollution response had no usable pollutant concentrations.")

    save_feature_store(pd.DataFrame([row]))

    # Recompute derived (lag/rolling) features over the whole history and
    # persist so downstream training always has them up to date. The grid
    # reindex keeps those lags anchored to real hours across any gaps left by
    # failed runs.
    full = add_derived_features(to_hourly_grid(load_feature_store()))
    save_feature_store(full)

    driver = dominant_pollutant(pollution)
    print(f"[{now}Z] AQI={row['aqi']} (driver: {driver}, OWM index: {row['aqi_raw']}) "
          f"written for {config.CITY_NAME}")


if __name__ == "__main__":
    run()
