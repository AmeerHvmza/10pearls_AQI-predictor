"""
Backfill historical (features, targets) so there is enough data to train on.

Pollution history comes from OpenWeather (free tier reaches back to ~Nov 2020)
and matching weather history comes from Open-Meteo's free archive API, which
needs no key. Both sources are queried in UTC and joined on the hour.

Run:
    python src/backfill.py --days 90
"""
import argparse
import datetime as dt
import json
import os
import sys
from urllib.parse import urlencode

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import save_feature_store, to_hourly_grid
from feature_pipeline import (
    owm_aqi_to_us_aqi, add_derived_features, _get_json, UpstreamError,
)

# Open-Meteo's reanalysis archive trails real time by roughly five days.
# Asking for more recent hours returns nulls, not an error.
ARCHIVE_LAG_DAYS = 5

# OpenWeather accepts long history ranges but is far more reliable in chunks.
CHUNK_DAYS = 30

WEATHER_FIELDS = {
    "temperature_2m": "temp",
    "relative_humidity_2m": "humidity",
    "surface_pressure": "pressure",
    "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_deg",
    "cloud_cover": "clouds",
}


def _redacted_url(url: str, params: dict) -> str:
    safe = {
        k: ("REDACTED" if str(k).lower() in {"appid", "api_key", "apikey", "key"} else v)
        for k, v in params.items()
    }
    return f"{url}?{urlencode(safe)}"


def _audited_get(url: str, params: dict, kind: str) -> dict:
    redacted = _redacted_url(url, params)
    print(f"HTTP GET {redacted}", flush=True)
    data = _get_json(url, params)
    os.makedirs(config.DATA_DIR, exist_ok=True)
    if kind == "pollution":
        n_list = len(data.get("list") or [])
        print(f"  -> HTTP 200, list entries: {n_list}", flush=True)
        path = os.path.join(config.DATA_DIR, "gapfill_first_pollution.json")
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            print(f"  -> saved first pollution JSON -> {path}", flush=True)
    elif kind == "weather":
        hourly = (data.get("hourly") or {}).get("time") or []
        print(f"  -> HTTP 200, hourly timestamps: {len(hourly)}", flush=True)
        path = os.path.join(config.DATA_DIR, "gapfill_first_weather.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        print(f"  -> saved weather JSON -> {path}", flush=True)
    return data


def fetch_pollution_history(lat, lon, start_utc: dt.datetime, end_utc: dt.datetime) -> list:
    """Fetch pollution history in chunks. `start_utc`/`end_utc` must be
    timezone-aware so that `.timestamp()` yields true UTC epochs."""
    assert start_utc.tzinfo is not None and end_utc.tzinfo is not None

    entries = []
    chunk_start = start_utc
    while chunk_start < end_utc:
        chunk_end = min(chunk_start + dt.timedelta(days=CHUNK_DAYS), end_utc)
        print(f"  pollution {chunk_start:%Y-%m-%d} -> {chunk_end:%Y-%m-%d}")
        data = _audited_get(config.AIR_POLLUTION_HISTORY_URL, {
            "lat": lat, "lon": lon,
            "start": int(chunk_start.timestamp()),
            "end": int(chunk_end.timestamp()),
            "appid": config.OPENWEATHER_API_KEY,
        }, kind="pollution")
        entries.extend(data.get("list") or [])
        chunk_start = chunk_end
    return entries


def pollution_history_to_frame(entries: list) -> pd.DataFrame:
    rows = []
    for entry in entries:
        components = entry.get("components")
        if not components:
            continue
        ts = dt.datetime.fromtimestamp(entry["dt"], dt.timezone.utc).replace(
            tzinfo=None, minute=0, second=0, microsecond=0
        )
        rows.append({
            "timestamp": ts,
            "aqi_raw": entry.get("main", {}).get("aqi"),
            **components,
        })
    return pd.DataFrame(rows)


def fetch_weather_history_openmeteo(lat, lon, start_date: str, end_date: str) -> pd.DataFrame:
    """Free, no-API-key historical weather via Open-Meteo's archive API.

    `timezone=UTC` makes the returned `time` strings tz-naive UTC, matching the
    tz-naive UTC timestamps used everywhere else in the project.
    """
    data = _audited_get("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": ",".join(WEATHER_FIELDS),
        "timezone": "UTC",
    }, kind="weather")
    hourly = data.get("hourly")
    if not hourly or not hourly.get("time"):
        raise UpstreamError("Open-Meteo returned no hourly weather data.")

    df = pd.DataFrame(hourly)
    df["timestamp"] = pd.to_datetime(df["time"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df.drop(columns=["time"]).rename(columns=WEATHER_FIELDS)


def run(days: int):
    if not config.OPENWEATHER_API_KEY:
        raise SystemExit(
            "OPENWEATHER_API_KEY is not set.\n"
            "  Local: put OPENWEATHER_API_KEY in a project-root .env or export it."
        )

    # Keep these tz-aware: `.timestamp()` on a naive datetime is interpreted as
    # LOCAL time, which would shift the requested window by the machine's UTC
    # offset and desynchronise it from the Open-Meteo side of the join.
    end_utc = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_utc = end_utc - dt.timedelta(days=days)

    weather_end = min(end_utc, end_utc - dt.timedelta(days=ARCHIVE_LAG_DAYS))
    if weather_end < end_utc:
        print(f"Note: Open-Meteo's archive lags ~{ARCHIVE_LAG_DAYS} days, so weather "
              f"history stops at {weather_end:%Y-%m-%d}. The hourly feature "
              f"pipeline fills the remainder going forward.")

    try:
        print(f"Fetching {days} days of pollution history...")
        entries = fetch_pollution_history(config.LATITUDE, config.LONGITUDE,
                                          start_utc, end_utc)

        print("Fetching matching weather history from Open-Meteo (free archive)...")
        weather_df = fetch_weather_history_openmeteo(
            config.LATITUDE, config.LONGITUDE,
            start_utc.strftime("%Y-%m-%d"), weather_end.strftime("%Y-%m-%d"),
        )
    except UpstreamError as exc:
        raise SystemExit(f"Backfill aborted: {exc}")

    pollution_df = pollution_history_to_frame(entries)
    if pollution_df.empty:
        raise SystemExit("No pollution history returned for this location/period.")

    # Both sides are tz-naive UTC on the hour, so this is a plain inner join.
    merged = pollution_df.merge(weather_df, on="timestamp", how="inner")

    # The archive returns nulls (not errors) for hours it hasn't assimilated.
    weather_cols = list(WEATHER_FIELDS.values())
    complete = merged.dropna(subset=weather_cols)
    dropped_null = len(merged) - len(complete)
    unmatched = len(pollution_df) - len(merged)
    if unmatched or dropped_null:
        print(f"  {unmatched} pollution hour(s) had no weather row; "
              f"{dropped_null} more had null weather values. Both were skipped.")

    if complete.empty:
        raise SystemExit("Pollution and weather history did not overlap on any hour.")

    df = complete.copy()
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day"] = ts.dt.day
    df["day_of_week"] = ts.dt.dayofweek
    df["month"] = ts.dt.month
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    pollutant_cols = [c for c in ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
                      if c in df.columns]
    df["aqi"] = df[pollutant_cols].apply(lambda r: owm_aqi_to_us_aqi(r.to_dict()), axis=1)
    df = df.dropna(subset=["aqi"])

    df = add_derived_features(to_hourly_grid(df))
    save_feature_store(df)
    print(f"Backfilled {len(df)} hourly rows "
          f"({df['timestamp'].min()} -> {df['timestamp'].max()} UTC).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90,
                         help="How many days of history to backfill (OpenWeather free tier caps around 365).")
    args = parser.parse_args()
    run(args.days)
