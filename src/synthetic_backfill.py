"""
Synthetic backfill: generates realistic historical AQI + weather data for
Karachi so the full pipeline can run end-to-end while the OpenWeather API
key activates (takes up to 2 hours for new keys).

Once your key is active, re-run:
    python src/backfill.py --days 90
to replace this with real data and retrain.

Run:
    python src/synthetic_backfill.py --days 90
"""
import argparse
import datetime as dt
import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import save_feature_store, to_hourly_grid
from feature_pipeline import owm_aqi_to_us_aqi, add_derived_features

np.random.seed(42)


def generate_synthetic_data(days: int) -> pd.DataFrame:
    """
    Generates hourly synthetic data mimicking Karachi's real climate and
    AQI patterns:
    - Temperature: 25-40°C with diurnal and seasonal variation
    - Humidity: 40-90% inversely correlated with temp
    - PM2.5: base 40-80 µg/m³ (typical for Karachi) with rush-hour spikes
    - Wind: monsoon-influenced with seasonal shifts
    """
    end = dt.datetime.now(dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start = end - dt.timedelta(days=days)
    timestamps = pd.date_range(start, end, freq="h")
    n = len(timestamps)

    hours = np.array([t.hour for t in timestamps])
    day_of_year = np.array([t.timetuple().tm_yday for t in timestamps])

    # --- Temperature (°C) ---
    # Karachi: ~22°C winter, ~35°C summer, diurnal swing ~6°C
    seasonal_temp = 28 + 7 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
    diurnal_temp = 3 * np.sin(2 * np.pi * (hours - 6) / 24)
    temp = seasonal_temp + diurnal_temp + np.random.normal(0, 1.5, n)

    # --- Humidity (%) ---
    # Monsoon (Jul-Sep) higher humidity
    monsoon_boost = 15 * np.exp(-0.5 * ((day_of_year - 210) / 40) ** 2)
    humidity = 55 + monsoon_boost - 0.6 * (temp - 28) + np.random.normal(0, 5, n)
    humidity = np.clip(humidity, 20, 98)

    # --- Pressure (hPa) ---
    pressure = 1010 + 5 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 2, n)

    # --- Wind ---
    wind_speed = 3 + 2 * np.sin(2 * np.pi * (day_of_year - 150) / 365) + np.random.exponential(1.5, n)
    wind_deg = (220 + 40 * np.sin(2 * np.pi * day_of_year / 365) + np.random.normal(0, 30, n)) % 360

    # --- Clouds (%) ---
    clouds = 30 + monsoon_boost * 1.5 + np.random.normal(0, 15, n)
    clouds = np.clip(clouds, 0, 100)

    # --- Pollutants (µg/m³) ---
    # PM2.5 is the primary AQI driver in Karachi
    # Rush hour spikes at 8-10 AM and 5-8 PM
    rush_morning = 15 * np.exp(-0.5 * ((hours - 9) / 1.5) ** 2)
    rush_evening = 20 * np.exp(-0.5 * ((hours - 18) / 2) ** 2)
    # Winter is worse (inversion layer)
    winter_factor = 1 + 0.4 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    # Wind disperses pollutants
    wind_factor = 1 - 0.15 * np.clip(wind_speed / 10, 0, 1)

    pm25_base = 45 * winter_factor * wind_factor
    pm25 = pm25_base + rush_morning + rush_evening + np.random.exponential(8, n)
    pm25 = np.clip(pm25, 2, 300)

    pm10 = pm25 * (1.5 + np.random.normal(0, 0.2, n))
    pm10 = np.clip(pm10, 5, 500)

    # Other pollutants correlated with traffic patterns
    co = 300 + 200 * (rush_morning + rush_evening) / 20 + np.random.normal(0, 50, n)
    no = 5 + 15 * (rush_morning + rush_evening) / 20 + np.random.exponential(3, n)
    no2 = 15 + 25 * (rush_morning + rush_evening) / 20 + np.random.normal(0, 5, n)
    o3 = 40 + 30 * np.sin(2 * np.pi * (hours - 14) / 24) + np.random.normal(0, 10, n)  # peaks afternoon
    o3 = np.clip(o3, 5, 150)
    so2 = 8 + np.random.exponential(4, n)
    nh3 = 5 + np.random.exponential(3, n)

    # OWM raw AQI (1-5 scale)
    aqi_raw = np.clip(np.round(pm25 / 25), 1, 5).astype(int)

    rows = []
    for i, ts in enumerate(timestamps):
        pollution = {
            "aqi_raw": int(aqi_raw[i]),
            "co": round(float(co[i]), 2),
            "no": round(float(no[i]), 2),
            "no2": round(float(no2[i]), 2),
            "o3": round(float(o3[i]), 2),
            "so2": round(float(so2[i]), 2),
            "pm2_5": round(float(pm25[i]), 2),
            "pm10": round(float(pm10[i]), 2),
            "nh3": round(float(nh3[i]), 2),
        }
        row = {
            "timestamp": ts.to_pydatetime().replace(tzinfo=None),
            "hour": ts.hour,
            "day": ts.day,
            "day_of_week": ts.weekday(),
            "month": ts.month,
            "is_weekend": int(ts.weekday() >= 5),
            "temp": round(float(temp[i]), 1),
            "humidity": round(float(humidity[i]), 1),
            "pressure": round(float(pressure[i]), 1),
            "wind_speed": round(float(wind_speed[i]), 1),
            "wind_deg": round(float(wind_deg[i]), 1),
            "clouds": round(float(clouds[i]), 1),
            **pollution,
        }
        row["aqi"] = owm_aqi_to_us_aqi(pollution)
        rows.append(row)

    df = pd.DataFrame(rows)
    return add_derived_features(to_hourly_grid(df))


def run(days: int):
    print(f"Generating {days} days of synthetic Karachi AQI data...")
    df = generate_synthetic_data(days)
    save_feature_store(df)
    print(f"Synthetic backfill complete: {len(df)} hourly rows.")
    print(f"AQI range: {df['aqi'].min():.0f} – {df['aqi'].max():.0f}")
    print(f"Mean AQI: {df['aqi'].mean():.1f}")
    print(f"\nOnce your OpenWeather key activates, replace with real data:")
    print(f"  python src/backfill.py --days 90")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=90)
    args = parser.parse_args()
    run(args.days)
