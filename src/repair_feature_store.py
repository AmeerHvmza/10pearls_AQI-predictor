"""
One-off maintenance: recompute the `aqi` column from the stored pollutant
concentrations using the corrected EPA conversion.

Rows written before the breakpoint-truncation fix could land in the apparent
gap between two EPA buckets (e.g. PM2.5 = 12.03) and fall through to a
hardcoded 500.0. Concentrations were stored correctly, so the AQI can be
rebuilt in place without re-fetching anything from the API.

Run:
    python src/repair_feature_store.py            # report only
    python src/repair_feature_store.py --apply    # rewrite the store
"""
import argparse
import os
import shutil
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_feature_store, save_feature_store, to_hourly_grid, aqi_category
from feature_pipeline import owm_aqi_to_us_aqi, add_derived_features

POLLUTANT_COLS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]


def run(apply: bool):
    df = load_feature_store()
    if df.empty:
        raise SystemExit("Feature store is empty; nothing to repair.")

    available = [c for c in POLLUTANT_COLS if c in df.columns]
    if not available:
        raise SystemExit("No pollutant concentration columns found; cannot recompute AQI.")

    old_aqi = df["aqi"].copy()
    new_aqi = df[available].apply(lambda r: owm_aqi_to_us_aqi(r.to_dict()), axis=1)

    changed = (old_aqi.round(1) != new_aqi.round(1)) & new_aqi.notna()
    spurious_500 = (old_aqi == 500.0) & (new_aqi < 500.0)

    print(f"Rows: {len(df)}")
    print(f"AQI values changed: {int(changed.sum())}")
    print(f"  of which spurious 500s corrected: {int(spurious_500.sum())}")

    if spurious_500.any():
        sample = pd.DataFrame({
            "timestamp": df.loc[spurious_500, "timestamp"],
            "pm2_5": df.loc[spurious_500, "pm2_5"],
            "old_aqi": old_aqi[spurious_500],
            "new_aqi": new_aqi[spurious_500].round(1),
        }).head(10)
        sample["new_category"] = sample["new_aqi"].apply(aqi_category)
        print("\nSample of corrected rows:")
        print(sample.to_string(index=False))

    alert = config.ALERT_THRESHOLD
    print(f"\nRows at/above the alert threshold ({alert}): "
          f"{int((old_aqi >= alert).sum())} -> {int((new_aqi >= alert).sum())}")
    print(f"Mean AQI: {old_aqi.mean():.1f} -> {new_aqi.mean():.1f}")
    print(f"Max AQI:  {old_aqi.max():.1f} -> {new_aqi.max():.1f}")

    if not apply:
        print("\nDry run. Re-run with --apply to rewrite the feature store.")
        return

    backup = config.FEATURES_PATH + ".bak"
    shutil.copy2(config.FEATURES_PATH, backup)
    print(f"\nBacked up original -> {backup}")

    df["aqi"] = new_aqi
    df = df.dropna(subset=["aqi"])
    df = add_derived_features(to_hourly_grid(df))

    # save_feature_store merges against what is already on disk, and the stale
    # AQI values are still there. Write the repaired frame directly instead.
    os.remove(config.FEATURES_PATH)
    save_feature_store(df)
    print("Feature store repaired. Re-run src/train_pipeline.py to retrain on corrected targets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Rewrite the feature store (default is a dry run).")
    args = parser.parse_args()
    run(args.apply)
