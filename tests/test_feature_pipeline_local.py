"""Smoke: feature_pipeline completes with Hopsworks off (parquet only)."""
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, ROOT)
sys.path.insert(0, SRC)

os.environ.pop("HOPSWORKS_ENABLED", None)

import config  # noqa: E402
import feature_pipeline  # noqa: E402
import utils  # noqa: E402


FAKE_POLLUTION = {
    "aqi_raw": 2,
    "co": 220.0,
    "no": 0.4,
    "no2": 12.0,
    "o3": 48.0,
    "so2": 3.0,
    "pm2_5": 18.0,
    "pm10": 31.0,
    "nh3": 1.2,
}
FAKE_WEATHER = {
    "temp": 31.2,
    "humidity": 58.0,
    "pressure": 1011.0,
    "wind_speed": 3.4,
    "wind_deg": 210.0,
    "clouds": 22.0,
}


class LocalOnlyFeaturePipeline(unittest.TestCase):
    def test_hopsworks_defaults_off(self):
        self.assertFalse(config.HOPSWORKS_ENABLED)

    def test_feature_pipeline_runs_without_hopsworks(self):
        with tempfile.TemporaryDirectory() as tmp:
            parquet = os.path.join(tmp, "features.parquet")
            hopsworks_calls = []

            def boom(*_args, **_kwargs):
                hopsworks_calls.append(1)
                raise AssertionError("Hopsworks must not be called when disabled")

            use_live = bool(config.OPENWEATHER_API_KEY)
            fetch_patches = []
            if not use_live:
                fetch_patches = [
                    patch.object(feature_pipeline, "fetch_pollution",
                                 return_value=FAKE_POLLUTION),
                    patch.object(feature_pipeline, "fetch_weather",
                                 return_value=FAKE_WEATHER),
                    patch.object(config, "OPENWEATHER_API_KEY", "test-key"),
                ]
            for p in fetch_patches:
                p.start()
            try:
                with patch.object(config, "DATA_DIR", tmp), \
                     patch.object(config, "FEATURES_PATH", parquet), \
                     patch.object(config, "HOPSWORKS_ENABLED", False), \
                     patch.object(utils, "_try_hopsworks_dual_write", boom), \
                     patch.object(utils, "_save_to_hopsworks", boom), \
                     patch.object(utils, "hopsworks_login", boom):
                    feature_pipeline.run()
            finally:
                for p in fetch_patches:
                    p.stop()

            self.assertEqual(hopsworks_calls, [])
            self.assertTrue(os.path.exists(parquet), "parquet store was not written")
            df = pd.read_parquet(parquet)
            self.assertFalse(df.empty)
            self.assertIn("aqi", df.columns)
            self.assertTrue(df["aqi"].notna().any())
            print(f"local-only smoke ok ({'live API' if use_live else 'mocked fetches'}): "
                  f"{len(df)} row(s) -> {parquet}")

    def test_hopsworks_failure_does_not_block_parquet_write(self):
        """Enabled dual-write must not make save_feature_store fail."""
        with tempfile.TemporaryDirectory() as tmp:
            parquet = os.path.join(tmp, "features.parquet")
            row = pd.DataFrame([{
                "timestamp": pd.Timestamp("2026-01-01 00:00:00"),
                "aqi": 42.0,
            }])
            with patch.object(config, "DATA_DIR", tmp), \
                 patch.object(config, "FEATURES_PATH", parquet), \
                 patch.object(config, "HOPSWORKS_ENABLED", True), \
                 patch.object(utils, "_save_to_hopsworks",
                              side_effect=RuntimeError("no hopsworks account")):
                utils.save_feature_store(row)
            self.assertTrue(os.path.exists(parquet))
            stored = pd.read_parquet(parquet)
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored.iloc[0]["aqi"], 42.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
