"""Loads the latest features + registered models and produces a 3-day forecast."""
import os
import sys
import joblib
import datetime as dt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import load_feature_store, aqi_category, to_hourly_grid
from feature_pipeline import dominant_pollutant
from train_pipeline import engineer_features, predict_with_interval


def load_model(horizon: int):
    path = os.path.join(config.MODELS_DIR, f"model_{horizon}h.joblib")
    if not os.path.exists(path):
        return None
    try:
        bundle = joblib.load(path)
    except Exception as exc:
        print(f"Could not load model_{horizon}h.joblib ({type(exc).__name__}: {exc})")
        return None

    # Keras networks are stored beside the bundle rather than inside it.
    if bundle.get("is_sequence"):
        keras_path = bundle.get("keras_path")
        if not keras_path or not os.path.exists(keras_path):
            print(f"Warning: {horizon}h bundle expects a Keras model at "
                  f"{keras_path}, which is missing; skipping this horizon.")
            return None
        try:
            from lstm_model import LSTMForecaster
            bundle["model"] = LSTMForecaster.load(
                keras_path, n_features=len(bundle["feature_cols"]),
                seq_len=bundle["seq_len"])
        except Exception as exc:
            print(f"Warning: could not load the {horizon}h LSTM "
                  f"({type(exc).__name__}: {exc}); skipping this horizon.")
            return None
    return bundle


def _prepared_history() -> pd.DataFrame:
    """Feature store on a continuous hourly grid with training features applied."""
    df = load_feature_store()
    if df.empty:
        return df
    return engineer_features(to_hourly_grid(df))


def get_forecast() -> pd.DataFrame:
    """Returns a DataFrame: horizon_hours, forecast_time, predicted_aqi,
    aqi_lower, aqi_upper, category, model_used."""
    df = _prepared_history()
    if df.empty:
        return pd.DataFrame()

    # The grid can end on placeholder rows for hours that were never measured;
    # anchor the forecast to the most recent row that actually has an AQI.
    df = df.sort_values("timestamp")
    observed = df[df["aqi"].notna()]
    if observed.empty:
        return pd.DataFrame()
    latest = observed.iloc[[-1]]
    anchor = latest["timestamp"].iloc[0]
    # Models predict the change from this value, so it is also what the
    # persistence baseline would forecast.
    current_aqi = float(latest["aqi_capped"].iloc[0])

    rows = []
    for horizon in config.HORIZONS:
        bundle = load_model(horizon)
        if bundle is None:
            continue

        feature_cols = bundle["feature_cols"]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            print(f"Warning: missing features for {horizon}h model: {missing[:5]}"
                  f"{'...' if len(missing) > 5 else ''}")
            continue

        if bundle.get("is_sequence"):
            # The LSTM needs the last `seq_len` consecutive hours ending at the
            # anchor, not just the anchor row.
            seq_len = bundle["seq_len"]
            window = df[df["timestamp"] <= anchor].tail(seq_len)
            if len(window) < seq_len:
                print(f"Warning: {horizon}h LSTM needs {seq_len} consecutive "
                      f"hours; only {len(window)} available.")
                continue
            X = window[feature_cols].astype(float).values
        else:
            X = latest[feature_cols].astype(float).values
        # Present-but-null is the common failure: the longest lag (48h) is NaN
        # until the store has that much history, and sklearn raises on NaN.
        # Report which feature is not ready instead of crashing the dashboard.
        if np.isnan(X).any():
            not_ready = [c for c, v in zip(feature_cols, X[-1]) if np.isnan(v)]
            print(f"Warning: {horizon}h model needs more history; "
                  f"null features: {not_ready[:5]}"
                  f"{'...' if len(not_ready) > 5 else ''}")
            continue

        try:
            X_scaled = bundle["scaler"].transform(X)
            if bundle.get("is_sequence"):
                # (timesteps, features) -> a batch of one sequence.
                X_scaled = X_scaled[np.newaxis, :, :].astype("float32")
            pred, lower, upper = predict_with_interval(bundle, X_scaled,
                                                       current_aqi=current_aqi)
            ml_value = float(np.clip(pred[0], 0, 500))
            ml_lower = float(lower[0]) if lower is not None else None
            ml_upper = float(upper[0]) if upper is not None else None
        except Exception as exc:
            print(f"Warning: {horizon}h prediction failed "
                  f"({type(exc).__name__}: {exc}); skipping.")
            continue

        # Training records whether the trained model actually beat "assume no
        # change" out of sample. When it did not, serving the model anyway
        # would knowingly publish the worse forecast.
        beats_baseline = bundle.get("beats_baseline", True)
        if beats_baseline:
            value, low, high = ml_value, ml_lower, ml_upper
            method = bundle.get("model_name", "unknown")
        else:
            value = current_aqi
            sigma = bundle.get("baseline_residual_std") or 0.0
            z = bundle.get("interval_z", 1.2816)
            low = float(np.clip(value - z * sigma, 0, 500))
            high = float(np.clip(value + z * sigma, 0, 500))
            method = "persistence"

        rows.append({
            "horizon_hours": horizon,
            "forecast_time": anchor + dt.timedelta(hours=horizon),
            "predicted_aqi": round(value, 1),
            "aqi_lower": round(low, 1) if low is not None else None,
            "aqi_upper": round(high, 1) if high is not None else None,
            "category": aqi_category(value),
            "method": method,
            "model_used": bundle.get("model_name", "unknown"),
            "beats_baseline": bool(beats_baseline),
            "ml_predicted_aqi": round(ml_value, 1),
        })
    return pd.DataFrame(rows)


def get_model_metrics() -> dict:
    """Registry metrics per horizon, for the dashboard's model panel."""
    import json
    out = {}
    for horizon in config.HORIZONS:
        path = os.path.join(config.MODELS_DIR, f"metrics_{horizon}h.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                out[horizon] = json.load(f)
        except (OSError, ValueError):
            continue
    return out


def shap_plot_status(horizon: int):
    """Return (path, is_stale) for a horizon's SHAP plot, or None.

    A plot older than the model it claims to explain was generated from
    different data — for example when `shap` was unavailable during the most
    recent training run and the previous PNG was left in place.
    """
    plot_path = os.path.join(config.MODELS_DIR, f"shap_summary_{horizon}h.png")
    model_path = os.path.join(config.MODELS_DIR, f"model_{horizon}h.joblib")
    if not os.path.exists(plot_path):
        return None
    stale = (os.path.exists(model_path)
             and os.path.getmtime(plot_path) < os.path.getmtime(model_path))
    return plot_path, stale


def get_current_aqi():
    df = load_feature_store()
    if df.empty:
        return None

    observed = df[df["aqi"].notna()] if "aqi" in df.columns else df
    if observed.empty:
        return None
    latest = observed.sort_values("timestamp").iloc[-1]
    timestamp = pd.to_datetime(latest["timestamp"]).to_pydatetime()
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    age_hours = max(0.0, (now - timestamp).total_seconds() / 3600)

    pollutants = {c: latest[c] for c in
                  ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
                  if c in latest.index and pd.notna(latest[c])}

    return {
        "timestamp": timestamp,
        "aqi": float(latest["aqi"]),
        "category": aqi_category(latest["aqi"]),
        "pm2_5": float(latest["pm2_5"]) if pd.notna(latest.get("pm2_5")) else None,
        "dominant_pollutant": dominant_pollutant(pollutants) if pollutants else None,
        "age_hours": age_hours,
        # Every lag feature the forecast depends on is anchored to this
        # timestamp, so a stale store means a stale forecast, not a wrong one.
        "is_stale": age_hours > config.STALE_AFTER_HOURS,
    }


def get_history(hours: int = 24 * 14) -> pd.DataFrame:
    """Recent observed AQI, oldest first, for the dashboard chart."""
    df = load_feature_store()
    if df.empty:
        return df
    cols = [c for c in ["timestamp", "aqi", "pm2_5", "is_imputed"] if c in df.columns]
    return df.sort_values("timestamp")[cols].tail(hours).reset_index(drop=True)


if __name__ == "__main__":
    current = get_current_aqi()
    print(current)
    if current and current["is_stale"]:
        print(f"WARNING: latest feature row is {current['age_hours']:.0f}h old; "
              f"run src/feature_pipeline.py to refresh.")
    print(get_forecast().to_string())
