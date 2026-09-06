"""Cached read-only accessors over `src/predict.py` and the feature store.

These are thin wrappers: no forecasting, scoring, or feature engineering
happens here, only caching and column selection for display.
"""
import bootstrap  # noqa: F401  (sys.path side effect)
import pandas as pd
import streamlit as st

import config
from predict import (
    get_current_aqi, get_forecast, get_history, get_model_metrics,
    get_shap_importance,
)
from utils import describe_gaps, load_feature_store

POLLUTANT_COLUMNS = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]


@st.cache_data(ttl=300, show_spinner="Loading feature store…")
def load_core():
    """Current reading, forecast, and registry metrics.

    Feature engineering over the full store is the slow part, so both pages
    share this one cached call.
    """
    store = load_feature_store()
    # After the hourly-grid reindex the store has no missing timestamps, but
    # hours that were never measured are present with a null AQI.
    unmeasured = int(store["aqi"].isna().sum()) if "aqi" in store.columns else 0
    forecast, forecast_issues = get_forecast()
    return {
        "current": get_current_aqi(),
        "forecast": forecast,
        "forecast_issues": forecast_issues,
        "metrics": get_model_metrics(),
        "unmeasured_hours": unmeasured + sum(g[2] for g in describe_gaps(store)),
    }


@st.cache_data(ttl=300, show_spinner=False)
def load_history(hours: int) -> pd.DataFrame:
    return get_history(hours)


@st.cache_data(ttl=300, show_spinner=False)
def load_shap_importance(horizon: int | None = None):
    """Ranked SHAP importances for one horizon, defaulting to the shortest —
    the one the Dashboard's headline forecast uses."""
    if horizon is None:
        horizon = min(config.HORIZONS)
    return get_shap_importance(horizon)


@st.cache_data(ttl=300, show_spinner=False)
def load_pollutants(hours: int = 24):
    """Latest pollutant concentrations plus a short hourly tail for sparklines.

    Returns (latest_values, recent_series). Only pollutants with a non-null
    current reading are included, so the grid never shows an empty card.
    """
    store = load_feature_store()
    if store.empty:
        return {}, {}

    available = [c for c in POLLUTANT_COLUMNS if c in store.columns]
    if not available:
        return {}, {}

    recent = store.sort_values("timestamp").tail(hours)
    latest_row = recent.iloc[-1]
    latest = {c: float(latest_row[c]) for c in available if pd.notna(latest_row[c])}
    series = {
        c: [float(v) for v in recent[c].dropna().tolist()]
        for c in latest
    }
    return latest, series
