"""
FastAPI serving layer for the AQI Predictor.

The Streamlit dashboard is the human interface; this is the machine interface,
so other services (or a grader with curl) can consume the same forecasts.
Both read from the identical feature store and model registry — there is no
second copy of the prediction logic here.

Run:
    uvicorn api.main:app --reload --port 8000

Then open http://localhost:8000/docs for interactive Swagger documentation.
"""
import datetime as dt
import os
import sys
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
from predict import (
    get_forecast, get_current_aqi, get_history, get_model_metrics,
)
from utils import load_feature_store, aqi_category

app = FastAPI(
    title="AQI Predictor API",
    description=(
        "3-day air quality forecasting for a configured city. Serves the same "
        "models and feature store as the Streamlit dashboard."
    ),
    version="1.0.0",
)

# The dashboard and any external client may run on a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    city: str
    feature_store_backend: str
    feature_rows: int
    models_available: List[int]
    latest_reading: Optional[dt.datetime] = None
    data_is_stale: Optional[bool] = None


class CurrentResponse(BaseModel):
    city: str
    timestamp: dt.datetime = Field(..., description="UTC, tz-naive")
    aqi: float
    category: str
    pm2_5: Optional[float] = None
    dominant_pollutant: Optional[str] = None
    age_hours: float
    is_stale: bool = Field(
        ..., description="True when the reading is older than STALE_AFTER_HOURS. "
                         "Forecasts are anchored to this timestamp, not to now."
    )


class ForecastPoint(BaseModel):
    horizon_hours: int
    forecast_time: dt.datetime
    predicted_aqi: float
    aqi_lower: Optional[float] = None
    aqi_upper: Optional[float] = None
    category: str
    method: str = Field(..., description="Model that produced this value, or "
                                         "'persistence' when the baseline won.")
    model_used: str
    beats_baseline: bool
    ml_predicted_aqi: float


class ForecastResponse(BaseModel):
    city: str
    anchored_to: dt.datetime
    is_stale: bool
    alert_threshold: int
    alert: bool
    forecast: List[ForecastPoint]
    issues: List[str] = Field(default_factory=list)


class HistoryPoint(BaseModel):
    timestamp: dt.datetime
    aqi: Optional[float] = None
    pm2_5: Optional[float] = None


class AlertResponse(BaseModel):
    city: str
    alert: bool
    threshold: int
    periods: List[str]
    message: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {
        "service": "AQI Predictor API",
        "city": config.CITY_NAME,
        "docs": "/docs",
        "endpoints": ["/health", "/current", "/forecast", "/history",
                      "/alerts", "/metrics"],
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    """Liveness plus a summary of what the service can actually serve."""
    store = load_feature_store()
    available = [h for h in config.HORIZONS
                 if os.path.exists(os.path.join(config.MODELS_DIR, f"model_{h}h.joblib"))]
    current = get_current_aqi()
    return HealthResponse(
        status="ok",
        city=config.CITY_NAME,
        feature_store_backend=config.FEATURE_STORE_BACKEND,
        feature_rows=len(store),
        models_available=available,
        latest_reading=current["timestamp"] if current else None,
        data_is_stale=current["is_stale"] if current else None,
    )


@app.get("/current", response_model=CurrentResponse, tags=["forecast"])
def current():
    """Most recent measured AQI."""
    result = get_current_aqi()
    if result is None:
        raise HTTPException(
            status_code=503,
            detail="Feature store is empty. Run src/backfill.py then "
                   "src/feature_pipeline.py.",
        )
    return CurrentResponse(city=config.CITY_NAME, **result)


@app.get("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast():
    """3-day AQI forecast, one point per configured horizon."""
    current_reading = get_current_aqi()
    if current_reading is None:
        raise HTTPException(status_code=503, detail="Feature store is empty.")

    df, issues = get_forecast()
    if df.empty:
        raise HTTPException(
            status_code=503,
            detail=issues or ["No horizon could be scored."],
        )

    points = [ForecastPoint(**row) for row in df.to_dict(orient="records")]
    triggered = (current_reading["aqi"] >= config.ALERT_THRESHOLD
                 or any(p.predicted_aqi >= config.ALERT_THRESHOLD for p in points))
    return ForecastResponse(
        city=config.CITY_NAME,
        anchored_to=current_reading["timestamp"],
        is_stale=current_reading["is_stale"],
        alert_threshold=config.ALERT_THRESHOLD,
        alert=triggered,
        forecast=points,
        issues=issues,
    )


@app.get("/history", response_model=List[HistoryPoint], tags=["forecast"])
def history(hours: int = Query(336, ge=1, le=24 * 365,
                               description="How many recent hours to return.")):
    """Recent observed AQI, oldest first."""
    df = get_history(hours)
    if df.empty:
        raise HTTPException(status_code=503, detail="Feature store is empty.")
    cols = [c for c in ["timestamp", "aqi", "pm2_5"] if c in df.columns]
    # NaN is not valid JSON; hours that were never measured serialise as null.
    return [
        HistoryPoint(**{k: (None if v != v else v) for k, v in row.items()})
        for row in df[cols].to_dict(orient="records")
    ]


@app.get("/alerts", response_model=AlertResponse, tags=["forecast"])
def alerts():
    """Whether hazardous AQI is expected now or within the forecast window."""
    current_reading = get_current_aqi()
    if current_reading is None:
        raise HTTPException(status_code=503, detail="Feature store is empty.")

    periods = []
    if current_reading["aqi"] >= config.ALERT_THRESHOLD:
        periods.append("now")

    df, _issues = get_forecast()
    if not df.empty:
        for row in df.itertuples():
            if row.predicted_aqi >= config.ALERT_THRESHOLD:
                periods.append(f"+{row.horizon_hours}h")

    if periods:
        message = (f"Unhealthy air (AQI >= {config.ALERT_THRESHOLD}) expected "
                   f"{', '.join(periods)}. Limit prolonged outdoor exertion.")
    else:
        message = (f"No AQI readings at or above {config.ALERT_THRESHOLD} "
                   f"expected in the next 3 days.")

    return AlertResponse(
        city=config.CITY_NAME,
        alert=bool(periods),
        threshold=config.ALERT_THRESHOLD,
        periods=periods,
        message=message,
    )


@app.get("/metrics", tags=["meta"])
def metrics():
    """Registry metrics per horizon, including the persistence comparison."""
    result = get_model_metrics()
    if not result:
        raise HTTPException(
            status_code=503,
            detail="No metrics found. Run src/train_pipeline.py.",
        )
    return result


@app.get("/categories", tags=["meta"])
def categories():
    """The EPA category scale this service reports against."""
    return [
        {"lower": lo, "upper": hi - 1, "label": label,
         "color": config.AQI_CATEGORY_COLORS[label],
         "alerting": lo >= config.ALERT_THRESHOLD}
        for lo, hi, label in config.AQI_CATEGORIES
    ]


@app.get("/categories/{aqi}", tags=["meta"])
def categorise(aqi: float):
    """Classify an arbitrary AQI value."""
    label = aqi_category(aqi)
    return {"aqi": aqi, "category": label,
            "color": config.AQI_CATEGORY_COLORS.get(label),
            "alerting": aqi >= config.ALERT_THRESHOLD}
