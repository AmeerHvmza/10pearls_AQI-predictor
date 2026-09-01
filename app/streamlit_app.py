"""
AQI Predictor dashboard.

Run:
    streamlit run app/streamlit_app.py
"""
import os
import sys
import datetime as dt
import html

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import config
from predict import (
    get_forecast, get_current_aqi, get_history, get_model_metrics, shap_plot_status,
)
from utils import aqi_category, category_color, describe_gaps, load_feature_store

st.set_page_config(
    page_title=f"AQI Predictor — {config.CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    # "auto" keeps the sidebar open on desktop but collapses it on narrow
    # viewports, where an expanded sidebar would cover the whole page.
    initial_sidebar_state="auto",
)


# ---------------------------------------------------------------------------
# Streamlit renamed the chart/image sizing argument; support both so the app
# runs on the version range pinned in requirements.txt.
# ---------------------------------------------------------------------------
def _stretch(render, *args, **kwargs):
    try:
        return render(*args, width="stretch", **kwargs)
    except TypeError:
        return render(*args, use_container_width=True, **kwargs)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ACCENT = "#22d3ee"
SURFACE = "rgba(255, 255, 255, 0.04)"
BORDER = "rgba(148, 163, 184, 0.18)"

# NOTE: this stylesheet must not contain blank lines. Streamlit renders
# markdown as CommonMark, where a raw HTML block is terminated by the first
# blank line — any CSS after one would be printed to the page as text.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
.stApp {
    background:
      radial-gradient(1100px 600px at 12% -8%, rgba(34, 211, 238, 0.10), transparent 60%),
      radial-gradient(900px 500px at 92% 4%, rgba(129, 140, 248, 0.10), transparent 55%),
      #0b1120;
}
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1400px;}
.hero {
    display: flex; flex-wrap: wrap; gap: 1.25rem;
    align-items: center; justify-content: space-between; margin-bottom: 1.5rem;
}
.hero-eyebrow {
    text-transform: uppercase; letter-spacing: .18em;
    font-size: .72rem; font-weight: 600; color: #7dd3fc; margin-bottom: .35rem;
}
.hero-title {
    font-size: clamp(1.9rem, 4.5vw, 3rem); font-weight: 800;
    line-height: 1.08; margin: 0; color: #f1f5f9; letter-spacing: -0.02em;
}
.hero-sub {color: #94a3b8; font-size: .95rem; margin-top: .5rem; max-width: 46ch;}
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px; padding: 1.15rem 1.25rem;
    backdrop-filter: blur(6px);
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.45);
    height: 100%;
}
.card-label {
    text-transform: uppercase; letter-spacing: .12em;
    font-size: .68rem; font-weight: 600; color: #94a3b8; margin-bottom: .5rem;
}
.card-value {font-size: 1.9rem; font-weight: 750; color: #f1f5f9; line-height: 1.1;}
.card-sub {font-size: .8rem; color: #94a3b8; margin-top: .35rem;}
.forecast-grid {
    display: grid; gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
    margin-bottom: .5rem;
}
.fc-card {
    position: relative; overflow: hidden;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px; padding: 1.1rem 1.2rem 1.2rem 1.5rem;
    box-shadow: 0 10px 28px rgba(2, 6, 23, 0.45);
    transition: transform .18s ease, border-color .18s ease;
}
.fc-card:hover {transform: translateY(-3px); border-color: rgba(34,211,238,0.45);}
.fc-swatch {position: absolute; left: 0; top: 0; bottom: 0; width: 6px;}
.fc-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: .5rem; margin-bottom: .7rem;
}
.fc-horizon {font-size: .9rem; font-weight: 650; color: #e2e8f0;}
.fc-when {font-size: .72rem; color: #94a3b8;}
.fc-value {font-size: 2.5rem; font-weight: 800; line-height: 1; letter-spacing: -0.02em;}
.fc-cat {
    display: inline-block; margin-top: .6rem; padding: .22rem .6rem;
    border-radius: 999px; font-size: .72rem; font-weight: 650;
}
.fc-range {font-size: .74rem; color: #94a3b8; margin-top: .6rem;}
.fc-method {
    font-size: .66rem; color: #64748b; margin-top: .45rem;
    text-transform: uppercase; letter-spacing: .08em;
}
.banner {
    display: flex; align-items: flex-start; gap: .9rem;
    border-radius: 16px; padding: 1rem 1.2rem; margin: 1.1rem 0 1.4rem 0;
    border: 1px solid; font-size: .92rem; line-height: 1.5;
}
.banner-icon {font-size: 1.5rem; line-height: 1; flex-shrink: 0;}
.banner-title {font-weight: 700; margin-bottom: .18rem; font-size: .98rem;}
.banner-danger {
    background: linear-gradient(90deg, rgba(239,68,68,.16), rgba(239,68,68,.05));
    border-color: rgba(248,113,113,.45); color: #fecaca;
}
.banner-warn {
    background: linear-gradient(90deg, rgba(245,158,11,.15), rgba(245,158,11,.04));
    border-color: rgba(251,191,36,.42); color: #fde68a;
}
.banner-ok {
    background: linear-gradient(90deg, rgba(16,185,129,.14), rgba(16,185,129,.04));
    border-color: rgba(52,211,153,.38); color: #a7f3d0;
}
.banner-info {
    background: linear-gradient(90deg, rgba(56,189,248,.13), rgba(56,189,248,.04));
    border-color: rgba(56,189,248,.35); color: #bae6fd;
}
.banner code {
    background: rgba(15,23,42,.6); padding: .08rem .35rem;
    border-radius: 5px; font-size: .86em;
}
.section-title {
    font-size: 1.12rem; font-weight: 700; color: #f1f5f9;
    margin: 2rem 0 .3rem 0; letter-spacing: -0.01em;
}
.section-note {font-size: .82rem; color: #94a3b8; margin-bottom: .9rem;}
.pill {
    display: inline-block; padding: .2rem .55rem; border-radius: 999px;
    font-size: .7rem; font-weight: 600; border: 1px solid rgba(148,163,184,.3);
    color: #cbd5e1; background: rgba(148,163,184,.1); margin-right: .35rem;
}
[data-testid="stSidebar"] {
    background: #0d1526; border-right: 1px solid rgba(148,163,184,0.14);
}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
[data-testid="stImage"] img {border-radius: 12px;}
@media (max-width: 640px) {
    .block-container {padding-top: 1.4rem;}
    .hero {gap: .8rem;}
    .fc-value {font-size: 2.1rem;}
    .card-value {font-size: 1.55rem;}
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner="Loading feature store…")
def load_dashboard_data(history_hours: int):
    """Feature engineering over the full store is the slow part; cache it so
    sidebar interactions stay instant."""
    store = load_feature_store()
    # After the hourly-grid reindex the store has no missing timestamps, but
    # hours that were never measured are present with a null AQI.
    unmeasured = int(store["aqi"].isna().sum()) if "aqi" in store.columns else 0
    return {
        "current": get_current_aqi(),
        "forecast": get_forecast(),
        "history": get_history(history_hours),
        "metrics": get_model_metrics(),
        "unmeasured_hours": unmeasured + sum(g[2] for g in describe_gaps(store)),
    }


def fmt_ts(value) -> str:
    return pd.to_datetime(value).strftime("%d %b %Y, %H:%M") + " UTC"


MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "lstm": "LSTM (TensorFlow)",
    "persistence": "Persistence baseline",
}


def model_label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Plotly's colour validator rejects 8-digit #RRGGBBAA hex, so build an
    explicit rgba() string for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def human_age(hours: float) -> str:
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.1f} days ago"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        f"<div style='font-size:1.05rem;font-weight:750;color:#f1f5f9;'>🌫️ AQI Predictor</div>"
        f"<div style='font-size:.78rem;color:#94a3b8;margin-bottom:1.1rem;'>"
        f"Serverless 3-day air-quality forecasting</div>",
        unsafe_allow_html=True,
    )

    st.markdown("**Location**")
    preset_names = list(config.CITY_PRESETS)
    if config.CITY_NAME not in preset_names:
        preset_names.insert(0, config.CITY_NAME)
    selected_city = st.selectbox(
        "City", preset_names,
        index=preset_names.index(config.CITY_NAME),
        label_visibility="collapsed",
    )
    lat, lon = config.CITY_PRESETS.get(
        selected_city, (config.LATITUDE, config.LONGITUDE))
    st.caption(f"{lat:.4f}, {lon:.4f}")

    if selected_city != config.CITY_NAME:
        st.warning(
            f"The feature store only contains history for **{config.CITY_NAME}**. "
            f"To switch, set `AQI_CITY`/`AQI_LAT`/`AQI_LON` and re-run "
            f"`src/backfill.py`, then `src/train_pipeline.py`.",
            icon="⚠️",
        )

    st.divider()
    st.markdown("**View**")
    history_days = st.slider("History window (days)", 1, 60, 14)
    show_band = st.checkbox("Show uncertainty band", value=True)
    show_categories = st.checkbox("Shade AQI categories", value=True)

data = load_dashboard_data(history_days * 24)
current = data["current"]
forecast = data["forecast"]
history = data["history"]
metrics = data["metrics"]

if current is None:
    st.markdown(
        "<div class='banner banner-info'><div class='banner-icon'>📭</div><div>"
        "<div class='banner-title'>No data in the feature store yet</div>"
        "Run <code>python src/backfill.py --days 90</code> (or "
        "<code>python src/synthetic_backfill.py --days 90</code> without an API key), "
        "then <code>python src/train_pipeline.py</code>.</div></div>",
        unsafe_allow_html=True,
    )
    st.stop()

current_color = category_color(current["category"])

with st.sidebar:
    st.divider()
    st.markdown("**Data status**")
    st.markdown(
        f"<div style='font-size:.82rem;color:#cbd5e1;line-height:1.7;'>"
        f"Latest reading<br><b>{fmt_ts(current['timestamp'])}</b><br>"
        f"<span style='color:{'#fbbf24' if current['is_stale'] else '#34d399'};'>"
        f"● {human_age(current['age_hours'])}</span></div>",
        unsafe_allow_html=True,
    )
    if data["unmeasured_hours"]:
        st.caption(f"{data['unmeasured_hours']} hour(s) were never recorded; "
                   f"they are held on the hourly grid so lag features stay "
                   f"aligned to real time.")

    st.divider()
    st.markdown("**Explainability**")
    any_shap = False
    for horizon in config.HORIZONS:
        status = shap_plot_status(horizon)
        if status:
            any_shap = True
            st.markdown(f"[SHAP — {horizon}h model](#drivers)")
    if not any_shap:
        st.caption("SHAP plots appear after training.")

    st.divider()
    st.caption(f"Rendered {dt.datetime.now(dt.timezone.utc):%d %b %H:%M} UTC")


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hero_left, hero_right = st.columns([1.15, 1], gap="large")

with hero_left:
    st.markdown(
        f"""
<div class="hero"><div>
  <div class="hero-eyebrow">Air Quality Index · US EPA scale</div>
  <h1 class="hero-title">{html.escape(config.CITY_NAME)}</h1>
  <div class="hero-sub">
    Hourly feature pipeline, daily retraining, and a 3-day forecast —
    all running on GitHub Actions with no servers.
  </div>
</div></div>
        """,
        unsafe_allow_html=True,
    )

    driver = current.get("dominant_pollutant")
    pm25 = current.get("pm2_5")
    stat_cards = [
        ("Category", current["category"], "Current severity band", current_color),
        ("Dominant pollutant",
         {"pm2_5": "PM2.5", "pm10": "PM10", "o3": "Ozone", "no2": "NO₂",
          "so2": "SO₂", "co": "CO"}.get(driver, "—"),
         "Sets the overall AQI", "#f1f5f9"),
        ("PM2.5", f"{pm25:.1f}" if pm25 is not None else "—",
         "µg/m³ right now", "#f1f5f9"),
    ]
    cols = st.columns(len(stat_cards), gap="small")
    for col, (label, value, sub, color) in zip(cols, stat_cards):
        col.markdown(
            f"<div class='card'><div class='card-label'>{label}</div>"
            f"<div class='card-value' style='color:{color};font-size:1.35rem;'>"
            f"{html.escape(str(value))}</div>"
            f"<div class='card-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

with hero_right:
    gauge_steps = [
        {"range": [lo, min(hi, 500)],
         "color": hex_to_rgba(config.AQI_CATEGORY_COLORS[label], 0.22)}
        for lo, hi, label in config.AQI_CATEGORIES
    ]
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current["aqi"],
        number={"font": {"size": 64, "color": current_color,
                         "family": "Inter, sans-serif"},
                "valueformat": ".0f"},
        title={"text": f"<span style='font-size:.85rem;color:#94a3b8;'>"
                       f"CURRENT AQI</span>"},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1,
                     "tickcolor": "#475569", "tickfont": {"color": "#64748b", "size": 10},
                     "tickvals": [0, 50, 100, 150, 200, 300, 500]},
            "bar": {"color": current_color, "thickness": 0.28},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": gauge_steps,
            "threshold": {
                "line": {"color": "#f87171", "width": 3},
                "thickness": 0.82,
                "value": config.ALERT_THRESHOLD,
            },
        },
    ))
    gauge.update_layout(
        height=290, margin=dict(l=42, r=42, t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e2e8f0", "family": "Inter, sans-serif"},
    )
    _stretch(st.plotly_chart, gauge, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Alert banner
# ---------------------------------------------------------------------------
def banner(kind: str, icon: str, title: str, body: str):
    st.markdown(
        f"<div class='banner banner-{kind}'><div class='banner-icon'>{icon}</div>"
        f"<div><div class='banner-title'>{title}</div>{body}</div></div>",
        unsafe_allow_html=True,
    )


alerting = []
if current["aqi"] >= config.ALERT_THRESHOLD:
    alerting.append("right now")
if not forecast.empty:
    for row in forecast.itertuples():
        if row.predicted_aqi >= config.ALERT_THRESHOLD:
            alerting.append(f"in {row.horizon_hours}h")

if alerting:
    banner("danger", "⚠️",
           f"Unhealthy air expected (AQI ≥ {config.ALERT_THRESHOLD})",
           f"Elevated levels {', '.join(alerting)}. Limit prolonged outdoor "
           f"exertion; sensitive groups should stay indoors where possible.")
else:
    banner("ok", "✅", "No health alerts for the next 3 days",
           f"Forecast AQI stays below the alert threshold of "
           f"{config.ALERT_THRESHOLD} (Unhealthy).")

if current["is_stale"]:
    banner("warn", "🕒", "Forecast is anchored to stale data",
           f"The most recent reading is from {fmt_ts(current['timestamp'])} "
           f"({human_age(current['age_hours'])}). Every forecast below is relative "
           f"to that moment, not to now. Run "
           f"<code>python src/feature_pipeline.py</code> to refresh.")


# ---------------------------------------------------------------------------
# Forecast cards
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>3-day forecast</div>", unsafe_allow_html=True)

if forecast.empty:
    banner("info", "🧠", "No trained models found",
           "Run <code>python src/train_pipeline.py</code> to populate the model registry.")
else:
    st.markdown(
        "<div class='section-note'>Each horizon is measured from the latest reading. "
        "Ranges are an 80% band derived from the model's out-of-sample errors — "
        "indicative, not a calibrated prediction interval.</div>",
        unsafe_allow_html=True,
    )

    cards = []
    for row in forecast.itertuples():
        color = category_color(row.category)
        when = fmt_ts(row.forecast_time).replace(" UTC", "")
        if row.aqi_lower is not None and row.aqi_upper is not None:
            rng = f"Range {row.aqi_lower:.0f} – {row.aqi_upper:.0f}"
        else:
            rng = "Range unavailable"
        method = model_label(row.method)
        cards.append(f"""
<div class="fc-card">
  <div class="fc-swatch" style="background:{color};"></div>
  <div class="fc-head">
    <span class="fc-horizon">+{row.horizon_hours} hours</span>
    <span class="fc-when">{when}</span>
  </div>
  <div class="fc-value" style="color:{color};">{row.predicted_aqi:.0f}</div>
  <span class="fc-cat" style="background:{color}26;color:{color};border:1px solid {color}55;">
    {html.escape(row.category)}
  </span>
  <div class="fc-range">{rng}</div>
  <div class="fc-method">{html.escape(method)}</div>
</div>""")
    st.markdown(f"<div class='forecast-grid'>{''.join(cards)}</div>",
                unsafe_allow_html=True)

    served_by_baseline = forecast[~forecast["beats_baseline"]]
    if not served_by_baseline.empty:
        horizons = ", ".join(f"+{h}h" for h in served_by_baseline["horizon_hours"])
        banner("info", "📊", "Some horizons are served by the baseline",
               f"At {horizons}, a persistence baseline (\"AQI stays where it is\") "
               f"beat every trained model in cross-validation, so the honest "
               f"forecast is the baseline. The model's own prediction is shown "
               f"in the accuracy table below for comparison.")


# ---------------------------------------------------------------------------
# Continuous history + forecast chart
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Observed history and forecast</div>",
            unsafe_allow_html=True)
st.markdown("<div class='section-note'>Solid line is measured AQI; dashed is "
            "forecast. Breaks in the line are hours the pipeline never "
            "recorded.</div>", unsafe_allow_html=True)

fig = go.Figure()

# Scale to the data rather than the full 0-500 EPA range: real readings sit in
# a narrow band, and a fixed axis would flatten the series into a straight line.
observed_max = float(history["aqi"].max()) if not history.empty else 0.0
forecast_max = 0.0
if not forecast.empty:
    upper = forecast["aqi_upper"].dropna()
    forecast_max = float(max(forecast["predicted_aqi"].max(),
                             upper.max() if not upper.empty else 0))
y_max = max(observed_max, forecast_max, config.ALERT_THRESHOLD + 20) * 1.12
y_max = min(y_max, 500)

if show_categories:
    for lo, hi, label in config.AQI_CATEGORIES:
        if lo >= y_max:
            break
        fig.add_hrect(y0=lo, y1=min(hi, y_max),
                      fillcolor=config.AQI_CATEGORY_COLORS[label], opacity=0.07,
                      layer="below", line_width=0,
                      annotation_text=label if (hi - lo) > y_max * 0.12 else None,
                      annotation_position="right",
                      annotation_font=dict(size=9, color="rgba(203,213,225,0.5)"))

if not history.empty:
    fig.add_trace(go.Scatter(
        x=history["timestamp"], y=history["aqi"],
        mode="lines", name="Observed",
        line=dict(color=ACCENT, width=2.2),
        connectgaps=False,
        hovertemplate="%{x|%d %b %H:%M} UTC<br>AQI %{y:.0f}<extra>Observed</extra>",
    ))

if not forecast.empty:
    anchor_t = current["timestamp"]
    anchor_v = current["aqi"]
    fx = [anchor_t] + list(forecast["forecast_time"])
    fy = [anchor_v] + list(forecast["predicted_aqi"])

    has_band = forecast["aqi_lower"].notna().all() and forecast["aqi_upper"].notna().all()
    if show_band and has_band:
        lo = [anchor_v] + list(forecast["aqi_lower"])
        hi = [anchor_v] + list(forecast["aqi_upper"])
        # Upper bound first, then the lower bound reversed, so the fill closes.
        fig.add_trace(go.Scatter(
            x=fx + fx[::-1], y=hi + lo[::-1],
            fill="toself", fillcolor="rgba(167, 139, 250, 0.16)",
            line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
            name="80% range", showlegend=True,
        ))

    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines+markers", name="Forecast",
        line=dict(color="#a78bfa", width=2.6, dash="dash"),
        marker=dict(size=9, color="#a78bfa", line=dict(color="#0b1120", width=2)),
        hovertemplate="%{x|%d %b %H:%M} UTC<br>AQI %{y:.0f}<extra>Forecast</extra>",
    ))

    fig.add_vline(x=anchor_t, line_width=1, line_dash="dot",
                  line_color="rgba(148,163,184,0.55)")

fig.add_hline(y=config.ALERT_THRESHOLD, line_dash="dash", line_color="#f87171",
              line_width=1.4, annotation_text="Alert threshold",
              annotation_position="top left",
              annotation_font=dict(color="#fca5a5", size=11))

fig.update_layout(
    height=430, margin=dict(l=10, r=10, t=30, b=10),
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1", family="Inter, sans-serif", size=12),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
    xaxis=dict(showgrid=False, linecolor="rgba(148,163,184,0.25)", title=None),
    yaxis=dict(title="AQI", gridcolor="rgba(148,163,184,0.12)", zeroline=False,
               range=[0, y_max]),
)
_stretch(st.plotly_chart, fig, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Model accuracy
# ---------------------------------------------------------------------------
if metrics:
    st.markdown("<div class='section-title'>Model accuracy</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-note'>Out-of-sample results from time-series "
                "cross-validation, compared against a persistence baseline. Lower "
                "RMSE is better.</div>", unsafe_allow_html=True)

    rows = []
    for horizon, meta in sorted(metrics.items()):
        selected = meta.get("selected_metrics", {})
        baseline = meta.get("persistence_baseline", {})
        rows.append({
            "Horizon": f"+{horizon}h",
            "Best model": model_label(meta.get("deployed_model", "—")),
            "Model RMSE": round(selected.get("rmse", float("nan")), 1),
            "Baseline RMSE": round(baseline.get("rmse", float("nan")), 1),
            "Model R²": round(selected.get("r2", float("nan")), 3),
            "Serving": ("Model" if meta.get("beats_baseline") else "Baseline"),
        })
    _stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)

    trained_at = next((m.get("trained_at") for m in metrics.values() if m.get("trained_at")), None)
    if trained_at:
        st.caption(f"Last trained {fmt_ts(trained_at.replace('Z', '').split('+')[0])}")


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title' id='drivers'>What's driving the forecast</div>",
            unsafe_allow_html=True)

shap_entries = [(h, shap_plot_status(h)) for h in config.HORIZONS]
shap_entries = [(h, s) for h, s in shap_entries if s]

if not shap_entries:
    st.markdown("<div class='section-note'>SHAP plots appear here after training "
                "(<code>train_pipeline.py</code> generates them).</div>",
                unsafe_allow_html=True)
else:
    if any(stale for _, (_, stale) in shap_entries):
        banner("warn", "🔍", "Some SHAP plots predate the current models",
               "They were generated in an earlier training run, so they explain a "
               "previous version of the model. They refresh automatically the next "
               "time training runs somewhere with <code>shap</code> installed.")

    tabs = st.tabs([f"+{h}h model" for h, _ in shap_entries])
    for tab, (horizon, (path, stale)) in zip(tabs, shap_entries):
        with tab:
            if stale:
                st.markdown("<span class='pill'>⚠️ Older than the current model</span>",
                            unsafe_allow_html=True)
            _stretch(st.image, path)
            st.caption(
                f"SHAP summary for the {horizon}h model. Features are ranked by "
                f"average impact on the prediction; colour shows whether a high or "
                f"low feature value pushes the forecast up or down. The sample is "
                f"in-sample for the deployed model, so this shows what the model "
                f"relies on rather than how well it generalises."
            )

st.markdown(
    f"<div style='margin-top:2.5rem;padding-top:1.2rem;"
    f"border-top:1px solid rgba(148,163,184,0.14);color:#64748b;font-size:.78rem;'>"
    f"Data: OpenWeather (pollution) + Open-Meteo (weather) · "
    f"AQI on the US EPA 0–500 scale · Latest reading {fmt_ts(current['timestamp'])}"
    f"</div>",
    unsafe_allow_html=True,
)
