"""Palette, stylesheet, and small render helpers shared by both pages.

Visual layer only — nothing here reads or transforms model data.
"""
import math
import re

import bootstrap  # noqa: F401  (sys.path side effect)
import pandas as pd
import streamlit as st

import config
from feature_pipeline import aqi_sub_indices
from utils import aqi_category, category_color

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
GREEN = "#1F3B2C"          # deep forest accent, 12.2:1 on white
TEAL = "#0F766E"           # forecast series; distinct from the accent green
INK = "#212121"            # body text, 15:1 on white
MUTED = "#5A6B62"          # captions, 5.6:1 on white
BORDER = "#E3E8E4"
AMBER = "#92610B"          # stale/degraded state, 5.0:1 on white
RED = "#C21807"            # alert threshold marker

# The sidebar is a solid forest column, so anything rendered inside it needs
# the inverted pair rather than INK/MUTED.
SIDE_TEXT = "#E8EFE9"
SIDE_MUTED = "rgba(232, 239, 233, 0.66)"
SIDE_ACCENT = "#9FCDB1"    # fresh-data dot, labels, brand rule
SIDE_AMBER = "#FBBF24"     # stale-data dot on the dark column

# "Good" is pulled onto the forest accent so the page reads as one palette.
# Every other EPA hue is left exactly as published, because the yellow ->
# maroon ramp is what makes the severity scale legible at a glance.
CATEGORY_FILL_COLORS = {**config.AQI_CATEGORY_COLORS, "Good": GREEN}

# Translucent fills (gauge arcs, chart bands, pills) blend towards white, and
# the deep accent desaturates to grey at those alphas. This is the same forest
# hue lifted just enough to still read as green once blended.
GOOD_TINT = "#2E6B4F"

# The EPA hues are tuned for fills and swatches; several of them (notably the
# #ffd700 "Moderate" yellow) drop far below 4.5:1 when used as text on white.
# These are the same hues darkened just enough to pass AA as foreground — the
# severity ordering and colour family are unchanged, so the EPA reading of the
# scale still holds.
CATEGORY_TEXT_COLORS = {
    "Good": GREEN,
    "Moderate": "#8A6D00",
    "Unhealthy for Sensitive Groups": "#A15200",
    "Unhealthy": RED,
    "Very Unhealthy": "#6B21C7",
    "Hazardous": "#9D174D",
}


def category_fill_color(category: str) -> str:
    """Swatch/arc colour for `category` on the themed scale."""
    return CATEGORY_FILL_COLORS.get(category, category_color(category))


def category_tint(category: str) -> str:
    """Base colour for translucent fills of `category`."""
    return GOOD_TINT if category == "Good" else category_fill_color(category)


def category_text_color(category: str) -> str:
    """AA-safe foreground variant of the EPA colour for `category`."""
    return CATEGORY_TEXT_COLORS.get(category, category_color(category))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Plotly's colour validator rejects 8-digit #RRGGBBAA hex, so build an
    explicit rgba() string for translucent fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
# NOTE: this stylesheet must not contain blank lines. Streamlit renders
# markdown as CommonMark, where a raw HTML block is terminated by the first
# blank line — any CSS after one would be printed to the page as text.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}
/* Spacing scale: 8 / 16 / 24 / 32px, used for every gap and pad below. */
.stApp {background: #FAFAFA;}
#MainMenu, footer {visibility: hidden;}
/* Full-width forest strip behind Streamlit's toolbar so the bar is solid
   edge-to-edge (the native header is otherwise a transparent overlay). */
.stApp::before {
    content: ""; position: fixed; top: 0; left: 0; right: 0; height: 3.25rem;
    background: #1F3B2C; z-index: 999990;
}
header[data-testid="stHeader"] {
    background-color: #1F3B2C !important;
    background-image: none !important;
}
header[data-testid="stHeader"] * {color: #E8EFE9 !important;}
[data-testid="stDecoration"] {background: #1F3B2C !important; display: none;}
.stAppDeployButton, [data-testid="stAppDeployButton"],
div[data-testid="stToolbarActionButton"]:has(a) {display: none !important;}
a.top-reload {
    position: fixed; top: 0.42rem; right: 16px; z-index: 1000001;
    display: inline-flex; align-items: center; gap: 6px;
    color: #FFFFFF !important; font-weight: 600; font-size: .88rem;
    padding: 6px 14px; border: 1px solid rgba(255,255,255,.4);
    border-radius: 8px; text-decoration: none !important;
    background: transparent; letter-spacing: .02em;
}
a.top-reload:hover {
    background: rgba(255,255,255,.14); border-color: #FFFFFF;
    color: #FFFFFF !important; text-decoration: none !important;
}
/* Top padding clears Streamlit's fixed toolbar so the eyebrow is not clipped. */
.block-container {
    padding-top: 72px; padding-bottom: 48px;
    padding-left: 32px; padding-right: 32px; max-width: 1320px;
}
.stApp a, .stApp a:visited {color: #1F3B2C; text-decoration: none; font-weight: 500;}
.stApp a:hover {text-decoration: underline;}
.hero-eyebrow {
    text-transform: uppercase; letter-spacing: .16em;
    font-size: .72rem; font-weight: 600; color: #1F3B2C; margin-bottom: 8px;
}
.hero-title {
    font-size: clamp(1.9rem, 4.5vw, 2.8rem); font-weight: 700;
    line-height: 1.1; margin: 0; color: #1F3B2C; letter-spacing: -0.02em;
}
.hero-city {
    font-family: 'Fraunces', 'Times New Roman', serif !important;
    font-optical-sizing: auto;
    font-weight: 650;
    font-size: clamp(2.7rem, 6vw, 3.75rem);
    color: #1F3B2C !important;
    letter-spacing: -0.02em;
    line-height: 1.08;
}
.hero-sub {color: #5A6B62; font-size: .95rem; margin-top: 8px; max-width: 60ch;}
/* Secondary cards carry a forest left-edge so they read as a set. */
.card {
    background: #FFFFFF; border: 1px solid #E3E8E4;
    border-left: 3px solid #1F3B2C;
    border-radius: 14px; padding: 16px 20px; height: 100%;
    box-shadow: 0 1px 2px rgba(16, 24, 20, 0.04), 0 4px 12px rgba(16, 24, 20, 0.05);
}
.card-label {
    text-transform: uppercase; letter-spacing: .10em;
    font-size: .68rem; font-weight: 700; color: #1F3B2C; margin-bottom: 8px;
}
.card-value {font-size: 1.35rem; font-weight: 700; color: #212121; line-height: 1.2;}
.card-sub {font-size: .8rem; color: #5A6B62; margin-top: 6px; font-weight: 400;}
.forecast-grid {
    display: grid; gap: 16px;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    margin-bottom: 8px;
}
.fc-card {
    position: relative; overflow: hidden; background: #FFFFFF;
    border: 1px solid #E3E8E4; border-radius: 14px;
    padding: 16px 20px 16px 24px;
    box-shadow: 0 1px 2px rgba(16, 24, 20, 0.04), 0 4px 12px rgba(16, 24, 20, 0.05);
    transition: transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.fc-card:hover {
    transform: translateY(-2px); border-color: #1F3B2C;
    box-shadow: 0 2px 4px rgba(16, 24, 20, 0.05), 0 10px 24px rgba(16, 24, 20, 0.09);
}
.fc-swatch {position: absolute; left: 0; top: 0; bottom: 0; width: 5px;}
.fc-when {
    font-size: .72rem; font-weight: 700; color: #1F3B2C;
    text-transform: uppercase; letter-spacing: .10em; margin-bottom: 10px;
}
.fc-value {font-size: 2.4rem; font-weight: 700; line-height: 1; letter-spacing: -0.02em;}
.fc-cat {
    display: inline-block; margin-top: 10px; padding: 3px 10px;
    border-radius: 999px; font-size: .72rem; font-weight: 600;
}
.fc-range {
    display: flex; gap: 20px; margin-top: 14px;
    border-top: 1px solid #EDF1EE; padding-top: 12px;
}
.fc-range-label {
    font-size: .62rem; font-weight: 600; color: #5A6B62;
    text-transform: uppercase; letter-spacing: .10em;
}
.fc-range-value {font-size: .95rem; font-weight: 700; color: #212121;}
.fc-spark {margin-top: 12px;}
.fc-method {
    font-size: .64rem; color: #5A6B62; margin-top: 8px;
    text-transform: uppercase; letter-spacing: .08em;
}
.poll-grid {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
}
.poll-card {
    background: #FFFFFF; border: 1px solid #E3E8E4; border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 2px rgba(16, 24, 20, 0.04);
}
.poll-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 8px; margin-bottom: 8px;
}
.poll-name {
    font-size: .74rem; font-weight: 700; color: #1F3B2C;
    text-transform: uppercase; letter-spacing: .08em;
}
.poll-chip {
    padding: 2px 8px; border-radius: 999px;
    font-size: .62rem; font-weight: 600; white-space: nowrap;
}
.poll-value {font-size: 1.5rem; font-weight: 700; color: #212121; line-height: 1.1;}
.poll-unit {font-size: .72rem; color: #5A6B62; font-weight: 400; margin-left: 4px;}
.poll-spark {margin-top: 10px;}
.poll-foot {font-size: .66rem; color: #5A6B62; margin-top: 6px;}
/* Alerts keep the conventional red/amber semantics: the green branding stops
   here so severity stays unambiguous. */
.banner {
    display: flex; align-items: flex-start; gap: 14px;
    border-radius: 12px; padding: 16px 20px; margin: 16px 0 24px 0;
    border: 1px solid; border-left-width: 4px;
    font-size: .92rem; line-height: 1.55; background: #FFFFFF;
}
.banner-icon {font-size: 1.4rem; line-height: 1.2; flex-shrink: 0;}
.banner-title {font-weight: 700; margin-bottom: 2px; font-size: .98rem;}
.banner-danger {background: #FEF2F2; border-color: #FCA5A5; color: #991B1B;}
.banner-warn {background: #FFFBEB; border-color: #FCD34D; color: #854D0E;}
.banner-ok {background: #EDF2EE; border-color: #1F3B2C; color: #1F3B2C;}
.banner-info {background: #F4F7FA; border-color: #BBD3E6; color: #1F4B6E;}
.banner code {
    background: rgba(17, 24, 20, .07); padding: 1px 6px;
    border-radius: 4px; font-size: .86em; color: inherit;
}
.section-title {
    font-size: 1.15rem; font-weight: 700; color: #1F3B2C;
    margin: 32px 0 4px 0; letter-spacing: -0.01em;
}
.section-note {font-size: .82rem; color: #5A6B62; margin-bottom: 16px; max-width: 80ch;}
.pill {
    display: inline-block; padding: 3px 10px; border-radius: 999px;
    font-size: .7rem; font-weight: 600; border: 1px solid #E3E8E4;
    color: #5A6B62; background: #F4F7F4; margin-right: 6px;
}
/* The sidebar is a solid forest nav column: every child inherits light text
   so the branding reads as one block rather than a themed header on white. */
[data-testid="stSidebar"] {background: #1F3B2C; border-right: none;}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarHeader"],
[data-testid="stSidebarContent"] {background: #1F3B2C;}
[data-testid="stSidebar"] .block-container {padding-top: 24px;}
[data-testid="stSidebarCollapsedControl"] {
    background: #1F3B2C; border: 1px solid #1F3B2C;
}
[data-testid="stSidebar"] * {color: #E8EFE9;}
[data-testid="stSidebar"] hr {margin: 16px 0; border-color: rgba(232,239,233,.18);}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] * {
    color: rgba(232,239,233,.66);
}
[data-testid="stSidebar"] svg {fill: #E8EFE9; color: #E8EFE9;}
[data-testid="stSidebar"] a, [data-testid="stSidebar"] a:visited {color: #BFE3CC;}
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {color: rgba(232,239,233,.8);}
/* Inputs sit on a lighter forest tint so they stay visible on the dark column. */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.08); border-color: rgba(232,239,233,.28);
    color: #FFFFFF;
}
[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"] {
    color: rgba(232,239,233,.6);
}
[data-testid="stSidebar"] [data-testid="stAlertContainer"] * {color: #422006;}
.side-label {
    color: #9FCDB1; text-transform: uppercase; font-weight: 700;
    letter-spacing: .10em; font-size: .72rem; margin-bottom: 6px;
}
.side-brand {
    font-size: 1.05rem; font-weight: 700; color: #FFFFFF;
    border-left: 3px solid #9FCDB1; padding-left: 10px;
}
/* Native st.navigation links: light text, pale pill on the active page. */
[data-testid="stSidebarNav"] a {border-radius: 8px; padding-left: 8px;}
[data-testid="stSidebarNav"] a span {font-weight: 600; color: #E8EFE9;}
[data-testid="stSidebarNav"] li a:hover {background: rgba(255,255,255,.10);}
[data-testid="stSidebarNav"] a[aria-current="page"] {background: rgba(255,255,255,.16);}
[data-testid="stSidebarNav"] a[aria-current="page"] span {color: #FFFFFF;}
[data-testid="stSidebarCollapseButton"] svg,
[data-testid="stSidebarCollapsedControl"] svg {color: #E8EFE9;}
/* Ranked feature-importance bars. */
.imp-row {margin-bottom: 14px;}
.imp-head {
    display: flex; align-items: baseline; justify-content: space-between;
    gap: 12px; margin-bottom: 6px;
}
.imp-name {font-size: .84rem; color: #212121; font-weight: 500;}
.imp-value {
    font-size: .78rem; color: #5A6B62; text-align: right;
    font-variant-numeric: tabular-nums; flex-shrink: 0;
}
.imp-track {background: #EDF1EE; border-radius: 999px; height: 8px; width: 100%;}
.imp-fill {background: #1F3B2C; border-radius: 999px; height: 8px;}
[data-testid="stImage"] img {
    border-radius: 12px; border: 1px solid #E3E8E4; background: #FFFFFF;
}
[data-testid="stDataFrame"] {border-radius: 12px; overflow: hidden;}
/* Bordered containers (headline AQI card, chart) share the card treatment. */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF; border-radius: 16px; border-color: #E3E8E4;
    box-shadow: 0 1px 2px rgba(16, 24, 20, 0.04), 0 8px 24px rgba(16, 24, 20, 0.06);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.card-head) {overflow: hidden;}
.card-head {
    background: #1F3B2C; color: #FFFFFF;
    font-size: .95rem; font-weight: 700; letter-spacing: .04em;
    padding: 12px 20px;
    margin: -1rem -1rem 16px -1rem;
    border-radius: 15px 15px 0 0;
}
.card-head, .card-head * {color: #FFFFFF;}
.card-head-lg {
    font-size: 1.55rem; font-weight: 700; letter-spacing: -0.015em;
    padding: 16px 22px;
}
.metric-strip {
    display: grid; gap: 8px; grid-template-columns: repeat(3, 1fr);
    border-top: 1px solid #E3E8E4; padding-top: 16px; margin-top: 8px;
}
.ms-label {
    text-transform: uppercase; letter-spacing: .10em;
    font-size: .66rem; font-weight: 600; color: #5A6B62; margin-bottom: 4px;
}
.ms-value {font-size: 1rem; font-weight: 700; color: #212121; line-height: 1.3;}
.ms-sub {font-size: .72rem; color: #5A6B62; margin-top: 2px; font-weight: 400;}
.stButton > button {border-radius: 8px; font-weight: 600;}
/* Narrow viewports: one column, tighter gutters, nothing overlapping. */
@media (max-width: 720px) {
    .block-container {padding-left: 16px; padding-right: 16px; padding-top: 64px;}
    .forecast-grid {grid-template-columns: 1fr; gap: 12px;}
    .poll-grid {grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));}
    .imp-row {margin-bottom: 12px;}
    .metric-strip {grid-template-columns: 1fr; gap: 12px;}
    .fc-value {font-size: 2.1rem;}
    .card {padding: 14px 16px;}
    .banner {padding: 14px 16px; gap: 10px;}
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    # Real navigation (not a no-op <button>) so the browser actually reloads
    # the app; streamlit_app.py clears the data cache when ?reload=1 is set.
    st.markdown(
        '<a class="top-reload" href="?reload=1" target="_self">↻ Reload</a>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def stretch(render, *args, **kwargs):
    """Streamlit renamed the chart/image sizing argument; support both so the
    app runs on the version range pinned in requirements.txt."""
    try:
        return render(*args, width="stretch", **kwargs)
    except TypeError:
        return render(*args, use_container_width=True, **kwargs)


def banner(kind: str, icon: str, title: str, body: str) -> None:
    st.markdown(
        f"<div class='banner banner-{kind}'><div class='banner-icon'>{icon}</div>"
        f"<div><div class='banner-title'>{title}</div>{body}</div></div>",
        unsafe_allow_html=True,
    )


def card_head(title: str, *, large: bool = False) -> None:
    """Forest-green title strip for a bordered Streamlit container."""
    klass = "card-head card-head-lg" if large else "card-head"
    st.markdown(f"<div class='{klass}'>{title}</div>", unsafe_allow_html=True)


def section(title: str, note: str = "", anchor: str = "") -> None:
    anchor_attr = f" id='{anchor}'" if anchor else ""
    st.markdown(f"<div class='section-title'{anchor_attr}>{title}</div>",
                unsafe_allow_html=True)
    if note:
        st.markdown(f"<div class='section-note'>{note}</div>",
                    unsafe_allow_html=True)


def side_label(text: str) -> None:
    st.markdown(f"<div class='side-label'>{text}</div>", unsafe_allow_html=True)


MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "lstm": "LSTM (TensorFlow)",
    "persistence": "Persistence baseline",
}


def model_label(name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(name, name.replace("_", " ").title())


def fmt_ts(value) -> str:
    return pd.to_datetime(value).strftime("%d %b %Y, %H:%M") + " UTC"


def human_age(hours: float) -> str:
    if hours < 1:
        return "just now"
    if hours < 24:
        return f"{hours:.0f}h ago"
    return f"{hours / 24:.1f} days ago"


def sparkline_svg(values, *, color: str = GREEN, width: int = 240,
                  height: int = 40, dash_from: int | None = None,
                  fill: bool = True) -> str:
    """Inline SVG sparkline.

    Returned as markup rather than a chart object because these live inside
    HTML cards, where a Streamlit chart cannot be embedded. `dash_from` is the
    index where the line switches to dashed (used for the forecast segment).
    """
    pts = [float(v) for v in values
           if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if len(pts) < 2:
        return ""

    pad = 3
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    inner_w, inner_h = width - 2 * pad, height - 2 * pad
    step = inner_w / (len(pts) - 1)

    def xy(i, v):
        return (pad + i * step, pad + inner_h - (v - lo) / span * inner_h)

    coords = [xy(i, v) for i, v in enumerate(pts)]
    split = len(coords) - 1 if dash_from is None else max(1, min(dash_from, len(coords) - 1))
    head = coords[:split + 1]
    tail = coords[split:]

    def path(points):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

    parts = []
    if fill:
        area = (f"{path(head)} {head[-1][0]:.1f},{height - pad:.1f} "
                f"{head[0][0]:.1f},{height - pad:.1f}")
        parts.append(f"<polygon points='{area}' fill='{hex_to_rgba(color, 0.10)}'/>")
    parts.append(
        f"<polyline points='{path(head)}' fill='none' stroke='{color}' "
        f"stroke-width='1.8' stroke-linejoin='round' stroke-linecap='round'/>"
    )
    if len(tail) > 1:
        parts.append(
            f"<polyline points='{path(tail)}' fill='none' stroke='{color}' "
            f"stroke-width='1.8' stroke-dasharray='4 3' stroke-linecap='round'/>"
        )
        ex, ey = coords[-1]
        parts.append(f"<circle cx='{ex:.1f}' cy='{ey:.1f}' r='3' fill='{color}'/>")
    return (f"<svg viewBox='0 0 {width} {height}' width='100%' height='{height}' "
            f"preserveAspectRatio='none' role='img'>{''.join(parts)}</svg>")


# ---------------------------------------------------------------------------
# Pollutant reference data
# ---------------------------------------------------------------------------
# OpenWeather reports every component in ug/m3. `aqi_sub_indices` applies the
# EPA breakpoint tables (converting to ppb/ppm where the standard requires it),
# so the category shown per pollutant is the real EPA sub-index band. NO and
# NH3 are deliberately absent: the EPA publishes no AQI breakpoints for them,
# so those cards show the concentration without a category.
POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "o3": "O₃", "no2": "NO₂",
    "so2": "SO₂", "co": "CO", "no": "NO", "nh3": "NH₃",
}
POLLUTANT_ORDER = ["pm2_5", "pm10", "o3", "no2", "so2", "co", "no", "nh3"]


def pollutant_category(components: dict, name: str):
    """EPA sub-index category for one pollutant, or None when the EPA
    publishes no breakpoints for it."""
    subs = aqi_sub_indices(components)
    if name not in subs:
        return None
    return aqi_category(subs[name]), subs[name]


# ---------------------------------------------------------------------------
# Feature naming
# ---------------------------------------------------------------------------
FEATURE_LABELS = {
    "aqi": "Current AQI level",
    "aqi_capped": "Current AQI level",
    "aqi_change_rate": "AQI rate of change",
    "co": "CO concentration",
    "clouds": "Cloud cover",
    "dow_cos": "Day of week",
    "dow_sin": "Day of week",
    "hour": "Hour of day",
    "hour_cos": "Hour of day",
    "hour_sin": "Hour of day",
    "humidity": "Humidity",
    "is_weekend": "Weekend",
    "month": "Season (month)",
    "month_cos": "Season (month)",
    "month_sin": "Season (month)",
    "nh3": "NH₃ concentration",
    "no": "NO concentration",
    "no2": "NO₂ concentration",
    "o3": "Ozone concentration",
    "pm10": "PM10 concentration",
    "pm2_5": "PM2.5 concentration",
    "pressure": "Air pressure",
    "pressure_diff_6h": "6h pressure change",
    "so2": "SO₂ concentration",
    "temp": "Temperature",
    "temp_humidity": "Temperature × humidity",
    "wind_deg": "Wind direction",
    "wind_pm25_interaction": "Wind × PM2.5",
    "wind_speed": "Wind speed",
}

# Windowed AQI features are generated per lag, so match them by shape rather
# than listing every combination.
_FEATURE_PATTERNS = [
    (r"^aqi_lag_(\d+)h$", "AQI {0}h ago"),
    (r"^aqi_diff_(\d+)h$", "AQI change over {0}h"),
    (r"^aqi_trend_(\d+)h$", "AQI vs its {0}h average"),
    (r"^aqi_rmean_(\d+)h$", "{0}h average AQI"),
    (r"^aqi_rolling_mean_(\d+)h$", "{0}h average AQI"),
    (r"^aqi_rmin_(\d+)h$", "{0}h minimum AQI"),
    (r"^aqi_rmax_(\d+)h$", "{0}h maximum AQI"),
    (r"^aqi_rstd_(\d+)h$", "{0}h AQI variability"),
]


def feature_label(name: str) -> str:
    """Human-readable name for a model feature column."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    for pattern, template in _FEATURE_PATTERNS:
        match = re.match(pattern, name)
        if match:
            return template.format(*match.groups())
    return name.replace("_", " ").capitalize()