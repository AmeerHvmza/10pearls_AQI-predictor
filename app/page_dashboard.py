"""Dashboard page: current conditions, 3-day forecast, key pollutants."""
import html

import bootstrap  # noqa: F401  (sys.path side effect)
import plotly.graph_objects as go
import streamlit as st

import config
import data_access
import theme
from theme import (
    AMBER, BORDER, GREEN, INK, MUTED, RED,
    banner, card_head, category_fill_color, category_text_color, category_tint,
    fmt_ts, hex_to_rgba, human_age, model_label, section, side_label,
    sparkline_svg, stretch,
)


def _sidebar_location() -> None:
    side_label("Location")
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


def _sidebar_status(current, unmeasured_hours: int) -> None:
    st.divider()
    side_label("Data status")
    st.markdown(
        f"<div style='font-size:.82rem;color:{theme.SIDE_TEXT};line-height:1.7;'>"
        f"<span style='color:{theme.SIDE_MUTED};'>Latest reading</span><br>"
        f"<b>{fmt_ts(current['timestamp'])}</b><br>"
        f"<span style='color:"
        f"{theme.SIDE_AMBER if current['is_stale'] else theme.SIDE_ACCENT};"
        f"font-weight:600;'>● {human_age(current['age_hours'])}</span>"
        + (f"<br><span style='color:{theme.SIDE_MUTED};'>Last refreshed</span><br>"
           f"<b>{st.session_state['reloaded_at']}</b>"
           if st.session_state.get("reloaded_at") else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if unmeasured_hours:
        st.caption(f"{unmeasured_hours} hour(s) were never recorded; they are "
                   f"held on the hourly grid so lag features stay aligned to "
                   f"real time.")


def _gauge(current, current_color, current_text_color):
    gauge_steps = [
        # "Good" gets more weight than the pastel EPA bands so the brand
        # colour, not a washed-out sage, anchors the bottom of the dial.
        {"range": [lo, min(hi, 500)],
         "color": hex_to_rgba(category_tint(label),
                              0.55 if label == "Good" else 0.28)}
        for lo, hi, label in config.AQI_CATEGORIES
    ]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current["aqi"],
        number={"font": {"size": 86, "color": current_text_color,
                         "family": "Inter, sans-serif"},
                "valueformat": ".0f"},
        title={"text": f"<span style='font-size:.8rem;color:{MUTED};"
                       f"letter-spacing:.12em;'>CURRENT AQI</span>"},
        gauge={
            "axis": {"range": [0, 500], "tickwidth": 1,
                     "tickcolor": "#C9D6CC",
                     "tickfont": {"color": MUTED, "size": 10},
                     "tickvals": [0, 50, 100, 150, 200, 300, 500]},
            "bar": {"color": current_color, "thickness": 0.3,
                    "line": {"color": "rgba(255,255,255,0.85)", "width": 1}},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps": gauge_steps,
            "threshold": {
                "line": {"color": RED, "width": 3},
                "thickness": 0.82,
                "value": config.ALERT_THRESHOLD,
            },
        },
    ))
    fig.update_layout(
        height=340, margin=dict(l=42, r=42, t=54, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "family": "Inter, sans-serif"},
    )
    return fig


def _hero(current) -> None:
    st.markdown(
        f"""
<div>
  <div class="hero-eyebrow">Air Quality Index · US EPA scale</div>
  <h1 class="hero-title hero-city">{html.escape(config.CITY_NAME)}</h1>
  <div class="hero-sub">
    Latest measured air quality and the next three days, anchored to the most
    recent hourly reading in the feature store.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    current_color = category_fill_color(current["category"])
    current_text_color = category_text_color(current["category"])

    # Gestalt proximity: the gauge and the three readings that qualify it live
    # in a single card, clearly separated from the forecast row below.
    hero_left, hero_right = st.columns([1.25, 1], gap="large")

    with hero_left.container(border=True):
        card_head("Current AQI")
        stretch(st.plotly_chart, _gauge(current, current_color, current_text_color),
                config={"displayModeBar": False})

        driver = current.get("dominant_pollutant")
        pm25 = current.get("pm2_5")
        stats = [
            ("Category", current["category"], "Current severity band",
             current_text_color),
            ("Dominant pollutant",
             theme.POLLUTANT_LABELS.get(driver, "—"),
             "Sets the overall AQI", INK),
            ("PM2.5", f"{pm25:.1f}" if pm25 is not None else "—",
             "µg/m³ right now", INK),
        ]
        st.markdown(
            "<div class='metric-strip'>"
            + "".join(
                f"<div><div class='ms-label'>{label}</div>"
                f"<div class='ms-value' style='color:{color};'>"
                f"{html.escape(str(value))}</div>"
                f"<div class='ms-sub'>{sub}</div></div>"
                for label, value, sub, color in stats
            )
            + "</div>",
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            f"<div class='card'><div class='card-label'>What you are looking at</div>"
            f"<div class='card-sub' style='font-size:.88rem;line-height:1.6;"
            f"margin-top:4px;'>The dial shows the latest measured AQI for "
            f"{html.escape(config.CITY_NAME)} on the US EPA 0–500 scale. "
            f"The red marker at {config.ALERT_THRESHOLD} is the alert threshold; "
            f"the shaded arcs are the EPA severity bands."
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(
            f"<div class='card'><div class='card-label'>Latest reading</div>"
            f"<div class='card-value' style='font-size:1.05rem;'>"
            f"{fmt_ts(current['timestamp'])}</div>"
            f"<div class='card-sub' style='color:"
            f"{AMBER if current['is_stale'] else GREEN};font-weight:600;'>"
            f"● {human_age(current['age_hours'])}</div></div>",
            unsafe_allow_html=True,
        )


def _alerts(current, forecast) -> None:
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
               f"({human_age(current['age_hours'])}). Every forecast below is "
               f"relative to that moment, not to now. Run "
               f"<code>python src/feature_pipeline.py</code> to refresh.")


def _forecast_cards(current, forecast, forecast_issues) -> None:
    if forecast.empty:
        with st.container(border=True):
            card_head("3-Day Forecast", large=True)
            detail = "<br>".join(html.escape(i) for i in forecast_issues) or (
                "Run <code>python src/train_pipeline.py</code> to populate the "
                "model registry."
            )
            banner("info", "🧠", "Forecast unavailable", detail)
        return

    if forecast_issues:
        banner("warn", "⚠️", "Some horizons could not be scored",
               "<br>".join(html.escape(i) for i in forecast_issues))

    recent = data_access.load_history(12)
    observed_tail = ([float(v) for v in recent["aqi"].dropna().tolist()]
                     if not recent.empty else [])

    cards = []
    for row in forecast.itertuples():
        color = category_fill_color(row.category)
        tint = category_tint(row.category)
        text_color = category_text_color(row.category)
        # "%-d" is not portable to Windows, so strip the zero padding by hand.
        when = row.forecast_time.strftime("%b %d").replace(" 0", " ")
        if row.aqi_lower is not None and row.aqi_upper is not None:
            lo_txt, hi_txt = f"{row.aqi_lower:.0f}", f"{row.aqi_upper:.0f}"
        else:
            lo_txt = hi_txt = "—"

        spark = sparkline_svg(
            observed_tail + [float(row.predicted_aqi)],
            color=GREEN, height=44,
            dash_from=max(0, len(observed_tail) - 1),
        )
        cards.append(f"""
<div class="fc-card">
  <div class="fc-swatch" style="background:{color};"></div>
  <div class="fc-when">{html.escape(when)} · {row.horizon_hours} Hours</div>
  <div class="fc-value" style="color:{text_color};">{row.predicted_aqi:.0f}</div>
  <span class="fc-cat" style="background:{hex_to_rgba(tint, 0.14)};color:{text_color};
        border:1px solid {hex_to_rgba(tint, 0.45)};">
    {html.escape(row.category)}
  </span>
  <div class="fc-spark">{spark}</div>
  <div class="fc-range">
    <div><div class="fc-range-label">Min</div>
         <div class="fc-range-value">{lo_txt}</div></div>
    <div><div class="fc-range-label">Max</div>
         <div class="fc-range-value">{hi_txt}</div></div>
  </div>
  <div class="fc-method">{html.escape(model_label(row.method))}</div>
</div>""")

    with st.container(border=True):
        card_head("3-Day Forecast", large=True)
        st.markdown(
            "<div class='section-note'>Each horizon is measured from the latest "
            "reading. Ranges are an 80% band derived from the model's out-of-sample "
            "errors — indicative, not a calibrated prediction interval. The "
            "sparkline is the last 12 hours of measured AQI (solid) continued to "
            "that horizon's prediction (dashed).</div>",
            unsafe_allow_html=True,
        )
        st.markdown(f"<div class='forecast-grid'>{''.join(cards)}</div>",
                    unsafe_allow_html=True)

    served_by_baseline = forecast[~forecast["beats_baseline"]]
    if not served_by_baseline.empty:
        horizons = ", ".join(f"+{h}h" for h in served_by_baseline["horizon_hours"])
        banner("info", "📊", "Some horizons are served by the baseline",
               f"At {horizons}, a persistence baseline (\"AQI stays where it "
               f"is\") beat every trained model in cross-validation, so the "
               f"honest forecast is the baseline. The model's own prediction is "
               f"shown in the accuracy table on Model Analysis for comparison.")


def _pollutant_grid() -> None:
    latest, series = data_access.load_pollutants(24)
    if not latest:
        return

    section(
        "Key pollutants",
        "Concentrations from the latest reading, in µg/m³ as reported by "
        "OpenWeather. The band is that pollutant's own US EPA AQI sub-index — "
        "the same breakpoint tables the overall AQI is built from. NO and NH₃ "
        "have no EPA AQI breakpoints, so they are shown without a band. "
        "Sparklines are the last 24 hours.",
    )

    cards = []
    for name in theme.POLLUTANT_ORDER:
        if name not in latest:
            continue
        value = latest[name]
        graded = theme.pollutant_category(latest, name)
        if graded:
            category, sub_index = graded
            tint = category_tint(category)
            chip = (
                f"<span class='poll-chip' style='background:{hex_to_rgba(tint, 0.14)};"
                f"color:{category_text_color(category)};"
                f"border:1px solid {hex_to_rgba(tint, 0.45)};'>"
                f"{html.escape(category)}</span>"
            )
            foot = f"EPA sub-index {sub_index:.0f}"
        else:
            chip = ("<span class='poll-chip' style='background:#F4F7F4;"
                    f"color:{MUTED};border:1px solid {BORDER};'>No EPA band</span>")
            foot = "No EPA AQI breakpoints"

        spark = sparkline_svg(series.get(name, []), color=GREEN, height=34)
        cards.append(f"""
<div class="poll-card">
  <div class="poll-head">
    <span class="poll-name">{theme.POLLUTANT_LABELS[name]}</span>{chip}
  </div>
  <div class="poll-value">{value:,.2f}<span class="poll-unit">µg/m³</span></div>
  <div class="poll-spark">{spark}</div>
  <div class="poll-foot">{html.escape(foot)}</div>
</div>""")
    st.markdown(f"<div class='poll-grid'>{''.join(cards)}</div>",
                unsafe_allow_html=True)


def _trend_chart(history):
    """Compact 14-day AQI trend, straight from the feature store."""
    trace = dict(
        x=history["timestamp"], y=history["aqi"],
        mode="lines", name="AQI",
        line=dict(color=GREEN, width=2, shape="linear"),
        fill="tozeroy",
        connectgaps=False,
        hovertemplate="%{x|%d %b %H:%M} UTC<br>AQI %{y:.0f}<extra></extra>",
    )
    # Plotly 5.18+ supports a vertical gradient fill; fall back to a flat
    # tint on older versions so the chart still renders.
    try:
        fig = go.Figure(go.Scatter(
            **trace,
            fillgradient=dict(
                type="vertical",
                colorscale=[
                    [0, hex_to_rgba(GREEN, 0.00)],
                    [1, hex_to_rgba(GREEN, 0.22)],
                ],
            ),
        ))
    except TypeError:
        fig = go.Figure(go.Scatter(**trace, fillcolor=hex_to_rgba(GREEN, 0.10)))
    fig.update_layout(
        height=280, margin=dict(l=8, r=8, t=8, b=8), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(color=MUTED, family="Inter, sans-serif", size=11),
        hovermode="x unified",
        xaxis=dict(showgrid=False, linecolor="#D7E0D9", title=None,
                   tickformat="%d %b"),
        yaxis=dict(title="AQI", gridcolor="#EDF1EE", zeroline=False,
                   rangemode="tozero"),
    )
    return fig


def _trend_and_drivers() -> None:
    history = data_access.load_history(14 * 24)
    importance = data_access.load_shap_importance()

    if history.empty and not importance:
        return

    section("Recent trend and model drivers")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        if history.empty:
            st.caption("No history in the feature store yet.")
        else:
            with st.container(border=True):
                card_head("AQI trend (14 days)")
                stretch(st.plotly_chart, _trend_chart(history),
                        config={"displayModeBar": False})
                st.caption("Hourly measured AQI, exactly as recorded — gaps are "
                           "hours the pipeline never captured.")

    with right:
        if not importance:
            with st.container(border=True):
                card_head("SHAP feature importance")
                st.caption("Feature importances appear after the next training "
                           "run writes models/shap_importance_*.json.")
            return

        features = importance["features"][:5]
        top_value = max(f["mean_abs_shap"] for f in features) or 1.0
        rows = "".join(
            f"<div class='imp-row'>"
            f"<div class='imp-head'>"
            f"<div class='imp-name'>{html.escape(theme.feature_label(f['feature']))}</div>"
            f"<div class='imp-value'>{f['mean_abs_shap']:.2f}</div></div>"
            f"<div class='imp-track'><div class='imp-fill' "
            f"style='width:{max(4.0, f['mean_abs_shap'] / top_value * 100):.0f}%;'>"
            f"</div></div></div>"
            for f in features
        )
        with st.container(border=True):
            card_head("SHAP feature importance")
            st.markdown(
                f"<div class='card-sub' style='margin:-4px 0 14px 0;'>"
                f"Top 5 features driving the +{importance['horizon_hours']}h "
                f"{model_label(importance['model_name'])} forecast</div>"
                f"{rows}",
                unsafe_allow_html=True,
            )
            st.caption(
                f"Bars are mean |SHAP value| over the {importance['n_samples']} "
                f"most recent complete rows, scaled so the top feature fills the "
                f"track. Numbers are AQI points of average impact, not "
                f"percentages."
            )
            if importance.get("is_stale"):
                st.caption("⚠️ Generated before the current model was trained.")


def render_dashboard() -> None:
    data = data_access.load_core()
    current = data["current"]

    if current is None:
        st.markdown(
            "<div class='banner banner-info'><div class='banner-icon'>📭</div><div>"
            "<div class='banner-title'>No data in the feature store yet</div>"
            "Run <code>python src/backfill.py --days 90</code> with "
            "<code>OPENWEATHER_API_KEY</code> set, "
            "then <code>python src/train_pipeline.py</code>.</div></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    forecast = data["forecast"]
    with st.sidebar:
        _sidebar_location()
        _sidebar_status(current, data["unmeasured_hours"])

    _hero(current)
    _alerts(current, forecast)
    _forecast_cards(current, forecast, data.get("forecast_issues") or [])
    _trend_and_drivers()
    _pollutant_grid()

    st.markdown(
        f"<div style='margin-top:32px;padding-top:16px;"
        f"border-top:1px solid {BORDER};color:{MUTED};font-size:.78rem;'>"
        f"Data: OpenWeather (pollution) + Open-Meteo (weather) · "
        f"AQI on the US EPA 0–500 scale · "
        f"Latest reading {fmt_ts(current['timestamp'])}</div>",
        unsafe_allow_html=True,
    )
