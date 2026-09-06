"""Model Analysis page: history chart, accuracy table, SHAP explainability."""
import html

import bootstrap  # noqa: F401  (sys.path side effect)
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
import data_access
from predict import shap_plot_status
from theme import (
    BORDER, GREEN, INK, MUTED, RED, TEAL,
    banner, category_tint, fmt_ts, hex_to_rgba, model_label, section,
    side_label, stretch,
)


def _sidebar_settings():
    side_label("Chart settings")
    history_days = st.slider("History window (days)", 1, 60, 14)
    show_band = st.checkbox("Show uncertainty band", value=True)
    show_categories = st.checkbox("Shade AQI categories", value=True)

    st.divider()
    side_label("Explainability")
    any_shap = False
    for horizon in config.HORIZONS:
        if shap_plot_status(horizon):
            any_shap = True
            st.markdown(f"[SHAP — {horizon}h model](#drivers)")
    if not any_shap:
        st.caption("SHAP plots appear after training.")
    return history_days, show_band, show_categories


def _history_chart(current, forecast, history, show_band, show_categories):
    fig = go.Figure()

    # Scale to the data rather than the full 0-500 EPA range: real readings sit
    # in a narrow band, and a fixed axis would flatten the series into a line.
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
            fig.add_hrect(
                y0=lo, y1=min(hi, y_max),
                fillcolor=category_tint(label), opacity=0.07,
                layer="below", line_width=0,
                annotation_text=label if (hi - lo) > y_max * 0.12 else None,
                annotation_position="right",
                annotation_font=dict(size=9, color=MUTED),
            )

    if not history.empty:
        fig.add_trace(go.Scatter(
            x=history["timestamp"], y=history["aqi"],
            mode="lines", name="Observed",
            line=dict(color=GREEN, width=2.2),
            connectgaps=False,
            hovertemplate="%{x|%d %b %H:%M} UTC<br>AQI %{y:.0f}<extra>Observed</extra>",
        ))

    if not forecast.empty:
        anchor_t = current["timestamp"]
        anchor_v = current["aqi"]
        fx = [anchor_t] + list(forecast["forecast_time"])
        fy = [anchor_v] + list(forecast["predicted_aqi"])

        has_band = (forecast["aqi_lower"].notna().all()
                    and forecast["aqi_upper"].notna().all())
        if show_band and has_band:
            lo = [anchor_v] + list(forecast["aqi_lower"])
            hi = [anchor_v] + list(forecast["aqi_upper"])
            # Upper bound first, then the lower reversed, so the fill closes.
            fig.add_trace(go.Scatter(
                x=fx + fx[::-1], y=hi + lo[::-1],
                fill="toself", fillcolor=hex_to_rgba(TEAL, 0.14),
                line=dict(color="rgba(0,0,0,0)"), hoverinfo="skip",
                name="80% range", showlegend=True,
            ))

        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines+markers", name="Forecast",
            line=dict(color=TEAL, width=2.6, dash="dash"),
            marker=dict(size=9, color=TEAL, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="%{x|%d %b %H:%M} UTC<br>AQI %{y:.0f}<extra>Forecast</extra>",
        ))

        fig.add_vline(x=anchor_t, line_width=1, line_dash="dot",
                      line_color="#9AAFA1")

    fig.add_hline(y=config.ALERT_THRESHOLD, line_dash="dash", line_color=RED,
                  line_width=1.4, annotation_text="Alert threshold",
                  annotation_position="top left",
                  annotation_font=dict(color=RED, size=11))

    fig.update_layout(
        height=430, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FFFFFF",
        font=dict(color=INK, family="Inter, sans-serif", size=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False, linecolor="#D7E0D9", title=None),
        yaxis=dict(title="AQI", gridcolor="#EDF1EE", zeroline=False,
                   range=[0, y_max]),
    )
    return fig


def _accuracy_table(metrics) -> None:
    if not metrics:
        return
    section("Model accuracy",
            "Out-of-sample results from time-series cross-validation, compared "
            "against a persistence baseline. Lower RMSE and MAE are better; "
            "higher R² is better.")

    rows = []
    for horizon, meta in sorted(metrics.items()):
        selected = meta.get("selected_metrics", {})
        baseline = meta.get("persistence_baseline", {})
        rows.append({
            "Horizon": f"+{horizon}h",
            "Best model": model_label(meta.get("deployed_model", "—")),
            "Model RMSE": round(selected.get("rmse", float("nan")), 1),
            "Model MAE": round(selected.get("mae", float("nan")), 1),
            "Model R²": round(selected.get("r2", float("nan")), 3),
            "Baseline RMSE": round(baseline.get("rmse", float("nan")), 1),
            "Baseline MAE": round(baseline.get("mae", float("nan")), 1),
            "Serving": "Model" if meta.get("beats_baseline") else "Baseline",
        })
    stretch(st.dataframe, pd.DataFrame(rows), hide_index=True)

    trained_at = next((m.get("trained_at") for m in metrics.values()
                       if m.get("trained_at")), None)
    if trained_at:
        st.caption(f"Last trained "
                   f"{fmt_ts(trained_at.replace('Z', '').split('+')[0])}")


def _shap_section() -> None:
    section("What's driving the forecast", anchor="drivers")

    shap_entries = [(h, shap_plot_status(h)) for h in config.HORIZONS]
    shap_entries = [(h, s) for h, s in shap_entries if s]

    if not shap_entries:
        st.markdown("<div class='section-note'>SHAP plots appear here after "
                    "training (<code>train_pipeline.py</code> generates them)."
                    "</div>", unsafe_allow_html=True)
        return

    if any(stale for _, (_, stale) in shap_entries):
        banner("warn", "🔍", "Some SHAP plots predate the current models",
               "They were generated in an earlier training run, so they explain "
               "a previous version of the model. They refresh automatically the "
               "next time training runs somewhere with <code>shap</code> "
               "installed.")

    tabs = st.tabs([f"+{h}h model" for h, _ in shap_entries])
    for tab, (horizon, (path, stale)) in zip(tabs, shap_entries):
        with tab:
            if stale:
                st.markdown("<span class='pill'>⚠️ Older than the current "
                            "model</span>", unsafe_allow_html=True)
            stretch(st.image, path)
            st.caption(
                f"SHAP summary for the {horizon}h model. Features are ranked by "
                f"average impact on the prediction; colour shows whether a high "
                f"or low feature value pushes the forecast up or down. The "
                f"sample is in-sample for the deployed model, so this shows what "
                f"the model relies on rather than how well it generalises."
            )

    st.caption(
        "Training also writes the ranked mean |SHAP value| per feature to "
        "`models/shap_importance_{h}h.json`; the Dashboard's Primary factors "
        "panel reads the shortest horizon from those files."
    )


def render_analysis() -> None:
    data = data_access.load_core()
    current = data["current"]

    if current is None:
        st.markdown(
            "<div class='banner banner-info'><div class='banner-icon'>📭</div><div>"
            "<div class='banner-title'>No data in the feature store yet</div>"
            "Run <code>python src/backfill.py --days 90</code> with "
            "<code>OPENWEATHER_API_KEY</code> set, then "
            "<code>python src/train_pipeline.py</code>.</div></div>",
            unsafe_allow_html=True,
        )
        st.stop()

    with st.sidebar:
        history_days, show_band, show_categories = _sidebar_settings()

    st.markdown(
        """
<div>
  <div class="hero-eyebrow">Model analysis</div>
  <h1 class="hero-title">How the forecast is made</h1>
  <div class="hero-sub">
    Measured history against the current forecast, cross-validated accuracy
    per horizon, and the features the deployed models lean on.
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    forecast = data["forecast"]
    history = data_access.load_history(history_days * 24)

    section("Observed history and forecast",
            "Solid line is measured AQI; dashed is forecast. Breaks in the line "
            "are hours the pipeline never recorded.")
    with st.container(border=True):
        stretch(st.plotly_chart,
                _history_chart(current, forecast, history, show_band,
                               show_categories),
                config={"displayModeBar": False})

    _accuracy_table(data["metrics"])
    _shap_section()

    st.markdown(
        f"<div style='margin-top:32px;padding-top:16px;"
        f"border-top:1px solid {BORDER};color:{MUTED};font-size:.78rem;'>"
        f"Metrics come from the model registry written by "
        f"<code>src/train_pipeline.py</code> · "
        f"Latest reading {fmt_ts(current['timestamp'])}</div>",
        unsafe_allow_html=True,
    )
