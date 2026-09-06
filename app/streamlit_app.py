"""
AQI Predictor dashboard — entry point and navigation.

Two pages: "Dashboard" (current conditions and forecast) and "Model Analysis"
(history chart, accuracy, explainability). Page bodies live in
`page_dashboard.py` and `page_analysis.py`; shared styling in `theme.py`.

Run:
    streamlit run app/streamlit_app.py
"""
import datetime as dt
import os
import sys

# The sibling modules import `bootstrap` for their own path setup, but this
# file has to find it first: `streamlit run` puts the script's directory on
# sys.path, other runners (AppTest, plain imports) do not.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bootstrap  # noqa: E402,F401  (sys.path side effect)
import streamlit as st  # noqa: E402

import config  # noqa: E402
import page_analysis  # noqa: E402
import page_dashboard  # noqa: E402
import theme  # noqa: E402

st.set_page_config(
    page_title=f"AQI Predictor — {config.CITY_NAME}",
    page_icon="🌫️",
    layout="wide",
    # "auto" keeps the nav open on desktop but starts collapsed on narrow
    # viewports, where the sidebar overlays the whole page. The chevron in the
    # sidebar header toggles it either way.
    initial_sidebar_state="auto",
)
# ?reload=1 is the top-bar Reload link. Clear cached feature-store reads
# *before* any page runs so the next paint is a real reload, not a rerun
# of the same cached objects.
if st.query_params.get("reload") == "1":
    st.cache_data.clear()
    st.session_state["reloaded_at"] = (
        dt.datetime.now(dt.timezone.utc).strftime("%d %b %Y, %H:%M:%S") + " UTC"
    )
    # Drop the flag on this same run so the pages below read a cold cache.
    # Clearing query params may itself trigger a rerun; that is fine — the
    # cache is already empty either way.
    try:
        del st.query_params["reload"]
    except Exception:
        pass

theme.inject_css()

# Written before st.navigation so the brand sits above the nav links.
with st.sidebar:
    st.markdown(
        f"<div class='side-brand'>AQI Predictor</div>"
        f"<div style='font-size:.78rem;color:{theme.SIDE_MUTED};"
        f"margin:6px 0 16px 13px;'>Serverless 3-day air-quality forecasting"
        f"</div>",
        unsafe_allow_html=True,
    )

# st.navigation/st.Page are the native multipage API, available since 1.36 —
# the floor pinned in requirements.txt — so no manual radio nav is needed.
navigation = st.navigation([
    st.Page(page_dashboard.render_dashboard, title="Dashboard",
            icon=":material/monitoring:", default=True),
    st.Page(page_analysis.render_analysis, title="Model Analysis",
            icon=":material/insights:"),
])
navigation.run()

with st.sidebar:
    st.divider()
    st.caption(f"Rendered {dt.datetime.now(dt.timezone.utc):%d %b %H:%M} UTC")
