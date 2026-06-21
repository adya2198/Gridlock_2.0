"""TrafficGuard - Event Traffic Impact Forecasting & Resource Planner.

Streamlit application that:
1. Forecasts traffic impact of an event in advance (ML),
2. Recommends manpower / barricading / diversion (rule+ML hybrid),
3. Shows historical hotspot map + live alerts,
4. Lets operators log new events that feed back into the model (learning loop).

Run: streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import BLR_CENTER, CAUSE_DISRUPTION_WEIGHT, METRICS_PATH
from src.data_prep import clean
from src.feedback import add_event, feedback_count, resolve_event
from src.features import build_features, impact_tier
from src.model import feature_importance, load, predict_one, train

st.set_page_config(page_title="TrafficGuard", page_icon="🚦", layout="wide")

import plotly.graph_objects as go

# Per-mode design tokens. Both modes share the same CSS structure; only the
# token values, Plotly template, and map basemap differ.
THEMES = {
    "Night": {
        "bg": "#0f1419", "card": "#1a1f2b", "accent": "#00e5ff",
        "accent_hover": "#00b8cc", "text": "#e6e9ef", "muted": "#8b949e",
        "border": "#2d3748", "grid": "#2d3748", "primary_text": "#000000",
        "shadow": "0 4px 6px rgba(0,0,0,0.3)", "shadow_hover": "0 6px 12px rgba(0,0,0,0.4)",
        "map_style": "carto-darkmatter", "line_color": "#00e5ff",
    },
    "Day": {
        "bg": "#f5f7fa", "card": "#ffffff", "accent": "#0077b6",
        "accent_hover": "#005f8e", "text": "#1a1f2b", "muted": "#5c6675",
        "border": "#d8dee9", "grid": "#e2e8f0", "primary_text": "#ffffff",
        "shadow": "0 1px 3px rgba(0,0,0,0.1)", "shadow_hover": "0 4px 10px rgba(0,0,0,0.15)",
        "map_style": "carto-positron", "line_color": "#0077b6",
    },
}

with st.sidebar:
    st.markdown("### 🎨 Appearance")
    mode = st.radio(
        "Theme", ["Night", "Day"],
        index=0, horizontal=True, label_visibility="collapsed",
        format_func=lambda m: f"🌙 {m}" if m == "Night" else f"☀️ {m}",
    )

T = THEMES[mode]

st.markdown(f"""
<style>
:root {{
    --bg-color: {T['bg']};
    --card-bg: {T['card']};
    --accent: {T['accent']};
    --accent-hover: {T['accent_hover']};
    --text-main: {T['text']};
    --text-muted: {T['muted']};
    --border-color: {T['border']};
    --shadow: {T['shadow']};
    --shadow-hover: {T['shadow_hover']};
    --primary-text: {T['primary_text']};
}}

/* Global Background & Text */
.stApp {{
    background-color: var(--bg-color);
    color: var(--text-main);
}}

/* Sidebar follows the theme */
[data-testid="stSidebar"] {{
    background-color: var(--card-bg);
    border-right: 1px solid var(--border-color);
}}
[data-testid="stSidebar"] * {{
    color: var(--text-main);
}}

/* Headers */
h1, h2, h3, h4, h5, h6 {{
    color: var(--text-main) !important;
    font-weight: 600 !important;
}}

/* Metric Tiles */
[data-testid="stMetric"] {{
    background-color: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 16px;
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}}
[data-testid="stMetric"]:hover {{
    transform: translateY(-2px);
    box-shadow: var(--shadow-hover);
}}
[data-testid="stMetricValue"] {{
    color: var(--accent) !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background-color: transparent;
    border-bottom: 1px solid var(--border-color);
}}
.stTabs [data-baseweb="tab"] {{
    color: var(--text-muted);
}}
.stTabs [aria-selected="true"] {{
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}}

/* Buttons */
.stButton > button {{
    background-color: var(--card-bg);
    color: var(--text-main);
    border: 1px solid var(--border-color);
    border-radius: 6px;
    transition: all 0.2s ease;
}}
.stButton > button:hover {{
    border-color: var(--accent);
    color: var(--accent);
}}
.stButton > button[data-testid="baseButton-primary"] {{
    background-color: var(--accent);
    color: var(--primary-text);
    border: none;
}}
.stButton > button[data-testid="baseButton-primary"]:hover {{
    background-color: var(--accent-hover);
    color: var(--primary-text);
    border: none;
}}

/* Expanders */
.streamlit-expanderHeader {{
    background-color: var(--card-bg);
    color: var(--text-main);
    border-radius: 6px;
}}
[data-testid="stExpander"] {{
    border: 1px solid var(--border-color);
    border-radius: 6px;
    background-color: var(--card-bg);
}}

/* Inputs / Selectboxes */
.stSelectbox > div > div, .stDateInput > div > div, .stNumberInput > div > div {{
    background-color: var(--card-bg);
    color: var(--text-main);
    border-color: var(--border-color);
}}

/* Dataframes */
[data-testid="stDataFrame"] {{
    background-color: var(--card-bg);
}}

/* Info/Warning/Success Callouts */
.stAlert {{
    background-color: var(--card-bg) !important;
    color: var(--text-main) !important;
    border: 1px solid var(--border-color);
}}

/* Captions */
.stCaption {{
    color: var(--text-muted) !important;
}}
</style>
""", unsafe_allow_html=True)

CHART_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]),
        xaxis=dict(showgrid=True, gridcolor=T["grid"], zerolinecolor=T["grid"]),
        yaxis=dict(showgrid=True, gridcolor=T["grid"], zerolinecolor=T["grid"]),
    )
)
MAP_STYLE = T["map_style"]
LINE_COLOR = T["line_color"]

ALERT_COLORS = {"RED": "#ff4b4b", "ORANGE": "#ffa726", "YELLOW": "#fdd835", "GREEN": "#66bb6a"}
TIER_COLORS = {"SEVERE": "#ff4b4b", "HIGH": "#ffa726", "MODERATE": "#fdd835", "LOW": "#66bb6a"}


@st.cache_data(show_spinner="Loading & engineering data...")
def get_data() -> pd.DataFrame:
    return build_features(clean())


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load()


def reset_caches() -> None:
    get_data.clear()
    get_model.clear()

# Pull recommend lazily (rules-only module, cheap).
from src.recommend import recommend  # noqa: E402

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.title("🚦 TrafficGuard")
st.caption(
    "Forecast event-related traffic impact and get optimal manpower, "
    "barricading & diversion plans - learning from every new event."
)

df = get_data()
tm = get_model()

CAUSES = sorted(df["event_cause"].dropna().unique().tolist())
CORRIDORS = sorted(df["corridor"].dropna().unique().tolist())
ZONES = sorted(df["zone"].dropna().unique().tolist())

tab_dash, tab_forecast, tab_hotspot, tab_add, tab_model = st.tabs(
    ["📊 Dashboard", "🔮 Forecast & Plan", "🗺️ Hotspot Map", "➕ Log Event", "🧠 Model"]
)

# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total events", f"{len(df):,}")
    severe = int((df["impact_score"] >= 65).sum())
    c2.metric("Severe-impact events", f"{severe:,}")
    active = int((df["status"] == "active").sum())
    c3.metric("Currently active", f"{active:,}")
    c4.metric("Operator-logged", feedback_count())

    st.subheader("⚠️ Live alerts (active incidents, highest impact first)")
    live = df[df["status"] == "active"].sort_values("impact_score", ascending=False).head(10)
    if live.empty:
        st.info("No active incidents right now.")
    else:
        for _, r in live.iterrows():
            tier = impact_tier(r["impact_score"])
            rec = recommend(
                r["impact_score"],
                event_cause=r["event_cause"],
                requires_road_closure=bool(r["requires_road_closure"]),
                is_major_corridor=bool(r["is_major_corridor"]),
                is_peak=bool(r["is_peak"]),
                is_festival=bool(r["is_festival"]),
            )

            color = ALERT_COLORS[rec.alert_level]
            lat, lon = r["latitude"], r["longitude"]
            has_loc = pd.notna(lat) and pd.notna(lon)
            junction = r["junction"] if r["junction"] != "unknown" else r["corridor"]
            loc_line = (
                f"<div style='margin-top:6px;color:var(--text-muted);'><small>📍 {junction} ({lat:.4f}, {lon:.4f})</small></div>"
                if has_loc else ""
            )

            alert_col, btn_col = st.columns([8, 1])
            with alert_col:
                st.markdown(
                    f"<div style='border-left:6px solid {color};padding:12px 16px;"
                    f"margin-bottom:12px;background-color:var(--card-bg);border-radius:6px;"
                    f"box-shadow:0 4px 6px rgba(0,0,0,0.2);transition:transform 0.2s ease;'>"
                    f"<b style='color:{color};font-size:1.1em;'>{rec.alert_level} • {tier}</b> • "
                    f"<span style='color:var(--text-muted)'>({r['impact_score']:.0f}/100)</span> - <b>{r['event_cause']}</b> on "
                    f"<b>{r['corridor']}</b> ({r['zone']})<br>"
                    f"<div style='margin-top:8px;color:var(--text-main);'><small>Deploy {rec.officers} officers, {rec.barricade_units} "
                    f"barricades. {rec.diversion}</small></div>"
                    f"{loc_line}</div>",
                    unsafe_allow_html=True,
                )
            if has_loc:
                with st.expander("📍 View on map"):
                    st.map(
                        pd.DataFrame({"lat": [float(lat)], "lon": [float(lon)]}),
                        zoom=14,
                    )
                    st.markdown(
                        f"[Open in Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon})"
                    )

            with btn_col:
                if st.button("✅ Resolve", key=f"resolve_{r['id']}", help="Mark this incident as resolved"):
                    resolve_event(str(r["id"]))
                    reset_caches()
                    st.success(f"Marked {r['id']} resolved.")
                    st.rerun()

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Impact tier distribution")
        tiers = df["impact_score"].apply(impact_tier).value_counts()
        order = ["LOW", "MODERATE", "HIGH", "SEVERE"]
        tiers = tiers.reindex(order).fillna(0)
        fig = px.bar(
            x=tiers.index, y=tiers.values,
            color=tiers.index, color_discrete_map=TIER_COLORS,
            labels={"x": "Tier", "y": "Events"},
        )
        fig.update_layout(template=CHART_TEMPLATE, showlegend=False, height=320)
        st.plotly_chart(fig, use_container_width=True)
    with colB:
        st.subheader("Avg impact by hour of day")
        hourly = df.groupby("hour")["impact_score"].mean().reset_index()
        fig2 = px.line(hourly, x="hour", y="impact_score", markers=True)
        fig2.update_traces(line_color=LINE_COLOR, marker=dict(color=LINE_COLOR))
        fig2.update_layout(template=CHART_TEMPLATE, height=320, yaxis_title="Avg impact")
        st.plotly_chart(fig2, use_container_width=True)


# -----------------------------------------------------------------------------
# Forecast & Plan
# -----------------------------------------------------------------------------
with tab_forecast:
    st.subheader("🔮 Forecast the impact of an upcoming event")
    st.caption("Fill the event details — the model predicts impact and a plan.")

    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        f_cause = st.selectbox("Event cause", CAUSES, index=CAUSES.index("public_event") if "public_event" in CAUSES else 0)
        f_corr = st.selectbox("Corridor / road", CORRIDORS, index=0)
        f_zone = st.selectbox("Zone", ZONES, index=0)
    with fc2:
        f_date = st.date_input("Date", value=dt.date.today())
        f_hour = st.slider("Hour of day", 0, 23, 18)
        f_priority = st.selectbox("Priority", ["high", "low"], index=0)
    with fc3:
        f_closure = st.checkbox("Requires road closure", value=True)
        f_etype = st.selectbox("Event type", ["planned", "unplanned"], index=0)
        f_veh = st.selectbox("Vehicle type (if any)", ["unknown"] + sorted(df["veh_type"].unique().tolist()))

    if st.button("Forecast impact & generate plan", type="primary"):
        from src.calendar_features import build_calendar

        cal = build_calendar([f_date.year])
        cal_row = cal[cal["date"] == f_date]
        is_holiday = int(cal_row["is_public_holiday"].iloc[0]) if not cal_row.empty else 0
        is_fest = int(cal_row["is_festival"].iloc[0]) if not cal_row.empty else 0
        footfall = float(cal_row["festival_footfall"].iloc[0]) if not cal_row.empty else 0.0

        dow = f_date.weekday()
        is_major = int(any(k in f_corr for k in ["orr", "mysore", "bellary", "tumkur", "hosur", "magadi", "old madras", "bannerghata", "chord", "cbd"]))
        is_peak = int(f_hour in [8, 9, 10, 17, 18, 19, 20])

        # Estimate hotspot density from history for this zone/corridor.
        hist = df[(df["corridor"] == f_corr)]
        hotspot = float(hist["hotspot_density"].mean()) if not hist.empty else 0.3

        row = {
            "event_cause": f_cause, "event_type": f_etype, "veh_type": f_veh,
            "corridor": f_corr, "priority": f_priority, "zone": f_zone,
            "requires_road_closure": int(f_closure), "is_major_corridor": is_major,
            "hour": f_hour, "dow": dow, "month": f_date.month,
            "is_weekend": int(dow >= 5), "is_peak": is_peak,
            "is_public_holiday": is_holiday, "is_festival": is_fest,
            "festival_footfall": footfall,
            "cause_weight": CAUSE_DISRUPTION_WEIGHT.get(f_cause, 0.35),
            "hotspot_density": hotspot,
        }
        score = predict_one(tm, row)
        rec = recommend(
            score, event_cause=f_cause, requires_road_closure=f_closure,
            is_major_corridor=bool(is_major), is_peak=bool(is_peak),
            is_festival=bool(is_fest),
        )

        color = ALERT_COLORS[rec.alert_level]
        st.markdown(
            f"<h2 style='color:{color}'>Alert: {rec.alert_level} • "
            f"{rec.tier} impact ({score:.0f}/100)</h2>",
            unsafe_allow_html=True,
        )

        if is_fest or is_holiday:
            st.warning(f"📅 {f_date} is a holiday/festival - footfall amplified.")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("👮 Officers", rec.officers)
        m2.metric("👷 Wardens", rec.wardens)
        m3.metric("🚧 Barricade units", rec.barricade_units)
        m4.metric("🏗️ Tow/Crane", "Yes" if rec.tow_crane else "No")

        st.markdown(f"**🗺️ Diversion plan:** {rec.diversion}")
        st.markdown("**💡 Mitigation ideas to reduce congestion:**")
        for m in rec.mitigations:
            st.markdown(f"- {m}")
        with st.expander("Why this plan? (rationale)"):
            for r in rec.rationale:
                st.markdown(f"- {r}")


# -----------------------------------------------------------------------------
# Hotspot map
# -----------------------------------------------------------------------------
with tab_hotspot:
    st.subheader("🗺️ Historical impact hotspots")
    cause_filter = st.multiselect("Filter by cause", CAUSES, default=[])
    min_impact = st.slider("Min impact score", 0, 100, 40)

    plot_df = df.copy()
    if cause_filter:
        plot_df = plot_df[plot_df["event_cause"].isin(cause_filter)]
    plot_df = plot_df[plot_df["impact_score"] >= min_impact]
    plot_df = plot_df.dropna(subset=["latitude", "longitude"]).head(3000)

    if plot_df.empty:
        st.info("No events match the filter.")
    else:
        fig = px.scatter_mapbox(
            plot_df, lat="latitude", lon="longitude",
            color="impact_score", size="impact_score",
            color_continuous_scale="YlOrRd", size_max=15, zoom=10,
            center={"lat": BLR_CENTER[0], "lon": BLR_CENTER[1]},
            hover_data=["event_cause", "corridor", "zone", "impact_score"],
        )
        fig.update_layout(template=CHART_TEMPLATE, mapbox_style=MAP_STYLE, height=600, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Showing {len(plot_df):,} events. Brighter/bigger = higher impact.")


# -----------------------------------------------------------------------------
# Log event (feedback loop)
# -----------------------------------------------------------------------------
with tab_add:
    st.subheader("➕ Log a new event (feeds the learning loop)")
    st.caption("New events are stored and used the next time the model is retrained.")

    with st.form("add_event"):
        a1, a2 = st.columns(2)
        with a1:
            lat = st.number_input("Latitude", value=BLR_CENTER[0], format="%.6f")
            lon = st.number_input("Longitude", value=BLR_CENTER[1], format="%.6f")
            cause = st.selectbox("Cause", CAUSES, key="add_cause")
            corr = st.selectbox("Corridor", CORRIDORS, key="add_corr")
        with a2:
            zone = st.selectbox("Zone", ZONES, key="add_zone")
            prio = st.selectbox("Priority", ["high", "low"], key="add_prio")
            closure = st.checkbox("Requires road closure", key="add_closure")
            dur = st.number_input("Estimated duration (min)", value=60, min_value=0)
        submitted = st.form_submit_button("Save event", type="primary")
        if submitted:
            eid = add_event(
                latitude=lat, longitude=lon, event_cause=cause,
                requires_road_closure=closure, priority=prio,
                corridor=corr, zone=zone, duration_min=dur,
            )
            st.success(f"✅ Saved event {eid}. Total operator-logged: {feedback_count()}")
            st.info("Go to the Model tab and click Retrain to learn from it.")


# -----------------------------------------------------------------------------
# Model tab
# -----------------------------------------------------------------------------
with tab_model:
    st.subheader("🧠 Model performance & retraining")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MAE", metrics.get("mae"))
        m2.metric("R²", metrics.get("r2"))
        m3.metric("Tier accuracy", f"{metrics.get('tier_accuracy', 0) * 100:.0f}%")
        m4.metric("Train rows", metrics.get("n_train"))

    st.subheader("Feature importance")
    imp = feature_importance(tm)
    fig = px.bar(imp, x="importance", y="feature", orientation="h")
    fig.update_traces(marker_color=LINE_COLOR)
    fig.update_layout(template=CHART_TEMPLATE, height=420, yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("**Retrain the model** (incorporates all operator-logged events):")
    if st.button("🔄 Retrain now", type="primary"):
        with st.spinner("Retraining on full + feedback data..."):
            new_tm = train(save=True)
            reset_caches()
            st.success(f"Retrained! MAE={new_tm.metrics['mae']}, R²={new_tm.metrics['r2']}")
            st.rerun()