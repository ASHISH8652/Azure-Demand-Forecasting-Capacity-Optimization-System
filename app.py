import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Azure Capacity Intelligence",
    page_icon="🔷",
    layout="wide"
)

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "demand_forecasting_model.pkl"
FEATURE_PATH = "feature_list.pkl"
DATA_PATH = "feature_engineered_dataset.csv"
LOG_PATH = "streamlit_prediction_log.csv"

# =========================================================
# LOAD FILES
# =========================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error(f"Feature list file not found: {FEATURE_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    st.error("feature_engineered_dataset.csv not found.")
    st.stop()

# =========================================================
# PREP DATA
# =========================================================
if "time_stamp" in df.columns:
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
    df = df.sort_values("time_stamp").reset_index(drop=True)

if "region" not in df.columns:
    df["region"] = "Unknown"

if "service_type" not in df.columns:
    df["service_type"] = "Compute"

if "usage_units" not in df.columns:
    df["usage_units"] = 0.0

if "provisioned_capacity" not in df.columns:
    df["provisioned_capacity"] = df["usage_units"] + 100

if "cost_usd" not in df.columns:
    df["cost_usd"] = 1000.0

if "availability_pct" not in df.columns:
    df["availability_pct"] = 99.5

df["utilization_pct"] = np.where(
    df["provisioned_capacity"] > 0,
    (df["usage_units"] / df["provisioned_capacity"]) * 100,
    0
)

df["headroom_units"] = df["provisioned_capacity"] - df["usage_units"]
df["waste_units"] = np.where(df["headroom_units"] > 0, df["headroom_units"], 0)
df["waste_pct"] = np.where(df["provisioned_capacity"] > 0, (df["waste_units"] / df["provisioned_capacity"]) * 100, 0)
df["wasted_capacity_cost"] = np.where(
    df["provisioned_capacity"] > 0,
    (df["waste_units"] / df["provisioned_capacity"]) * df["cost_usd"],
    0
)
df["daily_growth_rate"] = df["usage_units"].pct_change().fillna(0) * 100

if "time_stamp" in df.columns:
    df["year"] = df["time_stamp"].dt.year
    df["month_name"] = df["time_stamp"].dt.strftime("%b %Y")
    df["month_period"] = df["time_stamp"].dt.to_period("M").astype(str)
    df["weekday"] = df["time_stamp"].dt.day_name()
else:
    df["year"] = 2026
    df["month_name"] = "Unknown"
    df["month_period"] = "Unknown"
    df["weekday"] = "Unknown"

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top left, #0b1f4a 0%, #050d1a 38%, #040916 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071326 0%, #08101d 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
.block-container {
    padding-top: 2.2rem;
    padding-bottom: 1rem;
    max-width: 1650px;
}
.main-title {
    font-size: 2.05rem;
    font-weight: 800;
    color: #7fb0ff;
    letter-spacing: 2px;
    margin-bottom: 0.2rem;
}
.sub-title {
    color: #8ea2d0;
    font-size: 0.95rem;
    margin-bottom: 0.8rem;
}
.section-title {
    color: #6f8fff;
    font-size: 0.9rem;
    font-weight: 800;
    letter-spacing: 3px;
    margin-top: 0.5rem;
    margin-bottom: 0.7rem;
}
.kpi-card {
    background: rgba(10, 20, 45, 0.95);
    border: 1px solid rgba(110, 140, 255, 0.18);
    border-radius: 18px;
    padding: 18px;
    min-height: 116px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.kpi-label {
    color: #8ea2d0;
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.kpi-value {
    color: #f8fbff;
    font-size: 1.95rem;
    font-weight: 800;
    margin-top: 8px;
}
.kpi-sub {
    color: #8ac7b8;
    font-size: 0.8rem;
    margin-top: 8px;
}
.panel {
    background: rgba(9, 18, 40, 0.96);
    border: 1px solid rgba(120, 150, 255, 0.12);
    border-radius: 18px;
    padding: 15px 16px;
    margin-bottom: 16px;
}
.panel-title {
    color: #dbe7ff;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
}
.small-note {
    color: #7d8fb3;
    font-size: 0.85rem;
}
.alert-box {
    border-radius: 14px;
    padding: 12px 16px;
    color: white;
    font-weight: 700;
    margin: 10px 0 14px 0;
}
.badge {
    display: inline-block;
    padding: 0.45rem 0.78rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    margin-bottom: 0.45rem;
    background: rgba(67,97,238,0.16);
    border: 1px solid rgba(118,169,255,0.18);
    color: #dbe7ff;
    font-size: 0.8rem;
    font-weight: 700;
}
.insight-box {
    background: linear-gradient(90deg, rgba(29,78,216,0.18), rgba(14,165,233,0.12));
    border: 1px solid rgba(118,169,255,0.18);
    border-radius: 14px;
    padding: 12px 16px;
    color: #eaf2ff;
    font-size: 0.94rem;
    margin-bottom: 16px;
}
.summary-box {
    background: rgba(12, 24, 52, 0.95);
    border: 1px solid rgba(120, 150, 255, 0.18);
    border-radius: 16px;
    padding: 16px;
}
.summary-box ul {
    margin: 0;
    padding-left: 1.15rem;
}
.summary-box li {
    color: #dbe7ff;
    margin-bottom: 0.38rem;
}
.chip-row {
    margin-top: 0.2rem;
    margin-bottom: 0.4rem;
}
.chip {
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    margin-bottom: 0.35rem;
    color: white;
    font-size: 0.76rem;
    font-weight: 700;
}
.chip-green { background: #16a34a; }
.chip-blue { background: #0284c7; }
.chip-orange { background: #ea580c; }
.chip-red { background: #dc2626; }
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0.7rem 0 1rem 0;
}
div.stButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.08);
    background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
    color: white;
    font-weight: 700;
}
div.stDownloadButton > button {
    width: 100%;
    height: 3rem;
    border-radius: 14px;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def load_logs():
    if os.path.exists(LOG_PATH):
        return pd.read_csv(LOG_PATH)
    return pd.DataFrame(columns=[
        "timestamp", "region", "service", "predicted_usage_units",
        "capacity_action", "recommendation_note",
        "provisioned_capacity", "availability_pct", "market_demand_index"
    ])

def save_log(new_row: pd.DataFrame):
    old = load_logs()
    merged = pd.concat([old, new_row], ignore_index=True)
    merged.to_csv(LOG_PATH, index=False)

def capacity_action(pred_value: float):
    if pred_value >= 900:
        return "Scale Up Capacity Immediately", "#dc2626", "High utilization risk"
    elif pred_value >= 700:
        return "Prepare Additional Capacity", "#ea580c", "Demand is rising"
    elif pred_value >= 500:
        return "Monitor Demand Closely", "#0284c7", "Moderate pressure"
    return "Capacity is Sufficient", "#16a34a", "Healthy operating buffer"

def align_features(input_df: pd.DataFrame):
    aligned = input_df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[feature_columns]
    return aligned

def get_key_insight(frame: pd.DataFrame, threshold: float):
    region_util = frame.groupby("region", as_index=False)["utilization_pct"].mean()
    top_region = region_util.sort_values("utilization_pct", ascending=False).iloc[0]["region"] if not region_util.empty else "N/A"
    service_headroom = frame.groupby("service_type", as_index=False)["headroom_units"].mean()
    top_headroom = service_headroom.sort_values("headroom_units", ascending=False).iloc[0]["service_type"] if not service_headroom.empty else "N/A"
    risk_events = int(((frame["utilization_pct"] / 100.0) >= threshold).sum())
    return f"{top_region} shows the highest utilization, while {top_headroom} has the strongest average headroom. Current threshold flags {risk_events} high-risk records."

def get_risk_chip_html(avg_util, wasted_cost, total_cost):
    html = '<div class="chip-row">'
    if avg_util < 60:
        html += '<span class="chip chip-green">Healthy Utilization</span>'
    elif avg_util < 75:
        html += '<span class="chip chip-blue">Moderate Utilization</span>'
    else:
        html += '<span class="chip chip-orange">High Utilization</span>'

    if total_cost > 0 and (wasted_cost / total_cost) > 0.30:
        html += '<span class="chip chip-red">High Waste Cost</span>'
    else:
        html += '<span class="chip chip-green">Controlled Waste</span>'

    html += '</div>'
    return html

def get_top_5_demand_months(frame):
    return (
        frame.groupby("month_name", as_index=False)["usage_units"]
        .sum()
        .sort_values("usage_units", ascending=False)
        .head(5)
    )

def get_service_util_headroom(frame):
    return frame.groupby("service_type", as_index=False).agg({
        "utilization_pct": "mean",
        "headroom_units": "mean"
    })

def get_monthly_waste_cost(frame):
    return frame.groupby("month_name", as_index=False)["wasted_capacity_cost"].sum()

def get_region_risk_heatmap_data(frame):
    risk_heat = frame.groupby(["region", "month_period"], as_index=False)["risk_event"].sum()
    if risk_heat.empty:
        return None
    return risk_heat.pivot(index="region", columns="month_period", values="risk_event").fillna(0)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.markdown("## 🔷 Azure Capacity Intel")
st.sidebar.markdown("---")

all_regions = sorted(df["region"].dropna().astype(str).unique().tolist())
all_services = sorted(df["service_type"].dropna().astype(str).unique().tolist())
all_years = sorted(df["year"].dropna().astype(int).unique().tolist())

selected_regions = st.sidebar.multiselect("Regions", all_regions, default=all_regions[:min(5, len(all_regions))] if all_regions else [])
selected_services = st.sidebar.multiselect("Service Type", all_services, default=all_services if all_services else [])
selected_years = st.sidebar.multiselect("Year", all_years, default=all_years if all_years else [])

risk_threshold = st.sidebar.slider(
    "Capacity Risk Threshold",
    0.40, 0.95, 0.65, 0.01,
    help="This threshold is used to flag high-utilization records as capacity risk events."
)

st.sidebar.caption("Utilization % alert level")

filtered_df = df.copy()
if selected_regions:
    filtered_df = filtered_df[filtered_df["region"].isin(selected_regions)]
if selected_services:
    filtered_df = filtered_df[filtered_df["service_type"].isin(selected_services)]
if selected_years:
    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]

if filtered_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

filtered_df["risk_event"] = (filtered_df["utilization_pct"] / 100.0) >= risk_threshold
filtered_df["underutilized_flag"] = filtered_df["utilization_pct"] < 50

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Summary")
st.sidebar.write(f"**Records selected:** {len(filtered_df):,}")
st.sidebar.write(f"**Regions selected:** {filtered_df['region'].nunique()}")
st.sidebar.write(f"**Services selected:** {filtered_df['service_type'].nunique()}")
st.sidebar.write(f"**Years selected:** {filtered_df['year'].nunique()}")

st.sidebar.markdown("---")
st.sidebar.markdown("### Quick Status")
st.sidebar.write(f"**Average Utilization:** {filtered_df['utilization_pct'].mean():.1f}%")
st.sidebar.write(f"**Average Headroom:** {filtered_df['headroom_units'].mean():,.0f}")
st.sidebar.write(f"**Risk Events:** {int(filtered_df['risk_event'].sum())}")
st.sidebar.write(f"**Underutilized Flags:** {int(filtered_df['underutilized_flag'].sum())}")

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">AZURE CAPACITY INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Milestone 4 • Forecast Integration & Capacity Planning Dashboard</div>', unsafe_allow_html=True)

st.markdown(
    f"""
    <div>
        <span class="badge">Active Regions: {filtered_df['region'].nunique()}</span>
        <span class="badge">Active Services: {filtered_df['service_type'].nunique()}</span>
        <span class="badge">Selected Years: {filtered_df['year'].nunique()}</span>
        <span class="badge">Risk Threshold: {risk_threshold:.2f}</span>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="insight-box"><b>Key Insight:</b> {get_key_insight(filtered_df, risk_threshold)}</div>',
    unsafe_allow_html=True
)

avg_util_all = filtered_df["utilization_pct"].mean()
total_cost_all = filtered_df["cost_usd"].sum()
wasted_cost_all = filtered_df["wasted_capacity_cost"].sum()
st.markdown(get_risk_chip_html(avg_util_all, wasted_cost_all, total_cost_all), unsafe_allow_html=True)

snap1, snap2, snap3, snap4 = st.columns(4)
with snap1:
    st.info(f"Top Region: {filtered_df.groupby('region')['utilization_pct'].mean().idxmax() if not filtered_df.empty else 'N/A'}")
with snap2:
    st.info(f"Top Service: {filtered_df.groupby('service_type')['usage_units'].mean().idxmax() if not filtered_df.empty else 'N/A'}")
with snap3:
    st.info(f"Peak Demand: {filtered_df['usage_units'].max():,.0f}")
with snap4:
    st.info(f"Min Headroom: {filtered_df['headroom_units'].min():,.0f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 KPI Overview",
    "📈 Demand Trends",
    "🌐 Regional Analysis",
    "🧠 Model & Forecast",
    "⚠️ Risk Alerts"
])

# =========================================================
# TAB 1 - KPI OVERVIEW
# =========================================================
with tab1:
    st.markdown('<div class="section-title">EXECUTIVE KPIS</div>', unsafe_allow_html=True)

    total_cost = filtered_df["cost_usd"].sum()
    wasted_cost = filtered_df["wasted_capacity_cost"].sum()
    avg_util = filtered_df["utilization_pct"].mean()
    total_incidents = int(filtered_df["risk_event"].sum())
    capacity_risk_events = int(filtered_df["risk_event"].sum())
    underutilized_flags = int(filtered_df["underutilized_flag"].sum())
    avg_headroom = filtered_df["headroom_units"].mean()
    avg_growth = filtered_df["daily_growth_rate"].mean()

    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)

    with r1c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Cost (USD)</div><div class="kpi-value">${total_cost/1e6:.2f}M</div><div class="kpi-sub">Filtered period</div></div>', unsafe_allow_html=True)
    with r1c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Wasted Capacity Cost</div><div class="kpi-value">${wasted_cost/1e6:.2f}M</div><div class="kpi-sub">{(wasted_cost/max(total_cost,1))*100:.1f}% of total spend</div></div>', unsafe_allow_html=True)
    with r1c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Utilization</div><div class="kpi-value">{avg_util:.1f}%</div><div class="kpi-sub">Across selected services</div></div>', unsafe_allow_html=True)
    with r1c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Total Incidents</div><div class="kpi-value">{total_incidents}</div><div class="kpi-sub">Risk threshold based</div></div>', unsafe_allow_html=True)

    with r2c1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Capacity Risk Events</div><div class="kpi-value">{capacity_risk_events}</div><div class="kpi-sub">{(capacity_risk_events/max(len(filtered_df),1))*100:.1f}% of records</div></div>', unsafe_allow_html=True)
    with r2c2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Underutilized Flags</div><div class="kpi-value">{underutilized_flags}</div><div class="kpi-sub">{(underutilized_flags/max(len(filtered_df),1))*100:.1f}% of records</div></div>', unsafe_allow_html=True)
    with r2c3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Headroom (Units)</div><div class="kpi-value">{avg_headroom:,.0f}</div><div class="kpi-sub">Available buffer</div></div>', unsafe_allow_html=True)
    with r2c4:
        st.markdown(f'<div class="kpi-card"><div class="kpi-label">Avg Daily Growth Rate</div><div class="kpi-value">{avg_growth:.3f}%</div><div class="kpi-sub">Across filtered period</div></div>', unsafe_allow_html=True)

    s1, s2 = st.columns([1.4, 1])

    with s1:
        util_status = "Healthy" if avg_util < 60 else "Moderate" if avg_util < 75 else "High"
        high_risk_regions = filtered_df.loc[filtered_df["risk_event"]].groupby("region").size().sort_values(ascending=False)
        top_risk_region = high_risk_regions.index[0] if not high_risk_regions.empty else "None"
        immediate_rec = "Reduce waste and monitor" if wasted_cost > total_cost * 0.3 else "Capacity profile is stable"

        st.markdown(
            '<div class="summary-box"><div class="panel-title">Executive Summary</div><ul>'
            f'<li>Current utilization status: <b>{util_status}</b></li>'
            f'<li>Cost efficiency: wasted cost is <b>{(wasted_cost/max(total_cost,1))*100:.1f}% of total spend</b></li>'
            f'<li>Highest-risk region: <b>{top_risk_region}</b></li>'
            f'<li>Immediate recommendation: <b>{immediate_rec}</b></li>'
            '</ul></div>',
            unsafe_allow_html=True
        )

    with s2:
        st.markdown(
            '<div class="panel"><div class="panel-title">Status Chips</div>'
            '<div class="chip-row">'
            '<span class="chip chip-green">Low Risk</span>'
            '<span class="chip chip-blue">Moderate Risk</span>'
            '<span class="chip chip-orange">High Risk</span>'
            '<span class="chip chip-red">Underutilized</span>'
            '</div>'
            '<div class="small-note">Color-coded labels improve dashboard readability during demo.</div></div>',
            unsafe_allow_html=True
        )

    st.markdown('<div class="section-title">COST COMPOSITION</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        pie_df = pd.DataFrame({
            "Category": ["Wasted Capacity", "Utilized Spend"],
            "Value": [wasted_cost, max(total_cost - wasted_cost, 0)]
        })
        fig_pie = px.pie(
            pie_df,
            names="Category",
            values="Value",
            title="Operational Cost Efficiency",
            color_discrete_sequence=px.colors.qualitative.Set2,
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        monthly_cost = filtered_df.groupby("month_name", as_index=False)[["cost_usd", "wasted_capacity_cost"]].sum()
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(x=monthly_cost["month_name"], y=monthly_cost["cost_usd"], name="Total Cost"))
        fig_cost.add_trace(go.Bar(x=monthly_cost["month_name"], y=monthly_cost["wasted_capacity_cost"], name="Wasted Capacity"))
        fig_cost.update_layout(
            title="Monthly Capacity Waste Trend",
            barmode="group",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10),
            xaxis_title="Month",
            yaxis_title="USD"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button("Download Filtered Data CSV", filtered_df.to_csv(index=False).encode("utf-8"), "filtered_dashboard_data.csv", "text/csv")
    with d2:
        risk_export = filtered_df[filtered_df["risk_event"]]
        st.download_button("Download Risk Records CSV", risk_export.to_csv(index=False).encode("utf-8"), "risk_records.csv", "text/csv")
    with d3:
        summary_report = pd.DataFrame({
            "Metric": ["Total Cost", "Wasted Cost", "Avg Utilization", "Risk Events", "Underutilized Flags", "Avg Headroom", "Avg Daily Growth"],
            "Value": [total_cost, wasted_cost, avg_util, capacity_risk_events, underutilized_flags, avg_headroom, avg_growth]
        })
        st.download_button("Download Summary Report CSV", summary_report.to_csv(index=False).encode("utf-8"), "executive_summary_report.csv", "text/csv")

# =========================================================
# TAB 2 - DEMAND TRENDS
# =========================================================
with tab2:
    st.markdown('<div class="section-title">USAGE & DEMAND OVER TIME</div>', unsafe_allow_html=True)

    metric_map = {
        "Usage Units": "usage_units",
        "Utilization Pct": "utilization_pct",
        "Cost Usd": "cost_usd",
        "Headroom Units": "headroom_units",
        "Wasted Capacity Cost": "wasted_capacity_cost"
    }
    primary_metric = st.selectbox("Primary Metric", list(metric_map.keys()), index=0)
    group_by = st.radio("Group by", ["service_type", "region"], horizontal=True)

    monthly_trend = filtered_df.groupby(["month_name", group_by], as_index=False)[metric_map[primary_metric]].mean()

    fig_line = px.line(
        monthly_trend,
        x="month_name",
        y=metric_map[primary_metric],
        color=group_by,
        markers=True,
        title=f"{primary_metric} Trend by {group_by.replace('_', ' ').title()}",
        template="plotly_dark"
    )
    fig_line.update_layout(
        paper_bgcolor="#0a142d",
        plot_bgcolor="#0a142d",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_line, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        growth_monthly = filtered_df.groupby("month_name", as_index=False)["daily_growth_rate"].mean()
        fig_growth = px.area(
            growth_monthly,
            x="month_name",
            y="daily_growth_rate",
            title="Avg Daily Growth Rate (%)",
            template="plotly_dark"
        )
        fig_growth.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    with c2:
        week_idx = filtered_df.groupby("weekday", as_index=False)["usage_units"].mean()
        weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        week_idx["weekday"] = pd.Categorical(week_idx["weekday"], categories=weekday_order, ordered=True)
        week_idx = week_idx.sort_values("weekday")
        base = week_idx["usage_units"].mean() if not week_idx.empty else 1
        week_idx["seasonality_index"] = week_idx["usage_units"] / base

        fig_week = px.bar(
            week_idx,
            x="weekday",
            y="seasonality_index",
            title="Weekly Seasonality Index",
            template="plotly_dark",
            color="weekday"
        )
        fig_week.add_hline(y=1, line_dash="dash", annotation_text="Baseline")
        fig_week.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            showlegend=False,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_week, use_container_width=True)

    st.markdown('<div class="section-title">ROLLING STATISTICS (30-DAY)</div>', unsafe_allow_html=True)
    rolling_df = filtered_df.copy()
    rolling_df["rolling_mean_30"] = rolling_df["usage_units"].rolling(30, min_periods=1).mean()
    rolling_df["rolling_std_30"] = rolling_df["usage_units"].rolling(30, min_periods=1).std().fillna(0)
    rolling_df["upper"] = rolling_df["rolling_mean_30"] + rolling_df["rolling_std_30"]
    rolling_df["lower"] = rolling_df["rolling_mean_30"] - rolling_df["rolling_std_30"]

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["rolling_mean_30"], mode="lines", name="30-Day Rolling Mean", line=dict(color="#4f8cff")))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["upper"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["lower"], mode="lines", fill="tonexty", name="Confidence Band", line=dict(width=0), fillcolor="rgba(79,140,255,0.18)"))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["usage_units"], mode="lines", name="Actual Usage", line=dict(color="#88f7d5", width=1)))
    fig_roll.update_layout(
        title="Actual Usage vs Rolling Mean",
        template="plotly_dark",
        paper_bgcolor="#0a142d",
        plot_bgcolor="#0a142d",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_roll, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        service_compare = get_service_util_headroom(filtered_df)
        fig_service_compare = go.Figure()
        fig_service_compare.add_trace(go.Bar(x=service_compare["service_type"], y=service_compare["utilization_pct"], name="Avg Utilization %"))
        fig_service_compare.add_trace(go.Bar(x=service_compare["service_type"], y=service_compare["headroom_units"], name="Avg Headroom Units"))
        fig_service_compare.update_layout(
            title="Utilization vs Headroom by Service",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_service_compare, use_container_width=True)

    with c4:
        waste_month = get_monthly_waste_cost(filtered_df)
        fig_waste_month = px.line(
            waste_month,
            x="month_name",
            y="wasted_capacity_cost",
            markers=True,
            title="Monthly Waste Cost Trend",
            template="plotly_dark"
        )
        fig_waste_month.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_waste_month, use_container_width=True)

    top_months = get_top_5_demand_months(filtered_df)
    fig_top_months = px.bar(
        top_months,
        x="month_name",
        y="usage_units",
        title="Top 5 High-Demand Months",
        template="plotly_dark",
        color="usage_units",
        color_continuous_scale="Blues"
    )
    fig_top_months.update_layout(
        paper_bgcolor="#0a142d",
        plot_bgcolor="#0a142d",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10)
    )
    st.plotly_chart(fig_top_months, use_container_width=True)

# =========================================================
# TAB 3 - REGIONAL ANALYSIS
# =========================================================
with tab3:
    st.markdown('<div class="section-title">REGIONAL CAPACITY BREAKDOWN</div>', unsafe_allow_html=True)

    region_agg = filtered_df.groupby("region", as_index=False).agg({
        "utilization_pct": "mean",
        "waste_pct": "mean",
        "cost_usd": "sum",
        "risk_event": "sum",
        "wasted_capacity_cost": "sum"
    })
    region_agg["risk_event"] = region_agg["risk_event"].astype(int)

    fig_bubble = px.scatter(
        region_agg,
        x="utilization_pct",
        y="waste_pct",
        size="cost_usd",
        color="risk_event",
        hover_name="region",
        title="Regional Capacity Risk Map",
        template="plotly_dark",
        color_continuous_scale="Turbo"
    )
    fig_bubble.add_vline(x=risk_threshold * 100, line_dash="dash", line_color="red", annotation_text="Risk threshold")
    fig_bubble.update_layout(
        paper_bgcolor="#0a142d",
        plot_bgcolor="#0a142d",
        font_color="white",
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Avg Utilization (%)",
        yaxis_title="Waste % of Total Cost"
    )
    st.plotly_chart(fig_bubble, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        top_waste = region_agg.sort_values("wasted_capacity_cost", ascending=False).head(10)
        fig_waste = px.bar(
            top_waste,
            x="wasted_capacity_cost",
            y="region",
            orientation="h",
            title="Top Waste Regions",
            template="plotly_dark"
        )
        fig_waste.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_waste, use_container_width=True)

    with c2:
        top_risk = region_agg.sort_values("risk_event", ascending=False).head(10)
        fig_risk = px.bar(
            top_risk,
            x="region",
            y="risk_event",
            title="Top Risk Regions",
            template="plotly_dark",
            color="risk_event",
            color_continuous_scale="Oranges"
        )
        fig_risk.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    heat_pivot = get_region_risk_heatmap_data(filtered_df)
    if heat_pivot is not None:
        fig_heat = px.imshow(
            heat_pivot,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Region Risk Heatmap"
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            margin=dict(l=10, r=10, t=50, b=10)
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("No heatmap data available.")

# =========================================================
# TAB 4 - MODEL & FORECAST
# =========================================================
with tab4:
    st.markdown('<div class="section-title">FORECAST SCENARIO ENGINE</div>', unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        input_region = st.selectbox("Forecast Region", all_regions, index=0 if all_regions else None)
    with fc2:
        input_service = st.selectbox("Forecast Service Type", all_services, index=0 if all_services else None)
    with fc3:
        input_year = st.number_input("Forecast Year", value=int(max(all_years)) if all_years else 2026)
    with fc4:
        input_market = st.number_input("Market Demand Index", value=120.0, help="Higher values simulate stronger business demand conditions.")

    gc1, gc2, gc3, gc4 = st.columns(4)
    with gc1:
        input_capacity = st.number_input("Provisioned Capacity", value=1000.0)
    with gc2:
        input_cost = st.number_input("Cost USD", value=5000.0)
    with gc3:
        input_availability = st.number_input("Availability %", value=99.5)
    with gc4:
        input_util = st.number_input("Capacity Utilization", value=0.75, help="Used to simulate current operational pressure in the forecast scenario.")

    recent = filtered_df.tail(30)
    payload = {
        "provisioned_capacity": input_capacity,
        "cost_usd": input_cost,
        "availability_pct": input_availability,
        "economic_indicator_index": recent["cost_usd"].mean() / 50 if "cost_usd" in recent.columns else 100,
        "market_demand_index": input_market,
        "product_launch_impact": 0,
        "year": input_year,
        "month": 3,
        "quarter": 1,
        "week_of_year": 12,
        "is_month_start": 0,
        "is_month_end": 0,
        "lag_1": recent["usage_units"].tail(1).mean() if not recent.empty else 700,
        "lag_2": recent["usage_units"].tail(2).mean() if not recent.empty else 680,
        "lag_4": recent["usage_units"].tail(4).mean() if not recent.empty else 650,
        "lag_8": recent["usage_units"].tail(8).mean() if not recent.empty else 620,
        "rolling_mean_3": recent["usage_units"].rolling(3).mean().iloc[-1] if len(recent) >= 3 else 690,
        "rolling_mean_6": recent["usage_units"].rolling(6).mean().iloc[-1] if len(recent) >= 6 else 670,
        "rolling_std_3": recent["usage_units"].rolling(3).std().iloc[-1] if len(recent) >= 3 else 20,
        "rolling_std_6": recent["usage_units"].rolling(6).std().iloc[-1] if len(recent) >= 6 else 30,
        "capacity_utilization": input_util,
        "growth_rate_1": recent["daily_growth_rate"].tail(1).mean() if "daily_growth_rate" in recent.columns else 0.05,
        "growth_rate_4": recent["daily_growth_rate"].tail(4).mean() if "daily_growth_rate" in recent.columns else 0.08,
        "spike_flag": 0,
        "region_East Us": 1 if input_region == "East Us" else 0,
        "region_Southeast Asia": 1 if input_region == "Southeast Asia" else 0,
        "region_West Europe": 1 if input_region == "West Europe" else 0,
        "region_Central India": 1 if input_region == "Central India" else 0,
        "service_Storage": 1 if input_service == "Storage" else 0,
        "service_type_Compute": 1 if input_service == "Compute" else 0,
        "service_type_Storage": 1 if input_service == "Storage" else 0
    }

    run_forecast = st.button("Run Forecast")

    if run_forecast:
        input_df = pd.DataFrame([payload])
        aligned_df = align_features(input_df)
        prediction = float(model.predict(aligned_df)[0])
        action, alert_color, note = capacity_action(prediction)

        save_log(pd.DataFrame({
            "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "region": [input_region],
            "service": [input_service],
            "predicted_usage_units": [prediction],
            "capacity_action": [action],
            "recommendation_note": [note],
            "provisioned_capacity": [input_capacity],
            "availability_pct": [input_availability],
            "market_demand_index": [input_market]
        }))

        st.markdown(f'<div class="alert-box" style="background:{alert_color};">{action} — {note}</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                title={"text": "Predicted Usage Units"},
                gauge={
                    "axis": {"range": [0, max(input_capacity * 1.5, prediction * 1.2)]},
                    "bar": {"color": "#4f8cff"},
                    "steps": [
                        {"range": [0, 500], "color": "#163d2c"},
                        {"range": [500, 700], "color": "#17406a"},
                        {"range": [700, 900], "color": "#5a3a15"},
                        {"range": [900, max(input_capacity * 1.5, prediction * 1.2)], "color": "#5f1720"}
                    ]
                }
            ))
            fig_gauge.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                font_color="white",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with c2:
            future_steps = list(range(1, 8))
            future_pred = [prediction * (1 + 0.02 * i) for i in range(7)]
            fig_future = px.line(
                x=future_steps,
                y=future_pred,
                markers=True,
                title="7-Step Forecast Outlook",
                template="plotly_dark"
            )
            fig_future.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                xaxis_title="Forecast Step",
                yaxis_title="Predicted Usage Units",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_future, use_container_width=True)

        current_demand = recent["usage_units"].tail(1).mean() if not recent.empty else 0
        capacity_buffer = input_capacity - prediction
        risk_level = "High" if prediction >= 900 else "Moderate" if prediction >= 700 else "Low"

        st.markdown("#### What-if Scenario Comparison")
        compare_df = pd.DataFrame({
            "Metric": ["Current Demand", "Forecast Demand", "Capacity Buffer", "Risk Level"],
            "Value": [round(current_demand, 2), round(prediction, 2), round(capacity_buffer, 2), risk_level]
        })
        st.dataframe(compare_df, use_container_width=True, hide_index=True)

        st.markdown("#### Scenario Interpretation")
        if capacity_buffer < 0:
            st.error("Forecasted demand is higher than available capacity. Additional provisioning is required.")
        elif capacity_buffer < 100:
            st.warning("Capacity buffer is low. The system is close to operating risk.")
        else:
            st.success("Capacity buffer is healthy under the selected scenario.")

# =========================================================
# TAB 5 - RISK ALERTS
# =========================================================
with tab5:
    st.markdown('<div class="section-title">THRESHOLD-BASED RISK MONITORING</div>', unsafe_allow_html=True)

    high_risk_df = filtered_df[filtered_df["risk_event"]].copy()
    under_df = filtered_df[filtered_df["underutilized_flag"]].copy()

    a1, a2, a3 = st.columns(3)
    with a1:
        st.metric("High Risk Records", len(high_risk_df))
    with a2:
        st.metric("Underutilized Records", len(under_df))
    with a3:
        st.metric("Current Threshold", f"{risk_threshold:.2f}")

    c1, c2 = st.columns(2)

    with c1:
        risk_by_region = high_risk_df.groupby("region", as_index=False).size().rename(columns={"size": "risk_count"})
        if not risk_by_region.empty:
            fig_alert = px.bar(
                risk_by_region.sort_values("risk_count", ascending=False),
                x="region",
                y="risk_count",
                title="High Risk Events by Region",
                template="plotly_dark",
                color="risk_count",
                color_continuous_scale="Reds"
            )
            fig_alert.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_alert, use_container_width=True)
        else:
            st.info("No high-risk alerts for selected filters.")

    with c2:
        if not under_df.empty:
            under_service = under_df.groupby("service_type", as_index=False).size().rename(columns={"size": "count"})
            fig_under = px.bar(
                under_service,
                x="service_type",
                y="count",
                title="Underutilized Flags by Service",
                template="plotly_dark",
                color="count",
                color_continuous_scale="Blues"
            )
            fig_under.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                margin=dict(l=10, r=10, t=50, b=10)
            )
            st.plotly_chart(fig_under, use_container_width=True)
        else:
            st.info("No underutilization alerts for selected filters.")

    st.markdown("#### Recent High-Risk Records")
    show_cols = [c for c in ["time_stamp", "region", "service_type", "usage_units", "provisioned_capacity", "utilization_pct", "wasted_capacity_cost"] if c in high_risk_df.columns]
    if not high_risk_df.empty and show_cols:
        st.dataframe(high_risk_df[show_cols].tail(20), use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk records to display.")

# =========================================================
# EXPORT PANEL
# =========================================================
st.markdown("### Reporting & Export")
e1, e2 = st.columns(2)

with e1:
    st.download_button(
        "Download Full Dashboard Data",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="azure_capacity_dashboard_full_data.csv",
        mime="text/csv"
    )

with e2:
    log_df = load_logs()
    st.download_button(
        "Download Forecast Audit Log",
        data=log_df.to_csv(index=False).encode("utf-8"),
        file_name="forecast_audit_log.csv",
        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    '<div class="small-note">Milestone 4 • Forecast Integration & Capacity Planning • Streamlit Executive Dashboard</div>',
    unsafe_allow_html=True
)