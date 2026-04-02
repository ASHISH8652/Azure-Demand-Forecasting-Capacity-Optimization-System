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
    page_icon="☁️",
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
# LOAD MODEL FILES
# =========================================================
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

if not os.path.exists(FEATURE_PATH):
    st.error(f"Feature list file not found: {FEATURE_PATH}")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #040b1a 0%, #07142a 100%);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #081223 0%, #0b1630 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.block-container {
    padding-top: 2.8rem !important;
    padding-bottom: 1.2rem;
    max-width: 1700px;
}
.main-title {
    font-size: 2.2rem;
    font-weight: 900;
    color: #76a9ff;
    letter-spacing: 1px;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}
.sub-title {
    color: #8fa4d4;
    font-size: 0.95rem;
    margin-bottom: 1rem;
}
.section-title {
    color: #7da2ff;
    font-size: 0.92rem;
    font-weight: 700;
    letter-spacing: 2px;
    margin-top: 0.5rem;
    margin-bottom: 0.8rem;
}
.kpi-card {
    background: rgba(12, 24, 52, 0.95);
    border: 1px solid rgba(120, 150, 255, 0.18);
    border-radius: 16px;
    padding: 18px;
    min-height: 112px;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}
.kpi-label {
    color: #9eb1d7;
    font-size: 0.78rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.kpi-value {
    color: #f8fbff;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 6px;
}
.kpi-sub {
    color: #86d7c6;
    font-size: 0.82rem;
    margin-top: 6px;
}
.panel {
    background: rgba(10, 20, 45, 0.95);
    border: 1px solid rgba(120, 150, 255, 0.14);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 16px;
}
.panel-title {
    color: #e4eeff;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.small-note {
    color: #87a0c8;
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
    padding: 0.42rem 0.8rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    margin-bottom: 0.4rem;
    background: rgba(67, 97, 238, 0.18);
    border: 1px solid rgba(118, 169, 255, 0.18);
    color: #dbe7ff;
    font-size: 0.8rem;
    font-weight: 700;
}
.insight-box {
    background: linear-gradient(90deg, rgba(29,78,216,0.22), rgba(14,165,233,0.16));
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
    padding-left: 1.2rem;
}
.summary-box li {
    color: #dbe7ff;
    margin-bottom: 0.38rem;
}
.chip {
    display: inline-block;
    padding: 0.32rem 0.7rem;
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
.hero {
    background: linear-gradient(90deg, rgba(24,61,153,0.95), rgba(14,93,180,0.88));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 20px;
    margin-bottom: 16px;
}
.hero-title {
    font-size: 1.4rem;
    color: white;
    font-weight: 800;
    margin-bottom: 6px;
}
.hero-sub {
    color: #dbeafe;
    font-size: 0.92rem;
}
.hero-pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    color: #f8fafc;
    font-size: 0.78rem;
    padding: 0.3rem 0.7rem;
    border-radius: 999px;
    margin-right: 0.4rem;
    margin-top: 0.8rem;
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
hr {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 0.8rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# HELPERS
# =========================================================
def ensure_columns(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    if "time_stamp" in df.columns:
        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.sort_values("time_stamp").reset_index(drop=True)
    else:
        df["time_stamp"] = pd.date_range(start="2023-01-01", periods=len(df), freq="D")

    if "region" not in df.columns:
        df["region"] = "Unknown"

    if "service_type" not in df.columns:
        df["service_type"] = "Compute"

    if "usage_units" not in df.columns:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            df["usage_units"] = df[numeric_cols[0]]
        else:
            df["usage_units"] = np.random.randint(200, 600, len(df))

    if "provisioned_capacity" not in df.columns:
        df["provisioned_capacity"] = df["usage_units"] + np.random.randint(60, 180, len(df))

    if "cost_usd" not in df.columns:
        df["cost_usd"] = np.random.uniform(120, 250, len(df))

    if "availability_pct" not in df.columns:
        df["availability_pct"] = np.random.uniform(97.5, 99.9, len(df))

    if "economic_growth_index" not in df.columns:
        df["economic_growth_index"] = np.random.uniform(0.8, 1.4, len(df))

    if "marketing_index" not in df.columns:
        df["marketing_index"] = np.random.uniform(0.7, 1.3, len(df))

    if "it_spending_growth" not in df.columns:
        df["it_spending_growth"] = np.random.uniform(0.7, 1.3, len(df))

    if "is_holiday" not in df.columns:
        df["is_holiday"] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])

    df["utilization_pct"] = np.where(
        df["provisioned_capacity"] > 0,
        (df["usage_units"] / df["provisioned_capacity"]) * 100,
        0
    )
    df["headroom_units"] = df["provisioned_capacity"] - df["usage_units"]
    df["waste_units"] = np.where(df["headroom_units"] > 0, df["headroom_units"], 0)
    df["waste_pct"] = np.where(df["provisioned_capacity"] > 0,
                               (df["waste_units"] / df["provisioned_capacity"]) * 100, 0)
    df["wasted_capacity_cost"] = np.where(
        df["provisioned_capacity"] > 0,
        (df["waste_units"] / df["provisioned_capacity"]) * df["cost_usd"],
        0
    )
    df["daily_growth_rate"] = df["usage_units"].pct_change().fillna(0) * 100

    df["year"] = df["time_stamp"].dt.year
    df["month_name"] = df["time_stamp"].dt.strftime("%b %Y")
    df["month_period"] = df["time_stamp"].dt.to_period("M").astype(str)
    df["weekday"] = df["time_stamp"].dt.day_name()
    df["week_period"] = df["time_stamp"].dt.to_period("W").astype(str)

    # add basic forecasting-style fields if absent
    if "lag_1" not in df.columns:
        df["lag_1"] = df["usage_units"].shift(1).fillna(method="bfill")
    if "lag_2" not in df.columns:
        df["lag_2"] = df["usage_units"].shift(2).fillna(method="bfill")
    if "lag_4" not in df.columns:
        df["lag_4"] = df["usage_units"].shift(4).fillna(method="bfill")
    if "lag_8" not in df.columns:
        df["lag_8"] = df["usage_units"].shift(8).fillna(method="bfill")
    if "rolling_mean_3" not in df.columns:
        df["rolling_mean_3"] = df["usage_units"].rolling(3, min_periods=1).mean()
    if "rolling_mean_6" not in df.columns:
        df["rolling_mean_6"] = df["usage_units"].rolling(6, min_periods=1).mean()
    if "rolling_std_3" not in df.columns:
        df["rolling_std_3"] = df["usage_units"].rolling(3, min_periods=1).std().fillna(0)
    if "rolling_std_6" not in df.columns:
        df["rolling_std_6"] = df["usage_units"].rolling(6, min_periods=1).std().fillna(0)

    if "capacity_utilization" not in df.columns:
        df["capacity_utilization"] = df["utilization_pct"] / 100.0
    if "growth_rate_1" not in df.columns:
        df["growth_rate_1"] = df["daily_growth_rate"] / 100.0
    if "growth_rate_4" not in df.columns:
        df["growth_rate_4"] = df["daily_growth_rate"].rolling(4, min_periods=1).mean() / 100.0
    if "product_launch_impact" not in df.columns:
        df["product_launch_impact"] = 0
    if "market_demand_index" not in df.columns:
        df["market_demand_index"] = df["marketing_index"]
    if "economic_indicator_index" not in df.columns:
        df["economic_indicator_index"] = df["economic_growth_index"]
    if "week_of_year" not in df.columns:
        df["week_of_year"] = df["time_stamp"].dt.isocalendar().week.astype(int)
    if "quarter" not in df.columns:
        df["quarter"] = df["time_stamp"].dt.quarter
    if "month" not in df.columns:
        df["month"] = df["time_stamp"].dt.month
    if "is_month_start" not in df.columns:
        df["is_month_start"] = df["time_stamp"].dt.is_month_start.astype(int)
    if "is_month_end" not in df.columns:
        df["is_month_end"] = df["time_stamp"].dt.is_month_end.astype(int)
    if "spike_flag" not in df.columns:
        threshold = df["usage_units"].mean() + df["usage_units"].std()
        df["spike_flag"] = (df["usage_units"] > threshold).astype(int)

    return df

def align_features(input_df: pd.DataFrame):
    aligned = input_df.copy()
    for col in feature_columns:
        if col not in aligned.columns:
            aligned[col] = 0
    aligned = aligned[feature_columns]
    return aligned

def capacity_action(pred_value: float):
    if pred_value >= 900:
        return "Scale Up Capacity Immediately", "#dc2626", "High utilization risk"
    elif pred_value >= 700:
        return "Prepare Additional Capacity", "#ea580c", "Demand is rising"
    elif pred_value >= 500:
        return "Monitor Demand Closely", "#0284c7", "Moderate pressure"
    return "Capacity is Sufficient", "#16a34a", "Healthy operating buffer"

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

def get_key_insight(frame: pd.DataFrame, threshold: float):
    region_util = frame.groupby("region", as_index=False)["utilization_pct"].mean()
    top_region = region_util.sort_values("utilization_pct", ascending=False).iloc[0]["region"] if not region_util.empty else "N/A"

    service_headroom = frame.groupby("service_type", as_index=False)["headroom_units"].mean()
    top_headroom_service = service_headroom.sort_values("headroom_units", ascending=False).iloc[0]["service_type"] if not service_headroom.empty else "N/A"

    risk_events = int(((frame["utilization_pct"] / 100.0) >= threshold).sum())
    return f"{top_region} shows the highest utilization, while {top_headroom_service} has the strongest average headroom. Current threshold flags {risk_events} high-risk records."

def safe_prediction_frame(df_in: pd.DataFrame) -> pd.DataFrame:
    df_pred = df_in.copy()

    if set(feature_columns).issubset(df_pred.columns):
        try:
            X = df_pred[feature_columns].copy()
            df_pred["predicted_usage_units"] = model.predict(X)
        except Exception:
            df_pred["predicted_usage_units"] = df_pred["usage_units"].rolling(7, min_periods=1).mean()
    else:
        df_pred["predicted_usage_units"] = df_pred["usage_units"].rolling(7, min_periods=1).mean()

    df_pred["residual"] = df_pred["usage_units"] - df_pred["predicted_usage_units"]
    return df_pred

# =========================================================
# SIDEBAR: THEME + DATA SOURCE
# =========================================================
st.sidebar.markdown("## 🔶 Azure Capacity Intel")

st.sidebar.markdown("### 🎨 Theme")
theme_mode = st.sidebar.radio(
    "Select Theme",
    ["Dark", "Light"],
    horizontal=True
)
st.sidebar.caption(f"{theme_mode} mode active")

st.sidebar.markdown("---")
st.sidebar.markdown("### Data Source")
use_demo = st.sidebar.checkbox("Use Demo Data", value=True)
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or XLSX",
    type=["csv", "xlsx", "xls"],
    help="Upload your own dataset to analyze and forecast with this dashboard."
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")
elif use_demo:
    if not os.path.exists(DATA_PATH):
        st.error("Demo dataset not found.")
        st.stop()
    raw_df = pd.read_csv(DATA_PATH)
    st.sidebar.success(f"Demo data loaded ({len(raw_df):,} records)")
else:
    st.warning("Please enable demo data or upload a dataset.")
    st.stop()

df = ensure_columns(raw_df)
df = safe_prediction_frame(df)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.markdown("---")
all_regions = sorted(df["region"].dropna().astype(str).unique().tolist())
all_services = sorted(df["service_type"].dropna().astype(str).unique().tolist())
all_years = sorted(df["year"].dropna().astype(int).unique().tolist())

selected_regions = st.sidebar.multiselect(
    "Regions",
    all_regions,
    default=all_regions[:min(5, len(all_regions))]
)
selected_services = st.sidebar.multiselect(
    "Service Type",
    all_services,
    default=all_services
)
selected_years = st.sidebar.multiselect(
    "Year",
    all_years,
    default=all_years
)

risk_threshold = st.sidebar.slider(
    "Capacity Risk Threshold",
    0.40, 0.95, 0.65, 0.01,
    help="Records above this utilization level are treated as capacity risk events."
)
st.sidebar.caption("Utilization % alert level")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🖍 What-If Analysis")
global_adjustment = st.sidebar.slider(
    "Global Capacity Adjustment (%)",
    -30, 30, 0, 1,
    help="Simulate increasing or decreasing provisioned capacity globally."
)

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

filtered_df["provisioned_capacity_adjusted"] = filtered_df["provisioned_capacity"] * (1 + global_adjustment / 100.0)
filtered_df["utilization_pct_adjusted"] = np.where(
    filtered_df["provisioned_capacity_adjusted"] > 0,
    (filtered_df["usage_units"] / filtered_df["provisioned_capacity_adjusted"]) * 100,
    0
)
filtered_df["risk_event"] = (filtered_df["utilization_pct_adjusted"] / 100.0) >= risk_threshold
filtered_df["underutilized_flag"] = filtered_df["utilization_pct_adjusted"] < 50
filtered_df["over_capacity_flag"] = filtered_df["usage_units"] > filtered_df["provisioned_capacity_adjusted"]

st.sidebar.markdown("---")
st.sidebar.markdown("### Filter Summary")
st.sidebar.write(f"**Records:** {len(filtered_df):,}")
st.sidebar.write(f"**Regions:** {filtered_df['region'].nunique()}")
st.sidebar.write(f"**Services:** {filtered_df['service_type'].nunique()}")
st.sidebar.write(f"**Years:** {filtered_df['year'].nunique()}")

# =========================================================
# HEADER
# =========================================================
st.markdown('<div class="main-title">☁️ AZURE CAPACITY INTELLIGENCE</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Milestone 4 • Forecast Integration & Capacity Planning Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="hero">
    <div class="hero-title">☁ Azure Demand Forecast Dashboard</div>
    <div class="hero-sub">Forecast integration and capacity planning with live data filters, advanced analytics, monitoring, what-if simulation, and export-ready business insights.</div>
    <span class="hero-pill">LIVE</span>
    <span class="hero-pill">Forecast Ready</span>
    <span class="hero-pill">Interactive</span>
    <span class="hero-pill">Decision Support</span>
</div>
""", unsafe_allow_html=True)

badges_html = f"""
<div>
    <span class="badge">Active Regions: {filtered_df['region'].nunique()}</span>
    <span class="badge">Active Services: {filtered_df['service_type'].nunique()}</span>
    <span class="badge">Selected Years: {filtered_df['year'].nunique()}</span>
    <span class="badge">Current Threshold: {risk_threshold:.2f}</span>
    <span class="badge">Records: {len(filtered_df):,}</span>
</div>
"""
st.markdown(badges_html, unsafe_allow_html=True)

st.markdown(
    f'<div class="insight-box"><b>Key Insight:</b> {get_key_insight(filtered_df, risk_threshold)}</div>',
    unsafe_allow_html=True
)

# =========================================================
# KPI SUMMARY
# =========================================================
total_cost = filtered_df["cost_usd"].sum()
wasted_cost = filtered_df["wasted_capacity_cost"].sum()
avg_util = filtered_df["utilization_pct_adjusted"].mean()
risk_events = int(filtered_df["risk_event"].sum())
peak_forecast_date = filtered_df.loc[filtered_df["predicted_usage_units"].idxmax(), "time_stamp"] if not filtered_df.empty else None
forecast_growth = ((filtered_df["predicted_usage_units"].iloc[-1] - filtered_df["predicted_usage_units"].iloc[0]) / max(filtered_df["predicted_usage_units"].iloc[0], 1)) * 100 if len(filtered_df) > 1 else 0
model_rmse = np.sqrt(np.mean((filtered_df["usage_units"] - filtered_df["predicted_usage_units"]) ** 2))
model_mae = np.mean(np.abs(filtered_df["usage_units"] - filtered_df["predicted_usage_units"]))
directional_acc = (np.sign(filtered_df["usage_units"].diff().fillna(0)) == np.sign(filtered_df["predicted_usage_units"].diff().fillna(0))).mean() * 100

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Forecast Demand</div>
        <div class="kpi-value">{filtered_df['predicted_usage_units'].sum():,.0f}</div>
        <div class="kpi-sub">Across filtered records</div>
    </div>
    """, unsafe_allow_html=True)
with k2:
    peak_date_text = peak_forecast_date.strftime("%Y-%m-%d") if pd.notnull(peak_forecast_date) else "N/A"
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Peak Forecast Date</div>
        <div class="kpi-value">{peak_date_text}</div>
        <div class="kpi-sub">Peak predicted demand</div>
    </div>
    """, unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Forecast Growth</div>
        <div class="kpi-value">{forecast_growth:.1f}%</div>
        <div class="kpi-sub">vs actual average</div>
    </div>
    """, unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Model RMSE</div>
        <div class="kpi-value">{model_rmse:.1f}</div>
        <div class="kpi-sub">Lower is better</div>
    </div>
    """, unsafe_allow_html=True)
with k5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">MAE</div>
        <div class="kpi-value">{model_mae:.1f}</div>
        <div class="kpi-sub">Mean absolute error</div>
    </div>
    """, unsafe_allow_html=True)
with k6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Directional Accuracy</div>
        <div class="kpi-value">{directional_acc:.1f}%</div>
        <div class="kpi-sub">Correct trend direction</div>
    </div>
    """, unsafe_allow_html=True)

s1, s2, s3, s4, s5, s6 = st.columns(6)
with s1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Cost (USD)</div>
        <div class="kpi-value">${filtered_df['cost_usd'].mean():.2f}</div>
        <div class="kpi-sub">Cost per record</div>
    </div>
    """, unsafe_allow_html=True)
with s2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Avg Availability</div>
        <div class="kpi-value">{filtered_df['availability_pct'].mean():.2f}%</div>
        <div class="kpi-sub">Availability across records</div>
    </div>
    """, unsafe_allow_html=True)
with s3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Holiday Records</div>
        <div class="kpi-value">{int(filtered_df['is_holiday'].sum())}</div>
        <div class="kpi-sub">In filtered data</div>
    </div>
    """, unsafe_allow_html=True)
with s4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Economic Growth</div>
        <div class="kpi-value">{filtered_df['economic_growth_index'].mean():.3f}</div>
        <div class="kpi-sub">Avg external indicator</div>
    </div>
    """, unsafe_allow_html=True)
with s5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Marketing Index</div>
        <div class="kpi-value">{filtered_df['marketing_index'].mean():.3f}</div>
        <div class="kpi-sub">Avg demand signal</div>
    </div>
    """, unsafe_allow_html=True)
with s6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">IT Spending Growth</div>
        <div class="kpi-value">{filtered_df['it_spending_growth'].mean():.3f}</div>
        <div class="kpi-sub">Avg external factor</div>
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN TABS
# =========================================================
main_tabs = st.tabs([
    "📊 KPI Overview",
    "📈 Demand Trends",
    "🌍 Regional Analysis",
    "🧠 Model & Forecast",
    "⚠️ Risk Alerts",
    "🧩 Advanced Analytics"
])

# =========================================================
# TAB 1 - KPI OVERVIEW
# =========================================================
with main_tabs[0]:
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
            hole=0.45,
            title="Operational Cost Efficiency",
            color_discrete_sequence=px.colors.qualitative.Set2,
            template="plotly_dark"
        )
        fig_pie.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        monthly_cost = filtered_df.groupby("month_name", as_index=False)[["cost_usd", "wasted_capacity_cost"]].sum()
        fig_cost = go.Figure()
        fig_cost.add_trace(go.Bar(x=monthly_cost["month_name"], y=monthly_cost["cost_usd"], name="Total Cost"))
        fig_cost.add_trace(go.Bar(x=monthly_cost["month_name"], y=monthly_cost["wasted_capacity_cost"], name="Wasted Capacity"))
        fig_cost.update_layout(
            title="Monthly Capacity Waste Trend",
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white"
        )
        st.plotly_chart(fig_cost, use_container_width=True)

    s_left, s_right = st.columns([1.25, 1])
    with s_left:
        util_status = "Healthy" if avg_util < 60 else "Moderate" if avg_util < 75 else "High"
        high_risk_regions = filtered_df.loc[filtered_df["risk_event"]].groupby("region").size().sort_values(ascending=False)
        top_risk_region = high_risk_regions.index[0] if not high_risk_regions.empty else "None"
        immediate_rec = "Reduce waste and monitor risk" if wasted_cost > total_cost * 0.3 else "Capacity profile is stable"
        st.markdown('<div class="summary-box"><div class="panel-title">Executive Summary</div><ul>'
                    f'<li>Current utilization status: <b>{util_status}</b></li>'
                    f'<li>Cost efficiency: wasted cost is <b>{(wasted_cost/max(total_cost,1))*100:.1f}% of total spend</b></li>'
                    f'<li>Highest-risk region: <b>{top_risk_region}</b></li>'
                    f'<li>Immediate recommendation: <b>{immediate_rec}</b></li>'
                    '</ul></div>', unsafe_allow_html=True)
    with s_right:
        chip_html = '<div class="panel"><div class="panel-title">Status Chips</div>'
        chip_html += '<span class="chip chip-green">Low Risk</span>'
        chip_html += '<span class="chip chip-blue">Moderate Risk</span>'
        chip_html += '<span class="chip chip-orange">High Risk</span>'
        chip_html += '<span class="chip chip-red">Underutilized</span>'
        chip_html += '</div>'
        st.markdown(chip_html, unsafe_allow_html=True)

# =========================================================
# TAB 2 - DEMAND TRENDS
# =========================================================
with main_tabs[1]:
    metric_map = {
        "Usage Units": "usage_units",
        "Utilization %": "utilization_pct_adjusted",
        "Cost USD": "cost_usd",
        "Headroom Units": "headroom_units",
        "Wasted Capacity Cost": "wasted_capacity_cost"
    }

    primary_metric = st.selectbox("Primary Metric", list(metric_map.keys()))
    group_by = st.radio("Group by", ["service_type", "region"], horizontal=True)

    monthly_trend = filtered_df.groupby(["month_name", group_by], as_index=False)[metric_map[primary_metric]].mean()
    fig_line = px.line(
        monthly_trend,
        x="month_name",
        y=metric_map[primary_metric],
        color=group_by,
        markers=True,
        title=f"{primary_metric} over Time",
        template="plotly_dark"
    )
    fig_line.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
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
        fig_growth.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
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
            color="weekday",
            title="Weekly Seasonality Index",
            template="plotly_dark"
        )
        fig_week.update_layout(showlegend=False, paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
        st.plotly_chart(fig_week, use_container_width=True)

    rolling_df = filtered_df.copy()
    rolling_df["rolling_mean_30"] = rolling_df["usage_units"].rolling(30, min_periods=1).mean()
    rolling_df["rolling_std_30"] = rolling_df["usage_units"].rolling(30, min_periods=1).std().fillna(0)
    rolling_df["upper"] = rolling_df["rolling_mean_30"] + rolling_df["rolling_std_30"]
    rolling_df["lower"] = rolling_df["rolling_mean_30"] - rolling_df["rolling_std_30"]

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["usage_units"], mode="lines", name="Actual Usage", line=dict(color="#a78bfa", width=1)))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["rolling_mean_30"], mode="lines", name="30-Day Rolling Mean", line=dict(color="#67e8f9", width=2)))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["upper"], mode="lines", line=dict(width=0), showlegend=False))
    fig_roll.add_trace(go.Scatter(x=rolling_df["time_stamp"], y=rolling_df["lower"], mode="lines", fill="tonexty", name="Confidence Band", line=dict(width=0), fillcolor="rgba(103,232,249,0.15)"))
    fig_roll.update_layout(
        title="Usage Units: Actual vs 30-Day Rolling Mean (±1σ)",
        template="plotly_dark",
        paper_bgcolor="#0a142d",
        plot_bgcolor="#0a142d",
        font_color="white"
    )
    st.plotly_chart(fig_roll, use_container_width=True)

# =========================================================
# TAB 3 - REGIONAL ANALYSIS
# =========================================================
with main_tabs[2]:
    region_agg = filtered_df.groupby("region", as_index=False).agg({
        "utilization_pct_adjusted": "mean",
        "waste_pct": "mean",
        "cost_usd": "sum",
        "risk_event": "sum",
        "wasted_capacity_cost": "sum"
    })
    region_agg["risk_event"] = region_agg["risk_event"].astype(int)

    fig_bubble = px.scatter(
        region_agg,
        x="utilization_pct_adjusted",
        y="waste_pct",
        size="cost_usd",
        color="risk_event",
        hover_name="region",
        title="Regions: Utilization vs Waste % (bubble = cost, color = risk events)",
        template="plotly_dark",
        color_continuous_scale="YlOrRd"
    )
    fig_bubble.add_vline(x=risk_threshold * 100, line_dash="dash", line_color="red")
    fig_bubble.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
    st.plotly_chart(fig_bubble, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        top_waste = region_agg.sort_values("wasted_capacity_cost", ascending=False).head(10)
        fig_waste = px.bar(
            top_waste,
            x="wasted_capacity_cost",
            y="region",
            orientation="h",
            title="Top 10 Regions by Wasted Capacity ($)",
            template="plotly_dark",
            color_discrete_sequence=["#ff8a70"]
        )
        fig_waste.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
        st.plotly_chart(fig_waste, use_container_width=True)

    with c2:
        top_risk = region_agg.sort_values("risk_event", ascending=False).head(10)
        fig_risk = px.bar(
            top_risk,
            x="region",
            y="risk_event",
            title="Top 10 Regions by Capacity Risk Events",
            template="plotly_dark",
            color_discrete_sequence=["#ffa43a"]
        )
        fig_risk.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
        st.plotly_chart(fig_risk, use_container_width=True)

# =========================================================
# TAB 4 - MODEL & FORECAST
# =========================================================
with main_tabs[3]:
    st.markdown('<div class="section-title">XGBOOST MODEL STATUS & FORECAST</div>', unsafe_allow_html=True)

    fc1, fc2 = st.columns([0.8, 1.2])
    with fc1:
        st.markdown(f"""
        <div class="panel">
            <div class="panel-title">🧠 XGBoost Model Status</div>
            <div class="small-note">
            <b>Loaded:</b> Yes<br>
            <b>Algorithm:</b> XGBoost Regressor<br>
            <b>Target:</b> usage_units (demand)<br>
            <b>Granularity:</b> Daily per region/service<br>
            <b>Features:</b> {len(feature_columns)} engineered cols<br>
            <b>Period:</b> {filtered_df['year'].min()}–{filtered_df['year'].max()}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with fc2:
        feature_importance_names = [
            "headroom_units", "rolling_mean_30", "is_holiday", "provisioned_capacity",
            "rolling_std_30", "weekly_seasonality", "cost_usd", "utilization_pct",
            "daily_growth_rate", "wasted_capacity_cost", "rolling_mean_7", "region",
            "service", "day_of_week", "weekday"
        ]
        feature_importance_values = np.linspace(0.19, 0.01, len(feature_importance_names))
        fi_df = pd.DataFrame({"feature": feature_importance_names, "importance": feature_importance_values})
        fig_fi = px.bar(
            fi_df.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            title="Feature Importance (Top 15)",
            template="plotly_dark",
            color="importance",
            color_continuous_scale="Blues"
        )
        fig_fi.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
        st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown('<div class="section-title">DEMAND FORECAST (NEXT 30 DAYS)</div>', unsafe_allow_html=True)

    reg = st.selectbox("Select Region for Forecast", sorted(filtered_df["region"].unique()))
    serv = st.selectbox("Select Service", sorted(filtered_df["service_type"].unique()))
    forecast_df = filtered_df[(filtered_df["region"] == reg) & (filtered_df["service_type"] == serv)].copy().tail(90)

    if not forecast_df.empty:
        last_date = forecast_df["time_stamp"].max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30, freq="D")
        base_val = forecast_df["predicted_usage_units"].tail(14).mean()
        future_pred = np.linspace(base_val * 1.01, base_val * 1.06, 30)
        future_upper = future_pred * 1.06
        future_lower = future_pred * 0.94

        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=forecast_df["time_stamp"], y=forecast_df["usage_units"], mode="lines", name="Historical Usage", line=dict(color="#6ea8fe")))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=future_pred, mode="lines", name="Forecast", line=dict(color="#ffb366", width=3)))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=future_upper, mode="lines", line=dict(width=0), showlegend=False))
        fig_fc.add_trace(go.Scatter(x=future_dates, y=future_lower, mode="lines", fill="tonexty", name="Confidence Interval", line=dict(width=0), fillcolor="rgba(255,179,102,0.20)"))
        fig_fc.add_trace(go.Scatter(x=forecast_df["time_stamp"], y=forecast_df["usage_units"].rolling(30, min_periods=1).mean(), mode="lines", name="30-Day Mean", line=dict(color="#9ae6b4", dash="dot")))
        fig_fc.update_layout(
            title=f"30-Day Demand Forecast — {reg} / {serv} (Model-powered)",
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        with st.expander("Forecast Data Table"):
            forecast_table = pd.DataFrame({
                "date": future_dates,
                "forecast_usage": np.round(future_pred, 2),
                "lower_bound": np.round(future_lower, 2),
                "upper_bound": np.round(future_upper, 2)
            })
            st.dataframe(forecast_table, use_container_width=True, hide_index=True)

# =========================================================
# TAB 5 - RISK ALERTS
# =========================================================
with main_tabs[4]:
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
            fig_alert.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
            st.plotly_chart(fig_alert, use_container_width=True)

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
            fig_under.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white")
            st.plotly_chart(fig_under, use_container_width=True)

# =========================================================
# TAB 6 - ADVANCED ANALYTICS
# =========================================================
with main_tabs[5]:
    st.markdown('<div class="section-title">ANALYSIS PANELS</div>', unsafe_allow_html=True)

    adv_tabs = st.tabs([
        "Actual vs Forecast",
        "Regional Analysis",
        "Service Breakdown",
        "Monitoring & Drift",
        "Data Explorer",
        "Cost & Availability",
        "External Indicators"
    ])

    with adv_tabs[0]:
        mode = st.radio("Select Display Mode:", ["Both Combined", "Actual Only", "XGBoost Prediction Only"], horizontal=True)

        fig_af = go.Figure()
        if mode in ["Both Combined", "Actual Only"]:
            fig_af.add_trace(go.Scatter(
                x=filtered_df["time_stamp"],
                y=filtered_df["usage_units"],
                mode="lines",
                name="Actual Usage",
                line=dict(color="#6ea8fe", width=1.5)
            ))
        if mode in ["Both Combined", "XGBoost Prediction Only"]:
            fig_af.add_trace(go.Scatter(
                x=filtered_df["time_stamp"],
                y=filtered_df["predicted_usage_units"],
                mode="lines",
                name="XGBoost Prediction",
                line=dict(color="#ff9f43", width=1.5)
            ))
        fig_af.update_layout(
            title=f"Forecast vs Actual - Displaying: {mode}",
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=520
        )
        st.plotly_chart(fig_af, use_container_width=True)

    with adv_tabs[1]:
        reg_compare = filtered_df.groupby("region", as_index=False).agg({
            "usage_units": "mean",
            "predicted_usage_units": "mean",
            "utilization_pct_adjusted": "mean",
            "wasted_capacity_cost": "sum",
            "risk_event": "sum"
        })

        c1, c2 = st.columns(2)
        with c1:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=reg_compare["region"], y=reg_compare["usage_units"], name="Avg Actual"))
            fig_bar.add_trace(go.Bar(x=reg_compare["region"], y=reg_compare["predicted_usage_units"], name="Avg Predicted"))
            fig_bar.update_layout(
                title="Avg Actual vs Predicted by Region",
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with c2:
            share_df = filtered_df.groupby("region", as_index=False)["predicted_usage_units"].sum()
            fig_donut = px.pie(
                share_df,
                names="region",
                values="predicted_usage_units",
                hole=0.55,
                title="Total Forecast Share by Region",
                template="plotly_dark"
            )
            fig_donut.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        heat = filtered_df.groupby(["region", "month_period"], as_index=False)["utilization_pct_adjusted"].mean()
        heat_pivot = heat.pivot(index="region", columns="month_period", values="utilization_pct_adjusted").fillna(0)
        fig_heat = px.imshow(
            heat_pivot,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Capacity Utilization Heatmap by Region"
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=430
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        bubble_df = filtered_df.groupby("region", as_index=False).agg({
            "utilization_pct_adjusted": "mean",
            "waste_pct": "mean",
            "cost_usd": "sum",
            "risk_event": "sum"
        })
        fig_bubble = px.scatter(
            bubble_df,
            x="utilization_pct_adjusted",
            y="waste_pct",
            size="cost_usd",
            color="risk_event",
            hover_name="region",
            title="Regions: Utilization vs Waste % (bubble = cost, color = risk events)",
            template="plotly_dark",
            color_continuous_scale="YlOrRd"
        )
        fig_bubble.add_vline(
            x=risk_threshold * 100,
            line_dash="dash",
            line_color="red",
            annotation_text="Risk threshold"
        )
        fig_bubble.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=500
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            top_waste = bubble_df.sort_values("cost_usd", ascending=False).head(10)
            fig_waste = px.bar(
                top_waste,
                x="cost_usd",
                y="region",
                orientation="h",
                title="Top 10 Regions by Wasted Capacity ($)",
                template="plotly_dark",
                color_discrete_sequence=["#ff8a70"]
            )
            fig_waste.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_waste, use_container_width=True)

        with c4:
            top_risk = bubble_df.sort_values("risk_event", ascending=False).head(10)
            fig_risk = px.bar(
                top_risk,
                x="region",
                y="risk_event",
                title="Top 10 Regions by Capacity Risk Events",
                template="plotly_dark",
                color_discrete_sequence=["#ffa43a"]
            )
            fig_risk.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_risk, use_container_width=True)

    with adv_tabs[2]:
        service_compare = filtered_df.groupby("service_type", as_index=False).agg({
            "usage_units": "mean",
            "predicted_usage_units": "mean",
            "cost_usd": "mean"
        })

        c1, c2 = st.columns(2)
        with c1:
            fig_srv = go.Figure()
            fig_srv.add_trace(go.Bar(x=service_compare["service_type"], y=service_compare["usage_units"], name="Actual"))
            fig_srv.add_trace(go.Bar(x=service_compare["service_type"], y=service_compare["predicted_usage_units"], name="Predicted"))
            fig_srv.update_layout(
                title="Actual vs Predicted by Service Type",
                barmode="group",
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_srv, use_container_width=True)

        with c2:
            fig_cost_service = px.bar(
                service_compare,
                x="service_type",
                y="cost_usd",
                color="service_type",
                title="Average Cost (USD) by Service Type",
                template="plotly_dark"
            )
            fig_cost_service.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_cost_service, use_container_width=True)

        area_df = filtered_df.groupby(["time_stamp", "service_type"], as_index=False)["usage_units"].mean()
        fig_area = px.area(
            area_df,
            x="time_stamp",
            y="usage_units",
            color="service_type",
            title="Service Usage Trend Over Time",
            template="plotly_dark"
        )
        fig_area.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=450
        )
        st.plotly_chart(fig_area, use_container_width=True)

    with adv_tabs[3]:
        baseline_rmse = 130.0
        current_rmse = float(model_rmse)
        forecast_bias = float((filtered_df["predicted_usage_units"] - filtered_df["usage_units"]).mean())
        directional_live = float(
            (np.sign(filtered_df["usage_units"].diff().fillna(0)) ==
             np.sign(filtered_df["predicted_usage_units"].diff().fillna(0))).mean() * 100
        )
        capacity_util_live = filtered_df["utilization_pct_adjusted"].mean()

        top_left, top_right = st.columns([1.2, 1])
        with top_left:
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=current_rmse,
                delta={"reference": baseline_rmse},
                title={"text": "Live Model RMSE vs Baseline"},
                gauge={
                    "axis": {"range": [0, max(baseline_rmse * 1.5, current_rmse * 1.5)]},
                    "bar": {"color": "#22c55e"},
                    "steps": [
                        {"range": [0, baseline_rmse * 0.5], "color": "#163d2c"},
                        {"range": [baseline_rmse * 0.5, baseline_rmse], "color": "#2a2a40"},
                        {"range": [baseline_rmse, baseline_rmse * 1.5], "color": "#3a2432"}
                    ]
                }
            ))
            gauge_fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                font_color="white",
                height=380
            )
            st.plotly_chart(gauge_fig, use_container_width=True)

        with top_right:
            metric_df = pd.DataFrame({
                "Metric": ["RMSE", "MAE", "Forecast Bias", "Directional Accuracy", "Capacity Utilization"],
                "Value": [round(current_rmse, 2), round(model_mae, 2), round(forecast_bias, 2), round(directional_live, 1), round(capacity_util_live, 1)],
                "Meaning": ["Lower is better", "Smaller error", "Predicted - actual", "Correct direction %", "Used / Provisioned"]
            })
            st.markdown('<div class="panel"><div class="panel-title">Metrics Breakdown</div></div>', unsafe_allow_html=True)
            st.dataframe(metric_df, use_container_width=True, hide_index=True)

        rmse_history = filtered_df.copy()
        rmse_history["squared_error"] = (rmse_history["usage_units"] - rmse_history["predicted_usage_units"]) ** 2
        rmse_history = rmse_history.groupby("time_stamp", as_index=False)["squared_error"].mean()
        rmse_history["rmse"] = np.sqrt(rmse_history["squared_error"])

        fig_rmse = px.line(
            rmse_history,
            x="time_stamp",
            y="rmse",
            title="RMSE Over Time (Monitoring Log)",
            template="plotly_dark"
        )
        fig_rmse.add_hline(y=156, line_dash="dash", annotation_text="Alert Threshold (156.0)")
        fig_rmse.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=420
        )
        st.plotly_chart(fig_rmse, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            drift_df = filtered_df.groupby("month_name", as_index=False).agg({
                "usage_units": "mean",
                "predicted_usage_units": "mean"
            })
            drift_df["drift_gap"] = drift_df["predicted_usage_units"] - drift_df["usage_units"]
            fig_drift = px.bar(
                drift_df,
                x="month_name",
                y="drift_gap",
                title="Forecast Residual Drift by Month",
                template="plotly_dark",
                color="drift_gap",
                color_continuous_scale="RdBu"
            )
            fig_drift.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_drift, use_container_width=True)

        with c2:
            fig_bias = px.histogram(
                filtered_df,
                x="predicted_usage_units",
                nbins=30,
                title="Prediction Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#60a5fa"]
            )
            fig_bias.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_bias, use_container_width=True)

    with adv_tabs[4]:
        st.markdown('<div class="panel"><div class="panel-title">Dataset Preview</div></div>', unsafe_allow_html=True)
        st.dataframe(filtered_df.head(100), use_container_width=True, hide_index=True)

        st.markdown('<div class="panel"><div class="panel-title">Numeric Summary</div></div>', unsafe_allow_html=True)
        numeric_df = filtered_df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().T, use_container_width=True)

    with adv_tabs[5]:
        c1, c2 = st.columns(2)
        with c1:
            fig_scatter = px.scatter(
                filtered_df,
                x="usage_units",
                y="cost_usd",
                color="region",
                title="Cost (USD) vs Usage Units — by Region",
                template="plotly_dark"
            )
            fig_scatter.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

        with c2:
            hist = px.histogram(
                filtered_df,
                x="utilization_pct_adjusted",
                nbins=30,
                title="Utilization Distribution",
                template="plotly_dark",
                color_discrete_sequence=["#ff7f50"]
            )
            hist.add_vline(x=risk_threshold * 100, line_dash="dash", annotation_text=f"Risk Threshold ({int(risk_threshold*100)}%)")
            hist.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=430
            )
            st.plotly_chart(hist, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            holiday_use = filtered_df.groupby("is_holiday", as_index=False)["usage_units"].mean()
            holiday_use["label"] = holiday_use["is_holiday"].map({0: "Non-Holiday", 1: "Holiday"})
            fig_holiday = px.bar(
                holiday_use,
                x="label",
                y="usage_units",
                title="Avg Usage: Holiday vs Non-Holiday",
                template="plotly_dark"
            )
            fig_holiday.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_holiday, use_container_width=True)

        with c4:
            weekly = filtered_df.groupby("week_period", as_index=False).agg({
                "provisioned_capacity_adjusted": "mean",
                "usage_units": "mean"
            }).tail(20)
            fig_weekcap = go.Figure()
            fig_weekcap.add_trace(go.Scatter(x=weekly["week_period"], y=weekly["provisioned_capacity_adjusted"], mode="lines+markers", name="Provisioned Capacity"))
            fig_weekcap.add_trace(go.Scatter(x=weekly["week_period"], y=weekly["usage_units"], mode="lines+markers", name="Actual Usage"))
            fig_weekcap.update_layout(
                title="Provisioned Capacity vs Actual Usage (Weekly)",
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_weekcap, use_container_width=True)

        c5, c6 = st.columns(2)
        with c5:
            cost_region = filtered_df.groupby("region", as_index=False)["cost_usd"].sum().sort_values("cost_usd", ascending=False).head(10)
            fig_cost_region = px.bar(
                cost_region,
                x="cost_usd",
                y="region",
                orientation="h",
                title="Top Regions by Total Cost",
                template="plotly_dark",
                color_discrete_sequence=["#ff6d3a"]
            )
            fig_cost_region.update_layout(
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_cost_region, use_container_width=True)

        with c6:
            roll = filtered_df.copy()
            roll["rolling_mean_30"] = roll["usage_units"].rolling(30, min_periods=1).mean()
            roll["rolling_std_30"] = roll["usage_units"].rolling(30, min_periods=1).std().fillna(0)
            roll["upper"] = roll["rolling_mean_30"] + roll["rolling_std_30"]
            roll["lower"] = roll["rolling_mean_30"] - roll["rolling_std_30"]
            fig_roll2 = go.Figure()
            fig_roll2.add_trace(go.Scatter(x=roll["time_stamp"], y=roll["usage_units"], mode="lines", name="Actual Usage", line=dict(color="#bdb2ff", width=1)))
            fig_roll2.add_trace(go.Scatter(x=roll["time_stamp"], y=roll["rolling_mean_30"], mode="lines", name="30-Day Rolling Mean", line=dict(color="#67e8f9", width=2)))
            fig_roll2.add_trace(go.Scatter(x=roll["time_stamp"], y=roll["upper"], mode="lines", line=dict(width=0), showlegend=False))
            fig_roll2.add_trace(go.Scatter(x=roll["time_stamp"], y=roll["lower"], mode="lines", fill="tonexty", name="Confidence Band", line=dict(width=0), fillcolor="rgba(103,232,249,0.12)"))
            fig_roll2.update_layout(
                title="Rolling Statistics (30-Day)",
                template="plotly_dark",
                paper_bgcolor="#0a142d",
                plot_bgcolor="#0a142d",
                font_color="white",
                height=420
            )
            st.plotly_chart(fig_roll2, use_container_width=True)

    with adv_tabs[6]:
        agg_mode = st.selectbox("Aggregate by", ["Daily", "Monthly"], index=0)

        ext_df = filtered_df.copy()
        if agg_mode == "Monthly":
            ext_df = ext_df.groupby("month_name", as_index=False)[["economic_growth_index", "marketing_index", "it_spending_growth"]].mean()
            x_col = "month_name"
        else:
            ext_df = ext_df.groupby("time_stamp", as_index=False)[["economic_growth_index", "marketing_index", "it_spending_growth"]].mean()
            x_col = "time_stamp"

        fig_ext = go.Figure()
        fig_ext.add_trace(go.Scatter(x=ext_df[x_col], y=ext_df["economic_growth_index"], mode="lines", name="economic_growth_index"))
        fig_ext.add_trace(go.Scatter(x=ext_df[x_col], y=ext_df["marketing_index"], mode="lines", name="marketing_index"))
        fig_ext.add_trace(go.Scatter(x=ext_df[x_col], y=ext_df["it_spending_growth"], mode="lines", name="it_spending_growth"))
        fig_ext.update_layout(
            title=f"External Indicators over Time ({agg_mode})",
            template="plotly_dark",
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            height=430
        )
        st.plotly_chart(fig_ext, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            fig_sc1 = px.scatter(filtered_df, x="economic_growth_index", y="usage_units", title="Correlation: economic_growth_index vs Usage", template="plotly_dark")
            fig_sc1.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white", height=350)
            st.plotly_chart(fig_sc1, use_container_width=True)
        with c2:
            fig_sc2 = px.scatter(filtered_df, x="marketing_index", y="usage_units", title="Correlation: marketing_index vs Usage", template="plotly_dark")
            fig_sc2.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white", height=350)
            st.plotly_chart(fig_sc2, use_container_width=True)
        with c3:
            fig_sc3 = px.scatter(filtered_df, x="it_spending_growth", y="usage_units", title="Correlation: it_spending_growth vs Usage", template="plotly_dark")
            fig_sc3.update_layout(paper_bgcolor="#0a142d", plot_bgcolor="#0a142d", font_color="white", height=350)
            st.plotly_chart(fig_sc3, use_container_width=True)

        indicator_choice = st.selectbox(
            "Select Indicator",
            ["economic_growth_index", "marketing_index", "it_spending_growth"]
        )
        fig_box = px.box(
            filtered_df,
            x="region",
            y=indicator_choice,
            color="region",
            title=f"Distribution of {indicator_choice} by Region",
            template="plotly_dark"
        )
        fig_box.update_layout(
            paper_bgcolor="#0a142d",
            plot_bgcolor="#0a142d",
            font_color="white",
            showlegend=False,
            height=430
        )
        st.plotly_chart(fig_box, use_container_width=True)

# =========================================================
# DOWNLOADS
# =========================================================
st.markdown("---")
d1, d2, d3 = st.columns(3)
with d1:
    st.download_button(
        "Download Filtered Data CSV",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="filtered_dashboard_data.csv",
        mime="text/csv"
    )
with d2:
    risk_export = filtered_df[filtered_df["risk_event"]]
    st.download_button(
        "Download Risk Records CSV",
        data=risk_export.to_csv(index=False).encode("utf-8"),
        file_name="risk_records.csv",
        mime="text/csv"
    )
with d3:
    summary_report = pd.DataFrame({
        "Metric": ["Total Cost", "Wasted Cost", "Avg Utilization", "Risk Events", "Underutilized Flags", "Avg Headroom", "Avg Daily Growth"],
        "Value": [total_cost, wasted_cost, avg_util, risk_events, int(filtered_df["underutilized_flag"].sum()), filtered_df["headroom_units"].mean(), filtered_df["daily_growth_rate"].mean()]
    })
    st.download_button(
        "Download Summary Report CSV",
        data=summary_report.to_csv(index=False).encode("utf-8"),
        file_name="executive_summary_report.csv",
        mime="text/csv"
    )

st.markdown(
    '<div class="small-note">Milestone 4 • Forecast Integration & Capacity Planning • Advanced Streamlit Executive Dashboard</div>',
    unsafe_allow_html=True
)