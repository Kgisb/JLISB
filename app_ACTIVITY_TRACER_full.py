
# -------------------------------------------------------------
# JetLearn – Unified App (with Activity tracer under Performance)
# -------------------------------------------------------------
# How to run locally:
#   streamlit run app_ACTIVITY_TRACER_full.py
#
# Notes
# - Loads Master_sheet-DB.csv in the same folder by default.
# - Robust column resolver for common header variants.
# - "Activity tracer" implements MTD/Cohort logic on Last Activity Date
#   with a seek bar for Number of Sales Activities and group-by for
#   JetLearn Deal Source, Academic Counsellor, Country.
#
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, datetime, timedelta
from calendar import monthrange
import re
import os

# ----------------------
# Page config & styling
# ----------------------
st.set_page_config(page_title="JetLearn – Unified App", page_icon="📊", layout="wide")

st.markdown(
    """
    <style>
      .block-container { max-width: 1400px !important; padding-top: 1.0rem !important; }
      .stAltairChart, .stDataFrame {
        border: 1px solid #e7e8ea; border-radius: 16px; padding: 14px; background: #ffffff;
        box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
      }
      div[data-testid="stMetric"] {
        background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
        border: 1px solid #eef0f2; border-radius: 16px; padding: 12px 14px;
        box-shadow: 0 1px 6px rgba(16,24,40,.05);
      }
      button[role="tab"] {
        border-radius: 999px !important; padding: 8px 14px !important; margin-right: 6px !important;
        border: 1px solid #e7e8ea !important;
      }
      button[role="tab"][aria-selected="true"] {
        background: #111827 !important; color: #ffffff !important; border-color: #111827 !important;
      }
      .section-title { font-weight:700; margin:.4rem 0 .25rem; }
      .muted { color:#6b7280; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Helpers
# ----------------------
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    # Try exact > case-insensitive matches
    for c in candidates:
        if c in df.columns:
            return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series, dayfirst=True) -> pd.Series:
    try:
        s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
        return s
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str): return "Other"
    v = value.strip().lower()
    if "math" in v: return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal(s)?\s*$", re.IGNORECASE)

def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col or dealstage_col not in df.columns:
        return df, 0
    col = df[dealstage_col].astype(str)
    keep = ~col.str.match(INVALID_RE)
    removed = int((~keep).sum())
    return df.loc[keep].copy(), removed

# ----------------------
# Sidebar: Navigation
# ----------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    MASTER_SECTIONS = {
        "Performance": [
            "Activity tracer",  # NEW pill
            # (You can add or retain other pills of your app here)
            # "Sales Activity", "Leaderboard", "Comparison"
        ],
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    sub_views = MASTER_SECTIONS.get(master, [])
    if "nav_sub" not in st.session_state:
        st.session_state["nav_sub"] = sub_views[0] if sub_views else ""
    if st.session_state["nav_sub"] not in sub_views and sub_views:
        st.session_state["nav_sub"] = sub_views[0]
    view = st.session_state["nav_sub"]

    # Track filter (Both/AI Coding/Math)
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)

# ----------------------
# Data: load & mapping
# ----------------------
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"
st.caption(f"Data source default: **{DEFAULT_DATA_PATH}** (override below if needed)")
data_src = st.text_input("CSV path", value=DEFAULT_DATA_PATH, key="data_src_input")
df = load_csv(data_src)

# Column mapping (robust, common variants)
dealstage_col         = find_col(df, ["Deal Stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
create_col            = find_col(df, ["Create Date","Created Date","Create_Date","Created At","Deal Create Date"])
last_activity_col     = find_col(df, ["Last Activity Date","Last activity date","Last_Activity_Date"])
num_sales_act_col     = find_col(df, ["Number of Sales Activities","Number of sales activities","Sales Activities Count","Sales_Activities_Count"])
counsellor_col        = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor","Deal Owner"])
country_col           = find_col(df, ["Country","Country Name"])
source_col            = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
pipeline_col          = find_col(df, ["Pipeline"])

# Exclude "1.2 Invalid deal(s)"
df, removed_invalid = exclude_invalid_deals(df, dealstage_col)
if removed_invalid:
    st.caption(f"Excluded **{removed_invalid}** rows with Deal Stage = '1.2 Invalid deal(s)'.")

# Global quick filters (consistent with your app style)
with st.sidebar.expander("Filters", expanded=True):
    def options(series: pd.Series):
        vals = sorted([str(v) for v in series.dropna().unique()])
        return ["All"] + vals

    sel_counsellors = ["All"]
    sel_countries   = ["All"]
    sel_sources     = ["All"]

    if counsellor_col:
        sel_counsellors = st.multiselect("Academic Counsellor", options(df[counsellor_col]), default=["All"])
    if country_col:
        sel_countries   = st.multiselect("Country", options(df[country_col]), default=["All"])
    if source_col:
        sel_sources     = st.multiselect("JetLearn Deal Source", options(df[source_col]), default=["All"])

def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if counsellor_col and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

df_f = apply_global_filters(df)

# Track filter
if track != "Both" and pipeline_col and pipeline_col in df_f.columns:
    df_f = df_f[ df_f[pipeline_col].map(normalize_pipeline) == track ]

# ----------------------
# Breadcrumb
# ----------------------
st.title("📊 JetLearn – Unified App")
st.markdown(f"<div class='muted'>Path: <b>{master}</b> › <b>{view}</b> • Track: <b>{track}</b></div>", unsafe_allow_html=True)

# =====================================================
# NEW PILL: Performance ▶ Activity tracer
# =====================================================
def render_performance_activity_tracer(df_scope: pd.DataFrame):
    st.subheader("Performance — Activity tracer")

    # Guard columns
    required = {
        "Deal Stage": dealstage_col,
        "Last Activity Date": last_activity_col,
        "Number of Sales Activities": num_sales_act_col,
        "Create Date": create_col,
    }
    missing = [k for k, v in required.items() if not v or v not in df_scope.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    # MTD vs Cohort
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="act_mode")

    # Date range — based on Last Activity Date
    today = date.today()
    month_start, month_end = month_bounds(today)
    dr = st.date_input("Date range (by Last Activity Date)", value=(month_start, today), key="act_dates")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start_d, end_d = dr
    else:
        start_d, end_d = today.replace(day=1), today
    if end_d < start_d:
        start_d, end_d = end_d, start_d

    # Seek bar for Number of Sales Activities
    s = df_scope[num_sales_act_col]
    s_num = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    min_sa = int(s_num.min()) if len(s_num) else 0
    max_sa = int(s_num.max()) if len(s_num) else 10
    sa_min, sa_max = st.slider("Number of Sales Activities (range filter)",
                               min_value=min_sa, max_value=max_sa,
                               value=(min_sa, max_sa), key="sa_range")

    # Group-by controls
    dim_opts = []
    if dealstage_col: dim_opts.append("Deal Stage")
    if counsellor_col: dim_opts.append("Academic Counsellor")
    if source_col: dim_opts.append("JetLearn Deal Source")
    if country_col: dim_opts.append("Country")

    grp_dims = st.multiselect("Group by", dim_opts, default=["Deal Stage"], key="act_dims")

    # Prepare dates
    d = df_scope.copy()
    d["_last_act"] = coerce_datetime(d[last_activity_col]).dt.date
    d["_create"]   = coerce_datetime(d[create_col]).dt.date
    d["_sa"]       = pd.to_numeric(d[num_sales_act_col], errors="coerce").fillna(0).astype(int)

    # Apply window (Last Activity Date) & seek bar
    mask_date = d["_last_act"].between(start_d, end_d)
    mask_sa   = d["_sa"].between(sa_min, sa_max)

    # MTD rule: count only if Create Date is in the same month as the END of selected range
    if mode == "MTD":
        ms, me = month_bounds(end_d)  # month-of selected range's end
        mask_mtd = d["_create"].between(ms, me)
        mask = mask_date & mask_sa & mask_mtd
    else:
        mask = mask_date & mask_sa

    d = d.loc[mask].copy()
    if d.empty:
        st.info("No rows after filters. Try widening the date range or seek bar.")
        return

    # Label columns for grouping
    if dealstage_col: d["Deal Stage"] = d[dealstage_col].fillna("Unknown").astype(str)
    if counsellor_col: d["Academic Counsellor"] = d[counsellor_col].fillna("Unknown").astype(str)
    if source_col: d["JetLearn Deal Source"] = d[source_col].fillna("Unknown").astype(str)
    if country_col: d["Country"] = d[country_col].fillna("Unknown").astype(str)

    # KPIs
    total_contacts = int(d["_sa"].sum())
    deals_count    = int(d.shape[0])
    avg_contacts   = (total_contacts / deals_count) if deals_count else 0.0

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total contacts (sum of #SA)", total_contacts)
    with c2: st.metric("Deals (rows)", deals_count)
    with c3: st.metric("Avg contacts / deal", f"{avg_contacts:.2f}")

    # ---------- Chart 1: Stacked by Deal Stage over time (sum of #SA) ----------
    dd = d.copy()
    dd["_day"] = pd.to_datetime(dd["_last_act"]).dt.date

    # Build aggregation
    group_for_chart = ["_day"]
    if "Deal Stage" in grp_dims:
        group_for_chart.append("Deal Stage")

    chart_df = (
        dd.groupby(group_for_chart, dropna=False)["_sa"]
          .sum()
          .reset_index()
          .rename(columns={"_sa": "Contacts"})
    )

    if "Deal Stage" in group_for_chart:
        ch = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("_day:T", title=None),
            y=alt.Y("Contacts:Q", title="Contacts (sum of #SA)"),
            color=alt.Color("Deal Stage:N", legend=alt.Legend(title="Deal Stage")),
            tooltip=[alt.Tooltip("_day:T", title="Date"), alt.Tooltip("Contacts:Q")]
        ).properties(height=320, title="Contacts by date (stacked by Deal Stage)")
    else:
        ch = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("_day:T", title=None),
            y=alt.Y("Contacts:Q", title="Contacts (sum of #SA)"),
            tooltip=[alt.Tooltip("_day:T", title="Date"), alt.Tooltip("Contacts:Q")]
        ).properties(height=320, title="Contacts by date")

    st.altair_chart(ch, use_container_width=True)

    # ---------- Table: breakdown by selected group dims ----------
    group_for_table = []
    for g in grp_dims:
        if g in d.columns:
            group_for_table.append(g)

    if not group_for_table:
        d["_All"] = "All"
        group_for_table = ["_All"]

    table_df = (
        d.groupby(group_for_table, dropna=False)
         .agg(
             Deals=("Deal Stage", "count"),
             Total_Contacts=("_sa", "sum"),
             Avg_Contacts_per_Deal=("_sa", "mean"),
         )
         .reset_index()
    )
    table_df["Avg_Contacts_per_Deal"] = table_df["Avg_Contacts_per_Deal"].round(2)

    st.markdown("### Breakdown")
    st.dataframe(table_df, use_container_width=True, hide_index=True)

    # Download
    st.download_button(
        "Download CSV — Activity tracer (breakdown)",
        table_df.to_csv(index=False).encode("utf-8"),
        file_name="activity_tracer_breakdown.csv",
        mime="text/csv"
    )

    # Footnote about mode
    st.caption(
        f"Window: **{start_d} → {end_d}** • Mode: **{mode}** "
        f"• MTD counts deals whose **Create Date** is in the same month as the range end."
    )


# ----------------------
# Router
# ----------------------
if master == "Performance" and view == "Activity tracer":
    render_performance_activity_tracer(df_f)
else:
    st.info("Select a view from the sidebar.")
