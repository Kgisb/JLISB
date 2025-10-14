
# -------------------------------------------------------------
# JetLearn – Unified App (Activity Composition, MTD vs Cohort)
# -------------------------------------------------------------
# Run:
#   streamlit run app_ACTIVITY_COMPOSITION_full.py
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, datetime, timedelta
from calendar import monthrange
import os
import re

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
      .muted { color:#6b7280; font-size:.85rem; }
      .section-title { font-weight:700; margin:.4rem 0 .25rem; }
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
    # Try exact then case-insensitive
    for c in candidates:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None

def to_dt(series: pd.Series, dayfirst=True):
    try:
        return pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=dayfirst)
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def to_int(series: pd.Series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    from calendar import monthrange
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

INVALID_RE = re.compile(r"^\s*1\.2\s*invalid\s*deal(s)?\s*$", re.IGNORECASE)

def exclude_invalid_deals(df: pd.DataFrame, dealstage_col: str | None) -> tuple[pd.DataFrame, int]:
    if not dealstage_col or dealstage_col not in df.columns:
        return df, 0
    col = df[dealstage_col].astype(str)
    keep = ~col.str.match(INVALID_RE)
    removed = int((~keep).sum())
    return df.loc[keep].copy(), removed

def normalize_pipeline(value: str) -> str:
    if not isinstance(value, str): return "Other"
    v = value.strip().lower()
    if "math" in v: return "Math"
    if "ai" in v or "coding" in v or "ai-coding" in v or "ai coding" in v:
        return "AI Coding"
    return "Other"

# ----------------------
# Sidebar: Navigation
# ----------------------
with st.sidebar:
    st.header("JetLearn • Navigation")
    MASTER_SECTIONS = {
        "Performance": ["Activity tracer"],  # Single focused pill per your spec
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    sub_views = MASTER_SECTIONS[master]
    view = st.radio("View", sub_views, index=0, key="nav_sub")

    # Track filter (Both/AI Coding/Math)
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)

# ----------------------
# Data loading
# ----------------------
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"
st.caption(f"Data source default: **{DEFAULT_DATA_PATH}** (override below if needed)")
csv_path = st.text_input("CSV path", value=DEFAULT_DATA_PATH, key="csv_path")
df = load_csv(csv_path)

# Column mapping (robust to common variants)
dealstage_col      = find_col(df, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
create_col         = find_col(df, ["Create Date","Created Date","Create_Date","Created At","Deal Create Date"])
last_activity_col  = find_col(df, ["Last Activity Date","Last activity date","Last_Activity_Date"])
# Two candidate variables for aggregation / filters:
num_sales_act_col  = find_col(df, ["Number of Sales Activities","Number of sales activities","Sales Activities Count","Sales_Activities_Count"])
num_times_contact_col = find_col(df, ["Number of times contacted","Times Contacted","No. of times contacted","Number_of_times_contacted"])
counsellor_col     = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor","Deal Owner"])
country_col        = find_col(df, ["Country","Country Name"])
source_col         = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
pipeline_col       = find_col(df, ["Pipeline"])

# Exclude invalid
df, removed = exclude_invalid_deals(df, dealstage_col)
if removed:
    st.caption(f"Excluded **{removed}** rows with Deal Stage = '1.2 Invalid deal(s)'.")

# --------------
# Global filters
# --------------
with st.sidebar.expander("Filters", expanded=True):
    def opts(series: pd.Series):
        vals = sorted([str(v) for v in series.dropna().unique()])
        return ["All"] + vals

    sel_counsellors = ["All"]
    sel_countries   = ["All"]
    sel_sources     = ["All"]

    if counsellor_col:
        sel_counsellors = st.multiselect("Academic Counsellor", opts(df[counsellor_col]), default=["All"])
    if country_col:
        sel_countries   = st.multiselect("Country", opts(df[country_col]), default=["All"])
    if source_col:
        sel_sources     = st.multiselect("JetLearn Deal Source", opts(df[source_col]), default=["All"])

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
# Performance ▶ Activity tracer (composition by Deal Stage)
# =====================================================
def render_activity_tracer(df_scope: pd.DataFrame):
    st.subheader("Performance — Activity tracer (Deal Stage composition)")

    # Required columns
    needed = {
        "Deal Stage": dealstage_col,
        "Create Date": create_col,
        "Last Activity Date": last_activity_col,
    }
    missing = [k for k, v in needed.items() if not v or v not in df_scope.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    # Counting mode
    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ac_mode")

    # Date window is based on Last Activity Date
    today = date.today()
    m_start, m_end = month_bounds(today)
    dr = st.date_input("Date range (by Last Activity Date)", value=(m_start, today), key="ac_dates")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start_d, end_d = dr
    else:
        start_d, end_d = m_start, today
    if end_d < start_d:
        start_d, end_d = end_d, start_d

    # Select aggregation variable (Deals count or sums of contact variables)
    agg_choices = ["Deals (count)"]
    if num_times_contact_col: agg_choices.append("Sum: Number of times contacted")
    if num_sales_act_col:     agg_choices.append("Sum: Number of Sales Activities")

    agg_pick = st.selectbox("Aggregate by", agg_choices, index=0, key="ac_agg")

    # Optional seek bars (range filters) for the two variables if present
    # These control composition by restricting the population, irrespective of aggregation measure chosen
    if num_times_contact_col or num_sales_act_col:
        with st.expander("Optional range filters", expanded=False):
            if num_times_contact_col:
                col_ntc = to_int(df_scope[num_times_contact_col])
                min_ntc, max_ntc = int(col_ntc.min()), int(col_ntc.max())
                ntc_range = st.slider("Number of times contacted (filter)",
                                       min_value=min_ntc, max_value=max_ntc,
                                       value=(min_ntc, max_ntc), key="rng_ntc")
            else:
                ntc_range = None

            if num_sales_act_col:
                col_nsa = to_int(df_scope[num_sales_act_col])
                min_nsa, max_nsa = int(col_nsa.min()), int(col_nsa.max())
                nsa_range = st.slider("Number of Sales Activities (filter)",
                                       min_value=min_nsa, max_value=max_nsa,
                                       value=(min_nsa, max_nsa), key="rng_nsa")
            else:
                nsa_range = None
    else:
        ntc_range = nsa_range = None

    # Primary grouping (which composition to show per row)
    prim_opts = []
    if counsellor_col: prim_opts.append("Academic Counsellor")
    if country_col:    prim_opts.append("Country")
    if source_col:     prim_opts.append("JetLearn Deal Source")

    if not prim_opts:
        st.error("None of Academic Counsellor / Country / JetLearn Deal Source found.")
        return

    primary = st.selectbox("Show composition per", prim_opts, index=0, key="ac_primary")

    # Prepare working columns
    d = df_scope.copy()
    d["_create"] = to_dt(d[create_col]).dt.date
    d["_last"]   = to_dt(d[last_activity_col]).dt.date

    if num_times_contact_col:
        d["_ntc"] = to_int(d[num_times_contact_col])
    else:
        d["_ntc"] = 0
    if num_sales_act_col:
        d["_nsa"] = to_int(d[num_sales_act_col])
    else:
        d["_nsa"] = 0

    # Apply window and optional range filters
    mask_date = d["_last"].between(start_d, end_d)
    mask_ntc  = True if ntc_range is None else d["_ntc"].between(ntc_range[0], ntc_range[1])
    mask_nsa  = True if nsa_range is None else d["_nsa"].between(nsa_range[0], nsa_range[1])

    if mode == "MTD":
        ms, me = month_bounds(end_d)  # MTD relative to end of selected range
        mask_mtd = d["_create"].between(ms, me)
        mask = mask_date & mask_ntc & mask_nsa & mask_mtd
    else:
        mask = mask_date & mask_ntc & mask_nsa

    d = d.loc[mask].copy()
    if d.empty:
        st.info("No rows after filters. Try widening the date range or adjusting ranges.")
        return

    # Label columns for grouping
    if dealstage_col:   d["Deal Stage"] = d[dealstage_col].fillna("Unknown").astype(str)
    if counsellor_col:  d["Academic Counsellor"] = d[counsellor_col].fillna("Unknown").astype(str)
    if country_col:     d["Country"] = d[country_col].fillna("Unknown").astype(str)
    if source_col:      d["JetLearn Deal Source"] = d[source_col].fillna("Unknown").astype(str)

    # Choose aggregation value
    if agg_pick == "Deals (count)":
        d["_val"] = 1
        agg_title = "Deals (count)"
    elif agg_pick == "Sum: Number of times contacted" and num_times_contact_col:
        d["_val"] = d["_ntc"]
        agg_title = "Sum of Number of times contacted"
    elif agg_pick == "Sum: Number of Sales Activities" and num_sales_act_col:
        d["_val"] = d["_nsa"]
        agg_title = "Sum of Number of Sales Activities"
    else:
        d["_val"] = 1
        agg_title = "Deals (count)"

    # Compute composition: per primary group, broken by Deal Stage
    group_cols = [primary, "Deal Stage"]
    comp = d.groupby(group_cols, dropna=False)["_val"].sum().reset_index(name="Value")

    # Totals per primary dim
    totals = comp.groupby(primary, dropna=False)["Value"].sum().rename("Total")
    comp = comp.merge(totals, on=primary, how="left")
    comp["% of Total"] = np.where(comp["Total"] > 0, (comp["Value"] / comp["Total"]) * 100.0, 0.0).round(1)

    # KPI row (overall)
    overall_total = int(comp["Value"].sum())
    unique_prim = comp[primary].nunique()
    unique_stages = comp["Deal Stage"].nunique()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall total", overall_total)
    with c2: st.metric(f"Unique {primary}", unique_prim)
    with c3: st.metric("Deal Stages", unique_stages)

    # Chart: stacked composition per primary
    st.markdown("### Composition chart")
    ch = alt.Chart(comp).mark_bar().encode(
        x=alt.X(f"{primary}:N", sort="-y", title=primary),
        y=alt.Y("Value:Q", title=agg_title),
        color=alt.Color("Deal Stage:N", legend=alt.Legend(title="Deal Stage")),
        tooltip=[
            alt.Tooltip(f"{primary}:N", title=primary),
            alt.Tooltip("Deal Stage:N"),
            alt.Tooltip("Value:Q", title=agg_title),
            alt.Tooltip("% of Total:Q"),
        ],
    ).properties(height=360, title=f"{agg_title} by {primary} (stacked by Deal Stage)")

    st.altair_chart(ch, use_container_width=True)

    # Table: wide pivot (primary x deal stage) with totals and %
    st.markdown("### Composition table")
    pivot = comp.pivot_table(index=primary, columns="Deal Stage", values="Value", fill_value=0, aggfunc="sum")
    pivot["Total"] = pivot.sum(axis=1)
    # Percent table
    pct_tbl = pivot.div(pivot["Total"].replace(0, np.nan), axis=0) * 100.0
    pct_tbl = pct_tbl.drop(columns=["Total"], errors="ignore").round(1).add_suffix(" (%)")

    table = pivot.merge(pct_tbl, left_index=True, right_index=True, how="left").reset_index()
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download CSV — Composition table",
        table.to_csv(index=False).encode("utf-8"),
        file_name="activity_composition_table.csv",
        mime="text/csv"
    )

    st.caption(
        f"Window: **{start_d} → {end_d}** • Mode: **{mode}** • Aggregation: **{agg_title}** "
        f"• MTD counts only deals whose **Create Date** falls in the same month as the end of the selected range."
    )


# ----------------------
# Router
# ----------------------
if master == "Performance" and view == "Activity tracer":
    render_activity_tracer(df_f)
else:
    st.info("Select a view from the sidebar.")
