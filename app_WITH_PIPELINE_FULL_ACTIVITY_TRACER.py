
# -------------------------------------------------------------
# JetLearn – Unified App (FULL) with Activity tracer integrated
# -------------------------------------------------------------
# Run:
#   streamlit run app_WITH_PIPELINE_FULL_ACTIVITY_TRACER.py
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import date, timedelta
from calendar import monthrange
import re
import os

# ======================
# Page & styling
# ======================
st.set_page_config(page_title="JetLearn – Unified App", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 1400px !important; padding-top: 1.2rem !important; padding-bottom: 2.0rem !important; }
      .stAltairChart, .stDataFrame, .stTable {
          border: 1px solid #e7e8ea; border-radius: 16px; padding: 14px; background: #ffffff; box-shadow: 0 2px 10px rgba(16, 24, 40, 0.06);
      }
      div[data-testid="stMetric"] { background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%); border: 1px solid #eef0f2; border-radius: 16px; padding: 12px 14px; box-shadow: 0 1px 6px rgba(16,24,40,.05); }
      button[role="tab"] { border-radius: 999px !important; padding: 8px 14px !important; margin-right: 6px !important; border: 1px solid #e7e8ea !important; }
      button[role="tab"][aria-selected="true"] { background: #111827 !important; color: #ffffff !important; border-color: #111827 !important; }
      .muted { color:#6b7280; font-size:.85rem; }
      .section-title { font-weight:700; margin:.4rem 0 .25rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ======================
# Helpers
# ======================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low: return low[c.lower()]
    return None

def coerce_datetime(series: pd.Series) -> pd.Series:
    try:
        s = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
        return s
    except Exception:
        return pd.to_datetime(series, errors="coerce")

def month_bounds(d: date):
    start = date(d.year, d.month, 1)
    end = date(d.year, d.month, monthrange(d.year, d.month)[1])
    return start, end

def last_month_bounds(today: date):
    first_this = date(today.year, today.month, 1)
    last_prev = first_this - timedelta(days=1)
    return month_bounds(last_prev)

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

# ======================
# Sidebar: Navigation
# ======================
with st.sidebar:
    st.header("JetLearn • Navigation")
    MASTER_SECTIONS = {
        "Performance": [
            "Activity tracer",     # NEW integrated pill
            "Leaderboard",
            "Comparison",
            "Sales Activity",
        ],
        "Funnel & Movement": ["Funnel","Lead Movement","Stuck deals","Deal Velocity","Deal Decay","Carry Forward"],
        "Insights & Forecast": ["Predictibility","Business Projection","Buying Propensity","80-20","Trend & Analysis","Heatmap","Bubble Explorer","Master Graph"],
        "Marketing": ["Referrals","HubSpot Deal Score tracker","Marketing Lead Performance & Requirement"],
    }
    master = st.radio("Sections", list(MASTER_SECTIONS.keys()), index=0, key="nav_master")
    sub_views = MASTER_SECTIONS.get(master, [])
    if "nav_sub" not in st.session_state:
        st.session_state["nav_sub"] = sub_views[0] if sub_views else ""
    if st.session_state["nav_sub"] not in sub_views and sub_views:
        st.session_state["nav_sub"] = sub_views[0]
    view = st.radio("View", sub_views, index=sub_views.index(st.session_state["nav_sub"]) if sub_views else 0, key="nav_sub")

    # Track (Both / AI Coding / Math)
    track = st.radio("Track", ["Both", "AI Coding", "Math"], index=0)

# ======================
# Data: load & mapping
# ======================
DEFAULT_DATA_PATH = "Master_sheet-DB.csv"
st.caption(f"Data source default: **{DEFAULT_DATA_PATH}** (override below if needed)")
data_src = st.text_input("CSV path", value=DEFAULT_DATA_PATH, key="data_src_input")
df = load_csv(data_src)

# Column mapping
dealstage_col = find_col(df, ["Deal Stage","Deal stage","Stage","Deal Status","Stage Name","Deal Stage Name"])
create_col    = find_col(df, ["Create Date","Create date","Create_Date","Created At","Deal Create Date"])
pay_col       = find_col(df, ["Payment Received Date","Payment Received date","Payment_Received_Date","Payment Date","Paid At"])
pipeline_col  = find_col(df, ["Pipeline"])
counsellor_col = find_col(df, ["Student/Academic Counsellor","Academic Counsellor","Student/Academic Counselor","Counsellor","Counselor","Deal Owner"])
country_col    = find_col(df, ["Country","Country Name"])
source_col     = find_col(df, ["JetLearn Deal Source","Deal Source","Source"])
first_cal_sched_col = find_col(df, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
cal_resched_col     = find_col(df, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
cal_done_col        = find_col(df, ["Calibration Done Date","Calibration done date","Calibration_Done_Date"])
calibration_slot_col = find_col(df, ["Calibration Slot (Deal)", "Calibration Slot", "Cal Slot (Deal)", "Cal Slot"])
last_activity_col    = find_col(df, ["Last Activity Date","Last activity date","Last_Activity_Date"])
num_sales_act_col    = find_col(df, ["Number of Sales Activities","Number of sales activities","Sales Activities Count","Sales_Activities_Count"])
num_times_contact_col = find_col(df, ["Number of times contacted","Times Contacted","No. of times contacted","Number_of_times_contacted"])

# Exclude invalid
df, removed_invalid = exclude_invalid_deals(df, dealstage_col)
if removed_invalid:
    st.caption(f"Excluded **{removed_invalid}** rows with Deal Stage = '1.2 Invalid deal(s)'.")

# Global filters
with st.sidebar.expander("Filters (Global)", expanded=True):
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

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    if counsellor_col and "All" not in sel_counsellors:
        f = f[f[counsellor_col].astype(str).isin(sel_counsellors)]
    if country_col and "All" not in sel_countries:
        f = f[f[country_col].astype(str).isin(sel_countries)]
    if source_col and "All" not in sel_sources:
        f = f[f[source_col].astype(str).isin(sel_sources)]
    return f

df_f = apply_filters(df)

# Track filter
if track != "Both":
    if pipeline_col and pipeline_col in df_f.columns:
        df_f = df_f[ df_f[pipeline_col].map(normalize_pipeline).fillna("Other") == track ]
    else:
        st.warning("Pipeline column not found — Track filter cannot be applied.", icon="⚠️")

# ======================
# Title & breadcrumb
# ======================
st.title("📊 JetLearn – Unified App")
st.markdown(f"<div class='muted'>Path: <b>{master}</b> › <b>{view}</b> • Track: <b>{track}</b></div>", unsafe_allow_html=True)

# ======================
# Performance: Leaderboard
# ======================
def render_performance_leaderboard(
    df_f,
    counsellor_col,
    create_col,
    pay_col,
    first_cal_sched_col,
    cal_resched_col,
    cal_done_col,
    source_col,
):
    st.subheader("Performance — Leaderboard (Academic Counsellor)")
    if not counsellor_col or counsellor_col not in df_f.columns:
        st.warning("Academic Counsellor column not found.", icon="⚠️")
        return

    level = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="lb_mode")
    date_mode = st.radio("Date scope", ["This month", "Last month", "Custom"], index=0, horizontal=True, key="lb_scope")

    today = date.today()
    def _month_bounds(d: date):
        from calendar import monthrange
        start = date(d.year, d.month, 1)
        end = date(d.year, d.month, monthrange(d.year, d.month)[1])
        return start, end
    def _last_month_bounds(d: date):
        first_this = date(d.year, d.month, 1)
        last_prev = first_this - timedelta(days=1)
        return _month_bounds(last_prev)

    if date_mode == "This month":
        range_start, range_end = _month_bounds(today)
    elif date_mode == "Last month":
        range_start, range_end = _last_month_bounds(today)
    else:
        c1, c2 = st.columns(2)
        with c1: range_start = st.date_input("Start", value=today.replace(day=1), key="lb_start")
        with c2: range_end   = st.date_input("End",   value=_month_bounds(today)[1], key="lb_end")
        if range_end < range_start:
            st.error("End date cannot be before start date.")
            return

    def _dt(s): 
        try: return pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True).dt.date
        except: return pd.Series(pd.NaT, index=df_f.index)

    _C = _dt(df_f[create_col]) if (create_col and create_col in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    _P = _dt(df_f[pay_col])    if (pay_col and pay_col in df_f.columns)     else pd.Series(pd.NaT, index=df_f.index)
    _F = _dt(df_f[first_cal_sched_col]) if (first_cal_sched_col and first_cal_sched_col in df_f.columns) else pd.Series(pd.NaT, index=df_f.index)
    _R = _dt(df_f[cal_resched_col])     if (cal_resched_col and cal_resched_col in df_f.columns)         else pd.Series(pd.NaT, index=df_f.index)
    _D = _dt(df_f[cal_done_col])        if (cal_done_col and cal_done_col in df_f.columns)              else pd.Series(pd.NaT, index=df_f.index)

    SRC  = df_f[source_col].fillna("Unknown").astype(str).str.strip() if (source_col and source_col in df_f.columns) else pd.Series("Unknown", index=df_f.index)

    c_in = _C.between(range_start, range_end)
    p_in = _P.between(range_start, range_end)
    f_in = _F.between(range_start, range_end)
    r_in = _R.between(range_start, range_end)
    d_in = _D.between(range_start, range_end)

    if level == "MTD":
        enrol_mask = p_in & c_in
        f_mask = f_in & c_in
        r_mask = r_in & c_in
        d_mask = d_in & c_in
    else:
        enrol_mask = p_in
        f_mask = f_in
        r_mask = r_in
        d_mask = d_in

    grp = df_f[counsellor_col].fillna("Unknown").astype(str)
    tbl = pd.DataFrame({
        "Academic Counsellor": grp,
        "Deals": c_in.astype(int),
        "Enrolments": enrol_mask.astype(int),
        "First Cal": f_mask.astype(int),
        "Cal Rescheduled": r_mask.astype(int),
        "Cal Done": d_mask.astype(int),
    }).groupby("Academic Counsellor").sum(numeric_only=True).reset_index()

    metric = st.selectbox("Rank by", ["Enrolments","Deals","First Cal","Cal Rescheduled","Cal Done"], index=0, key="lb_rank")
    asc = st.checkbox("Ascending order", value=False, key="lb_asc")
    tbl = tbl.sort_values(metric, ascending=asc).reset_index(drop=True)
    tbl.index = tbl.index + 1
    st.dataframe(tbl, use_container_width=True)
    st.caption(f"Window: **{range_start} → {range_end}** • Mode: **{level}**")

# ======================
# Performance: Comparison
# ======================
def render_performance_comparison(
    df_f, create_col, pay_col, counsellor_col, country_col, source_col,
    first_cal_sched_col=None, cal_resched_col=None, cal_done_col=None
):
    import pandas as pd
    import numpy as np
    from datetime import date

    st.subheader("Performance — Comparison (Window A vs Window B)")
    st.caption("Compare metrics across two independently-filtered windows (A & B). MTD = payment & created within window; Cohort = payment within window (create anywhere).")

    def _col(df, primary, cands):
        if primary and primary in df.columns: return primary
        for c in cands:
            if c in df.columns: return c
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low: return low[c.lower()]
        return None

    _create = _col(df_f, create_col, ["Create Date","Created Date","Deal Create Date","CreateDate","Created On"])
    _pay    = _col(df_f, pay_col,    ["Payment Received Date","Payment Date","Enrolment Date","PaymentReceivedDate","Paid On"])
    _owner  = _col(df_f, counsellor_col, ["Student/Academic Counsellor","Academic Counsellor","Counsellor","Counselor","Deal Owner"])
    _cntry  = _col(df_f, country_col,    ["Country","Country Name"])
    _src    = _col(df_f, source_col,     ["JetLearn Deal Source","Deal Source","Source"])

    if _create is None or _pay is None:
        st.error("Required date columns not found (Create Date / Payment Received Date).")
        return

    d = df_f.copy()
    def _to_dt(s):
        if pd.api.types.is_datetime64_any_dtype(s): return s
        try:    return pd.to_datetime(s, dayfirst=True, errors="coerce")
        except: return pd.to_datetime(s, errors="coerce")

    d["_create"] = _to_dt(d[_create])
    d["_pay"]    = _to_dt(d[_pay])
    if _owner: d["_owner"] = d[_owner].fillna("Unknown").astype(str)
    if _cntry: d["_cntry"] = d[_cntry].fillna("Unknown").astype(str)
    if _src:   d["_src"]   = d[_src].fillna("Unknown").astype(str)

    _first  = _col(d, first_cal_sched_col, ["First Calibration Scheduled Date","First Calibration","First Cal Scheduled"])
    _resch  = _col(d, cal_resched_col,     ["Calibration Rescheduled Date","Cal Rescheduled","Rescheduled Date"])
    _done   = _col(d, cal_done_col,        ["Calibration Done Date","Cal Done","Trial Done Date"])

    def _to_dt2(colname):
        if not colname or colname not in d.columns: return None
        try:    return pd.to_datetime(d[colname], dayfirst=True, errors="coerce")
        except: return pd.to_datetime(d[colname], errors="coerce")
    d["_first"] = _to_dt2(_first)
    d["_resch"] = _to_dt2(_resch)
    d["_done"]  = _to_dt2(_done)

    mode = st.radio("Counting mode", ["MTD","Cohort"], index=0, horizontal=True, key="cmp_mode")
    metrics = st.multiselect(
        "Metrics to compare",
        ["Deals Created","Enrollments","First Cal","Cal Rescheduled","Cal Done"],
        default=["Enrollments","Deals Created"],
        key="cmp_metrics",
    )
    if not metrics:
        st.info("Select at least one metric to compare."); return

    dim_opts = []
    if _owner: dim_opts.append("Academic Counsellor")
    if _src:   dim_opts.append("JetLearn Deal Source")
    if _cntry: dim_opts.append("Country")
    if not dim_opts:
        st.warning("No grouping dimensions available."); return

    st.markdown("**Configure Windows**")
    colA, colB = st.columns(2)
    today = date.today()
    with colA:
        st.write("### Window A")
        dims_a = st.multiselect("Group by (A)", dim_opts, default=[dim_opts[0]], key="cmp_dims_a")
        date_a = st.date_input("Date range (A)", value=(today.replace(day=1), today), key="cmp_date_a")
    with colB:
        st.write("### Window B")
        dims_b = st.multiselect("Group by (B)", dim_opts, default=[dim_opts[0]], key="cmp_dims_b")
        date_b = st.date_input("Date range (B)", value=(today.replace(day=1), today), key="cmp_date_b")

    def _ensure_tuple(val):
        if isinstance(val, (list, tuple)) and len(val) == 2:
            return pd.to_datetime(val[0]), pd.to_datetime(val[1])
        return pd.to_datetime(val), pd.to_datetime(val)

    a_start, a_end = _ensure_tuple(date_a)
    b_start, b_end = _ensure_tuple(date_b)
    if a_end < a_start: a_start, a_end = a_end, a_start
    if b_end < b_start: b_start, b_end = b_end, b_start

    def _agg(df, dims, start, end):
        group_cols = []
        if dims:
            for dname in dims:
                if dname == "Academic Counsellor" and "_owner" in df: group_cols.append("_owner")
                if dname == "JetLearn Deal Source" and "_src" in df: group_cols.append("_src")
                if dname == "Country" and "_cntry" in df: group_cols.append("_cntry")
        if not group_cols:
            df = df.copy(); df["_dummy"] = "All"; group_cols = ["_dummy"]

        g = df.copy()
        m_create = g["_create"].between(start, end)
        m_pay    = g["_pay"].between(start, end)
        m_first  = g["_first"].between(start, end) if "_first" in g else pd.Series(False, index=g.index)
        m_resch  = g["_resch"].between(start, end) if "_resch" in g else pd.Series(False, index=g.index)
        m_done   = g["_done"].between(start, end)  if "_done"  in g else pd.Series(False, index=g.index)

        res = g[group_cols].copy()

        for m in metrics:
            if m == "Deals Created":
                cnt = g.loc[m_create].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Deals Created"), left_on=group_cols, right_index=True, how="left")
            elif m == "Enrollments":
                if mode == "MTD":
                    cnt = g.loc[m_pay & m_create].groupby(group_cols, dropna=False).size()
                else:
                    cnt = g.loc[m_pay].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Enrollments"), left_on=group_cols, right_index=True, how="left")
            elif m == "First Cal":
                cnt = g.loc[(m_first & m_create) if mode == "MTD" else m_first].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("First Cal"), left_on=group_cols, right_index=True, how="left")
            elif m == "Cal Rescheduled":
                cnt = g.loc[(m_resch & m_create) if mode == "MTD" else m_resch].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Cal Rescheduled"), left_on=group_cols, right_index=True, how="left")
            elif m == "Cal Done":
                cnt = g.loc[(m_done & m_create) if mode == "MTD" else m_done].groupby(group_cols, dropna=False).size()
                res = res.merge(cnt.rename("Cal Done"), left_on=group_cols, right_index=True, how="left")

        res = res.groupby(group_cols, dropna=False).first().fillna(0).reset_index()
        pretty = []
        for c in group_cols:
            if c == "_owner": pretty.append("Academic Counsellor")
            elif c == "_src": pretty.append("JetLearn Deal Source")
            elif c == "_cntry": pretty.append("Country")
            elif c == "_dummy": pretty.append("All")
            else: pretty.append(c)
        res = res.rename(columns=dict(zip(group_cols, pretty)))
        return res, pretty

    res_a, _ = _agg(d, dims_a, a_start, a_end)
    res_b, _ = _agg(d, dims_b, b_start, b_end)

    join_keys = [c for c in ["Academic Counsellor","JetLearn Deal Source","Country","All"] if c in res_a.columns and c in res_b.columns]
    if join_keys:
        merged = pd.merge(res_a, res_b, on=join_keys, how="outer", suffixes=(" (A)", " (B)"))
    else:
        ra = res_a.assign(_KeyLabel=res_a.astype(str).agg(" | ".join, axis=1))
        rb = res_b.assign(_KeyLabel=res_b.astype(str).agg(" | ".join, axis=1))
        merged = pd.merge(ra, rb, on="_KeyLabel", how="outer", suffixes=(" (A)", " (B)"))

    for m in metrics:
        colA = f"{m} (A)"; colB = f"{m} (B)"
        if colA in merged.columns and colB in merged.columns:
            a = pd.to_numeric(merged[colA], errors="coerce").fillna(0.0)
            b = pd.to_numeric(merged[colB], errors="coerce").fillna(0.0)
            merged[f"Δ {m} (B−A)"] = (b - a)
            pct = np.where(a != 0, (b / a) * 100.0, np.nan)
            merged[f"% Δ {m} (vs A)"] = pd.Series(pct, index=merged.index).round(1)

    key_cols = [c for c in ["Academic Counsellor","JetLearn Deal Source","Country","All"] if c in merged.columns]
    a_cols = [f"{m} (A)" for m in metrics if f"{m} (A)" in merged.columns]
    b_cols = [f"{m} (B)" for m in metrics if f"{m} (B)" in merged.columns]
    d_cols = [c for c in merged.columns if c.startswith("Δ ") or c.startswith("% Δ ")]

    final_cols = key_cols + a_cols + b_cols + d_cols
    final = merged[final_cols].fillna(0)

    st.dataframe(final, use_container_width=True, hide_index=True)
    st.download_button(
        "Download CSV — Comparison (A vs B)",
        final.to_csv(index=False).encode("utf-8"),
        file_name="performance_comparison_A_vs_B.csv",
        mime="text/csv"
    )

# ======================
# Performance: Sales Activity
# ======================
def render_performance_sales_activity(
    df_f,
    first_cal_sched_col: str | None,
    cal_resched_col: str | None,
    slot_col: str | None,
    counsellor_col: str | None,
    country_col: str | None,
    source_col: str | None,
):
    import pandas as pd
    import altair as alt
    from datetime import date, timedelta

    st.subheader("Performance — Sales Activity")
    uid = "perf_sales_activity"

    d = df_f.copy()

    def _pick_col(df: pd.DataFrame, preferred, candidates):
        if isinstance(preferred, str) and preferred in df.columns:
            return preferred
        for c in candidates:
            if c in df.columns:
                return c
        lc = {c.lower().strip(): c for c in df.columns}
        for c in candidates:
            k = c.lower().strip()
            if k in lc:
                return lc[k]
        return None

    first_col = _pick_col(d, first_cal_sched_col, ["First Calibration Scheduled Date","First calibration scheduled date","First_Calibration_Scheduled_Date"])
    resch_col = _pick_col(d, cal_resched_col, ["Calibration Rescheduled Date","Calibration rescheduled date","Calibration_Rescheduled_Date"])
    slot_col_res = _pick_col(d, slot_col, ["Calibration Slot (Deal)","Calibration Slot","Cal Slot (Deal)","Cal Slot"])
    explicit_booking = _pick_col(d, None, ["[Trigger] - Calibration Booking Date","Calibration Booking Date","Cal Booking Date","Booking Date (Calibration)"])

    book = st.selectbox("Booking Type", ["All","Pre-book","Sales-book"], index=0, key=f"sa_book_{uid}")
    metric = st.selectbox("Metric", ["Trial Scheduled","Trial Rescheduled","Calibration Booked"], index=0, key=f"sa_metric_{uid}")

    today = date.today()
    preset = st.radio("Range", ["Today","Yesterday","This Month","Custom"], index=2, horizontal=True, key=f"sa_rng_{uid}")
    if preset == "Today":
        start, end = today, today
    elif preset == "Yesterday":
        start = today - timedelta(days=1); end = start
    elif preset == "This Month":
        start = today.replace(day=1); end = today
    else:
        c1, c2 = st.columns(2)
        with c1: start = st.date_input("Start", value=today.replace(day=1), key=f"sa_start_{uid}")
        with c2: end = st.date_input("End", value=today, key=f"sa_end_{uid}")
        if start > end: start, end = end, start

    def _to_dt(s: pd.Series):
        try: return pd.to_datetime(s, errors="coerce", dayfirst=True, infer_datetime_format=True)
        except: return pd.to_datetime(pd.Series([None]*len(s)), errors="coerce")

    if metric == "Trial Scheduled":
        if not first_col:
            st.warning("First Calibration Scheduled Date column not found."); return
        d["_act_date"] = _to_dt(d[first_col])
    elif metric == "Trial Rescheduled":
        if not resch_col:
            st.warning("Calibration Rescheduled Date column not found."); return
        d["_act_date"] = _to_dt(d[resch_col])
    else:
        if explicit_booking:
            d["_act_date"] = _to_dt(d[explicit_booking])
        else:
            if not slot_col_res:
                st.warning("No booking date column or slot column available to derive booking date."); return
            def extract_date(txt):
                if not isinstance(txt, str) or txt.strip() == "": return None
                t = txt.strip()[:10].replace(".", "-").replace("/", "-")
                return pd.to_datetime(t, errors="coerce", dayfirst=True)
            d["_act_date"] = d[slot_col_res].apply(extract_date)

    if book != "All" and slot_col_res:
        if book == "Pre-book":
            d = d[d[slot_col_res].notna() & (d[slot_col_res].astype(str).strip() != "")]
        elif book == "Sales-book":
            d = d[d[slot_col_res].isna() | (d[slot_col_res].astype(str).strip() == "")]

    if "_act_date" not in d.columns:
        st.info("No activity date available after metric selection."); return

    d = d[d["_act_date"].notna()].copy()
    mask = (d["_act_date"].dt.date >= start) & (d["_act_date"].dt.date <= end)
    d = d[mask].copy()

    if d.empty:
        st.info("No rows in the selected window/filters."); return

    d["_d"] = d["_act_date"].dt.date
    counts = d.groupby("_d").size().reset_index(name="Count")

    c1,c2,c3 = st.columns(3)
    with c1: st.metric("Total", int(counts["Count"].sum()))
    with c2: st.metric("Days", counts.shape[0])
    with c3: st.metric("Avg / day", f"{(counts['Count'].mean() if counts.shape[0] else 0.0):.1f}")

    ch = alt.Chart(counts).mark_bar().encode(
        x=alt.X("_d:T", title=None),
        y=alt.Y("Count:Q", title="Count"),
        tooltip=[alt.Tooltip("_d:T", title="Date"), alt.Tooltip("Count:Q")]
    ).properties(height=320, title=f"{metric} — daily counts")
    st.altair_chart(ch, use_container_width=True)

    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.download_button("Download CSV", counts.to_csv(index=False).encode(), "sales_activity_counts.csv", "text/csv")

# ======================
# Performance: Activity tracer (Deal Stage composition)
# ======================
def render_activity_tracer(df_scope: pd.DataFrame):
    st.subheader("Performance — Activity tracer (Deal Stage composition)")

    needed = {
        "Deal Stage": dealstage_col,
        "Create Date": create_col,
        "Last Activity Date": last_activity_col,
    }
    missing = [k for k, v in needed.items() if not v or v not in df_scope.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return

    mode = st.radio("Counting mode", ["MTD", "Cohort"], index=0, horizontal=True, key="ac_mode")

    today = date.today()
    m_start, _ = month_bounds(today)
    dr = st.date_input("Date range (by Last Activity Date)", value=(m_start, today), key="ac_dates")
    if isinstance(dr, (list, tuple)) and len(dr) == 2:
        start_d, end_d = dr
    else:
        start_d, end_d = m_start, today
    if end_d < start_d:
        start_d, end_d = end_d, start_d

    # Aggregation measure
    agg_choices = ["Deals (count)"]
    if num_times_contact_col: agg_choices.append("Sum: Number of times contacted")
    if num_sales_act_col:     agg_choices.append("Sum: Number of Sales Activities")
    agg_pick = st.selectbox("Aggregate by", agg_choices, index=0, key="ac_agg")

    # Optional range filters to constrain population
    ntc_range = None; nsa_range = None
    with st.expander("Optional range filters (constrain population)", expanded=False):
        if num_times_contact_col:
            ntc_series = pd.to_numeric(df_scope[num_times_contact_col], errors="coerce").fillna(0).astype(int)
            min_ntc, max_ntc = int(ntc_series.min()), int(ntc_series.max())
            ntc_range = st.slider("Number of times contacted", min_value=min_ntc, max_value=max_ntc, value=(min_ntc, max_ntc), key="rng_ntc")
        if num_sales_act_col:
            nsa_series = pd.to_numeric(df_scope[num_sales_act_col], errors="coerce").fillna(0).astype(int)
            min_nsa, max_nsa = int(nsa_series.min()), int(nsa_series.max())
            nsa_range = st.slider("Number of Sales Activities", min_value=min_nsa, max_value=max_nsa, value=(min_nsa, max_nsa), key="rng_nsa")

    prim_opts = []
    if counsellor_col: prim_opts.append("Academic Counsellor")
    if country_col:    prim_opts.append("Country")
    if source_col:     prim_opts.append("JetLearn Deal Source")
    if not prim_opts:
        st.error("None of Academic Counsellor / Country / JetLearn Deal Source found.")
        return
    primary = st.selectbox("Show composition per", prim_opts, index=0, key="ac_primary")

    d = df_scope.copy()
    d["_create"] = coerce_datetime(d[create_col]).dt.date
    d["_last"]   = coerce_datetime(d[last_activity_col]).dt.date
    d["_ntc"] = pd.to_numeric(d[num_times_contact_col], errors="coerce").fillna(0).astype(int) if num_times_contact_col else 0
    d["_nsa"] = pd.to_numeric(d[num_sales_act_col], errors="coerce").fillna(0).astype(int) if num_sales_act_col else 0

    mask_date = d["_last"].between(start_d, end_d)
    mask_ntc  = True if ntc_range is None else d["_ntc"].between(ntc_range[0], ntc_range[1])
    mask_nsa  = True if nsa_range is None else d["_nsa"].between(nsa_range[0], nsa_range[1])

    if mode == "MTD":
        ms, me = month_bounds(end_d)
        mask_mtd = d["_create"].between(ms, me)
        mask = mask_date & mask_ntc & mask_nsa & mask_mtd
    else:
        mask = mask_date & mask_ntc & mask_nsa

    d = d.loc[mask].copy()
    if d.empty:
        st.info("No rows after filters. Try widening the date range or adjusting ranges.")
        return

    if dealstage_col:   d["Deal Stage"] = d[dealstage_col].fillna("Unknown").astype(str)
    if counsellor_col:  d["Academic Counsellor"] = d[counsellor_col].fillna("Unknown").astype(str)
    if country_col:     d["Country"] = d[country_col].fillna("Unknown").astype(str)
    if source_col:      d["JetLearn Deal Source"] = d[source_col].fillna("Unknown").astype(str)

    if agg_pick == "Deals (count)":
        d["_val"] = 1; agg_title = "Deals (count)"
    elif agg_pick == "Sum: Number of times contacted" and num_times_contact_col:
        d["_val"] = d["_ntc"]; agg_title = "Sum of Number of times contacted"
    elif agg_pick == "Sum: Number of Sales Activities" and num_sales_act_col:
        d["_val"] = d["_nsa"]; agg_title = "Sum of Number of Sales Activities"
    else:
        d["_val"] = 1; agg_title = "Deals (count)"

    group_cols = [primary, "Deal Stage"]
    comp = d.groupby(group_cols, dropna=False)["_val"].sum().reset_index(name="Value")
    totals = comp.groupby(primary, dropna=False)["Value"].sum().rename("Total")
    comp = comp.merge(totals, on=primary, how="left")
    comp["% of Total"] = np.where(comp["Total"] > 0, (comp["Value"] / comp["Total"]) * 100.0, 0.0).round(1)

    overall_total = int(comp["Value"].sum())
    unique_prim = comp[primary].nunique()
    unique_stages = comp["Deal Stage"].nunique()
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Overall total", overall_total)
    with c2: st.metric(f"Unique {primary}", unique_prim)
    with c3: st.metric("Deal Stages", unique_stages)

    st.markdown("### Composition chart")
    ch = alt.Chart(comp).mark_bar().encode(
        x=alt.X(f"{primary}:N", sort="-y", title=primary),
        y=alt.Y("Value:Q", title=agg_title),
        color=alt.Color("Deal Stage:N", legend=alt.Legend(title="Deal Stage")),
        tooltip=[alt.Tooltip(f"{primary}:N", title=primary),
                 alt.Tooltip("Deal Stage:N"),
                 alt.Tooltip("Value:Q", title=agg_title),
                 alt.Tooltip("% of Total:Q")],
    ).properties(height=360, title=f"{agg_title} by {primary} (stacked by Deal Stage)")
    st.altair_chart(ch, use_container_width=True)

    st.markdown("### Composition table")
    pivot = comp.pivot_table(index=primary, columns="Deal Stage", values="Value", fill_value=0, aggfunc="sum")
    pivot["Total"] = pivot.sum(axis=1)
    pct_tbl = pivot.div(pivot["Total"].replace(0, np.nan), axis=0) * 100.0
    pct_tbl = pct_tbl.drop(columns=["Total"], errors="ignore").round(1).add_suffix(" (%)")
    table = pivot.merge(pct_tbl, left_index=True, right_index=True, how="left").reset_index()

    st.dataframe(table, use_container_width=True, hide_index=True)
    st.download_button("Download CSV — Composition table", table.to_csv(index=False).encode("utf-8"), "activity_composition_table.csv", "text/csv")

    st.caption(
        f"Window: **{start_d} → {end_d}** • Mode: **{mode}** • Aggregation: **{agg_title}** "
        f"• MTD counts only deals whose **Create Date** falls in the same month as the end of the selected range."
    )

# ======================
# ROUTER
# ======================
if master == "Performance":
    if view == "Activity tracer":
        render_activity_tracer(df_f)
    elif view == "Leaderboard":
        render_performance_leaderboard(df_f, counsellor_col, create_col, pay_col, first_cal_sched_col, cal_resched_col, cal_done_col, source_col)
    elif view == "Comparison":
        render_performance_comparison(df_f, create_col, pay_col, counsellor_col, country_col, source_col, first_cal_sched_col, cal_resched_col, cal_done_col)
    elif view == "Sales Activity":
        render_performance_sales_activity(df_f, first_cal_sched_col, cal_resched_col, calibration_slot_col, counsellor_col, country_col, source_col)
    else:
        st.info("This view is part of Performance but not implemented in this build.")
elif master in ("Funnel & Movement", "Insights & Forecast", "Marketing"):
    st.info("This build focuses on the Performance suite (Activity tracer, Leaderboard, Comparison, Sales Activity).")
else:
    st.info("Pick a section from the sidebar.")
