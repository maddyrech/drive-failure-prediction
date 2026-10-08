"""Fleet health dashboard for an operations manager.

Run with:
    streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATABASE_URL, FIGURES  # noqa: E402

BLUE, GREY = "#2F6690", "#8C9BAB"
st.set_page_config(page_title="Drive fleet health", page_icon="💽", layout="wide")


@st.cache_resource
def engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


@st.cache_data(ttl=600)
def query(sql: str) -> pd.DataFrame:
    with engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


try:
    results = {r["name"]: json.loads(r["payload"]) for _, r in query("SELECT * FROM project_results").iterrows()}
    at_risk = query("SELECT * FROM at_risk_drives ORDER BY risk_rank")
    afr = query("SELECT * FROM afr_by_manufacturer ORDER BY afr_pct DESC")
    budget = query("SELECT * FROM budget_curve ORDER BY budget_pct")
    weekly = query(
        "SELECT DATE_TRUNC('week', date)::date AS week, SUM(failures) AS failures "
        "FROM daily_fleet GROUP BY 1 ORDER BY 1"
    )
except Exception as err:  # noqa: BLE001
    st.error(
        "Can't read the fleet data. Check the database is running and the pipeline has finished "
        "(python -m src.run_pipeline), then refresh this page."
    )
    st.caption(f"Details: {err}")
    st.stop()

stats_res, model_res = results.get("stats_results", {}), results.get("model_metrics", {})
start, end = stats_res.get("date_range", ["?", "?"])
active = model_res.get("active_drives_scored", len(at_risk))

st.title("Drive fleet health")
st.caption(
    f"Backblaze drive data from {start} to {end}. "
    f"Risk scores use each drive's readings on {model_res.get('scored_date', end)}."
)

# The decision this page exists for: which drives to check this week
st.subheader("Drives to check this week")
horizon = model_res.get("horizon_days", 30)
# Only offer budgets the model was actually tested on, and that the stored list covers
options = [float(b) for b in budget["budget_pct"] if b / 100 * active <= len(at_risk)]
if not options:
    st.warning("No tested budget fits the stored drive list. Re-run the model training step.")
    st.stop()
choice = st.select_slider(
    "How much of the fleet can the team check each week?",
    options=options,
    value=1.0 if 1.0 in options else options[-1],
    format_func=lambda b: f"{b:g}% ({round(b / 100 * active):,} drives)",
)
row = budget[budget["budget_pct"] == choice].iloc[0]
n = round(choice / 100 * active)
one_in = round(100 / row["precision_pct"]) if row["precision_pct"] > 0 else None
st.write(
    f"Checking the riskiest **{n:,}** drives ({choice:g}% of the fleet) caught **{row['recall_pct']:.0f}%** "
    f"of drives that failed within {horizon} days in the test weeks"
    + (f", and about **1 in {one_in}** checks found one." if one_in else ".")
)

table = at_risk.head(n).assign(age_years=lambda d: (d["age_days"] / 365).round(1))
counters = ["reallocated_sectors", "pending_sectors", "uncorrectable_errors"]
for col in counters:
    # Some makers don't report every SMART counter: say so instead of showing "None"
    table[col] = table[col].map(lambda v: "n/a" if pd.isna(v) else f"{int(v):,}")
st.dataframe(
    table[["risk_rank", "serial_number", "model", "manufacturer", "age_years", *counters, "risk_score"]],
    hide_index=True,
    width="stretch",
    height=420,
    column_config={
        "risk_rank": "Rank",
        "serial_number": "Serial",
        "model": "Model",
        "manufacturer": "Maker",
        "age_years": "Age (years)",
        "reallocated_sectors": "Reallocated sectors",
        "pending_sectors": "Pending sectors",
        "uncorrectable_errors": "Uncorrectable errors",
        "risk_score": st.column_config.ProgressColumn(
            "Risk", help="Relative risk from the model. Use it to rank drives, not as a probability.",
            min_value=0.0, max_value=1.0, format="%.2f"),
    },
)
st.caption("n/a: this drive model doesn't report that counter (common for HGST and Toshiba drives).")

st.divider()
c1, c2, c3 = st.columns(3)
c1.metric("Drives monitored", f"{stats_res.get('drives', 0):,}")
c2.metric("Failures in the period", f"{stats_res.get('failures', 0):,}")
c3.metric("Fleet failure rate (per year)", f"{stats_res.get('fleet_afr_pct', 0)}%")

left, right = st.columns(2)
with left:
    st.subheader("Failure rate by maker")
    fig = go.Figure(go.Bar(
        x=afr["afr_pct"], y=afr["manufacturer"], orientation="h", marker_color=BLUE,
        error_x=dict(type="data", symmetric=False,
                     array=afr["afr_high_pct"] - afr["afr_pct"],
                     arrayminus=afr["afr_pct"] - afr["afr_low_pct"]),
        hovertemplate="%{y}: %{x:.2f}% a year<extra></extra>",
    ))
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0),
                      xaxis_title="Annualised failure rate (%), 95% interval", yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, width="stretch")
    test = stats_res.get("manufacturer_test", {})
    if test:
        verdict = "a real difference, not chance" if test["p_value"] < 0.05 else "not clearly different"
        st.caption(f"Chi-square test adjusted for running time: p = {test['p_value']:.2g}, so {verdict}.")

with right:
    st.subheader("Failures per week")
    fig = go.Figure(go.Scatter(x=weekly["week"], y=weekly["failures"], mode="lines+markers",
                               line=dict(color=BLUE), hovertemplate="%{x}: %{y} failures<extra></extra>"))
    fig.update_layout(height=300, margin=dict(l=0, r=10, t=10, b=0), yaxis_title="Failures")
    st.plotly_chart(fig, width="stretch")

st.divider()
signs = stats_res.get("reallocated_sectors_test", {})
if signs:
    st.subheader("The clearest warning sign")
    st.write(
        f"Drives that reallocated even one sector failed **{signs['relative_risk']}x** as often "
        f"({signs['failure_risk_flagged_pct']}% vs {signs['failure_risk_clean_pct']}%, "
        f"95% interval {signs['relative_risk_ci95'][0]}x to {signs['relative_risk_ci95'][1]}x)."
    )

left, right = st.columns(2)
with left:
    st.subheader("How long drives last")
    if (FIGURES / "survival_by_manufacturer.png").exists():
        st.image(str(FIGURES / "survival_by_manufacturer.png"), width="stretch")
with right:
    st.subheader("Model vs the simple rule")
    rule, matched = model_res.get("simple_rule", {}), model_res.get("best_model_at_rule_rate", {})
    if rule and matched:
        st.write(
            f"Flagging every drive with sector errors means checking {rule['flag_rate_pct']}% of the fleet "
            f"and catches {rule['recall_pct']}% of upcoming failures. "
            f"Checking the same number of drives ranked by the model catches {matched['recall_pct']}%."
        )
    if (FIGURES / "alert_budget.png").exists():
        st.image(str(FIGURES / "alert_budget.png"), width="stretch")
