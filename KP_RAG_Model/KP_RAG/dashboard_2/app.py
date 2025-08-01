# ---------------------------------------------------------------------------------
# HR DASHBOARD v2  ·  All–in–one Streamlit script (dashboard_2/app.py)
# ---------------------------------------------------------------------------------
# 1.  To-Do/Build Checklist (auto-generated)
# • Load HR conversation CSV(s) – default to sample_data.csv if none supplied.
# • Provide global sidebar filters (Parent Category, Agent, Date range).
# • Use Instrument Sans font + navy/yellow brand colours.
# • Render 11 high-value charts inside collapsible expanders:
#     1  Category Volume vs Error Rate              (bar + line)
#     2  Sub-Topic Confusion Treemap                (treemap)
#     3  Agent Workload vs Error                    (scatter)
#     4  Temporal Error Heat-map                    (heatmap)
#     5  Conversation Length vs Error               (scatter + trend)
#     6  Positive-Feedback Rate by Category         (bar)
#     7  Duplicate Query Frequency                  (bar)
#     8  Low-Confidence Outlier Tracker             (table)
#     9  Parent ↔ Sub-Topic Mismatch Sunburst       (sunburst)
#    10  Knowledge Article Impact                   (violin/box)
#    11  Error Similarity Trend Over Time           (line)
# • Each chart ends with a placeholder insight line.
# • Uses Plotly template = "simple_white", colour palette navy (#001f3f) & yellow (#FFDC00).
# • Entire app lives in this single file; sample dataset stored at ../dashboard_2/sample_data.csv.
# ---------------------------------------------------------------------------------

import os
import glob
import textwrap
from datetime import datetime
from pathlib import Path

# Base libs
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Optional dependency check for statsmodels (required for Plotly trendlines)
try:
    import statsmodels.api as _sm  # noqa: F401
    _HAS_STATSMODELS = True
except ModuleNotFoundError:
    _HAS_STATSMODELS = False

# -----------------------------------------------------------------------------
# Constants & Theme
# -----------------------------------------------------------------------------
# Rework colour scheme: primary bright yellow, secondary black
PRIMARY = "#FFDC00"  # bright yellow for bars/points
SECONDARY = "#000000"  # black for contrast lines/alt series
FONT_FAMILY = "Instrument Sans, sans-serif"

st.set_page_config(page_title="HR Query Analytics", layout="wide")

# Inject basic CSS to register custom font if available --------------------------------
# Looks for InstrumentSans.ttf in project root.
font_path = Path(__file__).resolve().parent.parent / "InstrumentSans.ttf"
if font_path.exists():
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Instrument Sans';
            src: url('file://{font_path}') format('truetype');
        }}
        html, body, [class*="css"]  {{
            font-family: 'Instrument Sans', sans-serif;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def primary_bar(x, y, name=""):
    return go.Bar(x=x, y=y, name=name or "", marker_color=PRIMARY)


def secondary_line(x, y, name=""):
    return go.Scatter(x=x, y=y, name=name or "", mode="lines+markers", line=dict(color=SECONDARY, width=3))


# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_data(uploaded_files):
    """Read uploaded CSV(s) or fall back to bundled sample."""
    if uploaded_files:
        dfs = [pd.read_csv(f) for f in uploaded_files]
    else:
        sample_path = Path(__file__).parent / "sample_data.csv"
        dfs = [pd.read_csv(sample_path)]
    df = pd.concat(dfs, ignore_index=True)

    # Type casts
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    int_cols = ["hour_of_day", "conversation_length"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    float_cols = [
        "Parent Error Similarity Score",
        "Parent Category Similarity_Score",
        "SubCategory Similarity_Score",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure helper cols exist
    if "day_of_week" not in df.columns:
        df["day_of_week"] = df["Timestamp"].dt.day_name()
    if "hour_of_day" not in df.columns:
        df["hour_of_day"] = df["Timestamp"].dt.hour

    return df


# -----------------------------------------------------------------------------
# Sidebar – file upload & global filters (DATA SOURCE SECTION)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("Controls")
    uploaded_files = st.file_uploader(
        "Upload one or more weekly CSV exports",
        type="csv",
        accept_multiple_files=True,
    )

    df_full = load_data(uploaded_files)

    # Filters -----------------------------------------------------
    parent_options = sorted(df_full["Parent Category Topic"].dropna().unique())
    agent_options = sorted(df_full["Agent_ID"].dropna().unique())

    parent_filter = st.multiselect("Parent Category", parent_options)
    agent_filter = st.multiselect("Agent", agent_options)

    # Date range filter
    # Guard against empty or NaT
    min_ts, max_ts = df_full["Timestamp"].min(), df_full["Timestamp"].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        min_ts = max_ts = pd.Timestamp.today()

    date_range = st.date_input(
        "Date range",
        value=(min_ts.date(), max_ts.date()),
        min_value=min_ts.date(),
        max_value=max_ts.date(),
    )

    threshold = st.slider(
        "High-confusion threshold (Parent Error Similarity)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

# Apply filter mask -----------------------------------------------------------
mask = pd.Series(True, index=df_full.index)
if parent_filter:
    mask &= df_full["Parent Category Topic"].isin(parent_filter)
if agent_filter:
    mask &= df_full["Agent_ID"].isin(agent_filter)
start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
mask &= df_full["Timestamp"].between(start_date, end_date + pd.Timedelta(days=1))

data = df_full[mask].copy()

# -----------------------------------------------------------------------------
# Page Title
# -----------------------------------------------------------------------------
st.title("HR Conversation Analytics Dashboard (v2)")
st.caption(
    "Interactive insights for HR Agent-Assist performance.  "
    "Colours: Navy = volume/context, Yellow = confusion/alert.  "
)

# -----------------------------------------------------------------------------
# Utility: add placeholder insight line
# -----------------------------------------------------------------------------

def add_insight(text: str):
    st.markdown(f"*Insight ⇢ {text}*")


# -----------------------------------------------------------------------------
# Chart 1 – Category Volume vs Error Rate
# -----------------------------------------------------------------------------
with st.expander("1 · Category Volume vs Error Rate"):
    agg = (
        data.groupby("Parent Category Topic")
        .agg(count=("Parent Category Topic", "size"), mean_error=("Parent Error Similarity Score", "mean"))
        .reset_index()
        .sort_values("count", ascending=False)
    )

    fig = go.Figure()
    fig.add_trace(primary_bar(agg["Parent Category Topic"], agg["count"], name="Query Volume"))
    fig.add_trace(
        secondary_line(agg["Parent Category Topic"], agg["mean_error"], name="Avg Error Score")
    )

    fig.update_layout(
        title="Query Volume and Average Error by Parent Category",
        yaxis_title="Volume",
        yaxis2=dict(title="Avg Error Score", overlaying="y", side="right"),
        xaxis_tickangle=-45,
        template="simple_white",
        font=dict(family=FONT_FAMILY, color="black"),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Payroll / Compensation pairs high volume with elevated error rates – training focus suggested.")

# -----------------------------------------------------------------------------
# Chart 1A – Adjust volume threshold (additional knob)
# -----------------------------------------------------------------------------
with st.expander("1A · Category Volume ≥ Min Threshold"):
    min_vol = st.slider("Minimum query volume", min_value=1, max_value=int(data.shape[0]), value=5, step=1, key="cat_vol")
    filt_df = agg[agg["count"] >= min_vol]
    fig2 = px.bar(
        filt_df,
        x="Parent Category Topic",
        y="mean_error",
        color_discrete_sequence=[PRIMARY],
        labels={"mean_error": "Avg Error Score"},
        title=f"Categories with ≥{min_vol} Queries (Avg Error)",
    )
    fig2.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig2, use_container_width=True)
    add_insight("Quickly focus on high-traffic categories above chosen threshold.")

# -----------------------------------------------------------------------------
# Chart 2 – Sub-Topic Confusion Treemap
# -----------------------------------------------------------------------------
with st.expander("2 · Sub-Topic Confusion Treemap"):
    treemap_df = (
        data.groupby(["Parent Category Topic", "SubCategory Topic"])
        .agg(err=("SubCategory Similarity_Score", "mean"), n=("SubCategory Topic", "size"))
        .reset_index()
    )

    # Build labels & parents for go.Treemap to avoid pandas/narwhals bug with px.treemap
    parent_nodes = treemap_df["Parent Category Topic"].unique().tolist()
    labels = parent_nodes + treemap_df["SubCategory Topic"].tolist()
    parents = [""] * len(parent_nodes) + treemap_df["Parent Category Topic"].tolist()
    values = (
        [
            treemap_df[treemap_df["Parent Category Topic"] == p]["n"].sum()
            for p in parent_nodes
        ]
        + treemap_df["n"].tolist()
    )
    colors = (
        [
            treemap_df[treemap_df["Parent Category Topic"] == p]["err"].mean()
            for p in parent_nodes
        ]
        + treemap_df["err"].tolist()
    )

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            marker=dict(colors=colors, colorscale=[[0, SECONDARY], [1, PRIMARY]]),
            hovertemplate="<b>%{label}</b><br>Records: %{value}<br>Avg Confusion: %{color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(
        title="Volume (Size) & Avg Confusion (Color) by Sub-Topic",
        template="simple_white",
        font=dict(family=FONT_FAMILY, color="black"),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Small yet dark rectangles reveal niche but highly confusing subtopics like garnishments.")

# -----------------------------------------------------------------------------
# Chart 3 – Agent Workload vs Error Scatter
# -----------------------------------------------------------------------------
with st.expander("3 · Agent Workload vs Error"):
    agent_df = (
        data.groupby("Agent_ID")
        .agg(queries=("Agent_ID", "size"), mean_err=("Parent Error Similarity Score", "mean"))
        .reset_index()
    )
    fig = px.scatter(
        agent_df,
        x="queries",
        y="mean_err",
        text="Agent_ID",
        color_discrete_sequence=[PRIMARY],
        labels={"queries": "Total Queries", "mean_err": "Avg Error Score"},
        title="Agent Query Load vs Average Error Score",
    )
    fig.update_traces(textfont=dict(color="black", family=FONT_FAMILY))
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Agents in upper-right quadrant (high load, high error) may need coaching or workload balance.")

# Extra control: minimum queries filter
with st.expander("3A · Filter Agents by Minimum Load"):
    min_queries = st.slider("Minimum handled queries", 1, int(agent_df["queries"].max()), 5, key="min_agent_q")
    st.dataframe(agent_df[agent_df["queries"] >= min_queries])
    add_insight("Table highlights agents above selected workload filter.")

# -----------------------------------------------------------------------------
# Chart 4 – Temporal Error Heat-map
# -----------------------------------------------------------------------------
with st.expander("4 · Temporal Error Heat-map"):
    heat = (
        data.groupby(["day_of_week", "hour_of_day"])
        .agg(mean_err=("Parent Error Similarity Score", "mean"))
        .reset_index()
    )
    fig = px.density_heatmap(
        heat,
        x="hour_of_day",
        y="day_of_week",
        z="mean_err",
        color_continuous_scale=[[0, SECONDARY], [1, PRIMARY]],
        title="Average Error by Hour & Day of Week",
    )
    fig.update_layout(
        yaxis=dict(
            categoryorder="array",
            categoryarray=[
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ],
        ),
        template="simple_white",
        font=dict(family=FONT_FAMILY, color="black"),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Weekend late-night slots glow yellow → potential understaffing or resource bottlenecks.")

# -----------------------------------------------------------------------------
# Chart 5 – Conversation Length vs Error Correlation
# -----------------------------------------------------------------------------
with st.expander("5 · Length vs Error"):
    scatter_kwargs = dict(
        x="conversation_length",
        y="Parent Error Similarity Score",
        color_discrete_sequence=[PRIMARY],
        labels={"conversation_length": "Length (s)", "Parent Error Similarity Score": "Error Score"},
        title="Conversation Duration vs Error Score",
    )
    if _HAS_STATSMODELS:
        scatter_kwargs["trendline"] = "ols"
    else:
        st.info(
            "Install `statsmodels` for regression trendlines (pip install statsmodels). Showing scatter only."
        )

    fig = px.scatter(data, **scatter_kwargs)
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Positive slope implies longer conversations correlate with confusion – monitor >60 s calls.")

# Optional sub-filter by Parent Category
with st.expander("5A · Length vs Error (Category Filter)"):
    cat_select = st.multiselect("Select Categories", parent_options, key="len_cat")
    sub_df = data if not cat_select else data[data["Parent Category Topic"].isin(cat_select)]
    fig_sub = px.scatter(
        sub_df,
        x="conversation_length",
        y="Parent Error Similarity Score",
        color="Parent Category Topic",
        color_discrete_sequence=[PRIMARY],
        title="Duration vs Error by Selected Categories",
    )
    fig_sub.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig_sub, use_container_width=True)
    add_insight("Different slope patterns across categories point to domain-specific complexity.")

# -----------------------------------------------------------------------------
# Chart 6 – Positive-Feedback Rate by Category
# -----------------------------------------------------------------------------
with st.expander("6 · Sentiment by Category"):
    sent = (
        data.groupby(["Parent Category Topic", "Feedback"]).size().unstack(fill_value=0).reset_index()
    )
    if "positive" not in sent.columns:
        sent["positive"] = 0
    sent["positive_rate"] = sent["positive"] / sent.drop(columns=["Parent Category Topic"]).sum(axis=1)
    fig = px.bar(
        sent,
        x="Parent Category Topic",
        y="positive_rate",
        color_discrete_sequence=[PRIMARY],
        labels={"positive_rate": "Positive Feedback %"},
        title="Positive Feedback Ratio per Category",
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        xaxis_tickangle=-45,
        template="simple_white",
        font=dict(family=FONT_FAMILY, color="black"),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Enrollment & Benefits lags peers with a lower positivity ratio – investigate root causes.")

# -----------------------------------------------------------------------------
# Chart 7 – Duplicate Query Frequency
# -----------------------------------------------------------------------------
with st.expander("7 · Top Duplicate Queries"):
    dup = (
        data.groupby("Knowledge_Answer").size().reset_index(name="count").sort_values("count", ascending=False).head(15)
    )
    fig = px.bar(
        dup,
        y="Knowledge_Answer",
        x="count",
        orientation="h",
        color_discrete_sequence=[PRIMARY],
        title="Most Repeated Knowledge Answers",
    )
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("High-volume duplicates like garnishment status may merit self-serve FAQs.")

# Add knob for top N duplicates
with st.expander("7A · Adjust Top N Duplicates"):
    top_n = st.slider("Show top N phrases", 5, 30, 10, key="dup_top")
    dup_n = dup.head(top_n)
    st.bar_chart(dup_n.set_index("Knowledge_Answer"), color=PRIMARY)
    add_insight("Tune N to inspect narrower or broader duplicate sets.")

# -----------------------------------------------------------------------------
# Chart 8 – Low-Confidence Outlier Tracker
# -----------------------------------------------------------------------------
with st.expander("8 · Low-Confidence Queries (<0.30)"):
    low_conf = data[
        (data["Parent Category Similarity_Score"] < 0.30)
        | (data["SubCategory Similarity_Score"] < 0.30)
    ]
    st.dataframe(
        low_conf[
            [
                "Timestamp",
                "Agent_ID",
                "Parent Category Topic",
                "SubCategory Topic",
                "Parent Category Similarity_Score",
                "SubCategory Similarity_Score",
            ]
        ]
    )
    add_insight("Rows below 0.30 similarity feed a manual review queue for taxonomy tuning.")

# -----------------------------------------------------------------------------
# Chart 9 – Sub-Topic ↔ Parent Mismatch Sunburst
# -----------------------------------------------------------------------------
with st.expander("9 · Mismatched Parent/Sub-Topic"):
    # Build naive mapping: most-common parent per sub-topic
    mapping = (
        data.groupby(["SubCategory Topic", "Parent Category Topic"]).size().reset_index(name="n")
    )
    expected_parent = (
        mapping.sort_values("n", ascending=False)
        .groupby("SubCategory Topic")
        .first()["Parent Category Topic"]
    )
    data["Expected Parent"] = data["SubCategory Topic"].map(expected_parent)
    mism = data[data["Expected Parent"] != data["Parent Category Topic"]]

    if mism.empty:
        st.info("No mismatches detected with current filter set.")
    else:
        mismatch_counts = (
            mism.groupby(["Expected Parent", "Parent Category Topic"]).size().reset_index(name="n")
        )

        # Build Sunburst manually with go to bypass pandas/narwhals bug seen in px.sunburst
        root_nodes = mismatch_counts["Expected Parent"].unique().tolist()
        root_vals = (
            mismatch_counts.groupby("Expected Parent")["n"].sum().reindex(root_nodes).tolist()
        )

        labels = root_nodes + mismatch_counts["Parent Category Topic"].tolist()
        parents = [""] * len(root_nodes) + mismatch_counts["Expected Parent"].tolist()
        values = root_vals + mismatch_counts["n"].tolist()

        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                marker=dict(colors=values, colorscale=[[0, PRIMARY], [1, SECONDARY]]),
                hovertemplate="<b>%{label}</b><br>Records: %{value}<extra></extra>",
            )
        )
        fig.update_layout(
            title="Sub-Topic vs Assigned Parent Mismatches",
            template="simple_white",
            font=dict(family=FONT_FAMILY, color="black"),
        )
        st.plotly_chart(fig, use_container_width=True)
    add_insight("Parent/Sub-topic mismatches reveal taxonomy drift – correct mapping or model labels.")

# -----------------------------------------------------------------------------
# Chart 10 – Knowledge Article Impact
# -----------------------------------------------------------------------------
with st.expander("10 · Knowledge Article Impact"):
    data["Has_Knowledge"] = np.where(data["Knowledge"].str.strip() != "-", "With Article", "No Article")
    fig = px.box(
        data,
        x="Has_Knowledge",
        y="Parent Error Similarity Score",
        color="Has_Knowledge",
        color_discrete_map={"With Article": PRIMARY, "No Article": SECONDARY},
        title="Error Score Distribution vs Knowledge Article Availability",
    )
    fig.update_layout(template="simple_white", showlegend=False, font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Queries lacking documentation trend higher in error score – bolster knowledge base.")

# -----------------------------------------------------------------------------
# Chart 11 – Error Similarity Trend Over Time
# -----------------------------------------------------------------------------
with st.expander("11 · Error Trend Over Time"):
    time_df = (
        data.set_index("Timestamp")
        .resample("D")
        ["Parent Error Similarity Score"]
        .mean()
        .dropna()
        .reset_index()
    )
    fig = px.line(
        time_df,
        x="Timestamp",
        y="Parent Error Similarity Score",
        markers=True,
        color_discrete_sequence=[PRIMARY],
        title="Daily Average Parent Error Similarity Score",
    )
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Recent upward or downward trends spotlight policy roll-outs or training wins.")

# -----------------------------------------------------------------------------
# Chart 12 – Error Score Distribution by Category (NEW)
# -----------------------------------------------------------------------------
with st.expander("12 · Error Distribution by Category"):
    box_df = data[["Parent Category Topic", "Parent Error Similarity Score"]].dropna()
    fig = px.box(
        box_df,
        x="Parent Category Topic",
        y="Parent Error Similarity Score",
        points="all",
        color_discrete_sequence=[PRIMARY],
        title="Distribution of Error Scores per Category",
    )
    fig.update_layout(template="simple_white", font=dict(family=FONT_FAMILY, color="black"))
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Wide inter-quartile ranges signal inconsistent handling within categories.")

# -----------------------------------------------------------------------------
# Chart 13 – Hourly Volume & Error Overlay (NEW)
# -----------------------------------------------------------------------------
with st.expander("13 · Hourly Volume and Error"):
    hour_df = (
        data.groupby("hour_of_day").agg(
            volume=("hour_of_day", "size"),
            mean_err=("Parent Error Similarity Score", "mean"),
        ).reset_index()
    )
    fig = go.Figure()
    fig.add_trace(primary_bar(hour_df["hour_of_day"], hour_df["volume"], name="Volume"))
    fig.add_trace(secondary_line(hour_df["hour_of_day"], hour_df["mean_err"], name="Avg Error"))
    fig.update_layout(
        title="Volume & Error by Hour of Day",
        yaxis_title="Volume",
        yaxis2=dict(title="Avg Error", overlaying="y", side="right"),
        template="simple_white",
        font=dict(family=FONT_FAMILY, color="black"),
    )
    st.plotly_chart(fig, use_container_width=True)
    add_insight("Pinpoint under-performance windows to optimise staffing schedules.")

# -----------------------------------------------------------------------------
# Footer / Download filtered data
# -----------------------------------------------------------------------------
st.markdown("---")
st.download_button(
    label="Download filtered data as CSV",
    data=data.to_csv(index=False).encode("utf-8"),
    file_name="filtered_hr_data.csv",
    mime="text/csv",
)

st.caption("© 2025 KP Analytics • Built with Streamlit & Plotly by Agentic AI") 