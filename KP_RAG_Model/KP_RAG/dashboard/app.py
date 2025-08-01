import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from io import BytesIO
from PIL import Image
from typing import List
from pathlib import Path
import base64

# ------------------ CONFIGURATION ------------------ #
# Color palette constants
NAVY = "#001f3f"
YELLOW = "#FFD700"
WHITE = "#FFFFFF"

# Load local font file and embed as base64
FONT_NAME = "Instrument Sans"
FONT_PATH = "InstrumentSans.ttf"
font_css = ""
if Path(FONT_PATH).exists():
    try:
        font_data = Path(FONT_PATH).read_bytes()
        encoded = base64.b64encode(font_data).decode()
        font_css = f"""
        @font-face {{
            font-family: '{FONT_NAME}';
            src: url(data:font/ttf;base64,{encoded}) format('truetype');
            font-weight: 100 900;
            font-style: normal;
        }}
        """
    except Exception:
        font_css = ""

st.set_page_config(page_title="Knowledge Base Analytics Dashboard", layout="wide")

# Inject CSS (font + yellow accent overrides)
custom_css = f"""
<style>
{font_css}
:root {{
    --primary-color: {YELLOW};
    --secondary-color: {YELLOW};
    --text-color: {NAVY};
}}
html, body, [class*='css'] {{
    background-color: {WHITE};
    color: var(--text-color);
    font-family: '{FONT_NAME}', sans-serif;
}}
/* Headings */
h1, h2, h3, h4, h5, h6 {{ color: var(--text-color); }}
/* Buttons */
.stButton>button {{ background-color: var(--primary-color); color: {NAVY}; border:none; }}
.stButton>button:hover {{ background-color: var(--secondary-color); opacity:0.9; }}
/* Select & multiselect chips */
.stMultiSelect div[data-baseweb='tag'] {{ background-color: var(--primary-color)!important; color:{NAVY}!important; }}
/* Slider thumb & track */
div[data-testid=\"stSlider\"] span {{ background-color: var(--primary-color)!important; }}
/* Generic fix: convert default Streamlit red (#ff4b4b) */
*[style*='#ff4b4b'] {{ background-color: var(--primary-color)!important; border-color: var(--primary-color)!important; }}
</style>
"""

st.markdown(custom_css, unsafe_allow_html=True)

# Helper: consistently style Plotly figures

def style_fig(fig, *, title: str = ""):
    fig.update_layout(
        title=title,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font_color=NAVY,
        title_font_color=NAVY,
    )
    fig.update_xaxes(color=NAVY, title_font_color=NAVY, tickfont=dict(color=NAVY))
    fig.update_yaxes(color=NAVY, title_font_color=NAVY, tickfont=dict(color=NAVY))
    return fig

# ------------------ DATA LOADING ------------------ #
@st.cache_data(show_spinner="Loading data...")
def load_data(path: str) -> pd.DataFrame:
    """Read CSV and perform basic type coercion."""
    df = pd.read_csv(path, low_memory=False)

    # Ensure expected boolean fields parsed correctly
    bool_cols = ["answer_provided", "handled", "matched_score_threshold"]
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().map({"true": True, "false": False})

    num_cols = [
        "top_score",
        "avg_score",
        "answer_score",
        "answer_subtopic_score",
        "query_subtopic_score",
        "retrieved_context_word_count",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df

DATA_PATH = "query_analysis_results.csv"
df = load_data(DATA_PATH)

# Attempt to add article title mapping
try:
    if Path("article_titles.csv").exists():
        art_df = pd.read_csv("article_titles.csv")
        if art_df.shape[1] >= 2:
            article_title_map = dict(zip(art_df.iloc[:,0], art_df.iloc[:,1]))
        else:
            article_title_map = {}
    else:
        article_title_map = {}
except Exception:
    article_title_map = {}

# ------------------ SIDEBAR FILTERS ------------------ #
st.sidebar.header("🔍 Filters")

# Dynamic multiselects / sliders
answer_labels = df["answer_label"].dropna().unique().tolist() if "answer_label" in df.columns else []
selected_answers = st.sidebar.multiselect("Answer Label", options=answer_labels, default=answer_labels)

query_qualities = df["query_quality"].dropna().unique().tolist() if "query_quality" in df.columns else []
selected_qualities = st.sidebar.multiselect("Query Quality", options=query_qualities, default=query_qualities)

topics = df["preassigned_topic_label"].dropna().unique().tolist() if "preassigned_topic_label" in df.columns else []
selected_topics = st.sidebar.multiselect("Topic", options=topics, default=topics)  # default all

# Numeric range slider for answer_score
if "answer_score" in df.columns:
    min_score, max_score = float(df["answer_score"].min()), float(df["answer_score"].max())
    score_range = st.sidebar.slider("Answer Score Range", min_value=min_score, max_value=max_score, value=(min_score, max_score))
else:
    score_range = (None, None)

# Apply filters
filtered_df = df.copy()
if answer_labels:
    filtered_df = filtered_df[filtered_df["answer_label"].isin(selected_answers)]
if query_qualities:
    filtered_df = filtered_df[filtered_df["query_quality"].isin(selected_qualities)]
if topics:
    filtered_df = filtered_df[filtered_df["preassigned_topic_label"].isin(selected_topics)]
if score_range[0] is not None:
    filtered_df = filtered_df[(filtered_df["answer_score"] >= score_range[0]) & (filtered_df["answer_score"] <= score_range[1])]

# Replace original df reference for downstream charts
df = filtered_df

st.sidebar.write(f"Showing **{len(df):,}** records after filtering")

# Toggle raw data view
if st.sidebar.checkbox("Show raw data"):
    st.sidebar.dataframe(df.head(1000))

# Provide download of filtered data for further analysis
csv_bytes = df.to_csv(index=False).encode("utf-8")
st.sidebar.download_button(
    label="Download filtered rows as CSV",
    data=csv_bytes,
    file_name="filtered_query_analysis.csv",
    mime="text/csv",
)

# Helper for navy + yellow palette

def highlight_color(idx: int, highlight_indices: list[int]) -> str:
    return YELLOW if idx in highlight_indices else NAVY

# ------------------ LAYOUT ------------------ #
st.title("Knowledge Base Coverage Dashboard")

# Chart 1: Knowledge Base Coverage Overview (Pie)
with st.container():
    st.subheader("1 Knowledge Base Coverage Overview")
    if "handled" in df.columns:
        resolved = df["handled"].sum()
        unresolved = len(df) - resolved
    elif "answer_label" in df.columns:
        resolved = (df["answer_label"].str.upper() == "YES").sum()
        unresolved = len(df) - resolved
    else:
        st.warning("Could not locate 'handled' or 'answer_label' columns to compute resolution rate.")
        resolved = unresolved = 0

    pie_df = pd.DataFrame({
        "Status": ["Resolved", "Unresolved"],
        "Count": [resolved, unresolved],
    })
    colors = [YELLOW, NAVY]
    fig1 = px.pie(pie_df, values="Count", names="Status", color="Status", color_discrete_map={"Resolved": YELLOW, "Unresolved": NAVY})
    fig1 = style_fig(fig1, title="Knowledge Base Coverage Overview")
    fig1.update_traces(textinfo="percent+label", pull=[0.05, 0])
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown(
        f"**Insight:** The knowledge base currently resolves **{resolved / max(len(df),1):.0%}** of incoming queries. { 'Great coverage!' if resolved/len(df) > 0.8 else 'Consider enriching content to improve coverage.' }"
    )

# Chart 2: Query Quality Distribution (Bar)
with st.container():
    st.subheader("2 Query Quality Distribution")
    if "query_quality" not in df.columns:
        st.warning("Column 'query_quality' missing from data.")
    else:
        quality_counts = df["query_quality"].fillna("Unknown").value_counts().reset_index()
        quality_counts.columns = ["Quality", "Count"]
        most_common_idx = quality_counts["Count"].idxmax()
        colors = [highlight_color(i, [most_common_idx]) for i in quality_counts.index]
        fig2 = px.bar(quality_counts, x="Quality", y="Count", color_discrete_sequence=colors)
        fig2 = style_fig(fig2, title="Query Quality Distribution")
        st.plotly_chart(fig2, use_container_width=True)
        top_quality = quality_counts.loc[most_common_idx, "Quality"]
        st.markdown(f"**Insight:** Most queries are classified as **{top_quality}**, suggesting a focus on { 'maintaining clarity' if top_quality=='well-structured' else 'guiding users to craft clearer questions' }.")

# Chart 3: Diagnosis Breakdown (Horizontal Bar)
with st.container():
    st.subheader("3 Diagnosis Breakdown")
    diag_col = "query_diagnosis" if "query_diagnosis" in df.columns else None
    if not diag_col:
        st.warning("Column for diagnosis not found.")
    else:
        diag_counts = df[diag_col].fillna("Unknown").value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Count"]
        top_idx = diag_counts["Count"].idxmax()
        colors = [highlight_color(i, [top_idx]) for i in diag_counts.index]
        fig3 = px.bar(diag_counts, y="Diagnosis", x="Count", orientation="h", color_discrete_sequence=colors)
        fig3 = style_fig(fig3, title="Diagnosis Breakdown")
        st.plotly_chart(fig3, use_container_width=True)
        top_reason = diag_counts.loc[top_idx, "Diagnosis"]
        st.markdown(f"**Insight:** The primary failure reason is **{top_reason}**; prioritize addressing this root cause.")

# Chart 4: Answer Score Analysis (Histogram)
with st.container():
    st.subheader("4 Answer Score Distribution")
    if "answer_score" not in df.columns:
        st.warning("Column 'answer_score' missing.")
    else:
        # bins slider
        bin_count = st.slider("Select histogram bins", min_value=10, max_value=100, value=30, step=5, key="bins_hist")
        fig4 = px.histogram(df, x="answer_score", nbins=bin_count, color_discrete_sequence=[NAVY])
        # Highlight peak bin in yellow by overlaying
        counts, bins = np.histogram(df["answer_score"].dropna(), bins=bin_count)
        peak_bin_idx = counts.argmax()
        peak_start, peak_end = bins[peak_bin_idx], bins[peak_bin_idx + 1]
        fig4.add_vrect(
            x0=peak_start, x1=peak_end, fillcolor=YELLOW, opacity=0.3, line_width=0,
        )
        fig4 = style_fig(fig4, title="Answer Score Distribution")
        st.plotly_chart(fig4, use_container_width=True)
        st.markdown("**Insight:** The most common answer score range lies between **{:.2f} – {:.2f}**. Use this to calibrate evaluation thresholds.".format(peak_start, peak_end))

# Chart 5: Topics with Highest Failure Rates
with st.container():
    st.subheader("5 Topics with Highest Failure Rates")
    topic_col = "preassigned_topic_label" if "preassigned_topic_label" in df.columns else "answer_parent_topic_label" if "answer_parent_topic_label" in df.columns else None
    if not topic_col or "handled" not in df.columns:
        st.warning("Necessary columns for topic failure analysis not found.")
    else:
        failures = df[~df["handled"].astype(bool)]
        topic_fail_counts = failures[topic_col].fillna("Unknown").value_counts().head(10).reset_index()
        topic_fail_counts.columns = ["Topic", "Failures"]
        colors = [highlight_color(i, [0,1,2]) for i in topic_fail_counts.index]
        fig5 = px.bar(topic_fail_counts, y="Topic", x="Failures", orientation="h", color_discrete_sequence=colors)
        fig5 = style_fig(fig5, title="Topics with Highest Failure Rates")
        st.plotly_chart(fig5, use_container_width=True)
        st.markdown("**Insight:** Addressing the highlighted topics could reduce unresolved queries by **{:,.0f}**.".format(topic_fail_counts["Failures"].head(3).sum()))

# Chart 6: Sub-Topic Deep Dive
with st.container():
    st.subheader("6 Sub-Topic Failure Volume")
    subtopic_col = "preassigned_subtopic_label" if "preassigned_subtopic_label" in df.columns else "answer_subtopic_label" if "answer_subtopic_label" in df.columns else None
    if not subtopic_col or "handled" not in df.columns:
        st.warning("Subtopic columns or handled flag missing.")
    else:
        failures_local = df[~df["handled"].astype(bool)] if "handled" in df.columns else df
        sub_fail = failures_local[subtopic_col].fillna("Unknown").value_counts().head(10).reset_index()
        sub_fail.columns = ["Subtopic", "Failures"]
        colors = [highlight_color(i, [0,1,2]) for i in sub_fail.index]
        fig6 = px.bar(sub_fail, y="Subtopic", x="Failures", orientation="h", color_discrete_sequence=colors)
        fig6 = style_fig(fig6, title="Sub-Topic Failure Volume")
        st.plotly_chart(fig6, use_container_width=True)
        st.markdown("**Insight:** Concentrate curation efforts on these sub-topics to improve resolution rates.")

# Chart 7: Context Word Count Correlation
with st.container():
    st.subheader("7 Context Length vs Answer Score")
    if "retrieved_context_word_count" not in df.columns or "answer_score" not in df.columns:
        st.warning("Columns for correlation scatter missing.")
    else:
        x = df["retrieved_context_word_count"]
        y = df["answer_score"]
        fig7 = px.scatter(df, x=x, y=y, opacity=0.6, color_discrete_sequence=[NAVY])
        # Identify outliers (e.g., points beyond 95th percentile of word count)
        wc_threshold = np.percentile(x.dropna(), 95)
        outliers = df[x > wc_threshold]
        fig7.add_trace(go.Scatter(
            x=outliers["retrieved_context_word_count"],
            y=outliers["answer_score"],
            mode="markers",
            marker=dict(color=YELLOW, size=8),
            name="Outliers",
        ))
        fig7 = style_fig(fig7, title="Context Length vs Answer Score")
        st.plotly_chart(fig7, use_container_width=True)
        st.markdown("**Insight:** Most answers cluster below {} words. Extremely long contexts may not guarantee higher accuracy.".format(int(wc_threshold)))

# Chart 8: Top Matched Article Utilization
with st.container():
    st.subheader("8 Top Matched Article Utilization")
    if "top_match_article_id" not in df.columns:
        st.warning("Column 'top_match_article_id' missing.")
    else:
        # Topic-specific filter inside chart
        topic_for_articles = st.selectbox("Filter by topic (optional)", options=["All"] + topics, key="topic_for_articles")
        article_df_scope = df if topic_for_articles == "All" else df[df["preassigned_topic_label"] == topic_for_articles]
        
        article_stats = article_df_scope.groupby("top_match_article_id").agg(
            total_queries=("handled", "size"),
            unresolved=("handled", lambda x: (~x.astype(bool)).sum()),
        )
        article_stats["unresolved_ratio"] = article_stats["unresolved"] / article_stats["total_queries"]
        top_articles = article_stats.sort_values("unresolved", ascending=False).head(15).reset_index()
        # Map IDs to titles if available
        top_articles["Article"] = top_articles["top_match_article_id"].map(lambda x: article_title_map.get(x, x))
        colors = [YELLOW if row.unresolved_ratio > 0.5 else NAVY for _, row in top_articles.iterrows()]
        fig8 = px.bar(top_articles, x="Article", y="unresolved", color_discrete_sequence=colors, labels={"unresolved": "Unresolved Queries"})
        fig8 = style_fig(fig8, title="Top Matched Articles with Unresolved Queries")
        st.plotly_chart(fig8, use_container_width=True)
        st.markdown("**Insight:** Articles highlighted require content review as they frequently appear but still fail to resolve user queries.")

# Chart 9: Answer Provided vs Handled Ratio
with st.container():
    st.subheader("9 Answer Provided vs Handled")
    if "answer_provided" not in df.columns or "handled" not in df.columns:
        st.warning("Columns 'answer_provided' or 'handled' missing.")
    else:
        summary = (
            df.groupby(["answer_provided", "handled"]).size().unstack(fill_value=0)
        )
        summary = summary.rename(index={True: "Answer Provided", False: "No Answer"}, columns={True: "Handled", False: "Unresolved"})
        fig9 = go.Figure(
            data=[
                go.Bar(name="Handled", x=summary.index, y=summary["Handled"], marker_color=NAVY),
                go.Bar(name="Unresolved", x=summary.index, y=summary["Unresolved"], marker_color=YELLOW),
            ]
        )
        fig9.update_layout(barmode="stack", paper_bgcolor=WHITE, plot_bgcolor=WHITE)
        fig9 = style_fig(fig9, title="Answer Provided vs Handled")
        st.plotly_chart(fig9, use_container_width=True)
        mismatch_ratio = summary.loc["Answer Provided", "Unresolved"] / summary.loc["Answer Provided"].sum()
        st.markdown("**Insight:** {:.0%} of provided answers still leave the user unresolved.".format(mismatch_ratio))

# Chart 10: Recommendations Word Cloud
with st.container():
    st.subheader("10 KB Improvement Recommendations")
    rec_col = "kb_recommendation" if "kb_recommendation" in df.columns else None
    if not rec_col:
        st.warning("Column 'kb_recommendation' not found.")
    else:
        text = " ".join(df[rec_col].dropna().astype(str).tolist())
        if text.strip() == "":
            st.info("No recommendations available to generate word cloud.")
        else:
            wc = WordCloud(width=800, height=400, background_color=WHITE, colormap="cividis").generate(text)
            # Convert to image
            img = wc.to_image()
            buf = BytesIO()
            img.save(buf, format="PNG")
            st.image(buf.getvalue(), use_container_width=True)
            st.markdown("**Insight:** The larger words indicate more frequently suggested actions for knowledge base improvements.")

# Chart 11: Metric Correlation Heatmap
with st.container():
    st.subheader("11 Metric Correlation Heatmap")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.warning("Not enough numeric fields to compute correlation matrix.")
    else:
        corr = df[numeric_cols].corr()
        fig11 = px.imshow(corr, text_auto=True, color_continuous_scale=[[0, NAVY], [0.5, WHITE], [1, YELLOW]], aspect="auto")
        fig11 = style_fig(fig11, title="Metric Correlation Heatmap")
        st.plotly_chart(fig11, use_container_width=True)
        st.markdown("**Insight:** Strong correlations (yellow) indicate metrics that move together; prioritize decoupling if unintended.")

st.caption("Dashboard generated with Streamlit & Plotly | © 2025")

# ------------------ DRILL-DOWN SECTION ------------------ #
st.header("Topic & Subtopic Drill-Down")
if topics:
    drill_topic = st.selectbox("Choose a topic to explore its sub-topics", options=["All"] + topics)
    drill_df = df if drill_topic == "All" else df[df["preassigned_topic_label"] == drill_topic]
    if subtopic_col and len(drill_df):
        sub_counts = drill_df[subtopic_col].fillna("Unknown").value_counts().head(20).reset_index()
        sub_counts.columns = ["Subtopic", "Queries"]
        fig_sub = px.bar(sub_counts, y="Subtopic", x="Queries", orientation="h", color_discrete_sequence=[NAVY])
        fig_sub = style_fig(fig_sub)
        st.plotly_chart(fig_sub, use_container_width=True)

# ------------------ NEW CHART: Weakest Articles in Selected Scope ------------------ #
st.header("12 Weakest Articles within Scope")
scope_level = st.selectbox("Scope level", ["All", "Topic", "Subtopic"], key="weak_scope_level")
if scope_level == "Topic" and topics:
    scope_value = st.selectbox("Select Topic", options=topics, key="weak_topic")
    scope_df = df[df["preassigned_topic_label"] == scope_value]
elif scope_level == "Subtopic" and subtopic_col:
    subtopics_list = df[subtopic_col].dropna().unique().tolist()
    scope_value = st.selectbox("Select Subtopic", options=subtopics_list, key="weak_subtopic")
    scope_df = df[df[subtopic_col] == scope_value]
else:
    scope_df = df.copy()
# Quality filter inside this chart
if query_qualities:
    sel_q = st.multiselect("Query Quality filter", options=query_qualities, default=query_qualities, key="weak_quality")
    scope_df = scope_df[scope_df["query_quality"].isin(sel_q)]

if "handled" in scope_df.columns and "top_match_article_id" in scope_df.columns and len(scope_df):
    art_stats = scope_df.groupby("top_match_article_id").agg(
        total=("handled", "size"),
        unresolved=("handled", lambda x: (~x.astype(bool)).sum()),
    )
    art_stats["fail_rate"] = art_stats["unresolved"] / art_stats["total"]
    weakest = art_stats[art_stats["total"] >= 5].sort_values("fail_rate", ascending=False).head(15).reset_index()
    weakest["Article"] = weakest["top_match_article_id"].map(lambda x: article_title_map.get(x, x))
    colors_weak = [YELLOW if r.fail_rate > 0.5 else NAVY for _, r in weakest.iterrows()]
    fig_weak = px.bar(
        weakest,
        y="Article",
        x="unresolved",
        orientation="h",
        color_discrete_sequence=colors_weak,
        labels={"unresolved": "Unresolved Count"},
        hover_data={"fail_rate": ':.1%'}
    )
    fig_weak = style_fig(fig_weak, title="Weakest Articles (adjustable scope)")
    st.plotly_chart(fig_weak, use_container_width=True)
    st.markdown("Articles with highest unresolved ratios (yellow) within the selected scope and query quality need immediate attention.")
else:
    st.info("Insufficient data to compute weakest articles in the selected scope.")

# ------------------ Chart 13: Resolution Rate by Query Quality ------------------ #
with st.container():
    st.subheader("13 Resolution Rate by Query Quality")
    if "query_quality" in df.columns and "handled" in df.columns:
        qual_stats = df.groupby("query_quality").agg(
            total=("handled", "size"),
            resolved=("handled", "sum"),
        )
        qual_stats["resolution_rate"] = qual_stats["resolved"] / qual_stats["total"]
        qual_stats = qual_stats.reset_index().sort_values("resolution_rate", ascending=False)
        fig13 = px.bar(qual_stats, x="query_quality", y="resolution_rate", color_discrete_sequence=[NAVY])
        fig13 = style_fig(fig13, title="Resolution Rate by Query Quality")
        fig13.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig13, use_container_width=True)
        st.markdown("**Insight:** Target user education for query quality categories with low resolution rates.")
    else:
        st.info("Required columns missing to compute resolution rates by query quality.") 