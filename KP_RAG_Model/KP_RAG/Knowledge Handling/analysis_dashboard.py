import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

# --------------------- Helper Functions ---------------------
def compute_baseline_metrics(df):
    metrics = {}
    metrics['avg_top_score'] = df['top_score'].mean()
    metrics['median_top_score'] = df['top_score'].median()
    metrics['std_top_score'] = df['top_score'].std()
    metrics['percent_matched'] = df['matched_score_threshold'].mean() * 100
    metrics['answer_rate'] = df['answer_provided'].mean() * 100
    metrics['percent_no_results'] = (df['retrieved_article_ids'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0) == 0).mean() * 100
    return metrics


def show_overview(df):
    st.header("📈 Overview")
    metrics = compute_baseline_metrics(df)
    cols = st.columns(5)
    cols[0].metric("Avg Top Score", f"{metrics['avg_top_score']:.2f}")
    cols[1].metric("Median Top Score", f"{metrics['median_top_score']:.2f}")
    cols[2].metric("Matched ≥ Threshold", f"{metrics['percent_matched']:.1f}%")
    cols[3].metric("Answer Provided", f"{metrics['answer_rate']:.1f}%")
    cols[4].metric("No Results", f"{metrics['percent_no_results']:.1f}%")
    
    st.subheader("Top Score Distribution")
    hist = alt.Chart(df).mark_bar().encode(
        alt.X("top_score", bin=alt.Bin(maxbins=30)),
        y='count()'
    ).properties(width=600, height=300)
    st.altair_chart(hist, use_container_width=True)


def show_baseline(df):
    st.header("🔍 Baseline Diagnosis")
    # Lowest top scores
    st.subheader("Queries with Lowest Top Scores")
    low = df.nsmallest(10, 'top_score')[['query_id', 'user_query', 'top_score']]
    st.table(low)

    # No results
    st.subheader("Queries with No Results")
    no_res = df[df['retrieved_article_ids'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0) == 0]
    st.table(no_res[['query_id', 'user_query']].head(10))

    # Mode of top match segment type & article
    st.subheader("Most Common Match Types")
    seg_mode = df['top_match_segment_type'].mode()[0]
    art_mode = df['top_match_article_id'].mode()[0]
    st.write(f"Top segment type: **{seg_mode}**")
    st.write(f"Most matched article ID: **{art_mode}**")

    # Precision heatmap per article
    st.subheader("Precision Heatmap (% matched ≥ threshold by article)")
    pivot = (df.assign(matched=df['matched_score_threshold'].astype(int))
               .groupby('top_match_article_id')['matched']
               .mean()
               .reset_index())
    pivot.columns = ['article_id', 'precision_rate']
    heat = alt.Chart(pivot).mark_rect().encode(
        x=alt.X('article_id:N', sort='-y'),
        y=alt.value(0),
        color='precision_rate:Q'
    ).properties(height=100, width=600)
    st.altair_chart(heat)


def show_comparative(df):
    st.header("⚖️ Comparative Diagnosis")
    # Query quality tally
    st.subheader("Query Quality Distribution")
    qdist = df['query_quality'].value_counts().reset_index()
    qdist.columns = ['query_quality', 'count']
    chart_q = alt.Chart(qdist).mark_bar().encode(
        x='query_quality', y='count'
    )
    st.altair_chart(chart_q, use_container_width=True)

    # Matrix of query_quality vs diagnosis
    st.subheader("Query Quality vs Diagnosis")
    matrix = (df.groupby(['query_quality', 'query_diagnosis'])
                .size().reset_index(name='count'))
    heat = alt.Chart(matrix).mark_rect().encode(
        x='query_quality',
        y='query_diagnosis',
        color='count'
    ).properties(width=600, height=300)
    st.altair_chart(heat)

    # Top failing articles
    st.subheader("Top Failing Articles")
    art_fail = (df[df['answer_label'] == 'NO']
                .groupby('top_match_article_id')
                .size().reset_index(name='fail_count')
                .sort_values('fail_count', ascending=False)
                .head(10))
    st.table(art_fail)

    # Top failing queries
    st.subheader("Top Failing Queries (Lowest Answer Score)")
    q_fail = df[df['answer_label'] == 'NO'].nsmallest(10, 'answer_score')
    st.table(q_fail[['query_id', 'user_query', 'answer_score', 'answer_label']])

    # Cross-tab by functional area and subject
    if 'functional_area' in df.columns and 'functional_subject' in df.columns:
        st.subheader("Functional Area × Subject Failure Rates")
        ctab = pd.crosstab(df['functional_area'], df['functional_subject'], values=df['answer_label']=='NO', aggfunc='mean').fillna(0)
        st.dataframe(ctab)


def show_dynamic(df):
    st.header("⚙️ Dynamic / Exploratory")
    st.markdown("Use the filters below to slice the data interactively.")
    # Filters
    min_score, max_score = st.slider("Filter by top_score range", 0.0, 1.0, (0.0, 1.0))
    f_quality = st.multiselect("Filter Query Quality", df['query_quality'].unique(), default=df['query_quality'].unique())
    f_diag = st.multiselect("Filter Diagnosis", df['query_diagnosis'].unique(), default=df['query_diagnosis'].unique())
    dff = df[(df['top_score'].between(min_score, max_score)) &
             (df['query_quality'].isin(f_quality)) &
             (df['query_diagnosis'].isin(f_diag))]
    st.write(f"Showing {len(dff)} of {len(df)} queries")
    st.dataframe(dff)


def show_article_view(df):
    st.header("📄 Article-Level Analysis")
    art_ids = df['top_match_article_id'].unique()
    sel = st.selectbox("Select Article ID", art_ids)
    sub = df[df['top_match_article_id'] == sel]
    st.subheader(f"Metrics for Article {sel}")
    m = compute_baseline_metrics(sub)
    st.metric("Avg Top Score", f"{m['avg_top_score']:.2f}")
    st.metric("Avg Answer Score", f"{sub['answer_score'].mean():.2f}")
    st.metric("Answer Provided Rate", f"{sub['answer_provided'].mean()*100:.1f}%")
    st.subheader("Query Breakdown for this Article")
    st.dataframe(sub[['query_id','user_query','top_score','answer_score','answer_label','query_quality','query_diagnosis']])


def show_category_view(df):
    if 'predicted_category' not in df.columns or 'predicted_subcategory' not in df.columns:
        st.info("No taxonomy columns found. Please include 'predicted_category' and 'predicted_subcategory'.")
        return
    st.header("🗂 Category-Level Analysis")
    cats = df['predicted_category'].unique()
    cat_sel = st.selectbox("Select Category", cats)
    subcats = df[df['predicted_category'] == cat_sel]['predicted_subcategory'].unique()
    sub_sel = st.selectbox("Select Subcategory", subcats)
    sub = df[(df['predicted_category'] == cat_sel) & (df['predicted_subcategory'] == sub_sel)]
    st.subheader(f"Metrics for {cat_sel} / {sub_sel}")
    m = compute_baseline_metrics(sub)
    st.metric("Avg Top Score", f"{m['avg_top_score']:.2f}")
    st.metric("Avg Answer Score", f"{sub['answer_score'].mean():.2f}")
    st.metric("Answer Provided Rate", f"{sub['answer_provided'].mean()*100:.1f}%")
    st.subheader("Queries in this Bucket")
    st.dataframe(sub[['query_id','user_query','top_score','answer_score','answer_label','query_diagnosis']])

# --------------------- Main ---------------------
def main():
    st.set_page_config(page_title="KB QA Diagnostics", layout="wide")
    st.title("Knowledge Base QA Diagnostics Dashboard")

    uploaded_file = st.file_uploader("Upload your `query_analysis_results.csv`", type=["csv"])
    if not uploaded_file:
        st.info("Please upload a CSV file to begin analysis.")
        return
    df = pd.read_csv(uploaded_file)

    tabs = st.tabs(["Overview","Baseline","Comparative","Dynamic","Article","Category View"])
    with tabs[0]: show_overview(df)
    with tabs[1]: show_baseline(df)
    with tabs[2]: show_comparative(df)
    with tabs[3]: show_dynamic(df)
    with tabs[4]: show_article_view(df)
    with tabs[5]: show_category_view(df)

if __name__ == "__main__":
    main()
