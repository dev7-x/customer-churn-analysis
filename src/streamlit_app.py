import streamlit as st
import pandas as pd
import os
import plotly.express as px
import json

DATA = os.path.join(os.path.dirname(__file__),"..","data")
st.title("Churn Demo - At-risk Accounts")

@st.cache_data
def load():
    feats = pd.read_csv(os.path.join(DATA,'features_table.csv'))
    scores_path = os.path.join(DATA,'scored_batch.csv')
    scores = pd.read_csv(scores_path) if os.path.exists(scores_path) else None
    fi = pd.read_csv(os.path.join(DATA,'feature_importances.csv')) if os.path.exists(os.path.join(DATA,'feature_importances.csv')) else None
    metrics_path = os.path.join(DATA, 'performance_metrics.json')
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else None
    users = pd.read_csv(os.path.join(DATA, 'users.csv'))
    labels = pd.read_csv(os.path.join(DATA, 'labels.csv'))
    return feats, scores, fi, metrics, users, labels

feats, scores, fi, metrics, users, labels = load()

st.header("Model Performance")
if metrics is not None:
    st.metric("AUC", f"{metrics['auc']:.4f}")
    st.text("Classification Report:")
    st.json(metrics['classification_report'])
else:
    st.write("Run training to generate performance_metrics.json")

st.header("Data Insights")

# Churn rate by plan
churn_by_plan = pd.merge(users, labels, on='user_id')
churn_rate_by_plan = churn_by_plan.groupby('plan')['churn_label'].value_counts(normalize=True).unstack().fillna(0)
fig_plan = px.bar(churn_rate_by_plan, y=1, x=churn_rate_by_plan.index, title="Churn Rate by Plan", labels={'y':'Churn Rate', 'x':'Plan'})
st.plotly_chart(fig_plan, use_container_width=True)

# Churn rate by country
churn_rate_by_country = churn_by_plan.groupby('country')['churn_label'].value_counts(normalize=True).unstack().fillna(0)
fig_country = px.bar(churn_rate_by_country, y=1, x=churn_rate_by_country.index, title="Churn Rate by Country", labels={'y':'Churn Rate', 'x':'Country'})
st.plotly_chart(fig_country, use_container_width=True)

st.header("Feature importances")
if fi is not None:
    fig = px.bar(fi.sort_values('importance', ascending=True), x='importance', y='feature', orientation='h')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.write("Run training to generate feature_importances.csv")

st.header("Top at-risk accounts (by churn_prob)")
if scores is not None:
    top = scores.sort_values('churn_prob', ascending=False).head(50)
    st.dataframe(top[['user_id','churn_prob','sessions_30d','tickets_30d','days_since_last_login']])
    st.download_button("Download top 50", top.to_csv(index=False), file_name="top50.csv")
else:
    st.write("No scored_batch.csv found. Run scoring script or batch scoring to create it.")