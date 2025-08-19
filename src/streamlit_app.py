# src/streamlit_app.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px

DATA = os.path.join(os.path.dirname(__file__),"..","data")
st.title("Churn Demo - At-risk Accounts")

@st.cache_data
def load():
    feats = pd.read_csv(os.path.join(DATA,'features_table.csv'))
    scores_path = os.path.join(DATA,'scored_batch.csv')
    scores = pd.read_csv(scores_path) if os.path.exists(scores_path) else None
    fi = pd.read_csv(os.path.join(DATA,'feature_importances.csv')) if os.path.exists(os.path.join(DATA,'feature_importances.csv')) else None
    return feats, scores, fi

feats, scores, fi = load()

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
