import joblib, pandas as pd, os

DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data")
FEATURES = ["sessions_30d","avg_session_minutes_30d","tickets_30d","days_since_last_login","account_age_days"]

model = joblib.load(os.path.join(DATAPATH, "churn_rf.joblib"))
df = pd.read_csv(os.path.join(DATAPATH, "features_table.csv"))
X = df[FEATURES].fillna(0)
df["churn_prob"] = model.predict_proba(X)[:, 1]
df.to_csv(os.path.join(DATAPATH, "scored_batch.csv"), index=False)
print("✅ Saved data/scored_batch.csv")
