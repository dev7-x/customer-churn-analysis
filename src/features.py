# src/features.py
import pandas as pd
import os

DATAPATH = os.path.join(os.path.dirname(__file__), "..", "data")

# Load data
events = pd.read_csv(os.path.join(DATAPATH, 'events.csv'), parse_dates=['event_date'])
users = pd.read_csv(os.path.join(DATAPATH, 'users.csv'), parse_dates=['signup_date'])
support = pd.read_csv(os.path.join(DATAPATH, 'support.csv'), parse_dates=['ticket_date'])

# Labels (churn flag) — optional
labels_path = os.path.join(DATAPATH, 'labels.csv')
labels = pd.read_csv(labels_path)[['user_id', 'churn_label']] if os.path.exists(labels_path) else pd.DataFrame(columns=["user_id", "churn_label"])

# Reference date (latest event date)
max_date = events['event_date'].max()
cutoff = max_date - pd.Timedelta(days=30)

# Aggregate last 30 days activity
agg30 = (
    events[events['event_date'] >= cutoff]
    .groupby('user_id')
    .agg({
        'sessions': 'sum',
        'avg_session_minutes': 'mean'
    })
    .reset_index()
    .rename(columns={'sessions': 'sessions_30d', 'avg_session_minutes': 'avg_session_minutes_30d'})
)

# Support tickets in last 30 days
tickets30 = (
    support[support['ticket_date'] >= cutoff]
    .groupby('user_id')
    .size()
    .reset_index(name='tickets_30d')
)

# Last activity date
last_activity = (
    events.groupby('user_id')
    .event_date.max()
    .reset_index()
    .rename(columns={'event_date': 'last_activity'})
)

# Merge everything
feat = (
    users
    .merge(agg30, on='user_id', how='left')
    .merge(tickets30, on='user_id', how='left')
    .merge(last_activity, on='user_id', how='left')
    .merge(labels, on='user_id', how='left')   # ensure churn_label is included
    .fillna(0)
)

# Ensure datetime conversion
feat['signup_date'] = pd.to_datetime(feat['signup_date'], errors='coerce')
feat['last_activity'] = pd.to_datetime(feat['last_activity'], errors='coerce')

# Compute derived features
feat['days_since_last_login'] = (max_date - feat['last_activity']).dt.days.fillna(-1)
feat['account_age_days'] = (max_date - feat['signup_date']).dt.days.fillna(-1)

# Ensure churn_label column exists
if 'churn_label' not in feat.columns:
    feat['churn_label'] = 0  # default label if missing

# Save features
out = os.path.join(DATAPATH, "features_table.csv")
feat.to_csv(out, index=False)
print("✅ Saved features to", out, "with churn_label column")
